"""
Webcam Taiko Input (25 FPS Optimized) -- 2 zones (Left = DON, Right = KATSU)

Controls (click the OpenCV window first):
    SPACE        : toggle sending real keystrokes (starts OFF)
    [ / ]        : lower / raise strike sensitivity threshold
    a / d        : move center vertical divider left / right
    q or ESC     : quit
"""
import os
import sys
import time
import threading
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================ CONFIG ============================
CAM_INDEX = 1          # 1 = external webcam, 0 = built-in
CAM_W, CAM_H, CAM_FPS = 480, 360, 30   # lower res = sharper, zero-lag 25-30fps processing

# 25 FPS Hit Detection (tuned for 40ms frame intervals)
DOWN_THRESH = 0.30     # downward speed needed to arm a strike (lower = more sensitive)
DECEL_RATIO = 0.60     # higher ratio fires immediately on the 1st slowdown frame
ARM_TIMEOUT = 0.35     # slightly longer window to accommodate 40ms frames
REARM_FRAC = 0.40      # faster re-arming between consecutive hits
DEBOUNCE = 0.07        # per-zone cooldown in seconds
KEY_HOLD = 0.03        # key press duration in seconds
MAX_MISSED = 2         # tolerate up to 2 blurry frames without losing the swing

ZONE_KEYS = {
    0: ("DON", "f"),
    1: ("KAT", "j"),    
}
DON_COLOR = (60, 60, 230)
KAT_COLOR = (230, 150, 40)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ====================== KEY INJECTION BACKEND ======================
try:
    import pydirectinput as _pdi
    _pdi.PAUSE = 0
    _key_down, _key_up = _pdi.keyDown, _pdi.keyUp
    BACKEND = "pydirectinput"
except Exception:
    try:
        from pynput.keyboard import Controller as _Ctl
        _kb = _Ctl()
        _key_down, _key_up = _kb.press, _kb.release
        BACKEND = "pynput"
    except Exception:
        _key_down = _key_up = lambda k: None
        BACKEND = "none (print only)"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ========================= THREADED CAMERA =========================
class Camera:
    def __init__(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAM_INDEX, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Try changing CAM_INDEX.")

        self.frame, self.t, self.seq = None, 0.0, 0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                continue
            t = time.perf_counter()
            with self.lock:
                self.frame, self.t, self.seq = f, t, self.seq + 1

    def read(self):
        with self.lock:
            return self.frame, self.t, self.seq

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ============================ TRACKING ============================
class HandTrack:
    def __init__(self):
        self.x = self.y = self.t = 0.0
        self.vy = 0.0
        self.armed = False
        self.locked = False
        self.peak = 0.0
        self.arm_t = 0.0
        self.active = False
        self.missed = 0

    def update(self, x, y, t, frame_h, down_thresh):
        self.missed = 0
        if not self.active:
            self.x, self.y, self.t = x, y, t
            self.vy = 0.0
            self.active = True
            return False

        hit = False
        dt = t - self.t
        if dt > 0:
            raw_vy = (y - self.y) / frame_h / dt
            # Minimal smoothing (0.85 raw) so 25fps velocity spikes aren't dulled
            self.vy = 0.85 * raw_vy + 0.15 * self.vy

            if self.locked and self.vy < down_thresh * REARM_FRAC:
                self.locked = False

            if not self.locked:
                if not self.armed:
                    if self.vy > down_thresh:
                        self.armed, self.peak, self.arm_t = True, self.vy, t
                else:
                    self.peak = max(self.peak, self.vy)
                    if self.vy < self.peak * DECEL_RATIO:
                        hit = True
                        self.armed, self.locked = False, True
                    elif t - self.arm_t > ARM_TIMEOUT:
                        self.armed = False

        self.x, self.y, self.t = x, y, t
        return hit

    def mark_missed(self):
        self.missed += 1
        if self.missed > MAX_MISSED:
            self.active = False
            self.armed = False
            self.locked = False


# ============================== MAIN ==============================
def main():
    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        # Lowered confidence thresholds to catch slightly motion-blurred hands
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cam = Camera()
    win = "Taiko Hands (25 FPS)"
    cv2.namedWindow(win)

    tracks = {"Left": HandTrack(), "Right": HandTrack()}
    mid_x = 0.5
    down_thresh = DOWN_THRESH
    send_keys = False

    last_fire = {z: 0.0 for z in ZONE_KEYS}
    flash = {z: 0.0 for z in ZONE_KEYS}
    pending_release = []
    last_seq, fps, fps_t = -1, 0.0, time.perf_counter()
    last_ts_ms = -1

    print(f"Key backend: {BACKEND}")
    print("Keep hands open or in relaxed fists. Left = DON (f) | Right = KAT (j).")

    while True:
        now = time.perf_counter()

        for item in pending_release[:]:
            if now >= item[0]:
                _key_up(item[1])
                pending_release.remove(item)

        frame, t, seq = cam.read()
        if frame is None or seq == last_seq:
            time.sleep(0.0005)
            continue
        last_seq = seq

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = max(last_ts_ms + 1, int(t * 1000))
        last_ts_ms = ts_ms
        res = landmarker.detect_for_video(mp_image, ts_ms)

        seen_labels = set()
        hits = []

        if res.hand_landmarks and res.handedness:
            for lm_list, handedness_list in zip(res.hand_landmarks, res.handedness):
                label = handedness_list[0].category_name
                seen_labels.add(label)

                # Landmark 9 = Middle Finger Knuckle (much less motion blur than fingertip!)
                knuckle = lm_list[9]
                px, py = knuckle.x * w, knuckle.y * h

                if tracks[label].update(px, py, t, h, down_thresh):
                    hits.append(tracks[label])

        for lbl in ("Left", "Right"):
            if lbl not in seen_labels:
                tracks[lbl].mark_missed()

        for tr in hits:
            zone = 0 if tr.x < mid_x * w else 1
            if now - last_fire[zone] < DEBOUNCE:
                continue
            last_fire[zone] = flash[zone] = now
            label, key = ZONE_KEYS[zone]
            print(f"{label} ({key})")
            if send_keys:
                _key_down(key)
                pending_release.append((now + KEY_HOLD, key))

        # ------------------------- drawing -------------------------
        overlay = frame.copy()
        for col, (label, key) in ZONE_KEYS.items():
            x0, x1 = (0, int(mid_x * w)) if col == 0 else (int(mid_x * w), w)
            base = DON_COLOR if label == "DON" else KAT_COLOR
            lit = (now - flash[col]) < 0.12
            cv2.rectangle(overlay, (x0, 0), (x1, h), base, -1 if lit else 2)
            cv2.putText(overlay, f"{label} [{key.upper()}]", (x0 + 15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, base, 2)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        for lbl, tr in tracks.items():
            if tr.active:
                ring_color = (0, 255, 0) if tr.armed else (0, 255, 255)
                cv2.circle(frame, (int(tr.x), int(tr.y)), 12, ring_color, 3)

        fps = 0.9 * fps + 0.1 / max(1e-6, now - fps_t)
        fps_t = now
        status = "SENDING KEYS" if send_keys else "test mode (SPACE to send keys)"
        cv2.putText(frame, f"{fps:4.0f} fps | thresh {down_thresh:.2f} | {status}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win, frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord(" "):
            send_keys = not send_keys
        elif k == ord("["):
            down_thresh = max(0.15, down_thresh - 0.05)
        elif k == ord("]"):
            down_thresh += 0.05
        elif k == ord("a"):
            mid_x = max(0.2, mid_x - 0.02)
        elif k == ord("d"):
            mid_x = min(0.8, mid_x + 0.02)

    for _, key in pending_release:
        _key_up(key)
    landmarker.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()