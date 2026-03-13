"""
HandTalk — Step 4: Live Real-Time Sign Language Detection
==========================================================
THE MAIN APPLICATION.

Controls:
  Q         → Quit
  C         → Clear translated sentence
  B         → Backspace (remove last word)
  S         → Speak sentence aloud (pyttsx3 or espeak)
  P         → Pause / Resume detection
  F         → Toggle fullscreen
  H         → Toggle history panel
  K         → Toggle skeleton overlay
  +/-       → Increase / decrease confidence threshold
  R         → Reset session stats

Usage:
  python 4_live_detection.py
  python 4_live_detection.py --camera 1          # different camera
  python 4_live_detection.py --threshold 0.75    # custom confidence
  python 4_live_detection.py --demo              # demo mode (no model needed)
"""

import cv2
import numpy as np
import joblib
import time
import os
import sys
import argparse
import collections
import mediapipe as mp

# ── PATH SETUP ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.landmark_utils import extract_features, compute_finger_angles
from utils.gesture_buffer  import GestureBuffer, SentenceBuilder
from utils.draw_utils      import (
    draw_header, draw_gesture_panel, draw_sentence_panel,
    draw_confidence_rings, draw_history_panel,
    draw_skeleton_custom, draw_bounding_box,
    alpha_rect, rounded_rect, FONT, FONT_M,
    CYAN, GREEN, AMBER, RED, WHITE, GREY, DARK, SURFACE, BLACK
)

# ── CONFIG ─────────────────────────────────────────────────────
DEFAULT_CAMERA    = 0
DEFAULT_THRESHOLD = 0.70
MODEL_PATH        = os.path.join('models', 'handtalk_model.pkl')
ENCODER_PATH      = os.path.join('models', 'label_encoder.pkl')
WINDOW_TITLE      = 'HandTalk — Real-Time Sign Language Detection'

# Gesture emoji mapping for on-screen display
GESTURE_EMOJI = {
    'Hello':    '👋',
    'ThankYou': '🙏',
    'Yes':      '✅',
    'No':       '🚫',
    'Please':   '🤲',
    'Sorry':    '😔',
    'Help':     '🆘',
    'ILoveYou': '🤟',
    'Good':     '👍',
    'Bad':      '👎',
}


# ── ARGUMENT PARSER ────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='HandTalk Live Detection')
    p.add_argument('--camera',    type=int,   default=DEFAULT_CAMERA,    help='Webcam index')
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Confidence threshold')
    p.add_argument('--demo',      action='store_true',                    help='Demo mode (no model needed)')
    p.add_argument('--width',     type=int,   default=1280, help='Camera capture width')
    p.add_argument('--height',    type=int,   default=720,  help='Camera capture height')
    return p.parse_args()


# ── MODEL LOADER ───────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("        Run python 3_train_model.py first, or use --demo flag.")
        return None, None, None

    print(f"[INFO] Loading model from '{MODEL_PATH}'...")
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    classes = list(encoder.classes_)
    print(f"[INFO] Model loaded. Classes: {classes}\n")
    return model, encoder, classes


# ── DEMO PREDICTOR ─────────────────────────────────────────────
class DemoPredictor:
    """Fake predictor that cycles through gestures — no model needed."""

    DEMO_GESTURES = ['Hello', 'Good', 'ThankYou', 'Yes', 'ILoveYou',
                     'Please', 'Help', 'No', 'Sorry', 'Bad']

    def __init__(self):
        self._idx      = 0
        self._last     = time.time()
        self._interval = 2.5

    def predict(self, features):
        now = time.time()
        if now - self._last > self._interval:
            self._idx  = (self._idx + 1) % len(self.DEMO_GESTURES)
            self._last = now
        label = self.DEMO_GESTURES[self._idx]
        confidence = 0.85 + 0.1 * np.random.rand()
        probas = {g: 0.02 for g in self.DEMO_GESTURES}
        probas[label] = confidence
        return label, confidence, probas

    @property
    def classes_(self):
        return self.DEMO_GESTURES


# ── TTS HELPER ─────────────────────────────────────────────────
def speak_text(text):
    """Speak text using pyttsx3 (or fall back to espeak / say on macOS)."""
    if not text.strip():
        return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 155)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass
    # Fallback
    if sys.platform == 'darwin':
        os.system(f'say "{text}"')
    elif sys.platform.startswith('linux'):
        os.system(f'espeak "{text}" 2>/dev/null')


# ── SPLASH SCREEN ──────────────────────────────────────────────
def show_splash(cap):
    """Show a loading splash until a key is pressed."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Dark overlay
        overlay = np.zeros_like(frame)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Centre box
        bx, by = w//2 - 300, h//2 - 130
        rounded_rect(frame, bx, by, bx+600, by+260, 12, (14,22,34), -1, (30,52,72), 1)

        # Title
        cv2.putText(frame, 'HAND', (bx+30, by+80), FONT_M, 2.0, CYAN, 3)
        cv2.putText(frame, 'TALK', (bx+200, by+80), FONT_M, 2.0, WHITE, 2)

        cv2.putText(frame, 'Real-Time Sign Language Detection',
                    (bx+30, by+115), FONT, 0.52, GREY, 1)
        cv2.putText(frame, 'Python  |  OpenCV  |  MediaPipe  |  RandomForest',
                    (bx+30, by+140), FONT, 0.42, (60,90,110), 1)

        # Divider
        cv2.line(frame, (bx+30, by+158), (bx+570, by+158), (30,52,72), 1)

        # Controls reference
        controls = [
            ('Q', 'Quit'),
            ('C', 'Clear'),
            ('B', 'Backspace'),
            ('S', 'Speak'),
            ('P', 'Pause'),
            ('H', 'History'),
        ]
        for i, (key, action) in enumerate(controls):
            col = i % 3
            row = i // 3
            cx  = bx + 30 + col * 190
            cy  = by + 185 + row * 28
            cv2.rectangle(frame, (cx, cy-14), (cx+22, cy+4), SURFACE, -1)
            cv2.putText(frame, key, (cx+5, cy), FONT, 0.4, CYAN, 1)
            cv2.putText(frame, action, (cx+28, cy), FONT, 0.4, GREY, 1)

        # Animated prompt
        t   = time.time() - start
        alpha_blink = abs(math.sin(t * 2))
        prompt_color = tuple(int(c * alpha_blink) for c in WHITE)
        cv2.putText(frame, 'Press  SPACE  to begin',
                    (bx+150, by+245), FONT, 0.6, prompt_color, 1)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(' '), ord('\r'), 27):
            break


import math


# ── MAIN DETECTION LOOP ────────────────────────────────────────
def main():
    args = parse_args()

    # Load model
    if args.demo:
        print("[INFO] DEMO MODE — running without trained model.")
        predictor  = DemoPredictor()
        classes    = predictor.classes_
    else:
        model, encoder, classes = load_model()
        if model is None:
            return
        predictor = None  # will use model directly

    # MediaPipe Hands
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.55,
        model_complexity=1,
    )

    # Webcam
    print(f"[INFO] Opening camera {args.camera} at {args.width}×{args.height}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # State
    gesture_buffer  = GestureBuffer(window_size=10, min_confidence=args.threshold)
    sentence        = SentenceBuilder(max_words=15, cooldown_frames=22)
    history         = collections.deque(maxlen=8)

    paused          = False
    show_history    = True
    show_skeleton   = True
    fullscreen      = False
    threshold       = args.threshold

    # Stats
    frame_count     = 0
    detect_count    = 0
    session_start   = time.time()
    fps_times       = collections.deque(maxlen=30)

    current_gesture = None
    current_conf    = 0.0
    stable_gesture  = None

    # Splash
    show_splash(cap)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.width, args.height)

    print("\n[HandTalk] Live detection running. Press Q to quit.\n")

    # ── MAIN LOOP ───────────────────────────────────────────
    while True:
        t_frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame_count += 1

        # ── MEDIAPIPE ─────────────────────────────────────
        hand_detected = False
        conf_gesture  = 0.0
        conf_tracking = 0.0
        conf_context  = 0.0
        current_gesture = None
        current_conf    = 0.0

        if not paused:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results  = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_detected = True
                detect_count += 1

                hand_lm    = results.multi_hand_landmarks[0]
                handedness = (results.multi_handedness[0].classification[0].label
                              if results.multi_handedness else 'Right')
                track_conf = (results.multi_handedness[0].classification[0].score
                              if results.multi_handedness else 0.8)

                # Extract features
                features = extract_features(hand_lm)

                # Predict
                if args.demo:
                    label, confidence, probas = predictor.predict(features)
                else:
                    feat_2d    = features.reshape(1, -1)
                    pred_idx   = model.predict(feat_2d)[0]
                    probas_arr = model.predict_proba(feat_2d)[0]
                    confidence = float(probas_arr.max())
                    label      = encoder.inverse_transform([pred_idx])[0]
                    probas     = dict(zip(encoder.classes_, probas_arr.tolist()))

                current_gesture = label
                current_conf    = confidence
                conf_gesture    = confidence
                conf_tracking   = float(track_conf)
                conf_context    = confidence * 0.85  # derived heuristic

                # Buffer + stable prediction
                gesture_buffer.push(label, confidence)
                stable, fraction = gesture_buffer.get_stable_prediction()

                if stable and confidence >= threshold:
                    stable_gesture = stable
                    added = sentence.try_add(stable)
                    if added:
                        history.appendleft((stable, confidence))

                # Skeleton
                if show_skeleton:
                    draw_skeleton_custom(frame, hand_lm, handedness)

                # Bounding box
                draw_bounding_box(frame, hand_lm,
                                  label=label if confidence >= threshold else '',
                                  confidence=confidence)

        # ── HUD DRAWING ───────────────────────────────────
        # FPS
        fps_times.append(time.time())
        fps = int(len(fps_times) / max(fps_times[-1] - fps_times[0], 0.001)) if len(fps_times) > 1 else 0

        # HEADER
        draw_header(frame, fps, hand_detected,
                    model_name='Demo·ASL' if args.demo else 'RF·ASL')

        # GESTURE PANEL
        draw_gesture_panel(frame,
                           current_gesture if hand_detected else None,
                           current_conf,
                           stable=(stable_gesture is not None))

        # CONFIDENCE RINGS
        draw_confidence_rings(frame, conf_gesture, conf_tracking, conf_context)

        # HISTORY
        if show_history:
            draw_history_panel(frame, list(history), max_items=5)

        # SENTENCE
        draw_sentence_panel(frame, sentence.get_sentence())

        # PAUSED OVERLAY
        if paused:
            alpha_rect(frame, 0, 0, w, h, BLACK, alpha=0.45)
            (tw, th), _ = cv2.getTextSize('⏸ PAUSED', FONT_M, 1.5, 2)
            cv2.putText(frame, 'PAUSED', (w//2 - tw//2, h//2),
                        FONT_M, 1.5, AMBER, 3)
            cv2.putText(frame, 'Press P to resume', (w//2 - 120, h//2 + 50),
                        FONT, 0.55, GREY, 1)

        # SESSION STATS (bottom-left mini panel)
        elapsed   = int(time.time() - session_start)
        mm, ss    = divmod(elapsed, 60)
        det_rate  = f'{100*detect_count/max(frame_count,1):.0f}%'
        stats_txt = [
            f'Session:  {mm:02d}:{ss:02d}',
            f'Frames:   {frame_count}',
            f'Detected: {detect_count}  ({det_rate})',
            f'Threshold:{threshold:.2f}',
        ]
        py = h - 68
        for i, txt in enumerate(stats_txt):
            cv2.putText(frame, txt, (14, py + i*14),
                        FONT, 0.3, (50, 75, 95), 1)

        cv2.imshow(WINDOW_TITLE, frame)

        # ── KEY HANDLING ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("\n[HandTalk] Quit by user.")
            break

        elif key == ord('c'):
            sentence.clear()
            gesture_buffer.clear()
            print("[INFO] Sentence cleared.")

        elif key == ord('b'):
            sentence.backspace()
            print("[INFO] Backspace.")

        elif key == ord('s'):
            txt = sentence.get_sentence()
            print(f"[INFO] Speaking: '{txt}'")
            speak_text(txt)

        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

        elif key == ord('h'):
            show_history = not show_history

        elif key == ord('k'):
            show_skeleton = not show_skeleton

        elif key == ord('f'):
            fullscreen = not fullscreen
            flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

        elif key == ord('+') or key == ord('='):
            threshold = min(0.99, threshold + 0.05)
            print(f"[INFO] Threshold: {threshold:.2f}")

        elif key == ord('-'):
            threshold = max(0.30, threshold - 0.05)
            print(f"[INFO] Threshold: {threshold:.2f}")

        elif key == ord('r'):
            detect_count  = 0
            frame_count   = 0
            session_start = time.time()
            history.clear()
            sentence.clear()
            gesture_buffer.clear()
            print("[INFO] Session reset.")

    # ── CLEANUP ───────────────────────────────────────────
    elapsed = int(time.time() - session_start)
    mm, ss  = divmod(elapsed, 60)
    print(f"\n{'='*50}")
    print(f"  HandTalk Session Summary")
    print(f"{'='*50}")
    print(f"  Duration        : {mm:02d}:{ss:02d}")
    print(f"  Frames captured : {frame_count}")
    print(f"  Hands detected  : {detect_count}")
    print(f"  Final sentence  : {sentence.get_sentence() or '(empty)'}")
    print(f"{'='*50}\n")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
