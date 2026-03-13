"""
HandTalk — Step 1: Data Collection
===================================
Run this script to collect training samples for each gesture.
A webcam window will open. Follow the on-screen instructions:
  - Press SPACE to start recording a gesture class
  - Hold the gesture steady in front of the camera
  - The script auto-captures SAMPLES_PER_SIGN frames
  - Repeats for every sign in the SIGNS list

Output: data/raw/<SignName>/<frame_XXXX.jpg>
"""

import cv2
import os
import time

# ── CONFIG ─────────────────────────────────────────────────────
SIGNS            = ['Hello', 'ThankYou', 'Yes', 'No', 'Please',
                    'Sorry', 'Help', 'ILoveYou', 'Good', 'Bad']
SAMPLES_PER_SIGN = 200
WEBCAM_INDEX     = 0
DATA_DIR         = os.path.join('data', 'raw')

# ── COLORS & FONTS ─────────────────────────────────────────────
C_CYAN   = (255, 220, 0)    # BGR
C_GREEN  = (0, 230, 120)
C_RED    = (60, 60, 255)
C_WHITE  = (230, 230, 230)
C_BLACK  = (10, 10, 20)
C_DARK   = (20, 30, 45)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# ── HELPERS ────────────────────────────────────────────────────
def draw_overlay(frame, text, sub='', color=C_GREEN, progress=None):
    """Draw a styled HUD overlay on the frame."""
    h, w = frame.shape[:2]
    # Dark banner
    cv2.rectangle(frame, (0, 0), (w, 90), C_DARK, -1)
    cv2.rectangle(frame, (0, 0), (w, 90), (50, 80, 100), 1)
    # Logo
    cv2.putText(frame, 'HANDTALK', (14, 26), FONT, 0.55, C_CYAN, 2)
    cv2.putText(frame, 'DATA COLLECTION', (14, 46), FONT, 0.38, (100, 140, 160), 1)
    # Main text
    cv2.putText(frame, text, (14, 75), FONT, 0.7, color, 2)
    # Sub text
    if sub:
        cv2.putText(frame, sub, (w - 10 - len(sub)*9, 75), FONT, 0.45, (120, 140, 160), 1)
    # Progress bar
    if progress is not None:
        bar_w = w - 28
        cv2.rectangle(frame, (14, h-20), (14+bar_w, h-10), (30, 50, 60), -1)
        cv2.rectangle(frame, (14, h-20), (14+int(bar_w*progress), h-10), C_GREEN, -1)
        pct_text = f'{int(progress*100)}%'
        cv2.putText(frame, pct_text, (14+bar_w+6, h-10), FONT, 0.35, C_GREEN, 1)


def create_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for sign in SIGNS:
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)
    print(f"[INFO] Created data directories under '{DATA_DIR}/'")


# ── MAIN ───────────────────────────────────────────────────────
def main():
    create_dirs()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check WEBCAM_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n[HandTalk] Starting data collection for {len(SIGNS)} signs.")
    print(f"[HandTalk] {SAMPLES_PER_SIGN} samples per sign.\n")

    for sign_idx, sign in enumerate(SIGNS):
        # ── WAIT SCREEN ──────────────────────────────────────
        print(f"[INFO] Next sign: '{sign}' ({sign_idx+1}/{len(SIGNS)})")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Dim overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), C_DARK, -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            draw_overlay(frame,
                         f"NEXT: {sign}",
                         sub=f"Sign {sign_idx+1} of {len(SIGNS)}")

            # Centre instruction box
            box_x, box_y = w//2 - 220, h//2 - 60
            cv2.rectangle(frame, (box_x, box_y), (box_x+440, box_y+120),
                          (20, 35, 50), -1)
            cv2.rectangle(frame, (box_x, box_y), (box_x+440, box_y+120),
                          (50, 100, 130), 1)
            cv2.putText(frame, f"Prepare gesture:", (box_x+20, box_y+35),
                        FONT, 0.65, C_WHITE, 1)
            cv2.putText(frame, f"  \"{sign}\"", (box_x+20, box_y+70),
                        FONT, 0.9, C_CYAN, 2)
            cv2.putText(frame, "Press  SPACE  to begin", (box_x+20, box_y+105),
                        FONT, 0.45, C_GREEN, 1)

            cv2.imshow('HandTalk — Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Quit by user.")
                return

        # ── COUNTDOWN ────────────────────────────────────────
        for count in range(3, 0, -1):
            deadline = time.time() + 1.0
            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                draw_overlay(frame, f"Starting in {count}...", color=C_CYAN)
                cx, cy = w//2, h//2
                cv2.circle(frame, (cx, cy), 50, (40, 70, 90), -1)
                cv2.putText(frame, str(count), (cx-18, cy+16),
                            FONT, 1.8, C_CYAN, 3)
                cv2.imshow('HandTalk — Data Collection', frame)
                cv2.waitKey(1)

        # ── CAPTURE FRAMES ────────────────────────────────────
        saved = 0
        sign_dir = os.path.join(DATA_DIR, sign)

        while saved < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            progress = saved / SAMPLES_PER_SIGN
            draw_overlay(frame,
                         f"Recording: {sign}",
                         sub=f"{saved}/{SAMPLES_PER_SIGN}",
                         color=C_RED,
                         progress=progress)

            # REC dot
            if saved % 10 < 5:
                cv2.circle(frame, (w-30, 30), 8, C_RED, -1)
                cv2.putText(frame, 'REC', (w-65, 35), FONT, 0.4, C_RED, 1)

            # Save frame
            filename = os.path.join(sign_dir, f'frame_{saved:04d}.jpg')
            cv2.imwrite(filename, frame)
            saved += 1

            cv2.imshow('HandTalk — Data Collection', frame)
            cv2.waitKey(1)

        print(f"  [✓] Saved {saved} frames for '{sign}'")

    # ── DONE ─────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        draw_overlay(frame, "Collection Complete!", color=C_GREEN)
        cv2.putText(frame, f"Collected {len(SIGNS)*SAMPLES_PER_SIGN} total samples.",
                    (14, h//2), FONT, 0.65, C_WHITE, 1)
        cv2.putText(frame, "Run:  python 2_preprocess_data.py",
                    (14, h//2 + 40), FONT, 0.55, C_CYAN, 1)
        cv2.putText(frame, "Press Q to exit.", (14, h//2+80), FONT, 0.45, (120,140,160), 1)
        cv2.imshow('HandTalk — Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[HandTalk] Done! Next step: python 2_preprocess_data.py")


if __name__ == '__main__':
    main()
