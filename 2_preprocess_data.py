"""
HandTalk — Step 2: Preprocess Data & Extract Landmarks
========================================================
Reads all raw images from data/raw/<SignName>/, runs MediaPipe
Hands on each, extracts the 21 3D landmarks (63 features),
normalizes them relative to the wrist, and saves to CSV.

Output: data/processed/landmarks.csv
"""

import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────
SIGNS     = ['Hello', 'ThankYou', 'Yes', 'No', 'Please',
             'Sorry', 'Help', 'ILoveYou', 'Good', 'Bad']
RAW_DIR   = os.path.join('data', 'raw')
OUT_DIR   = os.path.join('data', 'processed')
OUT_CSV   = os.path.join(OUT_DIR, 'landmarks.csv')

# ── MEDIAPIPE SETUP ────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# ── FEATURE ENGINEERING ────────────────────────────────────────
def normalize_landmarks(hand_landmarks):
    """
    Normalize 21 landmarks relative to wrist (landmark 0).
    Also scale by the distance from wrist to middle-finger-MCP (landmark 9)
    so the representation is scale and translation invariant.

    Returns flat array of 63 floats: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """
    lm = hand_landmarks.landmark

    # Reference points
    wrist    = np.array([lm[0].x, lm[0].y, lm[0].z])
    mid_mcp  = np.array([lm[9].x, lm[9].y, lm[9].z])
    scale    = np.linalg.norm(mid_mcp - wrist) + 1e-8

    features = []
    for point in lm:
        norm_x = (point.x - wrist[0]) / scale
        norm_y = (point.y - wrist[1]) / scale
        norm_z = (point.z - wrist[2]) / scale
        features.extend([norm_x, norm_y, norm_z])

    return features  # 63 values


def build_csv_header():
    header = []
    for i in range(21):
        header += [f'lm{i}_x', f'lm{i}_y', f'lm{i}_z']
    header.append('label')
    return header


# ── MAIN ───────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[HandTalk] Extracting landmarks from raw images...\n")

    total_written   = 0
    total_skipped   = 0
    per_class_count = {}

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_header())

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            for sign in SIGNS:
                sign_dir = os.path.join(RAW_DIR, sign)
                if not os.path.isdir(sign_dir):
                    print(f"  [WARN] Directory not found: {sign_dir} — skipping.")
                    continue

                image_paths = sorted(Path(sign_dir).glob('*.jpg'))
                sign_count  = 0

                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        total_skipped += 1
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    if results.multi_hand_landmarks:
                        hand_lm  = results.multi_hand_landmarks[0]
                        features = normalize_landmarks(hand_lm)
                        writer.writerow(features + [sign])
                        sign_count    += 1
                        total_written += 1
                    else:
                        total_skipped += 1

                per_class_count[sign] = sign_count
                status = '✓' if sign_count > 0 else '✗'
                print(f"  [{status}] {sign:<12} → {sign_count} landmarks extracted")

    # ── SUMMARY ──────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Total rows written : {total_written}")
    print(f"  Total images skipped: {total_skipped}  (no hand detected)")
    print(f"  Output saved to    : {OUT_CSV}")
    print(f"{'─'*50}")
    print("\n[HandTalk] Done! Next step: python 3_train_model.py\n")


if __name__ == '__main__':
    main()
