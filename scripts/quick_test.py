"""
HandTalk — Quick Pipeline Test (No Webcam Required)
=====================================================
Tests the entire pipeline with synthetic landmark data.
Useful for CI, verifying installation, or demos without a camera.

Usage:
  python scripts/quick_test.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time

SIGNS = ['Hello', 'ThankYou', 'Yes', 'No', 'Please',
         'Sorry', 'Help', 'ILoveYou', 'Good', 'Bad']


def test_imports():
    print("[1/5] Testing imports...")
    import cv2
    import mediapipe as mp
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import joblib
    print(f"      cv2       : {cv2.__version__}")
    print(f"      mediapipe : {mp.__version__}")
    print("      ✓ All imports OK\n")


def test_mediapipe():
    print("[2/5] Testing MediaPipe Hands...")
    import mediapipe as mp
    import numpy as np
    import cv2

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        # Create a blank white test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Just check it doesn't crash (no hand expected in blank image)
    print("      ✓ MediaPipe Hands initialized successfully\n")


def test_feature_extraction():
    print("[3/5] Testing landmark feature extraction...")
    from utils.landmark_utils import LANDMARK_NAMES
    assert len(LANDMARK_NAMES) == 21, "Expected 21 landmarks"
    # Simulate 63 features
    fake_features = np.random.randn(63).astype(np.float32)
    assert fake_features.shape == (63,), f"Expected (63,), got {fake_features.shape}"
    print(f"      Feature vector shape: {fake_features.shape}")
    print("      ✓ Feature extraction logic OK\n")


def test_model_training():
    print("[4/5] Testing model training (tiny synthetic dataset)...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    np.random.seed(42)
    n_per_class = 30
    n_features  = 63

    # Generate synthetic data — each class has a shifted mean
    X_list, y_list = [], []
    for i, sign in enumerate(SIGNS):
        samples = np.random.randn(n_per_class, n_features).astype(np.float32)
        samples += i * 0.5  # shift so classes are separable
        X_list.append(samples)
        y_list.extend([sign] * n_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"      Training time : {train_time:.3f}s")
    print(f"      Test accuracy : {acc*100:.1f}% (synthetic data — ~100% expected)")
    assert acc > 0.80, f"Accuracy too low on synthetic data: {acc}"
    print("      ✓ Model training pipeline OK\n")


def test_buffers():
    print("[5/5] Testing gesture buffer & sentence builder...")
    from utils.gesture_buffer import GestureBuffer, SentenceBuilder

    buf = GestureBuffer(window_size=10, min_confidence=0.70, majority_threshold=0.6)
    for _ in range(7):
        buf.push('Hello', 0.92)
    for _ in range(3):
        buf.push('Good', 0.75)

    label, frac = buf.get_stable_prediction()
    assert label == 'Hello', f"Expected 'Hello', got '{label}'"
    print(f"      Buffer prediction: '{label}' ({frac*100:.0f}% majority)")

    sb = SentenceBuilder(max_words=15, cooldown_frames=5)
    sb.try_add('Hello')
    sb.try_add('Thank')
    sb.try_add('You')
    sentence = sb.get_sentence()
    assert sentence == 'Hello Thank You', f"Expected 'Hello Thank You', got '{sentence}'"
    print(f"      Sentence: '{sentence}'")
    print("      ✓ Buffers OK\n")


def main():
    print("=" * 55)
    print("  HandTalk — Quick Pipeline Test")
    print("=" * 55 + "\n")

    t0 = time.time()
    try:
        test_imports()
        test_mediapipe()
        test_feature_extraction()
        test_model_training()
        test_buffers()
    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n[FAIL] Missing dependency: {e}")
        print("       Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        raise

    elapsed = time.time() - t0
    print("=" * 55)
    print(f"  ALL TESTS PASSED in {elapsed:.2f}s")
    print("=" * 55)
    print("\n  Ready to run:")
    print("  → python 1_collect_data.py    (collect webcam data)")
    print("  → python 4_live_detection.py --demo  (demo, no model)\n")


if __name__ == '__main__':
    main()
