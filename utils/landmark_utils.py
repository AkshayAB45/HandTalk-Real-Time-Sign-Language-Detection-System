"""
utils/landmark_utils.py
========================
Shared landmark normalization and feature extraction utilities
used by both the preprocessing pipeline and live detection.
"""

import numpy as np


# Landmark indices for finger tips and key joints
WRIST         = 0
THUMB_TIP     = 4
INDEX_TIP     = 8
MIDDLE_TIP    = 12
RING_TIP      = 16
PINKY_TIP     = 20
INDEX_MCP     = 5
MIDDLE_MCP    = 9
RING_MCP      = 13
PINKY_MCP     = 17

# All 21 MediaPipe hand landmark names (for reference)
LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]


def extract_features(hand_landmarks):
    """
    Extract and normalize 63 features (21 landmarks × x,y,z)
    from a MediaPipe hand_landmarks object.

    Normalization:
      - Translate: subtract wrist position (landmark 0)
      - Scale: divide by wrist-to-middle-MCP distance (landmark 9)
      → Scale-invariant, translation-invariant, NOT rotation-invariant

    Args:
        hand_landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList

    Returns:
        np.ndarray of shape (63,) — float32
    """
    lm = hand_landmarks.landmark

    wrist   = np.array([lm[WRIST].x,      lm[WRIST].y,      lm[WRIST].z],      dtype=np.float32)
    mid_mcp = np.array([lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y, lm[MIDDLE_MCP].z], dtype=np.float32)
    scale   = float(np.linalg.norm(mid_mcp - wrist)) + 1e-8

    features = np.zeros(63, dtype=np.float32)
    for i, point in enumerate(lm):
        features[i*3]   = (point.x - wrist[0]) / scale
        features[i*3+1] = (point.y - wrist[1]) / scale
        features[i*3+2] = (point.z - wrist[2]) / scale

    return features


def compute_finger_angles(hand_landmarks):
    """
    Compute the extension angle for each finger (0° = closed, 180° = open).
    Returns dict: {'thumb': float, 'index': float, ...}
    """
    lm = hand_landmarks.landmark

    def angle_between(a, b, c):
        """Angle at point b, formed by a-b-c."""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    # Simplified: use MCP-PIP-TIP angle per finger
    angles = {
        'thumb'  : angle_between(lm[2],  lm[3],  lm[4]),
        'index'  : angle_between(lm[5],  lm[6],  lm[8]),
        'middle' : angle_between(lm[9],  lm[10], lm[12]),
        'ring'   : angle_between(lm[13], lm[14], lm[16]),
        'pinky'  : angle_between(lm[17], lm[18], lm[20]),
    }
    return angles


def is_finger_extended(angles, finger, threshold=140.0):
    """Returns True if the given finger appears extended (open)."""
    return angles.get(finger, 0) > threshold


def hand_features_summary(hand_landmarks):
    """
    Returns a human-readable dict of hand state:
      { 'features': np.ndarray(63,), 'angles': dict, 'extended': dict }
    """
    feats   = extract_features(hand_landmarks)
    angles  = compute_finger_angles(hand_landmarks)
    extended = {f: is_finger_extended(angles, f) for f in angles}
    return {'features': feats, 'angles': angles, 'extended': extended}
