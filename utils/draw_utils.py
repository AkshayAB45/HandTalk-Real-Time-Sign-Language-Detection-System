"""
utils/draw_utils.py
====================
OpenCV-based drawing helpers for the HandTalk live detection HUD.
Provides panels, bars, labels, skeleton customization, etc.
"""

import cv2
import numpy as np
import math


# ── PALETTE ────────────────────────────────────────────────────
BLACK   = (10,  15,  22)
DARK    = (18,  28,  42)
SURFACE = (22,  42,  62)
CYAN    = (255, 210, 0)      # BGR — renders as blue-cyan on screen
CYAN2   = (255, 230, 50)
GREEN   = (150, 245, 80)
AMBER   = (40,  195, 255)
RED     = (70,  50,  255)
WHITE   = (220, 230, 240)
GREY    = (80,  110, 130)

FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT_M  = cv2.FONT_HERSHEY_DUPLEX


# ── PRIMITIVE HELPERS ──────────────────────────────────────────
def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    """Draw a semi-transparent filled rectangle."""
    roi     = frame[y1:y2, x1:x2].copy()
    overlay = np.full_like(roi, color[::-1] if len(color)==3 else color)
    cv2.addWeighted(overlay, 1-alpha, roi, alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def rounded_rect(frame, x1, y1, x2, y2, radius, color, thickness=-1, line_color=None, line_thickness=1):
    """Draw a rounded rectangle (filled or outlined)."""
    if thickness == -1:
        # Filled — approximate with rectangles + circles
        cv2.rectangle(frame, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(frame, (cx, cy), radius, color, -1)
    if line_color:
        cv2.rectangle(frame, (x1+radius, y1), (x2-radius, y2), line_color, line_thickness)
        cv2.rectangle(frame, (x1, y1+radius), (x2, y2-radius), line_color, line_thickness)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(frame, (cx, cy), radius, line_color, line_thickness)


def progress_bar(frame, x, y, width, height, value, max_val=1.0,
                 bg_color=DARK, fill_color=CYAN, border_color=SURFACE,
                 label=None, value_text=None):
    """Draw a horizontal progress bar."""
    pct = max(0.0, min(1.0, value / max_val))

    # Background
    cv2.rectangle(frame, (x, y), (x+width, y+height), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x+width, y+height), border_color, 1)

    # Fill
    fill_w = int(width * pct)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x+fill_w, y+height), fill_color, -1)

    # Label
    if label:
        cv2.putText(frame, label, (x, y-4), FONT, 0.32, GREY, 1)
    if value_text:
        tw, _ = cv2.getTextSize(value_text, FONT, 0.32, 1)[0], None
        cv2.putText(frame, value_text, (x+width-40, y-4), FONT, 0.32, fill_color, 1)


def ring_indicator(frame, cx, cy, radius, pct, color=CYAN,
                   bg_color=DARK, thickness=4, label=None):
    """Draw a circular progress ring."""
    # Background ring
    cv2.circle(frame, (cx, cy), radius, bg_color, thickness)
    # Progress arc
    angle = int(360 * pct)
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle,
                    color, thickness, cv2.LINE_AA)
    # Label in center
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.35, 1)
        cv2.putText(frame, label, (cx - tw//2, cy + th//2),
                    FONT, 0.35, color, 1)


# ── HEADER / FOOTER BARS ───────────────────────────────────────
def draw_header(frame, fps, hand_detected, model_name='RF · ASL'):
    """Draw the top HUD header bar."""
    h, w = frame.shape[:2]
    bar_h = 52

    alpha_rect(frame, 0, 0, w, bar_h, BLACK, alpha=0.3)
    cv2.line(frame, (0, bar_h), (w, bar_h), SURFACE, 1)

    # Logo
    cv2.putText(frame, 'HAND', (14, 32), FONT_M, 0.9, CYAN, 2)
    cv2.putText(frame, 'TALK', (78, 32), FONT_M, 0.9, WHITE, 1)

    # Separator
    cv2.line(frame, (145, 12), (145, 40), SURFACE, 1)

    # FPS
    fps_color = GREEN if fps >= 25 else AMBER if fps >= 15 else RED
    cv2.putText(frame, f'FPS: {fps:02d}', (158, 20), FONT, 0.38, GREY, 1)
    cv2.putText(frame, f'{fps:02d}', (195, 38), FONT, 0.65, fps_color, 2)

    # Model
    cv2.putText(frame, 'MODEL', (260, 20), FONT, 0.34, GREY, 1)
    cv2.putText(frame, model_name, (255, 38), FONT, 0.45, CYAN2, 1)

    # Hand detected indicator
    det_color = GREEN if hand_detected else RED
    det_label = 'HAND ● DETECTED' if hand_detected else 'NO HAND DETECTED'
    cv2.putText(frame, det_label, (w - 20 - len(det_label)*9, 32),
                FONT, 0.42, det_color, 1)


def draw_gesture_panel(frame, gesture, confidence, stable, x=14, y=62):
    """Draw the current gesture prediction panel."""
    panel_w = 340
    panel_h = 80

    rounded_rect(frame, x, y, x+panel_w, y+panel_h, 8,
                 (14, 22, 34), -1, (30, 52, 72), 1)

    # Label
    cv2.putText(frame, 'DETECTED GESTURE', (x+10, y+18),
                FONT, 0.32, GREY, 1)

    if gesture:
        # Gesture name
        font_scale = 1.0 if len(gesture) <= 8 else 0.72
        cv2.putText(frame, gesture, (x+10, y+56),
                    FONT_M, font_scale, CYAN, 2)
        # Confidence
        pct_text = f'{confidence*100:.1f}%'
        cv2.putText(frame, pct_text, (x+panel_w-60, y+56),
                    FONT, 0.55, GREEN if confidence > 0.85 else AMBER, 1)
        # Confidence bar
        progress_bar(frame, x+10, y+65, panel_w-20, 6,
                     confidence, 1.0,
                     fill_color=GREEN if confidence > 0.85 else AMBER,
                     label='', value_text='')
    else:
        cv2.putText(frame, '— — —', (x+10, y+52), FONT, 0.55, GREY, 1)


def draw_sentence_panel(frame, sentence, x=14):
    """Draw the translated sentence at the bottom of the frame."""
    h, w = frame.shape[:2]
    panel_h = 64
    y       = h - panel_h

    alpha_rect(frame, 0, y, w, h, BLACK, alpha=0.3)
    cv2.line(frame, (0, y), (w, y), SURFACE, 1)

    cv2.putText(frame, 'TRANSLATION', (14, y+15), FONT, 0.3, GREY, 1)

    # Sentence text — truncate left if too long
    disp = sentence if len(sentence) <= 55 else '...' + sentence[-52:]
    cv2.putText(frame, disp if disp else '(start signing...)',
                (14, y+44),
                FONT_M, 0.7,
                WHITE if disp else GREY,
                1 if disp else 1)

    # Key hints
    hints = 'C=clear  B=backspace  S=speak  Q=quit'
    (tw, _th), _base = cv2.getTextSize(hints, FONT, 0.28, 1)
    cv2.putText(frame, hints, (w - tw - 12, y+56),
                FONT, 0.28, GREY, 1)


def draw_confidence_rings(frame, conf_gesture, conf_track, conf_context):
    """Draw three small confidence rings in the top-right area."""
    h, w = frame.shape[:2]
    base_x = w - 160
    base_y = 70
    r      = 22
    gap    = 55

    labels = ['SIGN', 'TRACK', 'CTX']
    values = [conf_gesture, conf_track, conf_context]
    colors = [CYAN, GREEN, AMBER]

    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        cx = base_x + i * gap
        cy = base_y
        ring_indicator(frame, cx, cy, r, val, color=col, thickness=3,
                       label=f'{int(val*100)}%')
        cv2.putText(frame, lbl, (cx - len(lbl)*4, cy+r+14),
                    FONT, 0.28, GREY, 1)


def draw_history_panel(frame, history, max_items=5):
    """Draw recent gestures list on the right side."""
    h, w  = frame.shape[:2]
    panel_w = 160
    x       = w - panel_w - 10
    y_start = 130
    item_h  = 26

    cv2.putText(frame, 'RECENT', (x, y_start-8), FONT, 0.3, GREY, 1)

    for i, (gesture, conf) in enumerate(history[-max_items:]):
        y    = y_start + i * item_h
        alpha = 1.0 - i * 0.15
        color = tuple(int(c * alpha) for c in CYAN)

        rounded_rect(frame, x, y, x+panel_w, y+item_h-2, 4,
                     (12, 20, 32), -1, (25, 42, 58), 1)
        cv2.putText(frame, gesture[:14], (x+8, y+16), FONT, 0.38, color, 1)
        conf_text = f'{conf*100:.0f}%'
        cv2.putText(frame, conf_text, (x+panel_w-38, y+16),
                    FONT, 0.32, GREEN if conf > 0.85 else AMBER, 1)


def draw_skeleton_custom(frame, hand_landmarks, handedness='Right'):
    """
    Draw a custom colored hand skeleton with gradient-colored connections.
    Uses the raw pixel coordinates rather than MediaPipe's default style.
    """
    import mediapipe as mp
    mp_hands = mp.solutions.hands

    h, w  = frame.shape[:2]
    lm    = hand_landmarks.landmark

    def pt(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Connections by finger
    connections = [
        # Thumb
        [(0,1),(1,2),(2,3),(3,4)],
        # Index
        [(0,5),(5,6),(6,7),(7,8)],
        # Middle
        [(0,9),(9,10),(10,11),(11,12)],
        # Ring
        [(0,13),(13,14),(14,15),(15,16)],
        # Pinky
        [(0,17),(17,18),(18,19),(19,20)],
        # Palm
        [(5,9),(9,13),(13,17)],
    ]

    finger_colors = [AMBER, CYAN, GREEN, (200,150,255), (150,200,255), GREY]

    # Draw connections
    for finger_conn, color in zip(connections, finger_colors):
        for (a, b) in finger_conn:
            cv2.line(frame, pt(a), pt(b), color, 2, cv2.LINE_AA)

    # Draw landmark dots
    tip_indices = {4, 8, 12, 16, 20}
    for i in range(21):
        p = pt(i)
        if i in tip_indices:
            cv2.circle(frame, p, 7, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 7, CYAN, 2,  cv2.LINE_AA)
        elif i == 0:
            cv2.circle(frame, p, 8, CYAN, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, p, 4, GREY, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 4, SURFACE, 1, cv2.LINE_AA)


def draw_bounding_box(frame, hand_landmarks, label='', confidence=0.0):
    """Draw a stylized bounding box around the detected hand."""
    h, w  = frame.shape[:2]
    lm    = hand_landmarks.landmark

    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    margin = 20

    x1 = max(0,   int(min(xs)) - margin)
    y1 = max(0,   int(min(ys)) - margin)
    x2 = min(w-1, int(max(xs)) + margin)
    y2 = min(h-1, int(max(ys)) + margin)

    corner_len = 18
    thickness  = 2

    # Draw only corners
    for (px, py), dx, dy in [
        ((x1, y1),  1,  1),
        ((x2, y1), -1,  1),
        ((x1, y2),  1, -1),
        ((x2, y2), -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx*corner_len, py), CYAN, thickness)
        cv2.line(frame, (px, py), (px, py + dy*corner_len), CYAN, thickness)

    # Label above box
    if label:
        label_text = f'{label} {confidence*100:.0f}%'
        (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.45, 1)
        lx = x1
        ly = y1 - 8
        if ly < 20:
            ly = y1 + th + 8
        cv2.putText(frame, label_text, (lx, ly), FONT, 0.45, CYAN, 1)
