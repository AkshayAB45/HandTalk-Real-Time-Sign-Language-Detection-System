"""
HandTalk — Model Evaluation & Benchmarking
===========================================
Loads the trained model and runs a full evaluation:
  - Per-class accuracy breakdown
  - Confusion matrix visualization
  - Inference speed benchmark (FPS)
  - Comparison: threshold sensitivity analysis

Usage:
  python scripts/evaluate_model.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

CSV_PATH   = os.path.join('data', 'processed', 'landmarks.csv')
MODEL_PATH = os.path.join('models', 'handtalk_model.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
OUT_DIR    = os.path.join('models')

# ── PALETTE ─────────────────────────────────────────────────────
DARK_BG  = '#04080f'
SURFACE  = '#0d1a28'
CYAN     = '#00c8ff'
GREEN    = '#00f5a0'
AMBER    = '#ffcc00'
TEXT     = '#d8eeff'
TEXT2    = '#6a9abd'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   SURFACE,
    'axes.edgecolor':   '#1e3a50',
    'axes.labelcolor':  TEXT2,
    'axes.titlecolor':  TEXT,
    'xtick.color':      TEXT2,
    'ytick.color':      TEXT2,
    'text.color':       TEXT,
    'grid.color':       '#1a2e40',
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run python 3_train_model.py first.")
        sys.exit(1)
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def load_dataset(encoder):
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found: {CSV_PATH}")
        sys.exit(1)
    df   = pd.read_csv(CSV_PATH)
    X    = df.drop('label', axis=1).values.astype(np.float32)
    y    = encoder.transform(df['label'].values)
    return X, y


def benchmark_inference(model, X, n_trials=500):
    """Measure per-sample and throughput inference speed."""
    # Warm-up
    for _ in range(10):
        model.predict(X[:1])

    # Single-sample latency
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        model.predict(X[np.random.randint(len(X)):np.random.randint(len(X))+1])
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\n  Inference Benchmark ({n_trials} trials):")
    print(f"    Mean latency : {times.mean():.3f} ms")
    print(f"    P50          : {np.percentile(times, 50):.3f} ms")
    print(f"    P95          : {np.percentile(times, 95):.3f} ms")
    print(f"    P99          : {np.percentile(times, 99):.3f} ms")
    print(f"    Max FPS      : {1000/times.mean():.0f} fps  (model alone)")


def plot_per_class_accuracy(y_true, y_pred, classes, save_path):
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    colors = [GREEN if v >= 0.95 else AMBER if v >= 0.85 else '#ff3d5a'
              for v in per_class]
    bars = ax.bar(classes, per_class * 100, color=colors, alpha=0.8,
                  width=0.6, edgecolor='none')

    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9,
                color=bar.get_facecolor(), fontweight='bold')

    ax.axhline(95, color=CYAN, linewidth=1, linestyle='--', alpha=0.6, label='95% target')
    ax.set_ylim(0, 108)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Breakdown', pad=12, color=TEXT)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Per-class accuracy → {save_path}")


def plot_threshold_sensitivity(model, X, y, encoder, save_path):
    """Plot accuracy vs confidence threshold."""
    probas     = model.predict_proba(X)
    thresholds = np.arange(0.3, 0.99, 0.02)
    accs, coverages = [], []

    for thresh in thresholds:
        max_proba  = probas.max(axis=1)
        mask       = max_proba >= thresh
        if mask.sum() == 0:
            accs.append(0)
            coverages.append(0)
            continue
        preds   = probas[mask].argmax(axis=1)
        acc     = (preds == y[mask]).mean()
        coverage = mask.mean()
        accs.append(acc * 100)
        coverages.append(coverage * 100)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax1.plot(thresholds, accs, color=CYAN, linewidth=2, label='Accuracy')
    ax1.fill_between(thresholds, accs, alpha=0.12, color=CYAN)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy (%)', color=CYAN)
    ax1.tick_params(axis='y', labelcolor=CYAN)
    ax1.set_ylim(80, 102)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverages, color=AMBER, linewidth=2,
             linestyle='--', label='Coverage')
    ax2.fill_between(thresholds, coverages, alpha=0.08, color=AMBER)
    ax2.set_ylabel('Coverage (% predictions made)', color=AMBER)
    ax2.tick_params(axis='y', labelcolor=AMBER)

    ax1.axvline(0.70, color='#ff3d5a', linewidth=1.5, linestyle=':',
                label='Default threshold (0.70)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower left')

    ax1.set_title('Accuracy vs Coverage — Threshold Sensitivity', pad=12, color=TEXT)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Threshold sensitivity → {save_path}")


def main():
    print("=" * 60)
    print("  HandTalk — Model Evaluation")
    print("=" * 60 + "\n")

    model, encoder = load_artifacts()
    X, y           = load_dataset(encoder)
    classes        = encoder.classes_

    print(f"[INFO] Dataset: {len(X)} samples, {X.shape[1]} features, {len(classes)} classes")

    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)

    print(f"\n[INFO] Overall Accuracy: {acc*100:.2f}%")
    print("\n" + classification_report(y, y_pred, target_names=classes, digits=4))

    benchmark_inference(model, X)

    print("\n[INFO] Generating plots...")
    plot_per_class_accuracy(y, y_pred, classes,
        save_path=os.path.join(OUT_DIR, 'per_class_accuracy.png'))
    plot_threshold_sensitivity(model, X, y, encoder,
        save_path=os.path.join(OUT_DIR, 'threshold_sensitivity.png'))

    print(f"\n[HandTalk] Evaluation complete. Plots saved to '{OUT_DIR}/'")


if __name__ == '__main__':
    main()
