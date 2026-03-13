"""
HandTalk — Step 3: Train Random Forest Model
==============================================
Loads landmarks.csv, splits into train/test, trains a
Random Forest classifier with optimized hyperparameters,
prints full evaluation metrics, plots confusion matrix,
and saves the model + label encoder to models/.

Output:
  models/handtalk_model.pkl
  models/label_encoder.pkl
  models/confusion_matrix.png
  models/feature_importance.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (classification_report, confusion_matrix,
                                       accuracy_score)
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler

# ── CONFIG ─────────────────────────────────────────────────────
CSV_PATH    = os.path.join('data', 'processed', 'landmarks.csv')
MODELS_DIR  = 'models'
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# Random Forest hyperparameters (tuned)
RF_PARAMS = {
    'n_estimators'      : 200,
    'max_depth'         : None,
    'min_samples_split' : 2,
    'min_samples_leaf'  : 1,
    'max_features'      : 'sqrt',
    'class_weight'      : 'balanced',
    'random_state'      : RANDOM_SEED,
    'n_jobs'            : -1,
}

# ── STYLE ──────────────────────────────────────────────────────
DARK_BG   = '#04080f'
SURFACE   = '#0d1a28'
SURFACE2  = '#152538'
CYAN      = '#00c8ff'
GREEN     = '#00f5a0'
TEXT      = '#d8eeff'
TEXT2     = '#6a9abd'

plt.rcParams.update({
    'figure.facecolor'  : DARK_BG,
    'axes.facecolor'    : SURFACE,
    'axes.edgecolor'    : '#1e3a50',
    'axes.labelcolor'   : TEXT2,
    'axes.titlecolor'   : TEXT,
    'xtick.color'       : TEXT2,
    'ytick.color'       : TEXT2,
    'text.color'        : TEXT,
    'grid.color'        : '#1a2e40',
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.5,
    'font.family'       : 'monospace',
})


# ── LOAD DATA ──────────────────────────────────────────────────
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"[ERROR] '{CSV_PATH}' not found.\n"
            "Run python 2_preprocess_data.py first."
        )
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded {len(df)} samples, {df.shape[1]-1} features.")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


# ── PLOTS ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(DARK_BG)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix — Raw Counts', 'Confusion Matrix — Normalized'],
        ['d', '.2f']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=classes, yticklabels=classes,
            cmap='Blues', ax=ax,
            linewidths=0.5, linecolor='#0a1520',
            annot_kws={'size': 9, 'color': TEXT},
            cbar_kws={'shrink': 0.8},
        )
        ax.set_title(title, pad=14, fontsize=13, color=TEXT)
        ax.set_xlabel('Predicted Label', labelpad=8)
        ax.set_ylabel('True Label',      labelpad=8)
        ax.tick_params(axis='x', rotation=35)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('HandTalk — Model Evaluation', fontsize=16, color=CYAN,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Confusion matrix saved → {save_path}")


def plot_feature_importance(model, n_top, save_path):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:n_top]
    labels      = []
    for idx in indices:
        lm_id   = idx // 3
        coord   = ['X', 'Y', 'Z'][idx % 3]
        labels.append(f'LM{lm_id:02d}_{coord}')

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    bars = ax.barh(range(n_top), importances[indices], color=CYAN, alpha=0.8,
                   edgecolor='none', height=0.7)
    # Gradient effect via alpha
    for i, bar in enumerate(bars):
        bar.set_alpha(0.4 + 0.6 * (1 - i / n_top))

    ax.set_yticks(range(n_top))
    ax.set_yticklabels(labels[::-1] if False else labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gini)', labelpad=8)
    ax.set_title(f'Top {n_top} Most Important Landmark Features',
                 pad=12, fontsize=13, color=TEXT)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Feature importance saved → {save_path}")


def plot_cv_scores(cv_scores, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)

    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bars  = ax.bar(folds, cv_scores, color=GREEN, alpha=0.75, width=0.5, edgecolor='none')

    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10,
                color=GREEN, fontweight='bold')

    ax.axhline(cv_scores.mean(), color=CYAN, linewidth=1.5, linestyle='--',
               label=f'Mean: {cv_scores.mean():.4f}')
    ax.set_ylim(max(0, cv_scores.min() - 0.05), 1.02)
    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Stratified Cross-Validation', pad=12, color=TEXT)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] CV scores plot saved  → {save_path}")


# ── MAIN ───────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  HandTalk — Model Training")
    print("=" * 60 + "\n")

    # 1. Load
    df = load_data()
    X  = df.drop('label', axis=1).values.astype(np.float32)
    y_raw = df['label'].values

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"[INFO] Classes: {list(le.classes_)}\n")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"[INFO] Train samples: {len(X_train)}")
    print(f"[INFO] Test  samples: {len(X_test)}\n")

    # 4. Train Random Forest
    print("[INFO] Training Random Forest...")
    print(f"       n_estimators={RF_PARAMS['n_estimators']}, "
          f"max_features={RF_PARAMS['max_features']}, "
          f"n_jobs={RF_PARAMS['n_jobs']}\n")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("  [✓] Training complete.\n")

    # 5. Evaluate on test set
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    classes = le.classes_

    print("=" * 60)
    print(f"  TEST ACCURACY: {acc*100:.2f}%")
    print("=" * 60)
    print("\n" + classification_report(
        y_test, y_pred, target_names=classes, digits=4
    ))

    # 6. Cross-validation
    print("[INFO] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"       Fold scores : {cv_scores.round(4)}")
    print(f"       Mean ± Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # 7. Save model & encoder
    model_path = os.path.join(MODELS_DIR, 'handtalk_model.pkl')
    le_path    = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    joblib.dump(model, model_path)
    joblib.dump(le,    le_path)
    print(f"  [✓] Model saved   → {model_path}")
    print(f"  [✓] Encoder saved → {le_path}\n")

    # 8. Plots
    print("[INFO] Generating evaluation plots...")
    plot_confusion_matrix(
        y_test, y_pred, classes,
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix.png')
    )
    plot_feature_importance(
        model, n_top=20,
        save_path=os.path.join(MODELS_DIR, 'feature_importance.png')
    )
    plot_cv_scores(
        cv_scores,
        save_path=os.path.join(MODELS_DIR, 'cv_scores.png')
    )

    print(f"\n{'='*60}")
    print(f"  FINAL ACCURACY : {acc*100:.2f}%")
    print(f"  CV MEAN        : {cv_scores.mean()*100:.2f}%")
    print(f"  MODEL SIZE     : {os.path.getsize(model_path)/1024:.1f} KB")
    print(f"{'='*60}")
    print("\n[HandTalk] Done! Next step: python 4_live_detection.py\n")


if __name__ == '__main__':
    main()
