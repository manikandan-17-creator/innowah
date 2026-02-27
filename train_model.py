"""
INNOWAH 2026 - ML Model Training Script
────────────────────────────────────────
Trains a dementia risk classifier using:
  - Synthetic data generated from clinical parameter distributions (use until real data collected)
  - OR real CSV data collected from patients

Models trained:
  1. Random Forest (interpretable, fast)
  2. XGBoost (high performance)
  3. Logistic Regression (baseline)

Output:
  - saved_models/innowah_model.pkl   (best model)
  - saved_models/innowah_scaler.pkl  (feature scaler)
  - saved_models/model_report.txt    (performance metrics)

Usage:
  python train_model.py                         # train on synthetic data
  python train_model.py --data real_data.csv    # train on real CSV
  python train_model.py --data real_data.csv --augment  # mix real + synthetic
"""

import numpy as np
import pandas as pd
import joblib
import os
import argparse
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.pipeline import Pipeline
from feature_extractor import FeatureExtractor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Run: pip install xgboost")


OUTPUT_DIR = "saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

extractor = FeatureExtractor()
FEATURE_NAMES = extractor.get_feature_names()
N_FEATURES = len(FEATURE_NAMES)  # 35


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# Uses clinical parameter distributions from INNOWAH document
# Labels: 0=Normal, 1=Mild Risk (MCI), 2=High Risk (Dementia)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n_samples=3000, seed=42):
    """
    Generate realistic synthetic data using Gaussian distributions
    centered on clinically defined ranges for each risk class.
    """
    rng = np.random.default_rng(seed)
    features, labels = [], []

    # Class distribution: 60% Normal, 25% Mild, 15% High Risk
    class_counts = [
        int(n_samples * 0.60),
        int(n_samples * 0.25),
        n_samples - int(n_samples * 0.60) - int(n_samples * 0.25)
    ]

    # ── Distribution parameters per class ──────────────────────────────
    # Format: {feature_name: (mean_class0, mean_class1, mean_class2, std)}
    # Values are in NORMALIZED space (0–1, healthy=1)
    DISTRIBUTIONS = {
        # Software / Cognitive
        "immediate_recall":     (0.85, 0.60, 0.35, 0.10),
        "delayed_recall":       (0.80, 0.55, 0.30, 0.10),
        "cue_benefit_index":    (0.75, 0.55, 0.35, 0.12),
        "retention_ratio":      (0.82, 0.58, 0.32, 0.10),
        "orientation_score":    (0.90, 0.70, 0.45, 0.10),
        "serial7s_score":       (0.80, 0.58, 0.35, 0.12),
        "clock_drawing_score":  (0.85, 0.62, 0.38, 0.12),
        "reaction_time_norm":   (0.72, 0.55, 0.38, 0.10),
        "error_consistency":    (0.80, 0.58, 0.35, 0.12),
        "naming_task_score":    (0.88, 0.65, 0.40, 0.10),
        "sentence_repetition":  (0.85, 0.63, 0.40, 0.10),
        "verbal_fluency_norm":  (0.72, 0.52, 0.32, 0.10),
        "iadl_score_norm":      (0.90, 0.65, 0.38, 0.12),
        "mood_filter":          (0.15, 0.35, 0.55, 0.10),

        # IMU
        "gait_speed":           (0.75, 0.58, 0.38, 0.08),
        "stride_variability":   (0.80, 0.60, 0.40, 0.10),
        "turning_velocity":     (0.75, 0.55, 0.35, 0.10),
        "postural_sway":        (0.82, 0.60, 0.38, 0.10),

        # PPG/HRV
        "rmssd":                (0.50, 0.35, 0.20, 0.08),
        "sdnn":                 (0.52, 0.35, 0.22, 0.08),
        "lf_hf_ratio":          (0.72, 0.55, 0.35, 0.10),
        "spo2":                 (0.85, 0.65, 0.42, 0.08),
        "desat_events":         (0.90, 0.70, 0.45, 0.10),

        # EEG
        "alpha_power":          (0.72, 0.52, 0.32, 0.10),
        "theta_power":          (0.78, 0.58, 0.38, 0.10),
        "delta_power":          (0.80, 0.60, 0.38, 0.10),
        "beta_power":           (0.65, 0.48, 0.28, 0.10),
        "theta_alpha_ratio":    (0.78, 0.58, 0.38, 0.10),
        "dominant_frequency":   (0.72, 0.52, 0.32, 0.10),
        "signal_entropy":       (0.70, 0.50, 0.30, 0.10),
        "posterior_alpha":      (0.72, 0.52, 0.32, 0.10),

        # Activity / Temp
        "daily_steps":          (0.65, 0.45, 0.28, 0.10),
        "skin_temp":            (0.60, 0.55, 0.50, 0.08),

        # Aggregates
        "sensor_score":         (0.72, 0.52, 0.33, 0.08),
        "cognitive_score":      (0.74, 0.55, 0.35, 0.08),
    }

    for label, count in enumerate(class_counts):
        for _ in range(count):
            sample = []
            for feat in FEATURE_NAMES:
                if feat in DISTRIBUTIONS:
                    means = DISTRIBUTIONS[feat][:3]
                    std   = DISTRIBUTIONS[feat][3]
                    val   = rng.normal(means[label], std)
                    val   = float(np.clip(val, 0.0, 1.0))
                else:
                    val = float(rng.uniform(0.3, 0.9))
                sample.append(val)
            features.append(sample)
            labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Add controlled noise + correlation structure
    noise = rng.normal(0, 0.02, X.shape).astype(np.float32)
    X = np.clip(X + noise, 0.0, 1.0)

    print(f"Generated {len(X)} synthetic samples: "
          f"Normal={sum(y==0)}, MildRisk={sum(y==1)}, HighRisk={sum(y==2)}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER (real CSV)
# CSV must have columns matching FEATURE_NAMES + "label" (0/1/2)
# ─────────────────────────────────────────────────────────────────────────────
def load_real_data(csv_path: str):
    df = pd.read_csv(csv_path)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column (0=Normal, 1=MildRisk, 2=HighRisk)")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    print(f"Loaded {len(X)} real samples from {csv_path}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(X, y):
    print(f"\nTraining on {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Define candidate models ───────────────────────────────────────
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

    # ── Train + evaluate all models ───────────────────────────────────
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s) if hasattr(model, 'predict_proba') else None

        acc   = accuracy_score(y_test, y_pred)
        cv_sc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        auc   = roc_auc_score(y_test, y_prob, multi_class="ovr") if y_prob is not None else 0.0

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  CV Accuracy:    {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
        print(f"  ROC-AUC (OvR):  {auc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["Normal", "Mild Risk", "High Risk"]))

        results[name] = {
            "model":    model,
            "acc":      acc,
            "cv_mean":  cv_sc.mean(),
            "auc":      auc
        }

    # ── Select best model by AUC ──────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    print(f"\n✅ Best model: {best_name} (AUC={best['auc']:.4f})")

    # ── Feature importance ────────────────────────────────────────────
    if hasattr(best["model"], "feature_importances_"):
        importances = best["model"].feature_importances_
        top_idx     = np.argsort(importances)[::-1][:10]
        print("\nTop 10 Features:")
        for i in top_idx:
            print(f"  {FEATURE_NAMES[i]:30s}  {importances[i]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    joblib.dump(best["model"], os.path.join(OUTPUT_DIR, "innowah_model.pkl"))
    joblib.dump(scaler,        os.path.join(OUTPUT_DIR, "innowah_scaler.pkl"))
    print(f"\nModel saved → {OUTPUT_DIR}/innowah_model.pkl")

    # ── Write report ──────────────────────────────────────────────────
    y_pred_best = best["model"].predict(X_test_s)
    report_path = os.path.join(OUTPUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write(f"INNOWAH 2026 - Model Training Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Test Accuracy: {best['acc']:.4f}\n")
        f.write(f"CV Accuracy: {best['cv_mean']:.4f}\n")
        f.write(f"ROC-AUC: {best['auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred_best,
                target_names=["Normal", "Mild Risk", "High Risk"]))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred_best)))

        if hasattr(best["model"], "feature_importances_"):
            f.write("\n\nTop Feature Importances:\n")
            importances = best["model"].feature_importances_
            for i in np.argsort(importances)[::-1]:
                f.write(f"  {FEATURE_NAMES[i]:30s}  {importances[i]:.4f}\n")

    print(f"Report saved → {report_path}")
    return best["model"], scaler


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, default=None,  help="Path to real CSV data")
    parser.add_argument("--augment", action="store_true",     help="Mix real + synthetic data")
    parser.add_argument("--samples", type=int, default=3000,  help="Number of synthetic samples")
    args = parser.parse_args()

    if args.data:
        X_real, y_real = load_real_data(args.data)
        if args.augment:
            X_syn, y_syn = generate_synthetic_data(n_samples=args.samples)
            X = np.vstack([X_real, X_syn])
            y = np.concatenate([y_real, y_syn])
            print(f"Combined: {len(X)} samples total")
        else:
            X, y = X_real, y_real
    else:
        X, y = generate_synthetic_data(n_samples=args.samples)

    train(X, y)
