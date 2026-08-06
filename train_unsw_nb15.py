"""
Train & Evaluate Trust-Drift Fusion (TDF) Model on UNSW-NB15 Dataset
======================================================================
This script loads the UNSW-NB15 training and test datasets from GitHub URLs,
preprocesses the features, fits the TDF hybrid model (Isolation Forest +
Resource Drift + Random Forest with weight grid search), evaluates model
performance on the test set, and saves the fitted model and evaluation metrics.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ml_engine.tdf_model import TrustDriftFusionModel, TDFConfig
from ml_engine.unsw_nb15_loader import load_and_preprocess_unsw_nb15


def train_and_evaluate(
    data_dir: str = "data",
    subset_ratio: float = 1.0,
    save_model_name: str = "tdf_model_unsw_nb15.pkl",
):
    print("=" * 70)
    print("  Trust-Drift Fusion (TDF) — UNSW-NB15 Training & Evaluation")
    print("=" * 70)
    start_time = time.time()

    DATA_DIR = os.path.join(PROJECT_ROOT, data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load & Preprocess UNSW-NB15 Data
    # -------------------------------------------------------------
    print("\n[STEP 1/4] Loading & Preprocessing UNSW-NB15 Dataset...")
    data = load_and_preprocess_unsw_nb15(
        data_dir=DATA_DIR,
        subset_ratio=subset_ratio,
        random_state=42
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_cols = data["feature_cols"]
    attack_cat_test = data["attack_cat_test"]

    print(f"  -> Train shape: X={X_train.shape}, y={y_train.shape} (Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)})")
    print(f"  -> Test shape : X={X_test.shape}, y={y_test.shape} (Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)})")
    print(f"  -> Features   : {len(feature_cols)} total features")

    # -------------------------------------------------------------
    # 2. Configure & Train TDF Model
    # -------------------------------------------------------------
    print("\n[STEP 2/4] Training Trust-Drift Fusion (TDF) Model...")
    config = TDFConfig(
        iso_n_estimators=100,
        iso_contamination=0.05,
        rf_n_estimators=100,
        rf_max_depth=12,
        weight_step=0.05,
        val_split=0.2,
        alpha=0.3,
        initial_trust=100.0,
    )

    model = TrustDriftFusionModel(config=config)
    fit_start = time.time()
    model.fit(X_train, y_train, feature_cols=feature_cols)
    fit_duration = time.time() - fit_start

    w_as, w_rds, w_rf = model.state.w_as, model.state.w_rds, model.state.w_rf
    print(f"  -> Fitting complete in {fit_duration:.2f}s")
    print(f"  -> Optimal Fusion Weights: w_AS={w_as:.3f}, w_RDS={w_rds:.3f}, w_RF={w_rf:.3f}")
    print(f"  -> Optimal F1 Threshold : {model.state.optimal_threshold:.3f}")
    print(f"  -> Validation F1 Score  : {model.state.val_f1:.4f}")

    # -------------------------------------------------------------
    # 3. Evaluate Model on UNSW-NB15 Test Set
    # -------------------------------------------------------------
    print("\n[STEP 3/4] Evaluating Model on Test Set...")
    eval_start = time.time()

    # Scores (Hybrid Threat continuous scores)
    ht_scores = model.score(X_test)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    eval_duration = time.time() - eval_start

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, ht_scores)
    cm = confusion_matrix(y_test, preds).tolist()

    print("\n" + "=" * 50)
    print("  EVALUATION METRICS (UNSW-NB15 TEST SET)")
    print("=" * 50)
    print(f"  Accuracy   : {acc * 100:.2f}%")
    print(f"  Precision  : {prec * 100:.2f}%")
    print(f"  Recall     : {rec * 100:.2f}%")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Confusion Matrix [TN, FP / FN, TP]:")
    print(f"    [[{cm[0][0]:>6}, {cm[0][1]:>6}],")
    print(f"     [{cm[1][0]:>6}, {cm[1][1]:>6}]]")

    # Per Attack Category Breakdown
    print("\n  Per-Attack Category Detection Rates:")
    print("  " + "-" * 45)
    attack_breakdown = {}
    if attack_cat_test is not None:
        df_eval = pd.DataFrame({
            "attack_cat": attack_cat_test.values,
            "y_true": y_test,
            "y_pred": preds,
            "ht_score": ht_scores,
        })
        # Normalize category names
        df_eval["attack_cat"] = df_eval["attack_cat"].astype(str).str.strip()
        for cat, group in df_eval.groupby("attack_cat"):
            cat_count = len(group)
            cat_detected = (group["y_pred"] == 1).sum() if cat != "Normal" else (group["y_pred"] == 0).sum()
            cat_rate = cat_detected / cat_count * 100
            avg_ht = group["ht_score"].mean()
            attack_breakdown[cat] = {
                "count": int(cat_count),
                "correct": int(cat_detected),
                "accuracy_pct": float(round(cat_rate, 2)),
                "avg_ht_score": float(round(avg_ht, 4)),
            }
            if cat == "Normal":
                print(f"    Normal Traffic  : {cat_detected:>6}/{cat_count:<6} correctly classified ({cat_rate:>5.1f}%) | Avg HT: {avg_ht:.3f}")
            else:
                print(f"    {cat:<16}: {cat_detected:>6}/{cat_count:<6} detected ({cat_rate:>5.1f}%) | Avg HT: {avg_ht:.3f}")

    # Trust Score severity distribution on test set
    device_sim_scores = []
    for i, ht in enumerate(ht_scores[:1000]):  # Simulate sample sequence
        dev_id = f"device_{i % 10}"
        trust = model.trust_update(float(ht), device_id=dev_id)
        band = model.severity_band(trust)
        device_sim_scores.append((trust, band))

    # -------------------------------------------------------------
    # 4. Save Model & Metrics
    # -------------------------------------------------------------
    print("\n[STEP 4/4] Saving Model Artifact & Metrics...")

    # Save trained TDF model
    model_path = os.path.join(DATA_DIR, save_model_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  -> Saved model pickle: {model_path}")

    # Also update default tdf_model.pkl if requested
    default_model_path = os.path.join(DATA_DIR, "tdf_model.pkl")
    with open(default_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  -> Saved default model: {default_model_path}")

    # Save metrics JSON
    metrics = {
        "dataset": "UNSW-NB15",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_features": int(len(feature_cols)),
        "fit_time_sec": float(round(fit_duration, 2)),
        "eval_time_sec": float(round(eval_duration, 2)),
        "total_time_sec": float(round(time.time() - start_time, 2)),
        "fusion_weights": {
            "w_as": float(round(w_as, 4)),
            "w_rds": float(round(w_rds, 4)),
            "w_rf": float(round(w_rf, 4)),
        },
        "optimal_threshold": float(round(model.state.optimal_threshold, 4)),
        "validation_f1": float(round(model.state.val_f1, 4)),
        "test_metrics": {
            "accuracy": float(round(acc, 4)),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1_score": float(round(f1, 4)),
            "roc_auc": float(round(auc, 4)),
            "confusion_matrix": cm,
        },
        "per_attack_category": attack_breakdown,
    }

    metrics_path = os.path.join(DATA_DIR, "unsw_nb15_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Saved metrics report: {metrics_path}")

    print("\n" + "=" * 70)
    print(f"  SUCCESS! UNSW-NB15 model training & evaluation finished in {time.time() - start_time:.2f}s")
    print("=" * 70)
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
