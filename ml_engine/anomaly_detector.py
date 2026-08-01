"""
ML Engine — Trust-Drift Fusion (TDF) Anomaly Detection
======================================================
Replaces the old per-device-type Isolation Forest with a unified TDF model
that fuses three signals: Anomaly Score (AS), Resource Drift Score (RDS),
and Random Forest attack probability (RF).

The TDF model provides:
- .fit(X_train, y_train) - trains all three components + auto-tunes fusion weights
- .score(X) - returns continuous Hybrid Threat score in [0, 1]
- .predict(X) - returns binary predictions using F1-optimal threshold
- .predict_proba(X) - returns 2-column array [1-score, score]
- .trust_update(ht, device_id) - stateful EMA trust scoring per device
- .severity_band(trust_score) - maps trust to Trusted/Monitored/Elevated/Critical

Pipeline integration:
- train_tdf(baseline_features) -> trains on normal (baseline) + labeled attack data
- score_all_devices(model, all_features) -> scores all windows with TDF
"""

import os
import numpy as np
import pandas as pd
import warnings

from .tdf_model import TrustDriftFusionModel, TDFConfig

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FEATURE_COLS = [
    "total_bytes_out",
    "total_packets_out",
    "avg_bytes_per_flow",
    "num_flows",
    "unique_dst_ips",
    "unique_dst_ports",
    "unique_protocols",
    "external_ratio",
    "avg_duration",
]

DEVICE_TYPE_MAP = {
    "CCTV": "CCTV",
    "Router": "Router",
    "AccessController": "AccessController",
}


# ============================================
# DATA LOADING (unchanged from legacy)
# ============================================

def load_features(filename="feature_vectors.csv"):
    """Load feature vectors from data/ folder."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    # Rename window_start to window if needed
    if "window_start" in df.columns:
        df = df.rename(columns={"window_start": "window"})
    # Parse window as datetime
    df["window"] = pd.to_datetime(df["window"])
    # Add device_type if missing (derive from device_id)
    if "device_type" not in df.columns:
        def get_device_type(device_id):
            if "CCTV" in device_id:
                return "CCTV"
            elif "Router" in device_id:
                return "Router"
            elif "Access" in device_id:
                return "AccessController"
            return "Unknown"
        df["device_type"] = df["device_id"].apply(get_device_type)
    return df


def split_baseline_and_full(features_df, baseline_hours=48):
    """
    Split features into baseline (first N hours) and full dataset.
    Baseline = assumed clean data for training.
    """
    min_time = features_df["window"].min()
    baseline_cutoff = min_time + pd.Timedelta(hours=baseline_hours)

    baseline = features_df[features_df["window"] < baseline_cutoff].copy()
    return baseline, features_df.copy()


def load_labels_for_features(features_df):
    """
    Load attack labels from feature vectors (already aligned) or full_dataset.csv as fallback.
    Returns a Series of labels (0=normal, 1=attack) aligned with features_df index.
    """
    # First check if features_df already has a 'label' column (from feature extraction)
    if "label" in features_df.columns:
        return features_df["label"].astype(int)

    # Fallback: load from full_dataset.csv
    full_path = os.path.join(DATA_DIR, "full_dataset.csv")
    if not os.path.exists(full_path):
        # No labels available - assume all normal (baseline only)
        return pd.Series(0, index=features_df.index)

    full_df = pd.read_csv(full_path, parse_dates=["timestamp"])
    full_df["window"] = full_df["timestamp"].dt.floor("h")

    # Map labels: normal=0, attack=1
    label_map = {"normal": 0}
    attack_labels = full_df[full_df["label"] != "normal"]["label"].unique()
    for al in attack_labels:
        label_map[al] = 1

    full_df["label_num"] = full_df["label"].map(label_map).fillna(0).astype(int)

    # Aggregate to window level: max label (1 if any attack in window)
    window_labels = full_df.groupby(["device_id", "window"])["label_num"].max().reset_index()
    window_labels.columns = ["device_id", "window", "label"]

    # Merge with features
    merged = features_df.merge(window_labels, on=["device_id", "window"], how="left")
    labels = merged["label"].fillna(0).astype(int)

    return labels


# ============================================
# TDF TRAINING & SCORING
# ============================================

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "tdf_config.yaml")


def load_tdf_config(config_path=None):
    """Load TDF configuration from YAML file, with fallback to defaults."""
    path = config_path or CONFIG_PATH
    if os.path.exists(path):
        print(f"  Loading TDF config from: {path}")
        return TDFConfig.from_yaml(path)
    else:
        print(f"  Config not found at {path}, using defaults")
        return TDFConfig()


def train_tdf(baseline_features, config=None):
    """
    Train the TDF model on baseline (normal) + labeled attack data.

    The TDF model needs both normal and attack samples for supervised RF component.
    We use:
    - Baseline (normal) data for Isolation Forest + drift statistics
    - Labeled attack data from full_dataset.csv for Random Forest

    Args:
        baseline_features: DataFrame with baseline (normal) feature vectors
        config: Optional TDFConfig for hyperparameters

    Returns:
        Trained TrustDriftFusionModel
    """
    if config is None:
        config = load_tdf_config()

    print("=" * 55)
    print("  TDF Model Training")
    print("=" * 55)

    # Get labels for baseline data (should be all normal=0)
    baseline_labels = load_labels_for_features(baseline_features)
    n_baseline = len(baseline_features)
    print(f"  Baseline samples: {n_baseline} (labels: {baseline_labels.sum()} attacks)")

    # Also load attack samples from full dataset for supervised training
    full_features = load_features("feature_vectors.csv")
    full_labels = load_labels_for_features(full_features)

    # Combine baseline + attack samples for training
    # Use all data but weight appropriately
    X_all = full_features[FEATURE_COLS].values
    y_all = full_labels.values

    print(f"  Total training samples: {len(X_all)} (attacks: {y_all.sum()})")

    # Initialize and fit TDF model
    model = TrustDriftFusionModel(config=config)
    model.fit(X_all, y_all, feature_cols=FEATURE_COLS)

    # Print model info
    print(f"\n  TDF Model Summary:")
    print(f"    Fusion weights: AS={model.state.w_as:.3f}, RDS={model.state.w_rds:.3f}, RF={model.state.w_rf:.3f}")
    print(f"    Optimal threshold: {model.state.optimal_threshold:.4f}")
    print(f"    Val F1: {model.state.val_f1:.4f}")

    return model


def score_all_devices(model, all_features):
    """
    Score all device windows using the TDF model.

    Args:
        model: Trained TrustDriftFusionModel
        all_features: DataFrame with all feature vectors

    Returns:
        DataFrame with anomaly_score (HT), is_anomaly, trust_score, severity
    """
    X = all_features[FEATURE_COLS].values

    # Get Hybrid Threat scores (continuous risk in [0,1])
    ht_scores = model.score(X)

    # Get binary predictions
    is_anomaly = model.predict(X)

    # Get trust scores with EMA smoothing per device
    trust_scores = []
    severities = []

    for idx, row in all_features.iterrows():
        device_id = row["device_id"]
        ht = ht_scores[idx] if isinstance(idx, int) else ht_scores[len(trust_scores)]
        trust = model.trust_update(ht, device_id=device_id)
        severity = model.severity_band(trust)
        trust_scores.append(trust)
        severities.append(severity)

    # Build results DataFrame
    results = all_features.copy()
    results["anomaly_score"] = ht_scores  # HT in [0,1]
    results["is_anomaly"] = is_anomaly.astype(int)
    results["trust_score"] = trust_scores
    results["severity"] = severities

    # Add raw component scores for explainability
    components = model.get_signal_components(X)
    results["as_score"] = components["as"]
    results["rds_score"] = components["rds"]
    results["rf_score"] = components["rf"]

    return results


def compute_baseline(baseline_features):
    """
    Compute per-device baseline statistics (mean/std) for drift detection.
    Kept for backward compatibility with drift_detector.
    """
    stats = {}
    for device_id in baseline_features["device_id"].unique():
        device_data = baseline_features[baseline_features["device_id"] == device_id]
        device_stats = {}
        for feat in FEATURE_COLS:
            device_stats[feat] = {
                "mean": device_data[feat].mean(),
                "std": device_data[feat].std() if device_data[feat].std() > 0 else 1e-6,
            }
        stats[device_id] = device_stats
    return stats


# ============================================
# MAIN — Run anomaly detection pipeline
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("  ML Engine — TDF Anomaly Detection Pipeline")
    print("=" * 55)

    # Step 1: Load feature vectors
    print("\n[1/4] Loading feature vectors...")
    features = load_features("feature_vectors.csv")
    print(f"  Loaded {len(features)} feature vectors")
    print(f"  Devices: {features['device_id'].unique().tolist()}")
    print(f"  Time range: {features['window'].min()} to {features['window'].max()}")

    # Step 2: Split into baseline and full
    print("\n[2/4] Splitting baseline (first 48 hours)...")
    baseline, full = split_baseline_and_full(features, baseline_hours=48)
    print(f"  Baseline: {len(baseline)} windows")
    print(f"  Full:     {len(full)} windows")

    # Step 3: Train TDF model
    print("\n[3/4] Training TDF model...")
    model = train_tdf(baseline)

    # Step 4: Score all windows
    print("\n[4/4] Scoring all windows...")
    scored = score_all_devices(model, full)

    # Results summary
    print("\n" + "=" * 55)
    print("  TDF ANOMALY DETECTION RESULTS")
    print("=" * 55)

    for device_id in scored["device_id"].unique():
        dev_data = scored[scored["device_id"] == device_id]
        normal_data = dev_data[dev_data["is_anomaly"] == 0]
        anomaly_data = dev_data[dev_data["is_anomaly"] == 1]

        print(f"\n  {device_id}:")
        print(f"    Total windows:   {len(dev_data)}")
        print(f"    Normal windows:  {len(normal_data)}")
        print(f"    Anomaly windows: {len(anomaly_data)}")
        if len(anomaly_data) > 0:
            print(f"    Anomaly scores:  min={anomaly_data['anomaly_score'].min():.3f}  max={anomaly_data['anomaly_score'].max():.3f}")
        if len(normal_data) > 0:
            print(f"    Normal scores:   min={normal_data['anomaly_score'].min():.3f}  max={normal_data['anomaly_score'].max():.3f}")

        # Trust score summary
        print(f"    Trust scores:    min={dev_data['trust_score'].min():.1f}  max={dev_data['trust_score'].max():.1f}")
        print(f"    Severities:      {dev_data['severity'].value_counts().to_dict()}")

    # Save results
    output_path = os.path.join(DATA_DIR, "anomaly_scores.csv")
    scored.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print("=" * 55)

    # Save model
    model_path = os.path.join(DATA_DIR, "tdf_model.pkl")
    model.save(model_path)
    print(f"  Model saved to: {model_path}")