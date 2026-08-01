"""
Feature extraction from raw flow data to feature vectors.
Generates feature_vectors.csv from full_dataset.csv (which should include both normal and attack data).
"""

import pandas as pd
import numpy as np
import os
import sys

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


def is_internal_ip(ip: str) -> bool:
    """Return True if IP is a private/internal address (RFC-1918)."""
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or ip.startswith("172.16.") or ip.startswith("172.17.")
        or ip.startswith("172.18.") or ip.startswith("172.19.")
        or ip.startswith("172.20.") or ip.startswith("172.21.")
        or ip.startswith("172.22.") or ip.startswith("172.23.")
        or ip.startswith("172.24.") or ip.startswith("172.25.")
        or ip.startswith("172.26.") or ip.startswith("172.27.")
        or ip.startswith("172.28.") or ip.startswith("172.29.")
        or ip.startswith("172.30.") or ip.startswith("172.31.")
        or ip == "127.0.0.1"
    )


def extract_features_from_flows(df):
    """Extract feature vectors from flow-level data grouped by device_id and hour window."""
    df = df.copy()
    df["window"] = df["timestamp"].dt.floor("h")
    df["is_external"] = df["dst_ip"].apply(lambda ip: 0 if is_internal_ip(str(ip)) else 1)

    # Group by device_id and window
    grouped = df.groupby(["device_id", "window"])

    features = []

    for (device_id, window), group in grouped:
        feat = {
            "device_id": device_id,
            "window": window,
            "total_bytes_out": float(group["bytes_out"].sum()),
            "total_packets_out": float(group["packets_out"].sum()),
            "avg_bytes_per_flow": float(group["bytes_out"].mean()),
            "num_flows": float(len(group)),
            "unique_dst_ips": float(group["dst_ip"].nunique()),
            "unique_dst_ports": float(group["dst_port"].nunique()),
            "unique_protocols": float(group["protocol"].nunique()),
            "external_ratio": float(group["is_external"].mean()),
            "avg_duration": float(group["duration"].mean()),
        }
        features.append(feat)

    return pd.DataFrame(features)


def get_device_type(device_id: str) -> str:
    if "CCTV" in device_id:
        return "CCTV"
    elif "Router" in device_id:
        return "Router"
    elif "Access" in device_id:
        return "AccessController"
    return "Unknown"


def main():
    print("=" * 55)
    print("  Feature Extraction from Full Dataset")
    print("=" * 55)

    # Load full dataset (normal + attack)
    full_path = os.path.join(DATA_DIR, "full_dataset.csv")
    attack_path = os.path.join(DATA_DIR, "attack_data.csv")

    print(f"\n[1/3] Loading full dataset from {full_path}...")
    full_df = pd.read_csv(full_path, parse_dates=["timestamp"])
    print(f"  Loaded {len(full_df)} normal flows")

    if os.path.exists(attack_path):
        print(f"\n[2/3] Loading attack data from {attack_path}...")
        attack_df = pd.read_csv(attack_path, parse_dates=["timestamp"])
        print(f"  Loaded {len(attack_df)} attack flows")

        # Combine
        combined = pd.concat([full_df, attack_df], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        print("\n[2/3] No attack data found, using full dataset only")
        combined = full_df

    print(f"\n  Total flows: {len(combined)}")
    print(f"  Labels: {combined['label'].unique()}")
    print(f"  Label counts:\n{combined['label'].value_counts()}")

    # Extract features
    print("\n[3/3] Extracting feature vectors...")
    features_df = extract_features_from_flows(combined)

    # Add device_type
    features_df["device_type"] = features_df["device_id"].apply(get_device_type)

    # Add label (max label per window: 0=normal, 1=attack)
    # We need to get the max label per window from the combined data
    label_map = {"normal": 0, "attack_exfiltration": 1, "attack_c2": 1, "attack_scanning": 1}
    combined["label_num"] = combined["label"].map(label_map).fillna(0).astype(int)
    combined["window"] = combined["timestamp"].dt.floor("h")
    window_labels = combined.groupby(["device_id", "window"])["label_num"].max().reset_index()
    window_labels.columns = ["device_id", "window", "label"]

    features_df = features_df.merge(window_labels, on=["device_id", "window"], how="left")
    features_df["label"] = features_df["label"].fillna(0).astype(int)

    print(f"  Generated {len(features_df)} feature vectors")
    print(f"  Labels: {features_df['label'].value_counts().to_dict()}")

    # Save
    output_path = os.path.join(DATA_DIR, "feature_vectors.csv")
    features_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    # Also save a copy for reference
    features_df.to_csv(os.path.join(DATA_DIR, "feature_vectors_original.csv"), index=False)

    print("\n" + "=" * 55)
    print("  Feature Extraction Complete")
    print("=" * 55)


if __name__ == "__main__":
    main()