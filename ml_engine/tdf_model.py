"""
Trust-Drift Fusion (TDF) Model
===============================
Fuses three independent signals into a Hybrid Threat score, then converts
to a continuously updated Trust Score with severity bands.

Signals:
1. Anomaly Score (AS) - Isolation Forest (unsupervised)
2. Resource Drift Score (RDS) - Online Z-score drift detection
3. Random Forest Attack Probability (RF) - Supervised classifier

Fusion: HT = w_as * AS + w_rds * RDS + w_rf * RF
Trust Score: T_t = alpha * (100 - 100 * HT) + (1 - alpha) * T_{t-1}

Severity bands:
- Trusted:        >= 80
- Monitored:      60-79
- Elevated Risk:  35-59
- Critical:       < 35
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from itertools import product
import pickle
import json
import warnings
import os
import yaml

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TDFConfig:
    """Configuration for TDF model hyperparameters."""
    # Isolation Forest params
    iso_n_estimators: int = 100
    iso_contamination: float = 0.05
    iso_random_state: int = 42

    # Random Forest params
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_class_weight: str = "balanced"
    rf_random_state: int = 42

    # Weight grid search
    weight_step: float = 0.05
    val_split: float = 0.2
    val_random_state: int = 42

    # Trust Score EMA
    alpha: float = 0.3
    initial_trust: float = 100.0

    # Severity bands (Trust Score thresholds)
    severity_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Trusted": (80, 100),
        "Monitored": (60, 79),
        "Elevated Risk": (35, 59),
        "Critical": (0, 34),
    })

    # Drift detector params
    rds_z_cap: float = 4.0
    rds_min_std: float = 1e-6

    def to_dict(self) -> Dict:
        return {
            "iso_n_estimators": self.iso_n_estimators,
            "iso_contamination": self.iso_contamination,
            "iso_random_state": self.iso_random_state,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "rf_min_samples_split": self.rf_min_samples_split,
            "rf_min_samples_leaf": self.rf_min_samples_leaf,
            "rf_class_weight": self.rf_class_weight,
            "rf_random_state": self.rf_random_state,
            "weight_step": self.weight_step,
            "val_split": self.val_split,
            "val_random_state": self.val_random_state,
            "alpha": self.alpha,
            "initial_trust": self.initial_trust,
            "severity_bands": self.severity_bands,
            "rds_z_cap": self.rds_z_cap,
            "rds_min_std": self.rds_min_std,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TDFConfig":
        return cls(**d)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TDFConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> "TDFConfig":
        """Load config from YAML file with environment variable overrides."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Flatten nested YAML structure to match dataclass fields
        flat = {}

        # Isolation Forest
        if "isolation_forest" in data:
            flat.update({f"iso_{k}": v for k, v in data["isolation_forest"].items()})

        # Random Forest
        if "random_forest" in data:
            flat.update({f"rf_{k}": v for k, v in data["random_forest"].items()})

        # Fusion weights
        if "fusion_weights" in data:
            flat.update(data["fusion_weights"])

        # Trust Score
        if "trust_score" in data:
            flat.update(data["trust_score"])

        # Severity bands
        if "severity_bands" in data:
            flat["severity_bands"] = {k: tuple(v) for k, v in data["severity_bands"].items()}

        # Drift detector
        if "drift_detector" in data:
            dd = data["drift_detector"]
            flat["rds_z_cap"] = dd.get("z_cap", 4.0)
            rds_min_std = dd.get("min_std", 1e-6)
            flat["rds_min_std"] = float(rds_min_std) if isinstance(rds_min_std, str) else rds_min_std

        # Environment variable overrides
        env_overrides = {
            "iso_n_estimators": "TDF_ISO_N_ESTIMATORS",
            "iso_contamination": "TDF_ISO_CONTAMINATION",
            "rf_n_estimators": "TDF_RF_N_ESTIMATORS",
            "rf_max_depth": "TDF_RF_MAX_DEPTH",
            "weight_step": "TDF_WEIGHT_STEP",
            "val_split": "TDF_VAL_SPLIT",
            "alpha": "TDF_ALPHA",
            "initial_trust": "TDF_INITIAL_TRUST",
            "rds_z_cap": "TDF_RDS_Z_CAP",
            "rds_min_std": "TDF_RDS_MIN_STD",
        }

        for field, env_var in env_overrides.items():
            if env_var in os.environ:
                val = os.environ[env_var]
                # Convert type based on field
                if field in ["iso_n_estimators", "rf_n_estimators", "rf_max_depth"]:
                    flat[field] = int(val)
                elif field in ["iso_contamination", "weight_step", "val_split", "alpha", "initial_trust", "rds_z_cap", "rds_min_std"]:
                    flat[field] = float(val)

        return cls(**flat)


# ============================================
# MODEL STATE (for persistence)
# ============================================

@dataclass
class TDFState:
    """Persisted model state after fitting."""
    # Fitted models
    iso_forest: Optional[IsolationForest] = None
    rf_classifier: Optional[RandomForestClassifier] = None
    scaler: Optional[StandardScaler] = None

    # Drift statistics (computed on normal training data only)
    drift_mean: Optional[np.ndarray] = None
    drift_std: Optional[np.ndarray] = None

    # AS normalization (min/max of raw isolation forest scores on training data)
    as_min: float = 0.0
    as_max: float = 1.0

    # Optimized fusion weights
    w_as: float = 0.33
    w_rds: float = 0.33
    w_rf: float = 0.34

    # Optimal threshold for predict()
    optimal_threshold: float = 0.5

    # Best validation F1 score
    val_f1: float = 0.0

    # Feature columns used during training
    feature_cols: List[str] = field(default_factory=list)

    # Config used
    config: Optional[TDFConfig] = None

    def to_dict(self) -> Dict:
        return {
            "iso_forest": self.iso_forest,
            "rf_classifier": self.rf_classifier,
            "scaler": self.scaler,
            "drift_mean": self.drift_mean.tolist() if self.drift_mean is not None else None,
            "drift_std": self.drift_std.tolist() if self.drift_std is not None else None,
            "as_min": self.as_min,
            "as_max": self.as_max,
            "w_as": self.w_as,
            "w_rds": self.w_rds,
            "w_rf": self.w_rf,
            "optimal_threshold": self.optimal_threshold,
            "val_f1": self.val_f1,
            "feature_cols": self.feature_cols,
            "config": self.config.to_dict() if self.config else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TDFState":
        state = cls()
        state.iso_forest = d.get("iso_forest")
        state.rf_classifier = d.get("rf_classifier")
        state.scaler = d.get("scaler")
        state.drift_mean = np.array(d["drift_mean"]) if d.get("drift_mean") is not None else None
        state.drift_std = np.array(d["drift_std"]) if d.get("drift_std") is not None else None
        state.as_min = d.get("as_min", 0.0)
        state.as_max = d.get("as_max", 1.0)
        state.w_as = d.get("w_as", 0.33)
        state.w_rds = d.get("w_rds", 0.33)
        state.w_rf = d.get("w_rf", 0.34)
        state.optimal_threshold = d.get("optimal_threshold", 0.5)
        state.val_f1 = d.get("val_f1", 0.0)
        state.feature_cols = d.get("feature_cols", [])
        state.config = TDFConfig.from_dict(d["config"]) if d.get("config") else None
        return state


# ============================================
# TDF MODEL CLASS
# ============================================

class TrustDriftFusionModel:
    """
    Trust-Drift Fusion (TDF) Model.

    Fuses three signals into a Hybrid Threat score:
      - AS: Isolation Forest anomaly score (unsupervised)
      - RDS: Resource Drift Score (statistical, per-feature Z-score)
      - RF: Random Forest attack probability (supervised)

    Provides sklearn-compatible interface: fit(), score(), predict(), predict_proba()
    Plus stateful trust_update() for EMA-based trust scoring.
    """

    def __init__(self, config: Optional[TDFConfig] = None):
        self.config = config or TDFConfig()
        self.state = TDFState(config=self.config)
        self._is_fitted = False
        self._trust_history: Dict[str, float] = {}  # device_id -> last trust score

    # ---------------------------------------------------------
    # PUBLIC API: sklearn-compatible
    # ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols: Optional[List[str]] = None) -> "TrustDriftFusionModel":
        """
        Fit the TDF model on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0 = normal, 1 = attack/anomaly)
            feature_cols: Optional list of feature names

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        if feature_cols is None:
            feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

        self.state.feature_cols = feature_cols
        n_samples, n_features = X.shape

        # Split training data into train/val for weight tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.val_split,
            random_state=self.config.val_random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # -----------------------------------------------------
        # 1. Fit Isolation Forest on NORMAL data only (unsupervised)
        # -----------------------------------------------------
        normal_mask = (y_train == 0)
        if normal_mask.sum() == 0:
            raise ValueError("No normal (label=0) samples in training data for Isolation Forest")

        X_normal = X_train[normal_mask]

        self.state.scaler = StandardScaler()
        X_normal_scaled = self.state.scaler.fit_transform(X_normal)

        self.state.iso_forest = IsolationForest(
            n_estimators=self.config.iso_n_estimators,
            contamination=self.config.iso_contamination,
            random_state=self.config.iso_random_state,
        )
        self.state.iso_forest.fit(X_normal_scaled)

        # Compute AS normalization bounds on ALL training data (normal + attack)
        X_train_scaled = self.state.scaler.transform(X_train)
        iso_scores_train = self.state.iso_forest.score_samples(X_train_scaled)
        # score_samples: higher = more normal, lower = more anomalous
        # We want AS in [0,1] where higher = more anomalous
        raw_as = -iso_scores_train  # negative so higher = more anomalous
        self.state.as_min = float(raw_as.min())
        self.state.as_max = float(raw_as.max())
        if self.state.as_max - self.state.as_min < 1e-10:
            self.state.as_max = self.state.as_min + 1.0

        # -----------------------------------------------------
        # 2. Compute Drift Statistics (RDS) on NORMAL data only
        # -----------------------------------------------------
        self.state.drift_mean = X_normal.mean(axis=0)
        self.state.drift_std = X_normal.std(axis=0)
        self.state.drift_std = np.maximum(self.state.drift_std, self.config.rds_min_std)

        # -----------------------------------------------------
        # 3. Fit Random Forest Classifier (supervised)
        # -----------------------------------------------------
        X_train_scaled = self.state.scaler.transform(X_train)

        self.state.rf_classifier = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            class_weight=self.config.rf_class_weight,
            random_state=self.config.rf_random_state,
            n_jobs=-1,
        )
        self.state.rf_classifier.fit(X_train_scaled, y_train)

        # -----------------------------------------------------
        # 4. Compute three signals on validation set
        # -----------------------------------------------------
        X_val_scaled = self.state.scaler.transform(X_val)

        as_val = self._compute_as(X_val_scaled)
        rds_val = self._compute_rds(X_val)
        rf_val = self._compute_rf(X_val_scaled)

        # -----------------------------------------------------
        # 5. Grid search over weight simplex for best F1
        # -----------------------------------------------------
        self._grid_search_weights(as_val, rds_val, rf_val, y_val)

        self._is_fitted = True
        return self

    def _compute_as(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute Anomaly Score from Isolation Forest."""
        raw_scores = -self.state.iso_forest.score_samples(X_scaled)  # higher = more anomalous
        as_scores = (raw_scores - self.state.as_min) / (self.state.as_max - self.state.as_min + 1e-10)
        return np.clip(as_scores, 0.0, 1.0)

    def _compute_rds(self, X: np.ndarray) -> np.ndarray:
        """Compute Resource Drift Score from per-feature Z-scores."""
        z_scores = np.abs(X - self.state.drift_mean) / self.state.drift_std
        z_scores = np.clip(z_scores, 0, self.config.rds_z_cap)
        rds = z_scores.mean(axis=1) / self.config.rds_z_cap
        return np.clip(rds, 0.0, 1.0)

    def _compute_rf(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute Random Forest attack probability."""
        return self.state.rf_classifier.predict_proba(X_scaled)[:, 1]

    def _grid_search_weights(self, as_val: np.ndarray, rds_val: np.ndarray, rf_val: np.ndarray, y_val: np.ndarray):
        """Grid search over weight simplex to maximize validation F1."""
        step = self.config.weight_step
        best_f1 = -1.0
        best_weights = (0.33, 0.33, 0.34)
        best_threshold = 0.5

        # Generate all weight combinations that sum to 1
        weights_list = []
        n_steps = int(1.0 / step) + 1
        for i in range(n_steps):
            w_as = i * step
            for j in range(n_steps - i):
                w_rds = j * step
                w_rf = 1.0 - w_as - w_rds
                if w_rf >= -1e-10:  # allow small numerical error
                    weights_list.append((w_as, w_rds, max(0.0, w_rf)))

        # For each weight combo, find optimal threshold on validation set
        for w_as, w_rds, w_rf in weights_list:
            ht = w_as * as_val + w_rds * rds_val + w_rf * rf_val

            # Sweep thresholds to maximize F1
            thresholds = np.linspace(0.0, 1.0, 101)
            for thresh in thresholds:
                y_pred = (ht >= thresh).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = (w_as, w_rds, w_rf)
                    best_threshold = thresh

        self.state.w_as, self.state.w_rds, self.state.w_rf = best_weights
        self.state.optimal_threshold = best_threshold
        self.state.val_f1 = best_f1

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous Hybrid Threat score in [0, 1].

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            HT scores in [0, 1] (higher = more threatening)
        """
        self._check_fitted()
        X = np.asarray(X)
        X_scaled = self.state.scaler.transform(X)

        as_scores = self._compute_as(X_scaled)
        rds_scores = self._compute_rds(X)
        rf_scores = self._compute_rf(X_scaled)

        ht = (self.state.w_as * as_scores +
              self.state.w_rds * rds_scores +
              self.state.w_rf * rf_scores)

        return np.clip(ht, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels (0 = normal, 1 = threat) using optimal threshold.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Binary predictions (0 or 1)
        """
        ht = self.score(X)
        return (ht >= self.state.optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for sklearn compatibility.

        Returns:
            2-column array [P(class=0), P(class=1)] where class=1 is threat
        """
        ht = self.score(X)
        return np.column_stack([1 - ht, ht])

    # ---------------------------------------------------------
    # TRUST SCORE API (stateful EMA)
    # ---------------------------------------------------------

    def trust_update(self, ht: float, prev_trust: Optional[float] = None, device_id: Optional[str] = None) -> float:
        """
        Compute updated Trust Score using EMA.

        Args:
            ht: Hybrid Threat score in [0, 1]
            prev_trust: Previous trust score (0-100). If None and device_id provided,
                       uses stored history. If neither, uses initial_trust.
            device_id: Optional device identifier for stateful tracking

        Returns:
            Updated trust score in [0, 100]
        """
        alpha = self.config.alpha
        initial = self.config.initial_trust

        if prev_trust is None:
            if device_id is not None and device_id in self._trust_history:
                prev_trust = self._trust_history[device_id]
            else:
                prev_trust = initial

        # Trust = 100 - 100*HT (so HT=0 -> trust=100, HT=1 -> trust=0)
        current_trust = 100.0 - 100.0 * ht
        new_trust = alpha * current_trust + (1 - alpha) * prev_trust
        new_trust = np.clip(new_trust, 0.0, 100.0)

        if device_id is not None:
            self._trust_history[device_id] = new_trust

        return float(new_trust)

    def trust_score(self, X: np.ndarray, device_ids: Optional[List[str]] = None,
                    prev_trusts: Optional[List[float]] = None) -> np.ndarray:
        """
        Compute trust scores for a batch with optional EMA smoothing.

        Args:
            X: Feature matrix
            device_ids: Optional list of device IDs for stateful tracking
            prev_trusts: Optional list of previous trust scores

        Returns:
            Trust scores in [0, 100]
        """
        ht = self.score(X)
        n = len(ht)
        trusts = np.zeros(n)

        for i in range(n):
            device_id = device_ids[i] if device_ids else None
            prev = prev_trusts[i] if prev_trusts else None
            trusts[i] = self.trust_update(ht[i], prev, device_id)

        return trusts

    def severity_band(self, trust_score: float) -> str:
        """
        Map trust score to severity band.

        Bands:
            Trusted:        >= 80
            Monitored:      60-79
            Elevated Risk:  35-59
            Critical:       < 35
        """
        if trust_score >= 80:
            return "Trusted"
        elif trust_score >= 60:
            return "Monitored"
        elif trust_score >= 35:
            return "Elevated Risk"
        else:
            return "Critical"

    def severity_bands_batch(self, trust_scores: np.ndarray) -> List[str]:
        """Map array of trust scores to severity bands."""
        return [self.severity_band(float(t)) for t in trust_scores]

    def get_signal_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual signal components for explainability.

        Returns:
            Dict with keys: 'as', 'rds', 'rf', 'ht'
        """
        self._check_fitted()
        X = np.asarray(X)
        X_scaled = self.state.scaler.transform(X)

        as_scores = self._compute_as(X_scaled)
        rds_scores = self._compute_rds(X)
        rf_scores = self._compute_rf(X_scaled)
        ht = (self.state.w_as * as_scores +
              self.state.w_rds * rds_scores +
              self.state.w_rf * rf_scores)

        return {
            "as": as_scores,
            "rds": rds_scores,
            "rf": rf_scores,
            "ht": np.clip(ht, 0.0, 1.0),
        }

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------

    def save(self, path: str):
        """Save model to disk using pickle."""
        state_dict = self.state.to_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, path: str) -> "TrustDriftFusionModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        config = TDFConfig.from_dict(state_dict.pop("config")) if state_dict.get("config") else TDFConfig()
        model = cls(config=config)
        model.state = TDFState.from_dict(state_dict)
        model._is_fitted = True
        return model

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling this method. Call fit() first.")

    def get_params(self) -> Dict:
        """Get model parameters for sklearn compatibility."""
        return self.config.to_dict()

    def set_params(self, **params) -> "TrustDriftFusionModel":
        """Set model parameters for sklearn compatibility."""
        for k, v in params.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"TrustDriftFusionModel(fitted=True, "
                    f"w_as={self.state.w_as:.3f}, w_rds={self.state.w_rds:.3f}, w_rf={self.state.w_rf:.3f}, "
                    f"thresh={self.state.optimal_threshold:.3f})")
        return "TrustDriftFusionModel(fitted=False)"


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def create_tdf_model(config: Optional[TDFConfig] = None) -> TrustDriftFusionModel:
    """Factory function to create a TDF model with default config."""
    return TrustDriftFusionModel(config=config)


def load_tdf_model(path: str) -> TrustDriftFusionModel:
    """Load a TDF model from disk."""
    return TrustDriftFusionModel.load(path)


# ============================================
# SELF-TEST / DEMO
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trust-Drift Fusion (TDF) Model - Self Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_normal = 500
    n_attack = 100
    n_features = 9

    # Normal data: centered at 0
    X_normal = np.random.randn(n_normal, n_features) * 0.5
    y_normal = np.zeros(n_normal)

    # Attack data: shifted mean
    X_attack = np.random.randn(n_attack, n_features) * 0.5 + 2.0
    y_attack = np.ones(n_attack)

    X = np.vstack([X_normal, X_attack])
    y = np.hstack([y_normal, y_attack])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    feature_cols = [
        "total_bytes_out", "total_packets_out", "avg_bytes_per_flow",
        "num_flows", "unique_dst_ips", "unique_dst_ports",
        "unique_protocols", "external_ratio", "avg_duration"
    ]

    # Train TDF
    print("\n[1] Training TDF model...")
    config = TDFConfig(
        iso_n_estimators=50,
        rf_n_estimators=50,
        weight_step=0.1,  # Coarser for faster test
        val_split=0.3,
    )
    model = TrustDriftFusionModel(config=config)
    model.fit(X, y, feature_cols=feature_cols)
    print(f"    Fitted: {model}")

    # Test score/predict
    print("\n[2] Testing score() and predict()...")
    ht = model.score(X[:10])
    preds = model.predict(X[:10])
    proba = model.predict_proba(X[:10])
    print(f"    HT scores: {ht}")
    print(f"    Predictions: {preds}")
    print(f"    Probabilities: {proba}")

    # Test trust scoring
    print("\n[3] Testing trust_update()...")
    for ht_val in [0.1, 0.3, 0.6, 0.9]:
        trust = model.trust_update(ht_val, prev_trust=100.0)
        severity = model.severity_band(trust)
        print(f"    HT={ht_val:.1f} -> Trust={trust:.1f} -> {severity}")

    # Test stateful trust with device_id
    print("\n[4] Testing stateful trust with device_id...")
    device_id = "CCTV_01"
    for ht_val in [0.1, 0.2, 0.5, 0.3]:
        trust = model.trust_update(ht_val, device_id=device_id)
        severity = model.severity_band(trust)
        print(f"    HT={ht_val:.1f} -> Trust={trust:.1f} -> {severity}")

    # Test signal components
    print("\n[5] Testing signal decomposition...")
    components = model.get_signal_components(X[:3])
    for k, v in components.items():
        print(f"    {k}: {v}")

    # Test save/load
    print("\n[6] Testing save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model.save(f.name)
        loaded = TrustDriftFusionModel.load(f.name)
        print(f"    Loaded: {loaded}")
        ht_orig = model.score(X[:3])
        ht_loaded = loaded.score(X[:3])
        print(f"    Scores match: {np.allclose(ht_orig, ht_loaded)}")

    # Test severity bands
    print("\n[7] Testing severity band boundaries...")
    for score in [100, 80, 79, 60, 59, 35, 34, 0]:
        print(f"    Trust={score} -> {model.severity_band(score)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)