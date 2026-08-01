"""
Unit tests for Trust-Drift Fusion (TDF) Model.
Tests: fit/score/predict on synthetic data, weight grid search, severity bands,
trust_update EMA, save/load persistence, signal components.
"""

import numpy as np
import tempfile
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ml_engine.tdf_model import TrustDriftFusionModel, TDFConfig


def create_synthetic_data(n_normal=500, n_attack=100, n_features=9, seed=42):
    """Generate synthetic normal and attack data for testing."""
    np.random.seed(seed)
    # Normal data: centered at 0
    X_normal = np.random.randn(n_normal, n_features) * 0.5
    y_normal = np.zeros(n_normal)
    # Attack data: shifted mean
    X_attack = np.random.randn(n_attack, n_features) * 0.5 + 2.0
    y_attack = np.ones(n_attack)
    # Combine and shuffle
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([y_normal, y_attack])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def test_fit_score_predict():
    """Test fit(), score(), predict(), predict_proba() on synthetic data."""
    print("Test 1: fit/score/predict on synthetic data...")
    X, y = create_synthetic_data()
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

    config = TDFConfig(
        iso_n_estimators=50,
        rf_n_estimators=50,
        weight_step=0.1,
        val_split=0.3,
    )
    model = TrustDriftFusionModel(config=config)
    model.fit(X, y, feature_cols=feature_cols)

    # Test score() returns HT in [0, 1]
    ht = model.score(X)
    assert ht.shape == (len(X),), f"Expected shape {(len(X),)}, got {ht.shape}"
    assert np.all(ht >= 0) and np.all(ht <= 1), f"HT scores not in [0,1]: min={ht.min()}, max={ht.max()}"

    # Test predict() returns binary 0/1
    preds = model.predict(X)
    assert np.all(np.isin(preds, [0, 1])), "Predictions not binary"
    assert preds.shape == (len(X),)

    # Test predict_proba() returns 2-column array
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected shape {(len(X), 2)}, got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.allclose(proba[:, 1], ht), "proba[:,1] should equal HT score"

    print("  PASSED: fit/score/predict/predict_proba work correctly")


def test_weight_grid_search():
    """Test that grid search produces weights summing to 1."""
    print("Test 2: Weight grid search produces valid weights...")
    X, y = create_synthetic_data()
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

    config = TDFConfig(
        iso_n_estimators=30,
        rf_n_estimators=30,
        weight_step=0.1,
        val_split=0.3,
    )
    model = TrustDriftFusionModel(config=config)
    model.fit(X, y, feature_cols=feature_cols)

    w_as, w_rds, w_rf = model.state.w_as, model.state.w_rds, model.state.w_rf
    assert abs((w_as + w_rds + w_rf) - 1.0) < 1e-10, f"Weights sum to {w_as + w_rds + w_rf}, not 1"
    assert all(w >= 0 for w in [w_as, w_rds, w_rf]), "Negative weight found"
    assert model.state.optimal_threshold >= 0 and model.state.optimal_threshold <= 1, "Threshold out of range"
    assert model.state.val_f1 >= 0, "Validation F1 should be non-negative"

    print(f"  PASSED: weights=({w_as:.3f}, {w_rds:.3f}, {w_rf:.3f}), threshold={model.state.optimal_threshold:.3f}, val_f1={model.state.val_f1:.3f}")


def test_severity_band_boundaries():
    """Test severity band mapping at exact boundaries."""
    print("Test 3: Severity band boundaries...")
    model = TrustDriftFusionModel()

    # Test exact boundaries
    assert model.severity_band(100) == "Trusted"
    assert model.severity_band(80) == "Trusted"
    assert model.severity_band(79) == "Monitored"
    assert model.severity_band(60) == "Monitored"
    assert model.severity_band(59) == "Elevated Risk"
    assert model.severity_band(35) == "Elevated Risk"
    assert model.severity_band(34) == "Critical"
    assert model.severity_band(0) == "Critical"

    # Test batch version
    scores = np.array([100, 80, 79, 60, 59, 35, 34, 0])
    bands = model.severity_bands_batch(scores)
    expected = ["Trusted", "Trusted", "Monitored", "Monitored", "Elevated Risk", "Elevated Risk", "Critical", "Critical"]
    assert bands == expected, f"Expected {expected}, got {bands}"

    print("  PASSED: All severity boundaries correct")


def test_trust_update_ema():
    """Test trust_update EMA behavior (stateless and stateful)."""
    print("Test 4: Trust update EMA...")
    model = TrustDriftFusionModel(config=TDFConfig(alpha=0.3, initial_trust=100.0))

    # Stateless: prev_trust provided explicitly
    trust1 = model.trust_update(ht=0.0, prev_trust=100.0)  # HT=0 -> trust=100
    assert trust1 == 100.0, f"HT=0 should give trust=100, got {trust1}"

    trust2 = model.trust_update(ht=1.0, prev_trust=100.0)  # HT=1 -> trust=0, EMA: 0.3*0 + 0.7*100 = 70
    assert abs(trust2 - 70.0) < 1e-10, f"Expected 70.0, got {trust2}"

    trust3 = model.trust_update(ht=0.5, prev_trust=100.0)  # HT=0.5 -> trust=50, EMA: 0.3*50 + 0.7*100 = 85
    assert abs(trust3 - 85.0) < 1e-10, f"Expected 85.0, got {trust3}"

    # Stateful: device_id tracking
    model2 = TrustDriftFusionModel(config=TDFConfig(alpha=0.3, initial_trust=100.0))
    device = "TEST_DEVICE"
    t1 = model2.trust_update(0.0, device_id=device)  # 100
    t2 = model2.trust_update(0.2, device_id=device)  # 0.3*80 + 0.7*100 = 94
    t3 = model2.trust_update(0.5, device_id=device)  # 0.3*50 + 0.7*94 = 80.8
    t4 = model2.trust_update(0.3, device_id=device)  # 0.3*70 + 0.7*80.8 = 77.56

    assert t1 == 100.0, f"First update: {t1}"
    assert abs(t2 - 94.0) < 1e-10, f"Second update: {t2}"
    assert abs(t3 - 80.8) < 1e-10, f"Third update: {t3}"
    assert abs(t4 - 77.56) < 1e-10, f"Fourth update: {t4}"

    print("  PASSED: EMA trust scoring works (stateless and stateful)")


def test_save_load_persistence():
    """Test model save/load preserves all state."""
    print("Test 5: Model save/load persistence...")
    X, y = create_synthetic_data()
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

    config = TDFConfig(
        iso_n_estimators=30,
        rf_n_estimators=30,
        weight_step=0.1,
        val_split=0.3,
    )
    model = TrustDriftFusionModel(config=config)
    model.fit(X, y, feature_cols=feature_cols)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    try:
        model.save(temp_path)
        loaded = TrustDriftFusionModel.load(temp_path)

        # Check config preserved
        assert loaded.config.alpha == model.config.alpha
        assert loaded.config.weight_step == model.config.weight_step

        # Check state preserved
        assert np.allclose(loaded.state.w_as, model.state.w_as)
        assert np.allclose(loaded.state.w_rds, model.state.w_rds)
        assert np.allclose(loaded.state.w_rf, model.state.w_rf)
        assert np.allclose(loaded.state.optimal_threshold, model.state.optimal_threshold)
        assert np.allclose(loaded.state.val_f1, model.state.val_f1)
        assert loaded.state.feature_cols == model.state.feature_cols

        # Check scores match
        ht_orig = model.score(X[:10])
        ht_loaded = loaded.score(X[:10])
        assert np.allclose(ht_orig, ht_loaded), "Scores don't match after load"

        # Check trust history preserved
        loaded.trust_update(0.5, device_id="test")
        assert "test" in loaded._trust_history

    finally:
        os.unlink(temp_path)

    print("  PASSED: Model save/load preserves state and scores")


def test_signal_components():
    """Test get_signal_components returns all three signals + HT."""
    print("Test 6: Signal component decomposition...")
    X, y = create_synthetic_data()
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]

    config = TDFConfig(
        iso_n_estimators=30,
        rf_n_estimators=30,
        weight_step=0.1,
        val_split=0.3,
    )
    model = TrustDriftFusionModel(config=config)
    model.fit(X, y, feature_cols=feature_cols)

    components = model.get_signal_components(X[:5])
    assert "as" in components, "Missing AS component"
    assert "rds" in components, "Missing RDS component"
    assert "rf" in components, "Missing RF component"
    assert "ht" in components, "Missing HT component"
    assert all(k in components for k in ["as", "rds", "rf", "ht"])

    # All components should be in [0, 1]
    for k, v in components.items():
        assert v.shape == (5,), f"Component {k} wrong shape: {v.shape}"
        assert np.all(v >= 0) and np.all(v <= 1), f"Component {k} not in [0,1]"

    # HT should equal weighted sum
    ht_reconstructed = model.state.w_as * components["as"] + model.state.w_rds * components["rds"] + model.state.w_rf * components["rf"]
    assert np.allclose(components["ht"], np.clip(ht_reconstructed, 0, 1)), "HT != weighted sum of components"

    print("  PASSED: Signal components correct and HT = weighted sum")


def test_config_from_yaml():
    """Test loading config from YAML."""
    print("Test 7: Config from YAML...")
    config = TDFConfig.from_yaml(os.path.join(PROJECT_ROOT, "config", "tdf_config.yaml"))

    assert config.iso_n_estimators == 100
    assert config.rf_n_estimators == 200
    assert config.weight_step == 0.05
    assert config.alpha == 0.3
    assert config.rds_z_cap == 4.0
    assert config.severity_bands["Trusted"] == (80, 100)
    assert config.severity_bands["Critical"] == (0, 34)

    print("  PASSED: YAML config loads correctly")


def test_sklearn_compatibility():
    """Test sklearn-compatible get_params/set_params."""
    print("Test 8: sklearn get_params/set_params...")
    model = TrustDriftFusionModel()
    params = model.get_params()
    assert isinstance(params, dict)
    assert "iso_n_estimators" in params
    assert "alpha" in params

    model.set_params(alpha=0.5, iso_n_estimators=50)
    assert model.config.alpha == 0.5
    assert model.config.iso_n_estimators == 50

    print("  PASSED: sklearn get_params/set_params work")


def test_config_env_override():
    """Test environment variable config override."""
    print("Test 9: Config environment variable override...")
    import os
    os.environ["TDF_ALPHA"] = "0.7"
    os.environ["TDF_WEIGHT_STEP"] = "0.1"
    os.environ["TDF_ISO_N_ESTIMATORS"] = "75"

    try:
        config = TDFConfig.from_yaml(os.path.join(PROJECT_ROOT, "config", "tdf_config.yaml"))
        assert config.alpha == 0.7, f"Expected alpha=0.7, got {config.alpha}"
        assert config.weight_step == 0.1, f"Expected weight_step=0.1, got {config.weight_step}"
        assert config.iso_n_estimators == 75, f"Expected iso_n_estimators=75, got {config.iso_n_estimators}"
    finally:
        del os.environ["TDF_ALPHA"]
        del os.environ["TDF_WEIGHT_STEP"]
        del os.environ["TDF_ISO_N_ESTIMATORS"]

    print("  PASSED: Environment variable overrides work")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running TDF Model Unit Tests")
    print("=" * 60)

    tests = [
        test_fit_score_predict,
        test_weight_grid_search,
        test_severity_band_boundaries,
        test_trust_update_ema,
        test_save_load_persistence,
        test_signal_components,
        test_config_from_yaml,
        test_sklearn_compatibility,
        test_config_env_override,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)