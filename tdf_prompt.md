Replace the existing anomaly/intrusion-detection model in this project with a new model called
Trust-Drift Fusion (TDF), as described below. Find wherever the current model is defined, trained,
and called from (training script, inference/serving code, API endpoints, tests), and swap it out
for TDF end to end, keeping the surrounding project structure, data loading, and interfaces intact
wherever possible.

## What TDF is

TDF fuses three independent signals computed on the same input feature vector into a single
Hybrid Threat score, then converts that into a continuously updated Trust Score:

1. Anomaly Score (AS) — unsupervised, from an Isolation Forest fit on the training data.
   AS = clip( (-iso_forest.score_samples(X) - AS_min) / (AS_max - AS_min), 0, 1 )
   where AS_min/AS_max are the min/max of the raw negative score_samples values observed on the
   training (or a held-out) split, used to rescale into [0, 1].

2. Resource Drift Score (RDS) — an online/statistical drift signal. For each feature, track a
   running mean (mu) and standard deviation (sigma) from the Normal-labeled training rows only.
   For a new row x, compute the per-feature Z-score |x_i - mu_i| / sigma_i, then:
   RDS = clip( mean(|Z|) / 4.0, 0, 1 )
   (the /4.0 keeps RDS in a sane [0,1] range for typical Z-score magnitudes; tune if needed)

3. Random Forest attack probability (RF) — supervised, from a RandomForestClassifier
   (class_weight="balanced") trained on the labeled training data, using predict_proba(X)[:, 1].

Fuse them into a Hybrid Threat score:
   HT = w_as * AS + w_rds * RDS + w_rf * RF

Do NOT hardcode w_as/w_rds/w_rf. Instead, auto-tune the three weights by grid search
(step size 0.05, weights sum to 1, e.g. via itertools/product over the simplex) on a held-out
validation split carved out of the TRAINING data only (never touch the test set during tuning).
For each candidate weight triple, sweep classification thresholds on HT to find the one that
maximizes F1 on the validation split, then keep the weight triple + threshold combo with the
best validation F1.

Convert the final HT into a continuously updated Trust Score using an exponential moving average:
   T_t = alpha * (100 - 100 * HT) + (1 - alpha) * T_(t-1)
   (alpha = 0.3 default; T_(t-1) is the previous window's trust score, seeded at 100 for a new
   device/session; if the project doesn't have a stateful/streaming design, alpha and T_(t-1)
   can be no-ops and the model can just return 100 - 100*HT as a stateless equivalent — implement
   both a stateless `.score(X)` returning HT/risk-in-[0,1], and a stateful `.trust_update(HT, prev_trust)`
   helper for anything that already tracks per-device/session state.)

Map the Trust Score to four severity bands:
   Trusted        >= 80
   Monitored      60-79
   Elevated Risk  35-59
   Critical       <35

## Interface / integration requirements

- Give the model a `.fit(X_train, y_train)` and `.score(X)` (returns continuous risk in [0,1])
  method, matching whatever interface the current model in this project already exposes (check
  the current model class/module and mirror its method names/signatures exactly so nothing else
  in the codebase needs to change).
- If the current model exposes `.predict(X)` returning 0/1, add that too, using the F1-optimal
  threshold found during weight tuning.
- If the current model exposes `.predict_proba(X)`, return a 2-column array [1-score, score] to
  stay compatible with any sklearn-style downstream code (metrics, calibration plots, etc.).
- Preserve any existing model persistence (pickle/joblib/ONNX/etc.) — make sure the new model is
  also saveable/loadable the same way as before.
- Preserve existing logging/metrics code paths (accuracy, precision, recall, F1, ROC-AUC, MCC,
  balanced accuracy) — TDF should be evaluable with the exact same evaluation script/harness
  already in the project, no bespoke evaluation code.
- If there's an existing threshold/config file (YAML/JSON/env vars) for detection sensitivity,
  wire the severity bands and any tunable knobs (alpha, weight_step, n_estimators for both
  Isolation Forest and Random Forest, contamination) into that same config system instead of
  hardcoding them.

## Deliverables

1. Locate and replace the existing model implementation with TDF (new class, e.g.
   `TrustDriftFusionModel`, in the same module/location the old model lived).
2. Update all call sites (training entrypoint, inference/serving code, tests) to use the new
   class instead of the old one.
3. Update or add unit tests covering: fit/score/predict on a small synthetic fixture, weight
   grid-search producing weights that sum to 1, and severity-band mapping boundaries.
4. Do not delete the old model file outright — rename it with a `_legacy` suffix or move it to
   an `archive/` folder so it can still be diffed against if needed, unless the project's
   conventions clearly prefer a clean removal (check for a CONTRIBUTING.md or similar first).
5. Run the existing test suite and evaluation script afterward and report the resulting metrics.

Ask me before making changes if you're unsure which file(s) currently define the model, or if the
project's data schema doesn't cleanly map to "one row = one feature vector with a binary label."
