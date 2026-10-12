"""ML contract tests: model loading, required features, prediction schema, probability range,
model version, threshold logic and SHAP faithfulness."""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import load_registry
from src.models.evaluation import CLASSES, apply_decision_weights, compute_metrics, tune_decision_weights
from src.models.risk_model import ModelInputError
from tests.helpers import quick_model_service, synthetic_feature_frame


@pytest.fixture(scope="module")
def svc():
    return quick_model_service()


@pytest.fixture(scope="module")
def frame():
    return synthetic_feature_frame()


def test_model_loads_with_version_and_schema(svc):
    assert svc.classifier is not None and svc.classifier.model_version == "test-1"
    assert svc.classifier.feature_names == load_registry().names
    assert svc.status["classifier"] == "loaded"


def test_prediction_schema_and_probability_range(svc, frame):
    X, _ = frame
    pred = svc.classifier.predict(X.iloc[0].to_dict())
    assert pred.predicted_class in CLASSES
    assert set(pred.probabilities) == set(CLASSES)
    assert all(0 <= p <= 1 for p in pred.probabilities.values())
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert 0 <= pred.ml_risk_score <= 1 and 0 <= pred.high_risk_probability <= 1
    assert isinstance(pred.escalate, bool)


def test_missing_required_feature_rejected(svc, frame):
    X, _ = frame
    row = X.iloc[0].to_dict()
    row.pop("payment_days")
    with pytest.raises(ModelInputError, match="payment_days"):
        svc.classifier.predict(row)


def test_non_numeric_and_infinite_values_rejected(svc, frame):
    X, _ = frame
    with pytest.raises(ModelInputError):
        svc.classifier.predict({**X.iloc[0].to_dict(), "payment_days": "ninety"})
    with pytest.raises(ModelInputError):
        svc.classifier.predict({**X.iloc[0].to_dict(), "penalty_pct": float("inf")})


def test_nan_features_are_allowed(svc, frame):
    X, _ = frame
    row = {k: float("nan") for k in X.columns}
    assert svc.classifier.predict(row).predicted_class in CLASSES


def test_extra_columns_ignored_and_order_independent(svc, frame):
    X, _ = frame
    shuffled = X.iloc[[0]][list(reversed(X.columns))].assign(extra=1.0)
    np.testing.assert_allclose(svc.classifier.predict_proba(shuffled), svc.classifier.predict_proba(X.iloc[[0]]))


def test_model_beats_chance_on_its_data(svc, frame):
    X, y = frame
    m = compute_metrics(y, svc.classifier.predict_proba(X))
    assert m["f1_macro"] > 0.5 and m["roc_auc_ovr_macro"] > 0.8


def test_shap_values_are_faithful_to_the_model(svc, frame):
    """base value + sum(SHAP) must equal the model's output for the explained class (RF: probability)."""
    X, _ = frame
    row = X.iloc[5].to_dict()
    exp = svc.explainer.explain(row, top_k=len(X.columns))
    k = CLASSES.index(exp.explained_class)
    model_out = svc.classifier.predict_proba(row)[0, k]
    assert exp.base_value + sum(c.shap_value for c in exp.contributions) == pytest.approx(model_out, abs=1e-6)
    assert all(c.display for c in exp.contributions)


def test_global_importance_covers_all_features(svc, frame):
    X, _ = frame
    rows = svc.explainer.global_importance(X.head(50))
    assert len(rows) == len(X.columns) and rows[0]["mean_abs_shap"] >= rows[-1]["mean_abs_shap"]


@pytest.mark.parametrize("family", ["logistic_regression", "random_forest", "xgboost", "lightgbm"])
@pytest.mark.parametrize("imbalance", ["none", "class_weight", "smote"])
def test_every_candidate_family_trains_predicts_and_explains(frame, family, imbalance):
    from src.explainability.shap_explainer import RiskExplainer
    from src.models.risk_model import RiskModelBundle, fit_pipeline, make_pipeline

    X, y = frame
    params = {} if family == "logistic_regression" else {"n_estimators": 30}
    pipe = fit_pipeline(make_pipeline(family, imbalance, params=params), family, imbalance, X, y)
    proba = pipe.predict_proba(X.head(5))
    assert proba.shape == (5, len(CLASSES)) and np.allclose(proba.sum(1), 1, atol=1e-6)
    bundle = RiskModelBundle(pipe, list(X.columns), load_registry().version, 0.3, family=family)
    explainer = RiskExplainer(bundle, background=X.head(50))
    assert len(explainer.explain(X.iloc[0].to_dict(), top_k=3).contributions) == 3


def test_decision_weights_shift_predictions_towards_critical():
    proba = np.array([[0.4, 0.1, 0.2, 0.3], [0.7, 0.1, 0.1, 0.1]])
    assert apply_decision_weights(proba, None).tolist() == [0, 0]
    assert apply_decision_weights(proba, [1, 1, 1, 2]).tolist() == [3, 0]


def test_tune_decision_weights_meets_constraints_when_possible(frame):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    proba = np.full((400, 4), 0.1)
    proba[np.arange(400), y] += 0.3
    proba[y == 3, 0] += 0.35  # CRITICAL is systematically under-predicted at argmax
    proba /= proba.sum(1, keepdims=True)
    res = tune_decision_weights(y, proba, 0.9, 0.8)
    assert res["feasible"] and res["critical_recall"] >= 0.8 and res["weights"][3] > 1


def test_anomaly_detector_scores_in_unit_interval(svc, frame):
    X, _ = frame
    s = svc.anomaly.score(X.head(30))
    assert ((s >= 0) & (s <= 1)).all()
    extreme = X.head(1).copy()
    extreme["contract_value_log10"], extreme["payment_days"], extreme["penalty_pct"] = 9.5, 365, 60
    assert svc.anomaly.score(extreme)[0] > np.median(s)


def test_feature_frame_has_no_label_leakage(frame):
    X, _ = frame
    assert not {"risk_label", "risk_points", "label_reasons"} & set(X.columns)
    assert isinstance(X, pd.DataFrame)
