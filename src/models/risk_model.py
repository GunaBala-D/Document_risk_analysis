"""Risk classifier: model factories and the serving bundle.

The bundle couples the fitted pipeline with everything needed to use it safely:
the exact feature list (schema), class order, feature version, the tuned escalation
threshold and the registry version. Inference validates inputs against that schema
instead of silently reordering or filling columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.evaluation import CLASSES, apply_decision_weights, high_risk_score

# Severity weight of each class for the continuous ML risk component (0..1).
CLASS_SEVERITY = np.array([0.0, 1 / 3, 2 / 3, 1.0])
MODEL_FAMILIES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
IMBALANCE_STRATEGIES = ["none", "class_weight", "smote"]


class ModelInputError(ValueError):
    pass


def make_pipeline(family: str, imbalance: str = "none", params: dict | None = None,
                  random_state: int = 42) -> Pipeline:
    """Build an (imblearn) pipeline: impute -> [scale] -> [SMOTE] -> classifier.

    class_weight for XGBoost is applied through sample weights at fit time
    (see ``fit_sample_weight``) because XGBClassifier has no class_weight argument.
    """
    params = dict(params or {})
    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if family == "logistic_regression":
        steps.append(("scale", StandardScaler()))
        clf = LogisticRegression(max_iter=3000, C=params.pop("C", 1.0),
                                 class_weight="balanced" if imbalance == "class_weight" else None,
                                 random_state=random_state)
    elif family == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 300), random_state=random_state, n_jobs=-1,
            class_weight="balanced_subsample" if imbalance == "class_weight" else None, **params,
        )
    elif family == "xgboost":
        clf = XGBClassifier(
            n_estimators=params.pop("n_estimators", 300), learning_rate=params.pop("learning_rate", 0.1),
            max_depth=params.pop("max_depth", 4), objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=random_state, n_jobs=4, **params,
        )
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=params.pop("n_estimators", 300), learning_rate=params.pop("learning_rate", 0.05),
            num_leaves=params.pop("num_leaves", 15), objective="multiclass",
            class_weight="balanced" if imbalance == "class_weight" else None,
            random_state=random_state, n_jobs=4, verbose=-1, **params,
        )
    else:
        raise ValueError(f"Unknown model family {family}")
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def fit_sample_weight(family: str, imbalance: str, y: np.ndarray) -> np.ndarray | None:
    if family == "xgboost" and imbalance == "class_weight":
        from sklearn.utils.class_weight import compute_sample_weight

        return compute_sample_weight("balanced", y)
    return None


def fit_pipeline(pipe: Pipeline, family: str, imbalance: str, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    w = fit_sample_weight(family, imbalance, y)
    if w is not None:
        pipe.fit(X, y, clf__sample_weight=w)
    else:
        pipe.fit(X, y)
    return pipe


@dataclass
class Prediction:
    predicted_class: str
    probabilities: dict[str, float]
    ml_risk_score: float  # expected severity in [0, 1]
    high_risk_probability: float
    escalate: bool


@dataclass
class RiskModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    feature_version: str
    escalation_threshold: float
    model_name: str = "document-risk-classifier"
    model_version: str = "local"
    family: str = ""
    metadata: dict = field(default_factory=dict)
    decision_weights: list[float] | None = None  # tuned multi-class thresholds (None = argmax)

    def validate_input(self, features: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ModelInputError(f"Missing required features: {missing}")
        df = df[self.feature_names]
        try:
            df = df.astype(float)
        except (TypeError, ValueError) as exc:
            raise ModelInputError(f"Non-numeric feature values: {exc}") from exc
        if np.isinf(df.to_numpy()).any():
            raise ModelInputError("Infinite feature values are not allowed")
        return df

    def predict_proba(self, features: dict[str, float] | pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self.validate_input(features))

    def predict(self, features: dict[str, float]) -> Prediction:
        proba = self.predict_proba(features)[0]
        hr = float(high_risk_score(proba[None, :])[0])
        return Prediction(
            predicted_class=CLASSES[int(apply_decision_weights(proba[None, :], self.decision_weights)[0])],
            probabilities={c: round(float(p), 6) for c, p in zip(CLASSES, proba, strict=True)},
            ml_risk_score=float(np.dot(proba, CLASS_SEVERITY)),
            high_risk_probability=hr,
            escalate=bool(hr >= self.escalation_threshold),
        )

    def transform_for_explainer(self, features: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
        """Apply the pre-processing steps (not SMOTE) so SHAP sees what the classifier sees."""
        X = self.validate_input(features)
        values = X
        for name, step in self.pipeline.steps[:-1]:
            if name == "smote":
                continue
            values = step.transform(values)
        return pd.DataFrame(np.asarray(values), columns=self.feature_names, index=X.index)

    @property
    def classifier(self):
        return self.pipeline.steps[-1][1]


def nan_to_none(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}
