"""SHAP explanations for the production risk classifier.

* Tree models (Random Forest, XGBoost) -> shap.TreeExplainer (exact, fast).
* Logistic regression -> shap.LinearExplainer on the scaled inputs.

Local explanations explain the *predicted class*: each feature's contribution to that
class's model output (log-odds for XGBoost/LR, probability for Random Forest). Global
importance is mean |SHAP| over a sample, averaged across classes. All numbers come
from the explainer for the exact model version being served.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import shap

from src.features.builder import load_registry
from src.models.evaluation import CLASSES
from src.models.risk_model import RiskModelBundle


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float | None
    shap_value: float
    direction: str  # "increases" | "decreases" (the predicted class's score)
    rank: int
    display: str = ""  # self-describing text, e.g. "Termination clause: missing"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalExplanation:
    explained_class: str
    output_space: str
    base_value: float
    contributions: list[FeatureContribution]

    @property
    def top_risk_factors(self) -> list[FeatureContribution]:
        """Positive contributors to a risky prediction (empty when the prediction is LOW)."""
        if self.explained_class == "LOW":
            return []
        return [c for c in self.contributions if c.shap_value > 0]

    def to_dict(self) -> dict:
        return {"explained_class": self.explained_class, "output_space": self.output_space,
                "base_value": self.base_value, "contributions": [c.to_dict() for c in self.contributions]}


def _to_3d(values, n_features: int) -> np.ndarray:
    """Normalise SHAP outputs to shape (n_samples, n_features, n_classes)."""
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[1] != n_features and arr.shape[2] == n_features:
        arr = np.transpose(arr, (0, 2, 1))
    return arr


class RiskExplainer:
    def __init__(self, bundle: RiskModelBundle, background: pd.DataFrame | None = None,
                 feature_labels: dict[str, str] | None = None):
        self.bundle = bundle
        self.labels = feature_labels or {}
        clf = bundle.classifier
        name = type(clf).__name__
        if name in {"XGBClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "LGBMClassifier"}:
            self.explainer = shap.TreeExplainer(clf)
            self.output_space = "probability" if name in {"RandomForestClassifier", "ExtraTreesClassifier"} \
                else "log-odds"
        elif name == "LogisticRegression":
            if background is None:
                raise ValueError("LinearExplainer needs background data")
            bg = bundle.transform_for_explainer(background)
            self.explainer = shap.LinearExplainer(clf, bg)
            self.output_space = "log-odds"
        else:
            raise ValueError(f"No SHAP explainer configured for {name}")

    def _shap(self, X_model: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = _to_3d(self.explainer.shap_values(X_model), X_model.shape[1])
        base = np.atleast_1d(np.asarray(self.explainer.expected_value, dtype=float))
        return values, base

    def explain(self, features: dict[str, float], class_name: str | None = None, top_k: int = 8) -> LocalExplanation:
        raw = self.bundle.validate_input(features)
        X_model = self.bundle.transform_for_explainer(raw)
        values, base = self._shap(X_model)
        if class_name is None:
            class_name = CLASSES[int(self.bundle.predict_proba(raw)[0].argmax())]
        k = CLASSES.index(class_name)
        sv = values[0, :, k] if values.shape[2] > 1 else values[0, :, 0]
        order = np.argsort(-np.abs(sv))[:top_k]
        registry = load_registry()
        contributions = []
        for rank, j in enumerate(order, start=1):
            feat = self.bundle.feature_names[j]
            v = raw.iloc[0, j]
            value = None if pd.isna(v) else float(v)
            contributions.append(FeatureContribution(
                feature=feat, label=self.labels.get(feat, feat), value=value, shap_value=float(sv[j]),
                direction="increases" if sv[j] > 0 else "decreases", rank=rank,
                display=registry.describe(feat, value),
            ))
        return LocalExplanation(class_name, self.output_space, float(base[k] if base.size > 1 else base[0]),
                                contributions)

    def global_importance(self, X: pd.DataFrame) -> list[dict]:
        X_model = self.bundle.transform_for_explainer(X)
        values, _ = self._shap(X_model)
        per_class = np.abs(values).mean(axis=0)  # (features, classes)
        overall = per_class.mean(axis=1)
        rows = []
        for j in np.argsort(-overall):
            feat = self.bundle.feature_names[j]
            rows.append({
                "feature": feat, "label": self.labels.get(feat, feat),
                "mean_abs_shap": float(overall[j]),
                "by_class": {c: float(per_class[j, i]) for i, c in enumerate(CLASSES[: per_class.shape[1]])},
            })
        return rows
