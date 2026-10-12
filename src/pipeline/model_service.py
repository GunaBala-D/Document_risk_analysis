"""Loads the production models once per process and reports their health.

Failure to load a model is not fatal: the analyzer runs without that component and the
result is flagged ``degraded``, so rules-based compliance checks keep working even if the
model registry is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.anomaly.detector import AnomalyDetector
from src.config import Settings, get_settings
from src.explainability.shap_explainer import RiskExplainer
from src.features.builder import load_registry
from src.logging_utils import get_logger
from src.models.risk_model import RiskModelBundle

log = get_logger(__name__)


@dataclass
class ModelService:
    classifier: RiskModelBundle | None = None
    explainer: RiskExplainer | None = None
    anomaly: AnomalyDetector | None = None
    classifier_metadata: dict = field(default_factory=dict)
    anomaly_metadata: dict = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> dict:
        return {
            "classifier": "loaded" if self.classifier else "unavailable",
            "classifier_version": self.classifier.model_version if self.classifier else None,
            "anomaly_detector": "loaded" if self.anomaly else "unavailable",
            "anomaly_version": self.anomaly.model_version if self.anomaly else None,
            "errors": self.errors,
        }

    @classmethod
    def load(cls, settings: Settings | None = None) -> ModelService:
        from src.models.registry import load_model_by_alias

        settings = settings or get_settings()
        svc = cls()
        registry = load_registry()
        try:
            pipe, meta, version = load_model_by_alias(settings.classifier_model_name, settings=settings)
            if meta.get("feature_names") != registry.names or meta.get("feature_version") != registry.version:
                raise RuntimeError(
                    f"Model feature schema ({meta.get('feature_version')}) does not match serving registry "
                    f"({registry.version})")
            svc.classifier = RiskModelBundle(pipe, meta["feature_names"], meta["feature_version"],
                                             float(meta["escalation_threshold"]), settings.classifier_model_name,
                                             str(version), meta.get("family", ""), meta,
                                             decision_weights=meta.get("decision_weights"))
            svc.classifier_metadata = meta
            background = pd.DataFrame(meta["shap_background"]) if meta.get("shap_background") else None
            svc.explainer = RiskExplainer(svc.classifier, background=background, feature_labels=registry.labels)
        except Exception as exc:
            log.error("classifier_load_failed", extra={"error": str(exc)})
            svc.errors["classifier"] = str(exc)
            svc.classifier = svc.explainer = None
        try:
            pipe, meta, version = load_model_by_alias(settings.anomaly_model_name, settings=settings)
            svc.anomaly = AnomalyDetector(pipe, meta["features"], meta["quantiles"], float(meta["flag_percentile"]),
                                          model_version=str(version))
            svc.anomaly_metadata = meta
        except Exception as exc:
            log.error("anomaly_load_failed", extra={"error": str(exc)})
            svc.errors["anomaly_detector"] = str(exc)
        return svc
