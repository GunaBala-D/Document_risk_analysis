"""Risk analysis stage: structured document -> structured risk signals.

features -> ML classification -> SHAP -> anomaly detection -> compliance rules -> risk score
-> deterministic recommended actions. The LLM is NOT called here; it only explains the
result afterwards (see src/llm), so analysis latency and correctness never depend on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from src.anomaly.detector import AnomalyFinding, anomaly_component_score
from src.compliance.engine import ComplianceEngine, RuleResult
from src.explainability.shap_explainer import LocalExplanation
from src.features.builder import (
    VendorContract,
    build_facts,
    build_features,
    contract_value_inr,
    load_registry,
)
from src.logging_utils import get_logger, timed
from src.models.risk_model import Prediction
from src.pipeline.document_processor import ProcessedDocument
from src.pipeline.model_service import ModelService
from src.scoring.risk_scorer import RiskScore, RiskScorer, business_score, compliance_score

log = get_logger(__name__)

LEVEL_ACTIONS = {
    "CRITICAL": "Escalate to the Risk Manager. Do not sign or renew until every CRITICAL/HIGH issue is resolved.",
    "HIGH": "Mandatory analyst review before approval; resolve the listed compliance issues with the vendor.",
    "MEDIUM": "Standard review: confirm the flagged items with the business owner before approval.",
    "LOW": "No material risk detected. Proceed with the standard approval workflow.",
}
ANOMALY_ACTIONS = {
    "vendor_value_deviation": "Verify the contract value and scope with the business owner; compare with the "
                              "vendor's previous contracts.",
    "vendor_contract_frequency": "Check whether related contracts were split to stay below an approval limit.",
    "vendor_payment_terms_deviation": "Confirm why payment terms differ from this vendor's usual terms.",
    "multivariate_isolation_forest": "Review the contract's commercial terms - its overall profile is unusual "
                                     "compared with historical contracts.",
}


@dataclass
class AnalysisResult:
    features: dict[str, float]
    prediction: Prediction | None
    explanation: LocalExplanation | None
    anomalies: list[AnomalyFinding]
    compliance: list[RuleResult]
    score: RiskScore
    recommended_actions: list[dict]
    needs_review: bool
    review_reasons: list[str]
    model_versions: dict[str, str | None]
    component_errors: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    value_inr: float | None = None

    @property
    def anomaly_score(self) -> float | None:
        mv = [a.score for a in self.anomalies if a.anomaly_type == "multivariate_isolation_forest"]
        return mv[0] if mv else None

    def to_report(self, document_id: str) -> dict:
        """Business risk report (the API's /risk payload)."""
        return {
            "document_id": document_id,
            "risk_score": self.score.risk_score,
            "risk_level": self.score.risk_level,
            "ml_predicted_class": self.prediction.predicted_class if self.prediction else None,
            "ml_probabilities": self.prediction.probabilities if self.prediction else None,
            "anomaly_score": self.anomaly_score,
            "anomalies": [a.description for a in self.anomalies if a.is_anomaly],
            "compliance_violations": [r.rule_name for r in self.compliance if r.type == "violation"],
            "compliance_warnings": [r.rule_name for r in self.compliance if r.type == "warning"],
            "top_risk_factors": [c.display for c in self.explanation.top_risk_factors[:5]] if self.explanation else [],
            "needs_review": self.needs_review,
            "degraded": self.score.degraded,
        }


def recommended_actions(level: str, compliance: list[RuleResult], anomalies: list[AnomalyFinding]) -> list[dict]:
    actions = [{"source": "risk_level", "priority": level, "action": LEVEL_ACTIONS[level]}]
    for r in compliance:
        actions.append({"source": r.rule_id, "priority": r.severity, "action": r.remediation})
    for a in anomalies:
        if a.is_anomaly:
            actions.append({"source": a.anomaly_type, "priority": "MEDIUM", "action": ANOMALY_ACTIONS[a.anomaly_type]})
    return actions


def analyze_document(doc: ProcessedDocument, as_of: date, vendor_history: list[VendorContract],
                     models: ModelService, engine: ComplianceEngine | None = None,
                     scorer: RiskScorer | None = None) -> AnalysisResult:
    registry = load_registry()
    engine = engine or ComplianceEngine()
    scorer = scorer or RiskScorer()
    timings: dict[str, float] = {}
    errors: dict[str, str] = dict(models.errors)

    with timed(timings, "feature_engineering_ms"):
        features = build_features(doc.text, doc.sections, doc.entities, as_of, vendor_history, registry)
        facts = build_facts(features, doc.entities, doc.sections, registry.config["fx_to_inr"])
        value_inr = contract_value_inr(doc.entities, registry.config["fx_to_inr"])

    prediction = explanation = None
    if models.classifier is not None:
        try:
            with timed(timings, "ml_inference_ms"):
                prediction = models.classifier.predict(features)
            with timed(timings, "shap_ms"):
                explanation = models.explainer.explain(features, prediction.predicted_class) if models.explainer else None
        except Exception as exc:
            log.exception("ml_inference_failed")
            errors["classifier"] = f"{type(exc).__name__}: {exc}"
            prediction = explanation = None

    anomalies: list[AnomalyFinding] = []
    if models.anomaly is not None:
        try:
            with timed(timings, "anomaly_detection_ms"):
                anomalies = models.anomaly.detect(features, vendor_history, as_of, value_inr)
        except Exception as exc:
            log.exception("anomaly_detection_failed")
            errors["anomaly_detector"] = f"{type(exc).__name__}: {exc}"

    with timed(timings, "compliance_ms"):
        compliance = engine.evaluate(facts)

    with timed(timings, "scoring_ms"):
        score = scorer.score(
            ml=prediction.ml_risk_score if prediction else None,
            anomaly=anomaly_component_score(anomalies) if models.anomaly is not None and anomalies else None,
            compliance=compliance_score(compliance, scorer.cfg["compliance"]),
            business=business_score(value_inr, features.get("days_to_expiry"), scorer.cfg["business"]),
            critical_violation=any(r.severity == "CRITICAL" and r.type == "violation" for r in compliance),
            ml_class=prediction.predicted_class if prediction else None,
        )

    reasons = []
    if score.risk_level in ("HIGH", "CRITICAL"):
        reasons.append(f"Risk level {score.risk_level}")
    if prediction and prediction.escalate:
        reasons.append(f"Model high-risk probability {prediction.high_risk_probability:.2f} >= escalation threshold")
    if score.degraded:
        reasons.append("Analysis ran in degraded mode (a model component was unavailable)")
    if doc.extraction.method in ("ocr", "mixed"):
        reasons.append("Text was obtained by OCR - verify extracted values")
    if any(a.is_anomaly for a in anomalies):
        reasons.append("Anomaly detected")

    return AnalysisResult(
        features=features, prediction=prediction, explanation=explanation, anomalies=anomalies,
        compliance=compliance, score=score,
        recommended_actions=recommended_actions(score.risk_level, compliance, anomalies),
        needs_review=bool(reasons), review_reasons=reasons,
        model_versions={
            "classifier": models.classifier.model_version if models.classifier else None,
            "anomaly_detector": models.anomaly.model_version if models.anomaly else None,
            "feature_version": registry.version, "compliance_rules": engine.version,
            "risk_scoring": scorer.cfg.get("version"),
        },
        component_errors=errors, timings_ms=timings, value_inr=value_inr,
    )


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))
