"""Application services: persistence and orchestration behind the HTTP routes."""

from __future__ import annotations

import hashlib
import math
import secrets
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database.models import (
    AnomalyRecord,
    AuditLog,
    ComplianceViolation,
    Document,
    DocumentEntity,
    DocumentFeatures,
    Prediction,
    RiskFactor,
    User,
    VendorContractRecord,
)
from src.compliance.engine import count_policy_violations
from src.config import Settings
from src.extraction.entities import normalize_vendor_key
from src.features.builder import VendorContract, load_registry
from src.ingestion.validation import ValidatedFile, validate_upload
from src.llm.ollama_client import LLMUnavailable, OllamaExplainer
from src.logging_utils import get_logger, request_id_var
from src.models.risk_model import nan_to_none
from src.pipeline.analyzer import AnalysisResult, analyze_document
from src.pipeline.document_processor import process_document
from src.pipeline.model_service import ModelService
from src.preprocessing.doc_type import PHASE_MESSAGES

log = get_logger(__name__)


class ConflictError(Exception):
    def __init__(self, message: str, existing_id: str | None = None):
        super().__init__(message)
        self.existing_id = existing_id


class ExtractionFailed(Exception):
    pass


def audit(session: Session, user: User | None, action: str, entity_type: str, entity_id: str | None,
          details: dict | None = None) -> None:
    session.add(AuditLog(user_id=user.id if user else None, action=action, entity_type=entity_type,
                         entity_id=entity_id, request_id=request_id_var.get(), details=details or {}))


def new_document_id() -> str:
    return "DOC-" + secrets.token_hex(5).upper()


def store_upload(session: Session, settings: Settings, user: User, filename: str, data: bytes) -> Document:
    vf: ValidatedFile = validate_upload(filename, data, settings.max_upload_bytes, settings.max_pdf_pages)
    sha = hashlib.sha256(data).hexdigest()
    existing = session.scalar(select(Document).where(Document.sha256 == sha))
    if existing:
        raise ConflictError("This file was already uploaded.", existing.id)
    doc_id = new_document_id()
    folder = Path(settings.upload_dir) / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / vf.filename
    path.write_bytes(data)
    doc = Document(id=doc_id, filename=vf.filename, sha256=sha, file_size=vf.size_bytes, mime_type=vf.mime_type,
                   file_format=vf.file_format, storage_path=str(path), uploaded_by=user.id, page_count=vf.page_count,
                   processing_status="uploaded")
    session.add(doc)
    audit(session, user, "document.upload", "document", doc_id,
          {"filename": vf.filename, "size": vf.size_bytes, "format": vf.file_format, "sha256": sha})
    session.commit()
    return doc


def vendor_history(session: Session, vendor_key: str | None, document_id: str) -> list[VendorContract]:
    if not vendor_key:
        return []
    rows = session.scalars(select(VendorContractRecord).where(
        VendorContractRecord.vendor_key == vendor_key,
        (VendorContractRecord.document_id.is_(None)) | (VendorContractRecord.document_id != document_id),
    )).all()
    return [VendorContract(r.contract_date, r.value_inr, r.payment_days, r.violation_count, r.high_risk) for r in rows]


def _entity_row(doc_id: str, e) -> DocumentEntity:
    v = e.value
    return DocumentEntity(
        document_id=doc_id, name=e.name,
        value_text=v.isoformat() if isinstance(v, date) else str(v) if v is not None else None,
        value_num=float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None,
        value_date=v if isinstance(v, date) else None,
        source_text=(e.raw or "")[:300] or None, confidence=e.confidence, method=e.method,
        span_start=e.start, span_end=e.end,
    )


def _persist_prediction(session: Session, doc: Document, result: AnalysisResult, models: ModelService) -> Prediction:
    p = result.prediction
    s = result.score
    pred = Prediction(
        document_id=doc.id, model_name=models.classifier.model_name if models.classifier else None,
        model_version=result.model_versions["classifier"], anomaly_model_version=result.model_versions["anomaly_detector"],
        versions=result.model_versions,
        ml_class=p.predicted_class if p else None, ml_probabilities=p.probabilities if p else None,
        ml_component=s.components["ml"], high_risk_probability=p.high_risk_probability if p else None,
        escalate=p.escalate if p else None, anomaly_score=result.anomaly_score,
        anomaly_component=s.components["anomaly"], compliance_component=s.components["compliance"],
        business_component=s.components["business"], risk_score=s.risk_score, risk_level=s.risk_level,
        contributions=s.contributions, floors_applied=s.floors_applied, degraded=s.degraded,
        component_errors=result.component_errors, needs_review=result.needs_review,
        review_reasons=result.review_reasons, recommended_actions=result.recommended_actions,
        shap_explained_class=result.explanation.explained_class if result.explanation else None,
        shap_output_space=result.explanation.output_space if result.explanation else None,
        shap_base_value=result.explanation.base_value if result.explanation else None,
        timings_ms=result.timings_ms,
    )
    if result.explanation:
        for c in result.explanation.contributions:
            pred.risk_factors.append(RiskFactor(rank=c.rank, feature=c.feature, label=c.label, feature_value=c.value,
                                                shap_value=c.shap_value, direction=c.direction))
    for a in result.anomalies:
        pred.anomalies.append(AnomalyRecord(anomaly_type=a.anomaly_type, score=a.score, is_anomaly=a.is_anomaly,
                                            description=a.description, details=_json_safe(a.details)))
    for r in result.compliance:
        pred.violations.append(ComplianceViolation(rule_id=r.rule_id, rule_name=r.rule_name, severity=r.severity,
                                                   rule_type=r.type, message=r.message, remediation=r.remediation,
                                                   evidence=_json_safe(r.evidence)))
    session.add(pred)
    session.flush()
    return pred


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def analyze(session: Session, settings: Settings, models: ModelService, doc: Document, user: User,
            as_of: date | None = None) -> Prediction | None:
    """Run extraction + analysis for a stored document. Returns the new prediction, or None if the
    document type is not supported in V1. Raises ExtractionFailed if no text could be extracted."""
    doc.processing_status = "processing"
    session.commit()
    data = Path(doc.storage_path).read_bytes()
    processed = process_document(doc.filename, data, settings.max_upload_bytes, settings.max_pdf_pages,
                                 settings.tesseract_cmd or None)
    ex = processed.extraction
    doc.extraction_status, doc.extraction_method = ex.status, ex.method
    doc.page_count, doc.ocr_pages, doc.char_count = ex.page_count or doc.page_count, len(ex.ocr_pages), len(processed.text)

    if not processed.ok:
        doc.processing_status, doc.status_message = "failed", ex.error
        doc.timings_ms = processed.timings_ms
        audit(session, user, "document.analyze_failed", "document", doc.id, {"error": ex.error})
        session.commit()
        raise ExtractionFailed(ex.error or "Extraction failed")

    doc.document_type = processed.doc_type.doc_type
    doc.doc_type_confidence = processed.doc_type.confidence
    if not processed.doc_type.supported:
        doc.processing_status = "unsupported_type"
        doc.status_message = PHASE_MESSAGES[processed.doc_type.doc_type]
        doc.timings_ms = processed.timings_ms
        audit(session, user, "document.unsupported_type", "document", doc.id, {"type": doc.document_type})
        session.commit()
        return None

    session.execute(delete(DocumentEntity).where(DocumentEntity.document_id == doc.id))
    for e in processed.entities.values():
        session.add(_entity_row(doc.id, e))
    doc.sections = [s.to_dict() for s in processed.sections]
    vendor_name = processed.entities["vendor_name"].value if "vendor_name" in processed.entities else None
    doc.vendor_name, doc.vendor_key = vendor_name, normalize_vendor_key(vendor_name)

    as_of = as_of or doc.uploaded_at.date()
    history = vendor_history(session, doc.vendor_key, doc.id)
    result = analyze_document(processed, as_of, history, models)

    session.add(DocumentFeatures(document_id=doc.id, feature_version=result.model_versions["feature_version"],
                                 features=nan_to_none(result.features)))
    pred = _persist_prediction(session, doc, result, models)
    pred.llm_status = "pending" if settings.llm_enabled else "disabled"

    # Point-in-time vendor history row for future documents of the same vendor
    session.execute(delete(VendorContractRecord).where(VendorContractRecord.document_id == doc.id))
    if doc.vendor_key:
        pay = result.features.get("payment_days")
        session.add(VendorContractRecord(
            vendor_key=doc.vendor_key, vendor_name=vendor_name, contract_date=as_of, value_inr=result.value_inr,
            payment_days=None if pay is None or math.isnan(pay) else pay,
            violation_count=count_policy_violations(result.compliance),
            high_risk=result.score.risk_level in ("HIGH", "CRITICAL"), source="analyzed", document_id=doc.id,
        ))

    doc.latest_prediction_id = pred.id
    doc.processing_status, doc.status_message = "analyzed", None
    doc.review_status = "pending" if result.needs_review else "not_required"
    doc.timings_ms = {**processed.timings_ms, **result.timings_ms}
    audit(session, user, "document.analyze", "document", doc.id,
          {"prediction_id": pred.id, "risk_level": pred.risk_level, "risk_score": pred.risk_score,
           "model_version": pred.model_version, "degraded": pred.degraded})
    session.commit()
    log.info("document_analyzed", extra={
        "document_id": doc.id, "risk_level": pred.risk_level, "risk_score": pred.risk_score,
        "model_version": pred.model_version, "timings_ms": doc.timings_ms, "degraded": pred.degraded})
    return pred


def entity_values(session: Session, doc_id: str) -> dict:
    rows = session.scalars(select(DocumentEntity).where(DocumentEntity.document_id == doc_id)).all()
    return {r.name: (r.value_num if r.value_num is not None else r.value_text) for r in rows}


def factor_display(f: RiskFactor) -> str:
    """Self-describing factor text, e.g. 'Termination clause: missing' (never just a label + raw 0/1)."""
    return load_registry().describe(f.feature, f.feature_value)


def evidence_for(session: Session, pred: Prediction) -> dict:
    """The ONLY input the LLM receives: structured outputs of the deterministic pipeline (no raw text)."""
    cls = pred.shap_explained_class
    return {
        "document_id": pred.document_id,
        "risk_score": pred.risk_score, "risk_level": pred.risk_level,
        "score_components_0_to_1": {"ml": pred.ml_component, "anomaly": pred.anomaly_component,
                                    "compliance": pred.compliance_component, "business": pred.business_component},
        "ml_predicted_class": pred.ml_class,
        "model_risk_factors": [{
            "factor": factor_display(f),
            "effect": f"pushes the model towards {cls}" if f.shap_value > 0 else f"pushes the model away from {cls}",
            "shap_contribution": round(f.shap_value, 3),
        } for f in sorted(pred.risk_factors, key=lambda f: f.rank)[:5]],
        "compliance_violations": [{"rule": v.rule_name, "rule_id": v.rule_id, "severity": v.severity, "type": v.rule_type,
                                   "finding": v.message} for v in pred.violations],
        "anomalies": [{"type": a.anomaly_type, "finding": a.description} for a in pred.anomalies if a.is_anomaly],
        "extracted_entities": entity_values(session, pred.document_id),
        "recommended_actions": [a["action"] for a in (pred.recommended_actions or [])],
    }


def generate_llm_explanation(session_factory, settings: Settings, prediction_id: int) -> None:
    """Background task: ask Ollama to explain the structured result. Never raises."""
    session: Session = session_factory()
    try:
        pred = session.get(Prediction, prediction_id)
        if pred is None:
            return
        if not settings.llm_enabled:
            pred.llm_status = "disabled"
            session.commit()
            return
        explainer = OllamaExplainer(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
        try:
            out = explainer.explain(evidence_for(session, pred))
            pred.llm_status, pred.llm_explanation = "ready", out.to_dict()
            pred.llm_model, pred.llm_latency_ms, pred.llm_error = out.model, out.latency_ms, None
            log.info("llm_explanation_ready", extra={"document_id": pred.document_id, "llm_latency_ms": out.latency_ms,
                                                     "grounded": out.grounded})
        except LLMUnavailable as exc:
            pred.llm_status, pred.llm_error = "unavailable", str(exc)[:500]
            log.warning("llm_unavailable", extra={"document_id": pred.document_id, "error": str(exc)[:200]})
        session.commit()
    except Exception:
        log.exception("llm_background_task_failed", extra={"prediction_id": prediction_id})
    finally:
        session.close()


def now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = ["ConflictError", "ExtractionFailed", "analyze", "audit", "evidence_for", "factor_display",
           "generate_llm_explanation", "store_upload"]
