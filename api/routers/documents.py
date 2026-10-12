from __future__ import annotations

from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from api import services
from api.auth import require
from api.deps import (
    get_document_or_404,
    get_latest_prediction_or_404,
    get_models,
    get_session_factory,
    get_settings_dep,
)
from api.schemas import (
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    EntityOut,
    ErrorResponse,
    ExplanationOut,
    FeedbackIn,
    FeedbackOut,
    RiskReport,
    UploadResponse,
)
from database.models import Document, DocumentEntity, Prediction, ReviewFeedback, User
from database.session import get_session
from src.feedback.validation import consistency_flags

router = APIRouter(prefix="/documents", tags=["documents"])
ERRORS = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}}


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED,
             responses={**ERRORS, 409: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                        415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session),
                          settings=Depends(get_settings_dep), user: User = Depends(require("documents:write"))):
    """Upload a contract (PDF, DOCX, PNG, JPG). The file is validated by content signature, not extension."""
    data = await file.read(settings.max_upload_bytes + 1)
    doc = services.store_upload(session, settings, user, file.filename or "upload", data)
    return UploadResponse(document_id=doc.id, filename=doc.filename, file_format=doc.file_format,
                          size_bytes=doc.file_size, page_count=doc.page_count, processing_status=doc.processing_status)


@router.post("/{document_id}/analyze", response_model=RiskReport | dict,
             responses={**ERRORS, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def analyze_document(document_id: str, background: BackgroundTasks,
                     explain: bool = Query(True, description="Generate the optional Ollama explanation in the background"),
                     as_of: date | None = Query(None, description="Analysis date (defaults to the upload date)"),
                     session: Session = Depends(get_session), settings=Depends(get_settings_dep),
                     models=Depends(get_models), session_factory=Depends(get_session_factory),
                     user: User = Depends(require("documents:write"))):
    """Run extraction, ML, anomaly detection, compliance rules and risk scoring. Returns the risk report.
    Documents that are not vendor contracts are stored but not risk-scored (V1 scope)."""
    doc = get_document_or_404(session, document_id)
    if doc.processing_status == "processing":
        raise HTTPException(409, "Document is already being processed")
    pred = services.analyze(session, settings, models, doc, user, as_of)
    if pred is None:
        return {"document_id": doc.id, "processing_status": doc.processing_status,
                "document_type": doc.document_type, "message": doc.status_message}
    if explain and settings.llm_enabled:
        background.add_task(services.generate_llm_explanation, session_factory, settings, pred.id)
    return build_risk_report(session, doc, pred)


@router.get("", response_model=DocumentList, responses=ERRORS)
def list_documents(risk_level: str | None = None, document_type: str | None = None, vendor: str | None = None,
                   review_status: str | None = None, processing_status: str | None = None,
                   date_from: date | None = None, date_to: date | None = None, q: str | None = None,
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   session: Session = Depends(get_session), user: User = Depends(require("documents:read"))):
    stmt = select(Document, Prediction).outerjoin(Prediction, Prediction.id == Document.latest_prediction_id)
    if risk_level:
        stmt = stmt.where(Prediction.risk_level == risk_level.upper())
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    if vendor:
        stmt = stmt.where(Document.vendor_name.ilike(f"%{vendor}%"))
    if review_status:
        stmt = stmt.where(Document.review_status == review_status)
    if processing_status:
        stmt = stmt.where(Document.processing_status == processing_status)
    if date_from:
        stmt = stmt.where(func.date(Document.uploaded_at) >= date_from)
    if date_to:
        stmt = stmt.where(func.date(Document.uploaded_at) <= date_to)
    if q:
        stmt = stmt.where(or_(Document.filename.ilike(f"%{q}%"), Document.id.ilike(f"%{q}%"),
                              Document.vendor_name.ilike(f"%{q}%")))
    total = session.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = session.execute(stmt.order_by(Document.uploaded_at.desc()).limit(limit).offset(offset)).all()
    items = [DocumentSummary(
        id=d.id, filename=d.filename, uploaded_at=d.uploaded_at, document_type=d.document_type,
        processing_status=d.processing_status, vendor_name=d.vendor_name, review_status=d.review_status,
        risk_score=p.risk_score if p else None, risk_level=p.risk_level if p else None,
        extraction_method=d.extraction_method) for d, p in rows]
    return DocumentList(total=total or 0, items=items)


@router.get("/{document_id}", response_model=DocumentDetail, responses=ERRORS)
def get_document(document_id: str, session: Session = Depends(get_session),
                 user: User = Depends(require("documents:read"))):
    doc = get_document_or_404(session, document_id)
    ents = session.scalars(select(DocumentEntity).where(DocumentEntity.document_id == doc.id)).all()
    uploader = session.get(User, doc.uploaded_by) if doc.uploaded_by else None
    return DocumentDetail(
        id=doc.id, filename=doc.filename, file_format=doc.file_format, mime_type=doc.mime_type,
        file_size=doc.file_size, uploaded_at=doc.uploaded_at, uploaded_by=uploader.full_name if uploader else None,
        document_type=doc.document_type, doc_type_confidence=doc.doc_type_confidence, page_count=doc.page_count,
        extraction_method=doc.extraction_method, extraction_status=doc.extraction_status, ocr_pages=doc.ocr_pages,
        processing_status=doc.processing_status, status_message=doc.status_message, vendor_name=doc.vendor_name,
        review_status=doc.review_status, sections=doc.sections, timings_ms=doc.timings_ms,
        latest_prediction_id=doc.latest_prediction_id,
        entities=[EntityOut(name=e.name, value=e.value_num if e.value_num is not None else e.value_text,
                            source_text=e.source_text, confidence=e.confidence, method=e.method) for e in ents],
    )


def build_risk_report(session: Session, doc: Document, pred: Prediction) -> RiskReport:
    factors = sorted(pred.risk_factors, key=lambda f: f.rank)
    top = [services.factor_display(f) for f in factors if f.shap_value > 0] if pred.shap_explained_class not in (None, "LOW") else []
    return RiskReport(
        document_id=doc.id, prediction_id=pred.id, analyzed_at=pred.created_at, risk_score=pred.risk_score,
        risk_level=pred.risk_level, ml_predicted_class=pred.ml_class, ml_probabilities=pred.ml_probabilities,
        high_risk_probability=pred.high_risk_probability, anomaly_score=pred.anomaly_score,
        score_components={"ml": pred.ml_component, "anomaly": pred.anomaly_component,
                          "compliance": pred.compliance_component, "business": pred.business_component},
        score_contributions=pred.contributions, floors_applied=pred.floors_applied,
        compliance_violations=[{"rule_id": v.rule_id, "rule_name": v.rule_name, "severity": v.severity,
                                "type": v.rule_type, "message": v.message, "remediation": v.remediation,
                                "evidence": v.evidence} for v in pred.violations],
        anomalies=[{"type": a.anomaly_type, "score": a.score, "is_anomaly": a.is_anomaly,
                    "description": a.description, "details": a.details} for a in pred.anomalies],
        top_risk_factors=top[:5], recommended_actions=pred.recommended_actions or [],
        needs_review=pred.needs_review, review_reasons=pred.review_reasons, degraded=pred.degraded,
        component_errors=pred.component_errors, model_version=pred.model_version, model_versions=pred.versions,
        timings_ms=pred.timings_ms,
    )


@router.get("/{document_id}/risk", response_model=RiskReport, responses={**ERRORS, 409: {"model": ErrorResponse}})
def get_risk(document_id: str, session: Session = Depends(get_session),
             user: User = Depends(require("documents:read"))):
    doc = get_document_or_404(session, document_id)
    return build_risk_report(session, doc, get_latest_prediction_or_404(session, doc))


@router.get("/{document_id}/explanation", response_model=ExplanationOut,
            responses={**ERRORS, 409: {"model": ErrorResponse}})
def get_explanation(document_id: str, session: Session = Depends(get_session),
                    user: User = Depends(require("documents:read"))):
    """SHAP explanation of the model prediction + the optional LLM narrative (status: pending/ready/unavailable)."""
    doc = get_document_or_404(session, document_id)
    pred = get_latest_prediction_or_404(session, doc)
    return ExplanationOut(
        document_id=doc.id, prediction_id=pred.id,
        shap={"explained_class": pred.shap_explained_class, "output_space": pred.shap_output_space,
              "base_value": pred.shap_base_value,
              "factors": [{"rank": f.rank, "feature": f.feature, "label": f.label, "value": f.feature_value,
                           "shap_value": f.shap_value, "direction": f.direction, "display": services.factor_display(f)}
                          for f in sorted(pred.risk_factors, key=lambda f: f.rank)]},
        llm_status=pred.llm_status, llm_explanation=pred.llm_explanation, llm_model=pred.llm_model,
        llm_latency_ms=pred.llm_latency_ms, llm_error=pred.llm_error,
        recommended_actions=pred.recommended_actions or [],
    )


@router.post("/{document_id}/explanation/regenerate", status_code=202, responses=ERRORS)
def regenerate_explanation(document_id: str, background: BackgroundTasks, session: Session = Depends(get_session),
                           settings=Depends(get_settings_dep), session_factory=Depends(get_session_factory),
                           user: User = Depends(require("documents:write"))):
    doc = get_document_or_404(session, document_id)
    pred = get_latest_prediction_or_404(session, doc)
    pred.llm_status = "pending" if settings.llm_enabled else "disabled"
    session.commit()
    if settings.llm_enabled:
        background.add_task(services.generate_llm_explanation, session_factory, settings, pred.id)
    return {"document_id": doc.id, "llm_status": pred.llm_status}


@router.post("/{document_id}/feedback", response_model=FeedbackOut, status_code=201,
             responses={**ERRORS, 409: {"model": ErrorResponse}})
def submit_feedback(document_id: str, body: FeedbackIn, session: Session = Depends(get_session),
                    user: User = Depends(require("feedback:write"))):
    """Reviewer confirms or corrects the system's risk level. Feedback stays 'pending' until a second
    person (risk manager) validates it - only validated feedback is used for retraining."""
    doc = get_document_or_404(session, document_id)
    pred = get_latest_prediction_or_404(session, doc)
    label = pred.risk_level if body.decision == "confirm" else body.corrected_label
    flags = consistency_flags(label, pred.risk_level, pred.ml_class, body.decision,
                              [v.severity for v in pred.violations if v.rule_type == "violation"],
                              doc.extraction_method)
    fb = ReviewFeedback(document_id=doc.id, prediction_id=pred.id, reviewer_id=user.id,
                        model_prediction=pred.ml_class, model_risk_level=pred.risk_level,
                        model_risk_score=pred.risk_score, model_version=pred.model_version, decision=body.decision,
                        reviewer_label=label, comment=body.comment, validation_status="pending",
                        validation_flags=flags)
    session.add(fb)
    doc.review_status = "confirmed" if body.decision == "confirm" else "corrected"
    session.flush()
    services.audit(session, user, f"feedback.{body.decision}", "document", doc.id,
                   {"feedback_id": fb.id, "reviewer_label": label, "model_level": pred.risk_level, "flags": flags})
    session.commit()
    return fb
