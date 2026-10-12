from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import services
from api.auth import require
from api.schemas import FeedbackOut, ValidateIn
from database.models import Document, Prediction, ReviewFeedback, User
from database.session import get_session
from src.feedback.validation import can_validate

router = APIRouter(tags=["human review"])


@router.get("/reviews/queue")
def review_queue(limit: int = Query(100, ge=1, le=500), session: Session = Depends(get_session),
                 user: User = Depends(require("documents:read"))):
    """Documents awaiting human validation, highest risk first."""
    rows = session.execute(
        select(Document, Prediction).join(Prediction, Prediction.id == Document.latest_prediction_id)
        .where(Document.review_status == "pending")
        .order_by(Prediction.risk_score.desc(), Document.uploaded_at.asc()).limit(limit)
    ).all()
    return {"total": len(rows), "items": [{
        "document_id": d.id, "filename": d.filename, "vendor_name": d.vendor_name, "uploaded_at": d.uploaded_at,
        "risk_score": p.risk_score, "risk_level": p.risk_level, "ml_class": p.ml_class,
        "review_reasons": p.review_reasons, "degraded": p.degraded, "extraction_method": d.extraction_method,
        "n_violations": sum(1 for v in p.violations if v.rule_type == "violation"),
        "n_anomalies": sum(1 for a in p.anomalies if a.is_anomaly),
    } for d, p in rows]}


@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(validation_status: str | None = Query(None, pattern="^(pending|validated|rejected)$"),
                  limit: int = Query(100, ge=1, le=500), session: Session = Depends(get_session),
                  user: User = Depends(require("documents:read"))):
    stmt = select(ReviewFeedback).order_by(ReviewFeedback.created_at.desc()).limit(limit)
    if validation_status:
        stmt = stmt.where(ReviewFeedback.validation_status == validation_status)
    return session.scalars(stmt).all()


@router.post("/feedback/{feedback_id}/validate", response_model=FeedbackOut)
def validate_feedback(feedback_id: int, body: ValidateIn, session: Session = Depends(get_session),
                      user: User = Depends(require("feedback:validate"))):
    """Second-person validation (four-eyes). Only validated feedback enters the retraining dataset."""
    fb = session.get(ReviewFeedback, feedback_id)
    if fb is None:
        raise HTTPException(404, "Feedback not found")
    if fb.validation_status != "pending":
        raise HTTPException(409, f"Feedback already {fb.validation_status}")
    if not can_validate(fb.reviewer_id, user.id):
        raise HTTPException(403, "Reviewers cannot validate their own feedback (four-eyes principle)")
    if not body.approve and not (body.comment and body.comment.strip()):
        raise HTTPException(422, "A comment is required when rejecting feedback")
    fb.validation_status = "validated" if body.approve else "rejected"
    fb.validated_by, fb.validated_at, fb.validation_comment = user.id, services.now_utc(), body.comment
    services.audit(session, user, f"feedback.{fb.validation_status}", "feedback", str(fb.id),
                   {"document_id": fb.document_id, "reviewer_label": fb.reviewer_label})
    session.commit()
    return fb
