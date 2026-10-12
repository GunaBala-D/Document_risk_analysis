from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.auth import require
from database.models import AnomalyRecord, ComplianceViolation, Document, Prediction, ReviewFeedback, User
from database.session import get_session

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def overview(request: Request, session: Session = Depends(get_session), user: User = Depends(require("analytics:read"))):
    """Portfolio KPIs computed from stored analysis results (latest prediction per document)."""
    latest = select(Prediction).join(Document, Document.latest_prediction_id == Prediction.id).subquery()
    by_level = dict(session.execute(select(latest.c.risk_level, func.count()).group_by(latest.c.risk_level)).all())
    by_status = dict(session.execute(select(Document.processing_status, func.count())
                                     .group_by(Document.processing_status)).all())
    by_type = dict(session.execute(select(Document.document_type, func.count())
                                   .group_by(Document.document_type)).all())
    n_anomalous_docs = session.scalar(
        select(func.count(func.distinct(AnomalyRecord.prediction_id)))
        .where(AnomalyRecord.is_anomaly.is_(True), AnomalyRecord.prediction_id.in_(select(latest.c.id))))
    n_violations = session.scalar(
        select(func.count()).select_from(ComplianceViolation)
        .where(ComplianceViolation.rule_type == "violation", ComplianceViolation.prediction_id.in_(select(latest.c.id))))
    top_rules = session.execute(
        select(ComplianceViolation.rule_id, ComplianceViolation.rule_name, func.count())
        .where(ComplianceViolation.prediction_id.in_(select(latest.c.id)))
        .group_by(ComplianceViolation.rule_id, ComplianceViolation.rule_name)
        .order_by(func.count().desc()).limit(8)).all()
    pending_reviews = session.scalar(select(func.count()).select_from(Document).where(Document.review_status == "pending"))
    feedback = dict(session.execute(select(ReviewFeedback.validation_status, func.count())
                                    .group_by(ReviewFeedback.validation_status)).all())
    trend = session.execute(
        select(func.date(latest.c.created_at), latest.c.risk_level, func.count())
        .group_by(func.date(latest.c.created_at), latest.c.risk_level)
        .order_by(func.date(latest.c.created_at))).all()
    models = request.app.state.models
    return {
        "total_documents": sum(by_status.values()),
        "analyzed_documents": by_status.get("analyzed", 0),
        "risk_levels": {lvl: by_level.get(lvl, 0) for lvl in ("LOW", "MEDIUM", "HIGH", "CRITICAL")},
        "documents_with_anomalies": n_anomalous_docs or 0,
        "compliance_violations": n_violations or 0,
        "top_compliance_rules": [{"rule_id": r, "rule_name": n, "count": c} for r, n, c in top_rules],
        "pending_reviews": pending_reviews or 0,
        "feedback": feedback,
        "processing_status": by_status,
        "document_types": {k or "unknown": v for k, v in by_type.items()},
        "daily_risk_levels": [{"date": str(d), "risk_level": lvl, "count": c} for d, lvl, c in trend],
        "current_model_version": models.classifier.model_version if models.classifier else None,
        "model_status": models.status,
    }
