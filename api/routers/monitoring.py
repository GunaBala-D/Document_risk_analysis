from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import monitoring_service as ms
from api.auth import require
from database.models import MonitoringReport, User
from database.session import get_session

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
def summary(request: Request, session: Session = Depends(get_session), user: User = Depends(require("monitoring:read"))):
    """Data drift, prediction drift, performance on validated labels, service health, retraining decision.
    Computed live from operational data - nothing here is simulated."""
    return ms.full_summary(session, request.app.state.settings, request.app.state.models)


@router.post("/run")
def run_report(request: Request, session: Session = Depends(get_session),
               user: User = Depends(require("monitoring:run"))):
    """Compute and persist a monitoring snapshot (e.g. from a daily scheduler)."""
    return ms.full_summary(session, request.app.state.settings, request.app.state.models, persist=True)


@router.get("/history")
def history(limit: int = Query(30, ge=1, le=365), session: Session = Depends(get_session),
            user: User = Depends(require("monitoring:read"))):
    rows = session.scalars(select(MonitoringReport).order_by(MonitoringReport.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "created_at": r.created_at, "type": r.report_type, "model_version": r.model_version,
             "dataset_drift": (r.summary.get("data_drift") or {}).get("dataset_drift"),
             "drift_share": (r.summary.get("data_drift") or {}).get("drift_share"),
             "prediction_psi": (r.summary.get("prediction_drift") or {}).get("psi"),
             "retrain": (r.summary.get("retraining") or {}).get("retrain")} for r in rows]
