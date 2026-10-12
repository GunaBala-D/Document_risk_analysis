"""Builds monitoring reports from operational data (used by the API and the retraining job)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import DocumentFeatures, MonitoringReport, Prediction, RequestMetric, ReviewFeedback
from src.config import ARTIFACTS_DIR, Settings, load_yaml_config
from src.features.builder import load_registry
from src.logging_utils import get_logger
from src.monitoring.drift import feature_drift, performance_on_feedback, prediction_drift, service_health
from src.monitoring.retraining_policy import decide_retraining

log = get_logger(__name__)


@lru_cache(maxsize=4)
def load_reference(model_name: str, version: str, tracking_uri: str) -> pd.DataFrame:
    """Training-time feature distribution of the serving model (logged with its MLflow run)."""
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        run_id = MlflowClient().get_model_version(model_name, version).run_id
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference_features.parquet")
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("reference_from_registry_failed", extra={"error": str(exc)})
        return pd.read_parquet(ARTIFACTS_DIR / "reports" / "reference_features.parquet")


def _window_start(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def drift_report(session: Session, settings: Settings, models, window_days: int | None = None) -> dict:
    cfg = load_yaml_config("monitoring")
    window_days = window_days or cfg["drift"]["window_days"]
    registry = load_registry()
    since = _window_start(window_days)
    rows = session.execute(select(DocumentFeatures.features).where(
        DocumentFeatures.created_at >= since, DocumentFeatures.feature_version == registry.version)).scalars().all()
    current = pd.DataFrame(rows, columns=registry.names) if rows else pd.DataFrame(columns=registry.names)
    current = current.astype(float)

    if models.classifier is None:
        data = {"status": "model_unavailable"}
        pred = {"status": "model_unavailable"}
    else:
        ref = load_reference(models.classifier.model_name, models.classifier.model_version,
                             settings.mlflow_tracking_uri)
        data = feature_drift(ref, current, registry.names, cfg["drift"])
        classes = session.execute(select(Prediction.ml_class).where(
            Prediction.created_at >= since, Prediction.ml_class.is_not(None))).scalars().all()
        pred = prediction_drift(models.classifier_metadata.get("reference_prediction_distribution", {}),
                                list(classes), cfg["drift"])
    return {"window_days": window_days, "generated_at": datetime.now(UTC).isoformat(),
            "model_version": models.classifier.model_version if models.classifier else None,
            "data_drift": data, "prediction_drift": pred}


def performance_report(session: Session, models) -> dict:
    rows = session.execute(
        select(ReviewFeedback.reviewer_label, Prediction.ml_class, ReviewFeedback.created_at)
        .join(Prediction, Prediction.id == ReviewFeedback.prediction_id)
        .where(ReviewFeedback.validation_status == "validated", Prediction.ml_class.is_not(None))
    ).all()
    test_f1 = (models.classifier_metadata.get("test_metrics") or {}).get("f1_macro") if models.classifier else None
    return performance_on_feedback([r.ml_class for r in rows], [r.reviewer_label for r in rows], test_f1)


def health_report(session: Session) -> dict:
    cfg = load_yaml_config("monitoring")["service"]
    since = datetime.now(UTC) - timedelta(hours=cfg["window_hours"])
    rows = session.execute(select(RequestMetric.route, RequestMetric.status_code, RequestMetric.latency_ms,
                                  RequestMetric.ts).where(RequestMetric.ts >= since)).all()
    df = pd.DataFrame(rows, columns=["route", "status_code", "latency_ms", "ts"])
    out = service_health(df, cfg)
    out["window_hours"] = cfg["window_hours"]
    return out


def unused_validated_labels(session: Session) -> int:
    return len(session.execute(select(ReviewFeedback.id).where(
        ReviewFeedback.validation_status == "validated", ReviewFeedback.used_in_training_run.is_(None))).all())


def last_retraining_run(session: Session) -> dict | None:
    recent = session.scalars(select(MonitoringReport).where(MonitoringReport.report_type == "retraining_decision")
                             .order_by(MonitoringReport.created_at.desc()).limit(20)).all()
    r = next((x for x in recent if x.summary.get("candidate_version")), None)  # skip "no retraining" decisions
    if r is None:
        return None
    s = r.summary
    failures = [c["name"] for c in (s.get("quality_gate") or {}).get("checks", []) if not c.get("passed")]
    return {"at": r.created_at.isoformat(), "candidate_version": s.get("candidate_version"),
            "promoted": s.get("promoted"), "simulation": s.get("simulation"), "gate_failures": failures,
            "candidate_f1_macro": (s.get("candidate_test_metrics") or {}).get("f1_macro"),
            "production_f1_macro": (s.get("production_test_metrics") or {}).get("f1_macro")}


def full_summary(session: Session, settings: Settings, models, persist: bool = False) -> dict:
    drift = drift_report(session, settings, models)
    perf = performance_report(session, models)
    decision = decide_retraining(unused_validated_labels(session), drift["data_drift"].get("dataset_drift"),
                                 perf.get("degraded"))
    summary = {**drift, "performance": perf, "service_health": health_report(session),
               "retraining": {**decision.to_dict(), "last_run": last_retraining_run(session)}}
    if persist:
        session.add(MonitoringReport(report_type="drift", model_version=drift["model_version"],
                                     summary={k: v for k, v in summary.items() if k != "service_health"}))
        session.commit()
    return summary
