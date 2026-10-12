from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require
from database.models import User
from src.config import get_settings

router = APIRouter(prefix="/models", tags=["models"])

_HIDDEN = {"shap_background", "quantiles"}


@router.get("/current")
def current_model(request: Request, user: User = Depends(require("models:read"))):
    """The model versions currently serving predictions, with their recorded evaluation results."""
    models = request.app.state.models
    if models.classifier is None:
        raise HTTPException(503, {"error": "model_unavailable", "message": "No classifier is loaded",
                                  "details": models.errors})
    meta = {k: v for k, v in models.classifier_metadata.items() if k not in _HIDDEN}
    anomaly = {k: v for k, v in models.anomaly_metadata.items() if k not in _HIDDEN}
    return {"classifier": {"name": models.classifier.model_name, "version": models.classifier.model_version,
                           **meta},
            "anomaly_detector": {"version": models.anomaly.model_version if models.anomaly else None, **anomaly},
            "status": models.status}


@router.get("/versions")
def model_versions(user: User = Depends(require("models:read"))):
    """All registered versions with aliases and quality-gate outcome, read from the MLflow registry."""
    from mlflow import MlflowClient

    from src.models.registry import setup_mlflow

    settings = get_settings()
    try:
        setup_mlflow(settings)
        client = MlflowClient()
        out = {}
        for name in (settings.classifier_model_name, settings.anomaly_model_name):
            rows = []
            for found in client.search_model_versions(f"name='{name}'"):
                # search results don't carry aliases in MLflow 3 - fetch each version for them
                mv = client.get_model_version(name, found.version)
                run = client.get_run(mv.run_id)
                rows.append({
                    "version": mv.version, "aliases": list(getattr(mv, "aliases", []) or []),
                    "created_at": mv.creation_timestamp, "run_id": mv.run_id, "tags": mv.tags,
                    "quality_gate": run.data.tags.get("quality_gate"),
                    "pipeline": run.data.tags.get("pipeline"), "simulation": run.data.tags.get("simulation"),
                    "feature_version": run.data.tags.get("feature_version"),
                    "metrics": {k: v for k, v in run.data.metrics.items() if k.startswith("test_") or k.startswith("if_")},
                })
            out[name] = sorted(rows, key=lambda r: int(r["version"]), reverse=True)
        return out
    except Exception as exc:
        raise HTTPException(503, {"error": "registry_unavailable", "message": str(exc)}) from exc
