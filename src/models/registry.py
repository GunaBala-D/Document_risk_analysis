"""MLflow tracking + model registry helpers.

Lifecycle (MLflow aliases; MLflow deprecated fixed "stages" in favour of aliases):
    new version (development) -> @staging (candidate) -> quality gate -> @production
The previous production version keeps an @previous alias for one-step rollback.

Serving loads ``models:/<name>@production``. Every successful load also refreshes a
local last-known-good copy, so the API can still start if the MLflow backend is down.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

import joblib

from src.config import PROJECT_ROOT, Settings, get_settings
from src.logging_utils import get_logger

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

log = get_logger(__name__)

PRODUCTION, STAGING, PREVIOUS = "production", "staging", "previous"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def setup_mlflow(settings: Settings | None = None) -> str:
    """Point MLflow at the configured backend and ensure the experiment exists."""
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    if exp is None:
        artifact_location = None
        if settings.mlflow_tracking_uri.startswith("sqlite"):
            artifact_location = (PROJECT_ROOT / "mlruns").as_uri()
        exp_id = mlflow.create_experiment(settings.mlflow_experiment, artifact_location=artifact_location)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def log_sklearn_model(model, registered_name: str, metadata: dict, input_example=None) -> str:
    """Log a model + metadata in the active run and register it. Returns the new version."""
    kwargs = {"registered_model_name": registered_name, "input_example": input_example}
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    if "skops_trusted_types" in params:
        # MLflow 3 serialises sklearn models with skops (no arbitrary code on load). We trust
        # exactly the types of the pipeline we just built - nothing loaded from outside.
        import skops.io

        kwargs["skops_trusted_types"] = skops.io.get_untrusted_types(data=skops.io.dumps(model))
    mlflow.log_dict(metadata, "model_metadata.json")
    info = mlflow.sklearn.log_model(model, **kwargs)
    version = getattr(info, "registered_model_version", None)
    if version is None:
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        versions = client.search_model_versions(f"name='{registered_name}' and run_id='{run_id}'")
        version = max(int(v.version) for v in versions)
    client = MlflowClient()
    for key in ("feature_version", "dataset_version", "family", "imbalance"):
        if metadata.get(key) is not None:
            client.set_model_version_tag(registered_name, str(version), key, str(metadata[key]))
    return str(version)


def get_alias_version(name: str, alias: str) -> str | None:
    try:
        return MlflowClient().get_model_version_by_alias(name, alias).version
    except Exception:  # alias or model does not exist yet
        return None


def set_alias(name: str, alias: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(name, alias, str(version))


def promote_to_production(name: str, version: str) -> str | None:
    """Move @production to ``version``; keep the old one as @previous. Returns old version."""
    old = get_alias_version(name, PRODUCTION)
    if old and old != str(version):
        set_alias(name, PREVIOUS, old)
    set_alias(name, PRODUCTION, version)
    client = MlflowClient()
    client.set_model_version_tag(name, str(version), "lifecycle", "production")
    if old and old != str(version):
        client.set_model_version_tag(name, old, "lifecycle", "archived")
    return old


def load_version_metadata(name: str, version: str) -> dict:
    mv = MlflowClient().get_model_version(name, str(version))
    return mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/model_metadata.json")


def load_model_by_alias(name: str, alias: str = PRODUCTION, settings: Settings | None = None):
    """Load (model, metadata, version) from the registry; fall back to the local cache.

    Raises RuntimeError if neither the registry nor the cache can provide a model.
    """
    settings = settings or get_settings()
    cache = Path(settings.model_cache_dir) / name / alias
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        version = get_alias_version(name, alias)
        if version is None:
            raise LookupError(f"No '{alias}' alias for model {name}")
        model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        metadata = load_version_metadata(name, version)
        metadata["registry_version"] = version
        cache.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cache / "model.joblib")
        (cache / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return model, metadata, version
    except Exception as exc:
        if (cache / "model.joblib").exists():
            log.warning("model_registry_unavailable_using_cache", extra={"model": name, "error": str(exc)})
            metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
            metadata["loaded_from"] = "local_cache"
            return joblib.load(cache / "model.joblib"), metadata, metadata.get("registry_version", "cached")
        raise RuntimeError(f"Could not load model {name}@{alias}: {exc}") from exc


def clear_cache(name: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    shutil.rmtree(Path(settings.model_cache_dir) / name, ignore_errors=True)
