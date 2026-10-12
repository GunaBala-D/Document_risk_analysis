"""Central application settings.

All runtime configuration comes from environment variables (optionally a local .env file).
Policy configuration (compliance rules, scoring weights, quality gates) lives in YAML under
``configs/`` because it is reviewed and versioned like code, not like secrets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{(DATA_DIR / 'dre_local.db').as_posix()}"

    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_experiment: str = "document-risk-engine"
    classifier_model_name: str = "document-risk-classifier"
    anomaly_model_name: str = "document-anomaly-detector"
    model_cache_dir: Path = ARTIFACTS_DIR / "model_cache"

    llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 120.0

    tesseract_cmd: str = ""

    upload_dir: Path = DATA_DIR / "uploads"
    max_upload_mb: float = 10.0
    max_pdf_pages: int = 60

    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_yaml_config(name: str) -> dict:
    """Load ``configs/<name>.yaml``. Cached: policy files are read once per process."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
