from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.auth import hash_key
from api.main import create_app
from database.models import Base, User
from database.session import make_engine
from src.config import Settings
from tests.helpers import fake_api_key, quick_model_service

USERS = [("admin", "admin"), ("analyst", "compliance_analyst"), ("analyst2", "procurement_analyst"),
         ("manager", "risk_manager")]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        upload_dir=tmp_path / "uploads",
        llm_enabled=True,
        ollama_base_url="http://127.0.0.1:9",  # nothing listens here -> exercises the LLM-down path
        ollama_timeout_seconds=2,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        model_cache_dir=tmp_path / "cache",
        max_upload_mb=2,
    )


@pytest.fixture
def session_factory(settings):
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        for name, role in USERS:
            s.add(User(username=name, full_name=name.title(), role=role, api_key_hash=hash_key(fake_api_key(name))))
        s.commit()
    yield factory
    engine.dispose()


def make_client(settings, session_factory, models=None) -> TestClient:
    app = create_app(settings, model_service=models or quick_model_service(), session_factory=session_factory)
    return TestClient(app)


@pytest.fixture
def client(settings, session_factory):
    with make_client(settings, session_factory) as c:
        yield c


def hdr(user: str) -> dict:
    return {"X-API-Key": fake_api_key(user)}
