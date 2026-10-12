"""End-to-end API flow on a real PostgreSQL database (JSONB columns, date filters, aggregates,
Alembic migration). Runs only when TEST_DATABASE_URL points at a disposable database:

    TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/dre_test pytest tests/integration/test_postgres.py

The database is wiped and migrated from scratch.
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from api.auth import hash_key
from api.main import create_app
from database.models import Base, User
from src.config import PROJECT_ROOT, Settings, get_settings
from tests.api.conftest import hdr
from tests.conftest import make_contract, pdf_bytes
from tests.helpers import fake_api_key, quick_model_service

PG_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not PG_URL or not PG_URL.startswith("postgresql"),
                                 reason="TEST_DATABASE_URL (PostgreSQL) not set")]


@pytest.fixture(scope="module")
def pg_factory():
    engine = create_engine(PG_URL)
    with engine.begin() as c:
        Base.metadata.drop_all(c)
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    from alembic import command
    from alembic.config import Config

    os.environ["DATABASE_URL"] = PG_URL
    get_settings.cache_clear()
    try:
        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
        command.upgrade(cfg, "head")
    finally:
        os.environ.pop("DATABASE_URL", None)
        get_settings.cache_clear()
    assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        for name, role in [("analyst", "compliance_analyst"), ("manager", "risk_manager")]:
            s.add(User(username=name, full_name=name, role=role, api_key_hash=hash_key(fake_api_key(name))))
        s.commit()
    yield factory
    engine.dispose()


def test_full_flow_on_postgres(pg_factory, tmp_path):
    settings = Settings(database_url=PG_URL, upload_dir=tmp_path / "uploads", llm_enabled=False,
                        model_cache_dir=tmp_path / "cache")
    app = create_app(settings, model_service=quick_model_service(), session_factory=pg_factory)
    as_of = date(2025, 6, 1)
    with TestClient(app) as c:
        ids = []
        for seed, forced in [(3, []), (4, ["unlimited_liability", "missing_termination"]), (5, ["excessive_payment"])]:
            _, text_ = make_contract(seed=seed, forced=forced, submission=as_of)
            r = c.post("/documents/upload", headers=hdr("analyst"), files={"file": (f"c{seed}.pdf", pdf_bytes(text_))})
            assert r.status_code == 201, r.text
            doc_id = r.json()["document_id"]
            a = c.post(f"/documents/{doc_id}/analyze", headers=hdr("analyst"),
                       params={"as_of": as_of.isoformat(), "explain": "false"})
            assert a.status_code == 200, a.text
            ids.append(doc_id)

        # JSONB round-trip + relationships
        risk = c.get(f"/documents/{ids[1]}/risk", headers=hdr("analyst")).json()
        assert "COMPLIANCE-007" in {v["rule_id"] for v in risk["compliance_violations"]}
        assert isinstance(risk["score_components"]["compliance"], float)
        # date filters (func.date on timestamptz) and pagination
        today = date.today().isoformat()
        listed = c.get("/documents", headers=hdr("analyst"), params={"date_from": today, "date_to": today}).json()
        assert listed["total"] == 3
        # aggregates in analytics
        ov = c.get("/analytics/overview", headers=hdr("analyst")).json()
        assert ov["analyzed_documents"] == 3 and sum(ov["risk_levels"].values()) == 3
        # feedback + four-eyes validation
        fb = c.post(f"/documents/{ids[1]}/feedback", headers=hdr("analyst"),
                    json={"decision": "correct", "corrected_label": "CRITICAL", "comment": "Unlimited liability."}).json()
        ok = c.post(f"/feedback/{fb['id']}/validate", headers=hdr("manager"), json={"approve": True})
        assert ok.status_code == 200 and ok.json()["validation_status"] == "validated"
        # request metrics written through the middleware
        with pg_factory() as s:
            n = s.execute(text("select count(*) from request_metrics")).scalar()
        assert n >= 8
