"""End-to-end: Upload -> Extraction -> Feature generation -> Prediction -> Risk scoring -> Persistence."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy import select

from database.models import (
    AnomalyRecord,
    AuditLog,
    ComplianceViolation,
    Document,
    DocumentEntity,
    DocumentFeatures,
    Prediction,
    RequestMetric,
    RiskFactor,
    VendorContractRecord,
)
from src.features.builder import load_registry
from src.synthetic.generator import build_vendors, render_contract_text, sample_contract
from src.synthetic.render import write_image
from tests.api.conftest import hdr
from tests.conftest import TESSERACT_AVAILABLE, pdf_bytes

pytestmark = pytest.mark.integration


def _contract(seed, submission, value=None, forced=None, vendor_seed=99):
    vendor = build_vendors(1, np.random.default_rng(vendor_seed))[0]
    rng = np.random.default_rng(seed)
    spec = sample_contract(f"E2E-{seed}", vendor, submission, rng, seq=seed, forced=forced or [],
                           value_override_inr=value)
    return spec, render_contract_text(spec, rng)


def _upload_analyze(client, data, name, as_of):
    doc_id = client.post("/documents/upload", headers=hdr("analyst"), files={"file": (name, data)}).json()["document_id"]
    r = client.post(f"/documents/{doc_id}/analyze", headers=hdr("analyst"),
                    params={"as_of": as_of.isoformat(), "explain": "false"})
    assert r.status_code == 200, r.text
    return doc_id, r.json()


def test_full_pipeline_persists_every_stage(client, session_factory):
    spec, text = _contract(1, date(2025, 3, 1), forced=["missing_termination", "high_penalty"])
    doc_id, rep = _upload_analyze(client, pdf_bytes(text), "e2e.pdf", spec.submission_date)

    with session_factory() as s:
        doc = s.get(Document, doc_id)
        assert doc.processing_status == "analyzed" and doc.extraction_method == "native"
        assert doc.vendor_name == spec.vendor_name and doc.review_status in ("pending", "not_required")
        assert {"validation_ms", "extraction_ms", "feature_engineering_ms", "ml_inference_ms", "compliance_ms",
                "anomaly_detection_ms", "scoring_ms"} <= set(doc.timings_ms)

        ents = {e.name: e for e in s.scalars(select(DocumentEntity).where(DocumentEntity.document_id == doc_id))}
        assert ents["payment_days"].value_num == spec.payment_days
        assert ents["start_date"].value_date == spec.start_date
        assert ents["penalty_pct"].value_num == pytest.approx(spec.penalty_pct, abs=0.05)

        feats = s.scalars(select(DocumentFeatures).where(DocumentFeatures.document_id == doc_id)).one()
        assert feats.feature_version == load_registry().version
        assert list(feats.features) == load_registry().names and feats.features["has_termination_clause"] == 0

        pred = s.get(Prediction, doc.latest_prediction_id)
        assert pred.risk_score == rep["risk_score"] and pred.model_version == "test-1"
        assert s.scalars(select(RiskFactor).where(RiskFactor.prediction_id == pred.id)).all()
        assert s.scalars(select(AnomalyRecord).where(AnomalyRecord.prediction_id == pred.id)).all()
        rules = {v.rule_id for v in s.scalars(select(ComplianceViolation).where(ComplianceViolation.prediction_id == pred.id))}
        assert {"COMPLIANCE-001", "COMPLIANCE-008"} <= rules

        vh = s.scalars(select(VendorContractRecord).where(VendorContractRecord.document_id == doc_id)).one()
        assert vh.contract_date == spec.submission_date and vh.source == "analyzed"

        audits = s.scalars(select(AuditLog).where(AuditLog.entity_id == doc_id)).all()
        assert {"document.upload", "document.analyze"} <= {a.action for a in audits}
        # privacy: audit logs carry identifiers/status only, never contract text
        assert not any(spec.vendor_name in str(a.details) or "terminate" in str(a.details) for a in audits)

        assert s.scalars(select(RequestMetric).where(RequestMetric.route == "/documents/{document_id}/analyze")).all()


def test_reanalysis_creates_new_prediction_and_keeps_history_consistent(client, session_factory):
    spec, text = _contract(2, date(2025, 3, 1))
    doc_id, first = _upload_analyze(client, pdf_bytes(text), "re.pdf", spec.submission_date)
    second = client.post(f"/documents/{doc_id}/analyze", headers=hdr("analyst"),
                         params={"as_of": spec.submission_date.isoformat(), "explain": "false"}).json()
    assert second["prediction_id"] != first["prediction_id"]
    with session_factory() as s:
        assert len(s.scalars(select(Prediction).where(Prediction.document_id == doc_id)).all()) == 2
        assert len(s.scalars(select(VendorContractRecord).where(VendorContractRecord.document_id == doc_id)).all()) == 1


def test_vendor_history_is_point_in_time_across_documents(client, session_factory):
    """Earlier analysed contracts of a vendor feed later ones - but never the other way round."""
    for i, d in enumerate([date(2025, 1, 10), date(2025, 2, 10), date(2025, 3, 10)]):
        spec, text = _contract(10 + i, d, value=2_000_000)
        _upload_analyze(client, pdf_bytes(text), f"hist{i}.pdf", d)
    spec, text = _contract(20, date(2025, 4, 10), value=12_000_000)   # 6x the vendor's usual value
    doc_id, rep = _upload_analyze(client, pdf_bytes(text), "unusual.pdf", spec.submission_date)
    with session_factory() as s:
        f = s.scalars(select(DocumentFeatures).where(DocumentFeatures.document_id == doc_id)).one().features
    assert f["vendor_prior_contracts"] == 3
    assert f["value_vs_vendor_avg"] == pytest.approx(6.0, rel=0.02)
    dev = [a for a in rep["anomalies"] if a["type"] == "vendor_value_deviation"]
    assert dev and dev[0]["is_anomaly"] and "6.0x" in dev[0]["description"]


@pytest.mark.ocr
@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract not installed")
def test_scanned_image_goes_through_ocr_and_is_flagged_for_review(client, tmp_path):
    spec, text = _contract(30, date(2025, 3, 1))
    path = tmp_path / "scan.png"
    write_image(text, path)
    doc_id, rep = _upload_analyze(client, path.read_bytes(), "scan.png", spec.submission_date)
    detail = client.get(f"/documents/{doc_id}", headers=hdr("analyst")).json()
    assert detail["extraction_method"] == "ocr" and detail["ocr_pages"] == 1
    assert any("OCR" in r for r in rep["review_reasons"]) and rep["needs_review"]


def test_alembic_migration_builds_the_schema(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from database.models import Base
    from src.config import PROJECT_ROOT, get_settings

    url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
        command.upgrade(cfg, "head")
    finally:
        get_settings.cache_clear()
    tables = set(inspect(create_engine(url)).get_table_names())
    assert set(Base.metadata.tables) <= tables
