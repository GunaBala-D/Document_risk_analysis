import io
from datetime import date

import pytest
from docx import Document as DocxDocument

from src.pipeline.model_service import ModelService
from src.synthetic.generator import render_invoice_text
from tests.api.conftest import hdr, make_client
from tests.conftest import make_contract, pdf_bytes
from tests.helpers import quick_model_service

AS_OF = date(2025, 6, 1)


def contract_pdf(seed=3, forced=None) -> bytes:
    _, text = make_contract(seed=seed, forced=forced, submission=AS_OF)
    return pdf_bytes(text)


def upload(client, data: bytes, name="contract.pdf", user="analyst"):
    return client.post("/documents/upload", headers=hdr(user), files={"file": (name, data)})


def upload_and_analyze(client, data=None, name="contract.pdf", **params):
    r = upload(client, data or contract_pdf(), name)
    assert r.status_code == 201, r.text
    doc_id = r.json()["document_id"]
    a = client.post(f"/documents/{doc_id}/analyze", headers=hdr("analyst"),
                    params={"as_of": AS_OF.isoformat(), **params})
    return doc_id, a


# ------------------------------------------------------------------ health & auth
def test_health_reports_dependencies(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["checks"]["database"]["status"] == "up"
    assert body["checks"]["models"]["classifier"] == "loaded"
    assert body["checks"]["llm"]["status"] == "down"     # deliberately unreachable in tests
    assert body["status"] == "degraded"
    assert "X-Request-ID" in r.headers


def test_authentication_required(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    me = client.get("/auth/me", headers=hdr("manager")).json()
    assert me["role"] == "risk_manager" and "feedback:validate" in me["permissions"]


def test_openapi_docs_available(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/documents/upload", "/documents/{document_id}/analyze", "/documents/{document_id}",
              "/documents/{document_id}/risk", "/documents/{document_id}/explanation",
              "/documents/{document_id}/feedback", "/analytics/overview", "/models/current", "/health"]:
        assert p in paths


# ------------------------------------------------------------------ upload validation
def test_upload_pdf_created(client):
    r = upload(client, contract_pdf())
    assert r.status_code == 201
    body = r.json()
    assert body["document_id"].startswith("DOC-") and body["file_format"] == "pdf"
    assert body["processing_status"] == "uploaded"


def test_upload_duplicate_conflict(client):
    data = contract_pdf()
    first = upload(client, data).json()["document_id"]
    r = upload(client, data)
    assert r.status_code == 409 and r.json()["details"]["existing_document_id"] == first


@pytest.mark.parametrize("name,data,status,code", [
    ("evil.pdf", b"MZ" + b"\0" * 500, 415, "unsupported_content"),
    ("contract.exe", b"%PDF-1.4 whatever", 415, "unsupported_extension"),
    ("contract.png", None, 415, "extension_mismatch"),
    ("empty.pdf", b"", 422, "empty_file"),
    ("big.pdf", "BIG", 413, "file_too_large"),
], ids=["disguised_executable", "bad_extension", "extension_mismatch", "empty", "too_large"])
def test_upload_rejections(client, name, data, status, code):
    if data == "BIG":
        data = b"%PDF-" + b"0" * (3 * 1024 * 1024)
    r = upload(client, data if data is not None else contract_pdf(), name)
    assert r.status_code == status and r.json()["error"] == code


def test_analyst_cannot_validate_and_unknown_document_404(client):
    assert client.post("/feedback/1/validate", headers=hdr("analyst"), json={"approve": True}).status_code == 403
    assert client.get("/documents/DOC-NOPE", headers=hdr("analyst")).status_code == 404
    assert client.post("/documents/DOC-NOPE/analyze", headers=hdr("analyst")).status_code == 404


# ------------------------------------------------------------------ analyze / risk / explanation
def test_analyze_returns_structured_risk_report(client):
    doc_id, r = upload_and_analyze(client, contract_pdf(forced=["missing_termination", "excessive_payment"]),
                                   explain="false")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["document_id"] == doc_id and 0 <= rep["risk_score"] <= 100
    assert rep["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert {"ml", "anomaly", "compliance", "business"} == set(rep["score_components"])
    assert all(v is not None and 0 <= v <= 1 for v in rep["score_components"].values())
    rule_ids = {v["rule_id"] for v in rep["compliance_violations"]}
    assert "COMPLIANCE-001" in rule_ids
    assert rep["model_version"] == "test-1" and rep["degraded"] is False
    assert sum(p for p in rep["ml_probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert rep["recommended_actions"][0]["source"] == "risk_level"


def test_risk_level_consistent_with_score_thresholds(client):
    _, r = upload_and_analyze(client, explain="false")
    rep = r.json()
    expected = "LOW" if rep["risk_score"] <= 30 else "MEDIUM" if rep["risk_score"] <= 60 else \
        "HIGH" if rep["risk_score"] <= 80 else "CRITICAL"
    assert rep["risk_level"] == expected or rep["floors_applied"]


def test_document_detail_risk_and_explanation_endpoints(client):
    doc_id, _ = upload_and_analyze(client, explain="false")
    d = client.get(f"/documents/{doc_id}", headers=hdr("analyst")).json()
    assert d["processing_status"] == "analyzed" and d["document_type"] == "vendor_contract"
    names = {e["name"] for e in d["entities"]}
    assert {"vendor_name", "contract_value", "payment_days", "end_date"} <= names
    assert d["sections"] and d["timings_ms"]["extraction_ms"] >= 0
    risk = client.get(f"/documents/{doc_id}/risk", headers=hdr("analyst"))
    assert risk.status_code == 200 and risk.json()["prediction_id"] == d["latest_prediction_id"]
    exp = client.get(f"/documents/{doc_id}/explanation", headers=hdr("analyst")).json()
    assert exp["shap"]["factors"] and exp["shap"]["output_space"] == "probability"
    assert all(":" in f["display"] for f in exp["shap"]["factors"])
    assert exp["llm_status"] == "pending"   # explain=false -> not generated


def test_ollama_down_still_returns_full_risk_result(client):
    """LLM failure must not affect the ML/rule-based result."""
    doc_id, r = upload_and_analyze(client)   # explain=true -> background task hits unreachable Ollama
    assert r.status_code == 200 and r.json()["risk_score"] is not None
    exp = client.get(f"/documents/{doc_id}/explanation", headers=hdr("analyst")).json()
    assert exp["llm_status"] == "unavailable" and exp["llm_explanation"] is None and exp["llm_error"]
    assert exp["shap"]["factors"] and exp["recommended_actions"]


def test_classifier_unavailable_degrades_gracefully(settings, session_factory):
    full = quick_model_service()
    degraded = ModelService(classifier=None, explainer=None, anomaly=full.anomaly,
                            errors={"classifier": "simulated registry outage"})
    with make_client(settings, session_factory, degraded) as c:
        _, r = upload_and_analyze(c, contract_pdf(forced=["missing_termination"]), explain="false")
        rep = r.json()
        assert r.status_code == 200 and rep["degraded"] is True
        assert rep["ml_predicted_class"] is None and rep["score_components"]["ml"] is None
        assert "COMPLIANCE-001" in {v["rule_id"] for v in rep["compliance_violations"]}
        assert "classifier" in rep["component_errors"]
        assert c.get("/health").json()["checks"]["models"]["classifier"] == "unavailable"
        assert c.get("/models/current", headers=hdr("analyst")).status_code == 503


def test_invoice_is_stored_but_not_scored(client):
    import numpy as np

    from src.synthetic.generator import build_vendors

    text = render_invoice_text(np.random.default_rng(0), build_vendors(1, np.random.default_rng(0))[0], 1)
    doc_id, r = upload_and_analyze(client, pdf_bytes(text), "invoice.pdf")
    assert r.status_code == 200
    body = r.json()
    assert body["processing_status"] == "unsupported_type" and "Phase 2" in body["message"]
    assert client.get(f"/documents/{doc_id}/risk", headers=hdr("analyst")).status_code == 409


def test_docx_upload_and_analysis(client):
    _, text = make_contract(seed=21, submission=AS_OF)
    doc = DocxDocument()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    _, r = upload_and_analyze(client, buf.getvalue(), "contract.docx", explain="false")
    assert r.status_code == 200 and r.json()["risk_level"]


def test_blank_document_extraction_failure_is_reported(client):
    import pymupdf

    d = pymupdf.open()
    d.new_page()
    _, r = upload_and_analyze(client, d.tobytes(), "blank.pdf", explain="false")
    assert r.status_code == 422 and r.json()["error"] == "extraction_failed"


# ------------------------------------------------------------------ list / filters / analytics
def test_list_filters_and_analytics(client):
    upload_and_analyze(client, contract_pdf(seed=3), explain="false")
    upload_and_analyze(client, contract_pdf(seed=4, forced=["unlimited_liability", "missing_termination"]),
                       explain="false")
    all_docs = client.get("/documents", headers=hdr("analyst")).json()
    assert all_docs["total"] == 2
    lvl = all_docs["items"][0]["risk_level"]
    assert all(i["risk_level"] == lvl for i in
               client.get("/documents", headers=hdr("analyst"), params={"risk_level": lvl}).json()["items"])
    assert client.get("/documents", headers=hdr("analyst"), params={"vendor": "zzzz"}).json()["total"] == 0
    ov = client.get("/analytics/overview", headers=hdr("analyst")).json()
    assert ov["total_documents"] == 2 and ov["analyzed_documents"] == 2
    assert sum(ov["risk_levels"].values()) == 2 and ov["current_model_version"] == "test-1"
    assert client.get("/models/current", headers=hdr("analyst")).json()["classifier"]["version"] == "test-1"


# ------------------------------------------------------------------ human review
def test_feedback_validation_workflow(client):
    doc_id, r = upload_and_analyze(client, contract_pdf(forced=["unlimited_liability"]), explain="false")
    level = r.json()["risk_level"]
    queue = client.get("/reviews/queue", headers=hdr("analyst")).json()
    assert doc_id in [i["document_id"] for i in queue["items"]]

    bad = client.post(f"/documents/{doc_id}/feedback", headers=hdr("analyst"),
                      json={"decision": "correct", "corrected_label": "LOW"})
    assert bad.status_code == 422   # correction needs a comment

    fb = client.post(f"/documents/{doc_id}/feedback", headers=hdr("analyst"),
                     json={"decision": "correct", "corrected_label": "LOW", "comment": "Liability is insured."})
    assert fb.status_code == 201
    body = fb.json()
    assert body["validation_status"] == "pending" and body["model_risk_level"] == level
    assert "label_below_HIGH_despite_CRITICAL_violation" in body["validation_flags"]

    assert client.get(f"/documents/{doc_id}", headers=hdr("analyst")).json()["review_status"] == "corrected"
    assert client.post(f"/feedback/{body['id']}/validate", headers=hdr("admin"),
                       json={"approve": False}).status_code == 422          # rejection needs a comment
    ok = client.post(f"/feedback/{body['id']}/validate", headers=hdr("manager"),
                     json={"approve": True, "comment": "Checked insurance certificate."})
    assert ok.status_code == 200 and ok.json()["validation_status"] == "validated"
    again = client.post(f"/feedback/{body['id']}/validate", headers=hdr("manager"), json={"approve": True})
    assert again.status_code == 409
    assert len(client.get("/feedback", headers=hdr("manager"), params={"validation_status": "validated"}).json()) == 1


def test_reviewer_cannot_validate_own_feedback(client):
    doc_id, _ = upload_and_analyze(client, explain="false")
    fb = client.post(f"/documents/{doc_id}/feedback", headers=hdr("manager"), json={"decision": "confirm"}).json()
    r = client.post(f"/feedback/{fb['id']}/validate", headers=hdr("manager"), json={"approve": True})
    assert r.status_code == 403


# ------------------------------------------------------------------ monitoring
def test_monitoring_summary_reports_insufficient_data_honestly(client, monkeypatch):
    import api.monitoring_service as ms
    from tests.helpers import synthetic_feature_frame

    X, _ = synthetic_feature_frame()
    monkeypatch.setattr(ms, "load_reference", lambda *a, **k: X)
    upload_and_analyze(client, explain="false")
    s = client.get("/monitoring/summary", headers=hdr("analyst")).json()
    assert s["data_drift"]["status"] == "insufficient_data"
    assert s["performance"]["status"] == "insufficient_labels"
    assert s["retraining"]["retrain"] is False
    assert s["service_health"]["requests"] > 0
    assert client.post("/monitoring/run", headers=hdr("analyst")).status_code == 403


def test_database_unavailable_returns_controlled_503(settings):
    from sqlalchemy.orm import sessionmaker

    from database.session import make_engine

    broken = sessionmaker(bind=make_engine("sqlite:///Z:/definitely/not/here/x.db"))
    with make_client(settings, broken) as c:
        r = c.get("/documents", headers=hdr("analyst"))
        assert r.status_code == 503 and r.json()["error"] == "database_unavailable"
        assert c.get("/health").status_code == 503
