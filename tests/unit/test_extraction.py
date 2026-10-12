from datetime import date

import pytest

from src.extraction.entities import extract_entities, normalize_vendor_key, parse_date
from src.extraction.text_extraction import extract_text
from src.preprocessing.cleaning import clean_text
from src.preprocessing.doc_type import detect_document_type
from src.preprocessing.sections import detect_sections, missing_mandatory
from src.synthetic.render import write_image, write_scanned_pdf
from tests.conftest import TESSERACT_AVAILABLE, make_contract, pdf_bytes


def _analyse(text):
    clean = clean_text(text)
    sections = detect_sections(clean)
    return clean, sections, extract_entities(clean, sections)


@pytest.mark.parametrize("seed", range(12))
def test_entities_match_ground_truth_native_text(seed):
    spec, text = make_contract(seed=seed)
    _, _, ents = _analyse(text)
    assert ents["vendor_name"].value == spec.vendor_name
    assert ents["customer_name"].value == spec.customer_name
    assert ents["contract_number"].value == spec.contract_number
    assert ents["start_date"].value == spec.start_date
    assert ents["end_date"].value == spec.end_date
    assert ents["payment_days"].value == spec.payment_days
    assert ents["currency"].value == spec.currency
    assert abs(ents["contract_value"].value - spec.contract_value) <= 0.005 * spec.contract_value
    assert ents["governing_law"].value == spec.governing_law
    assert ents["renewal_months"].value == spec.renewal_months
    assert abs(ents["penalty_pct"].value - spec.penalty_pct) < 0.05


def test_foreign_governing_law_with_article():
    text = ("GOVERNING LAW:\nThis Agreement shall be governed by the laws of the State of Delaware, USA and the "
            "courts of\nthe State of Delaware, USA shall have exclusive jurisdiction.")
    _, _, ents = _analyse(text)
    assert ents["governing_law"].value == "the State of Delaware, USA"
    assert ents["jurisdiction"].value == "the State of Delaware, USA"


def test_missing_termination_clause_detected():
    spec, text = make_contract(seed=3, forced=["missing_termination"])
    _, sections, ents = _analyse(text)
    assert not spec.has_termination
    assert "termination" in missing_mandatory(sections)
    assert "termination_notice_days" not in ents


def test_payment_days_not_confused_with_termination_notice():
    spec, text = make_contract(seed=11)
    _, _, ents = _analyse(text)
    assert ents["payment_days"].value == spec.payment_days
    assert ents["termination_notice_days"].value == spec.termination_notice_days


def test_unlimited_liability_detected():
    _, text = make_contract(seed=5, forced=["unlimited_liability"])
    _, _, ents = _analyse(text)
    assert ents["liability_cap"].value == "unlimited"


def test_section_positions_are_offsets_into_text():
    _, text = make_contract(seed=2)
    clean, sections, _ = _analyse(text)
    for s in sections:
        if s.method == "heading":
            assert clean[s.start:s.end].startswith(s.heading)


def test_parse_date_formats():
    assert parse_date("05/03/2024") == date(2024, 3, 5)  # DD/MM/YYYY
    assert parse_date("2024-03-05") == date(2024, 3, 5)
    assert parse_date("5 March 2024") == date(2024, 3, 5)
    assert parse_date("March 5, 2024") == date(2024, 3, 5)
    assert parse_date("not a date") is None


def test_indian_amount_units():
    text = ("PAYMENT TERMS:\nThe total contract value is INR 1.20 Crore. Payment terms: Net 45 days from invoice.\n")
    _, sections, ents = _analyse(text)
    assert ents["contract_value"].value == pytest.approx(12_000_000)
    assert ents["currency"].value == "INR"


def test_vendor_key_normalisation():
    assert normalize_vendor_key("Apex Technologies Pvt. Ltd.") == normalize_vendor_key("APEX Technologies Private Limited")
    assert normalize_vendor_key("Apex Technologies Ltd.") == normalize_vendor_key("Apex Technologies Limited")
    # different legal entities must NOT share a vendor history
    assert normalize_vendor_key("Cedar Infotech LLP") != normalize_vendor_key("Cedar Infotech Pvt. Ltd.")
    assert normalize_vendor_key("Cedar Infotech Ltd.") != normalize_vendor_key("Cedar Infotech Private Limited")


def test_doc_type_detection():
    _, text = make_contract(seed=1)
    assert detect_document_type(clean_text(text)).doc_type == "vendor_contract"
    invoice = "TAX INVOICE\nInvoice Number: INV-1\nBill To: X\nCGST 9% SGST 9% Total Amount Due: INR 100"
    assert detect_document_type(invoice).doc_type == "invoice"
    po = "PURCHASE ORDER\nPO Number: PO-1\nShip To: X\nQuantity Unit Price\nDelivery Date: 1 May 2025"
    assert detect_document_type(po).doc_type == "purchase_order"
    assert detect_document_type("hello world, nothing here").doc_type == "unknown"


def test_native_pdf_extraction_does_not_use_ocr():
    _, text = make_contract(seed=4)
    res = extract_text(pdf_bytes(text), "pdf")
    assert res.status == "success" and res.method == "native" and not res.ocr_pages


@pytest.mark.ocr
@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract not installed")
def test_scanned_pdf_falls_back_to_ocr(tmp_path):
    spec, text = make_contract(seed=6)
    path = tmp_path / "scan.pdf"
    write_scanned_pdf(text, path)
    res = extract_text(path.read_bytes(), "pdf")
    assert res.method == "ocr" and res.ocr_pages
    _, _, ents = _analyse(res.text)
    assert ents["payment_days"].value == spec.payment_days


@pytest.mark.ocr
@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract not installed")
def test_png_is_ocrd(tmp_path):
    _, text = make_contract(seed=8)
    path = tmp_path / "scan.png"
    write_image(text, path)
    res = extract_text(path.read_bytes(), "png")
    assert res.status == "success" and res.method == "ocr"
    assert detect_document_type(clean_text(res.text)).doc_type == "vendor_contract"


def test_ocr_unavailable_reports_failure_cleanly(tmp_path):
    _, text = make_contract(seed=9)
    path = tmp_path / "scan.png"
    write_image(text, path)
    res = extract_text(path.read_bytes(), "png", tesseract_cmd="Z:/definitely/missing/tesseract.exe")
    assert res.status == "failed" and "Tesseract" in res.error
