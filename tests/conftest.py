from __future__ import annotations

import shutil
from datetime import date

import numpy as np
import pytest

from src.synthetic.generator import build_vendors, render_contract_text, sample_contract
from src.synthetic.render import text_to_pdf_document

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None or bool(
    __import__("os").environ.get("TESSERACT_CMD")
)


def make_contract(seed: int = 7, forced: list[str] | None = None, fmt: str = "pdf",
                  submission: date = date(2025, 6, 1)):
    rng = np.random.default_rng(seed)
    vendor = build_vendors(1, rng)[0]
    spec = sample_contract("TEST-1", vendor, submission, rng, seq=1, forced=forced or [])
    spec.render_format = fmt
    text = render_contract_text(spec, rng)
    return spec, text


def pdf_bytes(text: str) -> bytes:
    doc = text_to_pdf_document(text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def contract():
    return make_contract()


@pytest.fixture
def contract_pdf(contract):
    spec, text = contract
    return spec, pdf_bytes(text)
