"""Render generated document text to real files (PDF, DOCX, scanned PDF, PNG, JPG).

Rendering to genuine file formats (instead of feeding text straight into feature
engineering) means the training data goes through the *same* ingestion, extraction and
OCR code that production uploads go through.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf as fitz
from docx import Document

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN, LINE_H, WRAP = 50, 14, 95


def _wrapped_lines(text: str, width: int = WRAP) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width, break_on_hyphens=False) or [""])
    return lines


def text_to_pdf_document(text: str) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN + 10
    for line in _wrapped_lines(text):
        if y > PAGE_H - MARGIN:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN + 10
        page.insert_text((MARGIN, y), line, fontsize=10, fontname="helv")
        y += LINE_H
    return doc


def write_pdf(text: str, path: Path) -> None:
    doc = text_to_pdf_document(text)
    doc.save(path)
    doc.close()


def write_scanned_pdf(text: str, path: Path, dpi: int = 150) -> None:
    """A PDF whose pages are images only (no text layer) - forces the OCR fallback."""
    src = text_to_pdf_document(text)
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
    out.save(path)
    out.close()
    src.close()


def write_image(text: str, path: Path, dpi: int = 200) -> None:
    """Single tall page image (PNG/JPG), like a phone scan of a whole contract."""
    lines = _wrapped_lines(text, width=110)
    height = MARGIN * 2 + len(lines) * 11
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=height)
    y = MARGIN
    for line in lines:
        page.insert_text((30, y), line, fontsize=8.5, fontname="helv")
        y += 11
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        pix.save(path, jpg_quality=85)
    else:
        pix.save(path)
    doc.close()


def write_docx(text: str, path: Path) -> None:
    document = Document()
    for para in text.split("\n"):
        document.add_paragraph(para)
    document.save(path)


EXTENSIONS = {"pdf": ".pdf", "scan_pdf": ".pdf", "docx": ".docx", "png": ".png", "jpg": ".jpg"}


def render_document(text: str, fmt: str, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}{EXTENSIONS[fmt]}"
    if fmt == "pdf":
        write_pdf(text, path)
    elif fmt == "scan_pdf":
        write_scanned_pdf(text, path)
    elif fmt == "docx":
        write_docx(text, path)
    elif fmt in {"png", "jpg"}:
        write_image(text, path)
    else:
        raise ValueError(f"Unknown render format: {fmt}")
    return path
