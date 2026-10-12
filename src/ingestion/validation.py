"""Upload validation.

The extension is never trusted on its own. A file is accepted only if:
  1. its size is within limits,
  2. its extension is on the allow-list,
  3. its *content signature* (magic bytes) identifies a supported format,
  4. the signature agrees with the extension, and
  5. the file can actually be opened by the parser that will process it
     (catches truncated/malformed/encrypted files and DOCX zip bombs).
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf as fitz

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "png": "image/png",
    "jpeg": "image/jpeg",
}
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024  # zip-bomb guard
MAX_IMAGE_PIXELS = 60_000_000


class ValidationError(Exception):
    """Raised when an upload must be rejected. ``code`` is stable for API clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedFile:
    filename: str
    file_format: str  # pdf | docx | png | jpeg
    mime_type: str
    size_bytes: int
    page_count: int


def sanitize_filename(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:150] or "upload"


def sniff_format(data: bytes) -> str | None:
    """Identify the format from magic bytes only."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return None
    return None


def _check_pdf(data: bytes, max_pages: int) -> int:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for corrupt files
        raise ValidationError("malformed_file", f"PDF could not be parsed: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ValidationError("encrypted_file", "Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise ValidationError("malformed_file", "PDF has no pages.")
        if doc.page_count > max_pages:
            raise ValidationError("too_many_pages", f"PDF has {doc.page_count} pages; limit is {max_pages}.")
        return doc.page_count


def _check_docx(data: bytes) -> int:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(info.file_size for info in zf.infolist())
            if total > MAX_DOCX_UNCOMPRESSED:
                raise ValidationError("malformed_file", "DOCX expands beyond the allowed size.")
            bad = zf.testzip()
            if bad is not None:
                raise ValidationError("malformed_file", f"Corrupt member in DOCX: {bad}")
        from docx import Document

        Document(io.BytesIO(data))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("malformed_file", f"DOCX could not be parsed: {exc}") from exc
    return 1


def _check_image(data: bytes, fmt: str) -> int:
    try:
        pix = fitz.Pixmap(data)
    except Exception as exc:
        raise ValidationError("malformed_file", f"Image could not be decoded: {exc}") from exc
    if pix.width * pix.height > MAX_IMAGE_PIXELS:
        raise ValidationError("image_too_large", "Image resolution exceeds the allowed limit.")
    if pix.width < 200 or pix.height < 200:
        raise ValidationError("image_too_small", "Image is too small to OCR reliably.")
    return 1


def validate_upload(filename: str, data: bytes, max_bytes: int, max_pages: int = 60) -> ValidatedFile:
    if not data:
        raise ValidationError("empty_file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            "file_too_large", f"File is {len(data) / 1_048_576:.1f} MB; limit is {max_bytes / 1_048_576:.0f} MB."
        )
    safe_name = sanitize_filename(filename)
    ext = PurePath(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            "unsupported_extension", f"'{ext or 'no extension'}' is not supported. Use PDF, DOCX, PNG or JPG."
        )
    detected = sniff_format(data)
    if detected is None:
        raise ValidationError("unsupported_content", "File content does not match any supported format.")
    if detected != ALLOWED_EXTENSIONS[ext]:
        raise ValidationError(
            "extension_mismatch", f"File extension '{ext}' does not match its content ({detected})."
        )

    if detected == "pdf":
        pages = _check_pdf(data, max_pages)
    elif detected == "docx":
        pages = _check_docx(data)
    else:
        pages = _check_image(data, detected)

    return ValidatedFile(
        filename=safe_name, file_format=detected, mime_type=MIME_TYPES[detected],
        size_bytes=len(data), page_count=pages,
    )
