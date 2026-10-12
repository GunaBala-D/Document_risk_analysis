import io
import zipfile

import pytest

from src.ingestion.validation import ValidationError, sanitize_filename, sniff_format, validate_upload
from src.synthetic.render import write_docx
from tests.conftest import pdf_bytes

MAX = 5 * 1024 * 1024


def test_valid_pdf_accepted():
    vf = validate_upload("contract.pdf", pdf_bytes("VENDOR AGREEMENT\nSome text " * 20), MAX)
    assert vf.file_format == "pdf" and vf.mime_type == "application/pdf" and vf.page_count >= 1


def test_valid_docx_accepted(tmp_path):
    path = tmp_path / "c.docx"
    write_docx("Agreement text", path)
    vf = validate_upload("c.docx", path.read_bytes(), MAX)
    assert vf.file_format == "docx"


def test_extension_not_trusted_pdf_renamed_to_png():
    with pytest.raises(ValidationError) as exc:
        validate_upload("contract.png", pdf_bytes("hello world " * 20), MAX)
    assert exc.value.code == "extension_mismatch"


def test_unsupported_extension_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_upload("payload.exe", b"MZ\x90\x00" + b"\x00" * 100, MAX)
    assert exc.value.code == "unsupported_extension"


def test_disguised_executable_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_upload("invoice.pdf", b"MZ\x90\x00" + b"\x00" * 100, MAX)
    assert exc.value.code == "unsupported_content"


def test_empty_and_oversized_rejected():
    with pytest.raises(ValidationError) as e1:
        validate_upload("a.pdf", b"", MAX)
    assert e1.value.code == "empty_file"
    with pytest.raises(ValidationError) as e2:
        validate_upload("a.pdf", b"%PDF-" + b"0" * 2000, 1000)
    assert e2.value.code == "file_too_large"


def test_truncated_pdf_rejected():
    data = pdf_bytes("hello world " * 50)
    with pytest.raises(ValidationError) as exc:
        validate_upload("a.pdf", data[:40], MAX)
    assert exc.value.code == "malformed_file"


def test_zip_that_is_not_docx_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(ValidationError) as exc:
        validate_upload("a.docx", buf.getvalue(), MAX)
    assert exc.value.code == "unsupported_content"


def test_sniff_and_sanitize():
    assert sniff_format(b"\x89PNG\r\n\x1a\n....") == "png"
    assert sniff_format(b"\xff\xd8\xff\xe0....") == "jpeg"
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("C:\\Users\\x\\My <Contract>.pdf") == "My _Contract_.pdf"
