"""Document processing stage: bytes -> validated, extracted, structured document.

Validation -> text extraction (OCR fallback) -> cleaning -> document type detection ->
section detection -> entity extraction. Every stage is timed. Risk analysis (features,
models, rules, scoring) lives in ``src.pipeline.analyzer`` and only runs for supported
document types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.extraction.entities import Entity, extract_entities
from src.extraction.text_extraction import ExtractionResult, extract_text
from src.ingestion.validation import ValidatedFile, validate_upload
from src.logging_utils import timed
from src.preprocessing.cleaning import clean_text
from src.preprocessing.doc_type import DocTypeResult, detect_document_type
from src.preprocessing.sections import Section, detect_sections


@dataclass
class ProcessedDocument:
    file: ValidatedFile
    extraction: ExtractionResult
    text: str = ""
    doc_type: DocTypeResult | None = None
    sections: list[Section] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.extraction.status != "failed"


def process_document(
    filename: str, data: bytes, max_bytes: int, max_pages: int = 60, tesseract_cmd: str | None = None,
    validated: ValidatedFile | None = None,
) -> ProcessedDocument:
    """Raises ``ValidationError`` for rejected uploads; extraction failures are returned, not raised."""
    timings: dict[str, float] = {}
    with timed(timings, "validation_ms"):
        vf = validated or validate_upload(filename, data, max_bytes, max_pages)
    with timed(timings, "extraction_ms"):
        extraction = extract_text(data, vf.file_format, tesseract_cmd)
    doc = ProcessedDocument(file=vf, extraction=extraction, timings_ms=timings)
    if not doc.ok:
        return doc

    with timed(timings, "cleaning_ms"):
        doc.text = clean_text(extraction.text)
    with timed(timings, "doc_type_ms"):
        doc.doc_type = detect_document_type(doc.text)
    if not doc.doc_type.supported:
        return doc
    with timed(timings, "section_detection_ms"):
        doc.sections = detect_sections(doc.text)
    with timed(timings, "entity_extraction_ms"):
        doc.entities = extract_entities(doc.text, doc.sections)
    return doc
