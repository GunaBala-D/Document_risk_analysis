"""Text normalisation applied after extraction (native or OCR).

Keeps line structure (needed by section detection) while removing extraction artefacts:
unicode variants, hyphenated line breaks, page furniture and runs of whitespace.
"""

from __future__ import annotations

import re
import unicodedata

_PAGE_FURNITURE = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|-\s*\d+\s*-)\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
})


def clean_text(text: str) -> str:
    # NFKC folds compatibility characters (ligatures, full-width digits) but keeps the rupee sign.
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = []
    for raw in text.split("\n"):
        line = _SPACES.sub(" ", raw).strip()
        if _PAGE_FURNITURE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def flatten(text: str) -> str:
    """Single-line version used by regex extractors that must span wrapped lines."""
    return re.sub(r"\s+", " ", text).strip()
