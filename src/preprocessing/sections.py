"""Contract section detection.

Two-tier heuristic approach:
  1. Heading detection: lines that look like headings (numbered, "ARTICLE n", "Clause n:",
     ALL-CAPS, or "Title:") are mapped to canonical sections via synonym patterns. The
     section spans until the next detected heading.
  2. Clause-language fallback: if no heading was found for a mandatory section, strong
     clause language in the body (e.g. "may terminate this Agreement") still counts as the
     clause being present, with lower confidence.

Rules beat an ML/NLP model here: headings in contracts are highly conventional, the rules
are auditable, and every decision can be traced to a pattern.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

CANONICAL_SECTIONS = [
    "parties", "term", "payment", "termination", "renewal", "penalties",
    "liability", "confidentiality", "compliance", "jurisdiction",
]
MANDATORY_SECTIONS = ["payment", "termination", "liability", "confidentiality", "jurisdiction"]

# Order matters: more specific patterns first ("termination" before "term").
HEADING_SYNONYMS: list[tuple[str, re.Pattern]] = [
    ("termination", re.compile(r"\bterminat")),
    ("renewal", re.compile(r"\brenewal\b|\bextension\b")),
    ("parties", re.compile(r"\bparties\b|\brecitals\b")),
    ("penalties", re.compile(r"\bpenalt|\bliquidated damages\b|\bservice credits?\b")),
    ("liability", re.compile(r"\bliabilit|\bindemn")),
    ("confidentiality", re.compile(r"\bconfidential|\bnon-disclosure\b")),
    ("compliance", re.compile(r"\bcompliance\b|\banti-bribery\b")),
    ("jurisdiction", re.compile(r"\bgoverning law\b|\bjurisdiction\b|\bdispute resolution\b")),
    ("payment", re.compile(r"\bpayment\b|\bfees\b|\binvoicing\b|\bconsideration\b")),
    ("term", re.compile(r"^term\b|\bterm of\b|\bduration\b|\bcontract period\b|^term$")),
    ("general", re.compile(r"\bgeneral provisions\b|\bmiscellaneous\b|\bboilerplate\b")),
]

_HEADING_PATTERNS = [
    re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})*\.?)\s+(?P<title>[A-Za-z][A-Za-z &/,'-]{2,70})$"),
    re.compile(r"^(?:article|clause|section)\s+\d{1,2}\s*[-:.]?\s*(?P<title>[A-Za-z][A-Za-z &/,'-]{2,70})$",
               re.IGNORECASE),
    re.compile(r"^(?P<title>[A-Z][A-Za-z &/,'-]{2,60}):$"),
    re.compile(r"^(?P<title>[A-Z][A-Z &/,'-]{3,60})$"),
]

CLAUSE_FALLBACK: dict[str, re.Pattern] = {
    "termination": re.compile(r"\b(may|right to|entitled to) terminate this agreement\b", re.IGNORECASE),
    "liability": re.compile(r"\bliability\b[^.]{0,120}\b(shall not exceed|limited to|unlimited)\b", re.IGNORECASE),
    "confidentiality": re.compile(r"\bkeep confidential\b|\bconfidential information\b", re.IGNORECASE),
    "jurisdiction": re.compile(r"\bgoverned by the laws of\b", re.IGNORECASE),
    "payment": re.compile(r"\bnet \d{1,3} days\b|\bwithin [^.]{0,40}\(?\d{1,3}\)? days\b", re.IGNORECASE),
    "compliance": re.compile(r"\banti-bribery\b|\banti-corruption\b", re.IGNORECASE),
}


@dataclass
class Section:
    name: str
    heading: str
    start: int
    end: int
    confidence: float
    method: str  # heading | clause_language

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_heading(title: str) -> str | None:
    t = title.lower().strip()
    for name, pattern in HEADING_SYNONYMS:
        if pattern.search(t):
            return name
    return None


def _heading_title(line: str) -> str | None:
    if len(line) > 80 or len(line) < 4:
        return None
    for pattern in _HEADING_PATTERNS:
        m = pattern.match(line)
        if m:
            return m.group("title").strip()
    return None


def detect_sections(text: str) -> list[Section]:
    """Detect canonical sections in cleaned text. Offsets refer to ``text``."""
    headings: list[tuple[int, str, str]] = []  # (offset, canonical, heading line)
    offset = 0
    for line in text.split("\n"):
        title = _heading_title(line)
        if title:
            name = _classify_heading(title)
            if name:
                headings.append((offset, name, line))
        offset += len(line) + 1

    sections: list[Section] = []
    seen: set[str] = set()
    for i, (start, name, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if name == "general" or name in seen:
            continue
        seen.add(name)
        sections.append(Section(name, heading, start, end, 0.9, "heading"))

    for name, pattern in CLAUSE_FALLBACK.items():
        if name in seen:
            continue
        m = pattern.search(text)
        if m:
            seen.add(name)
            sections.append(Section(name, "", m.start(), m.end(), 0.6, "clause_language"))

    return sorted(sections, key=lambda s: s.start)


def section_text(text: str, sections: list[Section], name: str) -> str | None:
    for s in sections:
        if s.name == name:
            return text[s.start:s.end]
    return None


def missing_mandatory(sections: list[Section]) -> list[str]:
    present = {s.name for s in sections}
    return [m for m in MANDATORY_SECTIONS if m not in present]
