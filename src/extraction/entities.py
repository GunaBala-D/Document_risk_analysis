"""Structured entity extraction for vendor contracts.

Deterministic extractors (regex + section context + date/number normalisation). Each
field records value, the matched source text, character offsets in the cleaned text,
the method used and a confidence. Fields are searched inside their expected section
first (e.g. payment days inside "Payment Terms") which avoids classic mistakes like
reading a 30-day *termination notice* as the payment period.

Why not an LLM or spaCy NER here: the target fields are formulaic (amounts, dates,
day counts, defined party names), rules are exact, fast, testable and auditable, and
extraction accuracy is measured against ground truth (see training/build_dataset.py).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

from src.preprocessing.sections import Section, section_text


def _p(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    """Compile with literal spaces relaxed to \\s+ (and optional spaces ' ?' to \\s*) so
    matches survive line wrapping."""
    return re.compile(pattern.replace(" ?", r"\s*").replace(" ", r"\s+"), flags)


MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATE = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS},?\s+\d{{4}}"
    rf"|{MONTHS}\s+\d{{1,2}},\s+\d{{4}}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|\d{4}-\d{2}-\d{2})"
)
MONEY = (
    r"(?P<cur>INR|Rs\.?|₹|USD|US\$|\$|EUR|€)\s?"
    r"(?P<amt>\d[\d,]*(?:\.\d+)?)(?:/-)?(?:\s(?P<unit>crores?|cr|lakhs?|lacs?|million|mn)\b)?"
)
CURRENCY_MAP = {"inr": "INR", "rs": "INR", "rs.": "INR", "₹": "INR", "usd": "USD", "us$": "USD",
                "$": "USD", "eur": "EUR", "€": "EUR"}
UNIT_MULT = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
             "million": 1e6, "mn": 1e6}
VENDOR_ROLES = {"vendor", "supplier", "service provider", "contractor", "consultant"}
CUSTOMER_ROLES = {"customer", "company", "client", "buyer", "purchaser"}

ENTITY_FIELDS = [
    "vendor_name", "customer_name", "contract_number", "signing_date", "start_date", "end_date",
    "contract_value", "currency", "payment_days", "termination_notice_days", "penalty_pct",
    "renewal_months", "governing_law", "jurisdiction", "liability_cap",
]


@dataclass
class Entity:
    name: str
    value: object
    raw: str | None
    confidence: float
    method: str
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.value, date):
            d["value"] = self.value.isoformat()
        return d


_SUFFIX_CANON = [  # spelling variants of the SAME legal form -> one token; different forms stay different
    (re.compile(r"\bprivate\s+limited\b|\bpvt\.?\s*ltd\.?|\bpvt\.?\s+limited\b|\bprivate\s+ltd\.?"), " pvtltd "),
    (re.compile(r"\blimited\b|\bltd\.?"), " ltd "),
    (re.compile(r"\bl\.?l\.?p\.?(?=\s|$)"), " llp "),
    (re.compile(r"\binc\.?|\bincorporated\b"), " inc "),
]


def normalize_vendor_key(name: str | None) -> str | None:
    """Vendor identity key for point-in-time history.

    Canonicalises spellings of a legal form ("Pvt. Ltd." == "Private Limited") but keeps the
    form itself: "Cedar Infotech LLP" and "Cedar Infotech Pvt. Ltd." are different legal
    entities and must not share a vendor history. (Production would resolve identity via
    vendor master data / tax IDs - see docs.)
    """
    if not name:
        return None
    key = name.lower()
    for pattern, token in _SUFFIX_CANON:
        key = pattern.sub(token, key)
    key = re.sub(r"[^a-z0-9 ]", " ", key)
    return re.sub(r"\s+", " ", key).strip() or None


def parse_date(raw: str) -> date | None:
    raw = re.sub(r"\s+", " ", raw.strip())
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return date.fromisoformat(raw)
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", raw):
            return datetime.strptime(raw, "%d/%m/%Y").date()  # Indian convention: DD/MM/YYYY
        return dateparser.parse(raw, dayfirst=True).date()
    except (ValueError, OverflowError, TypeError):
        return None


def parse_money(match: re.Match) -> tuple[float, str] | None:
    try:
        amount = float(match.group("amt").replace(",", ""))
    except ValueError:
        return None
    unit = (match.group("unit") or "").lower()
    amount *= UNIT_MULT.get(unit, 1.0)
    cur = CURRENCY_MAP.get(match.group("cur").lower(), "INR")
    return amount, cur


class ContractEntityExtractor:
    def __init__(self, text: str, sections: list[Section]):
        self.text = text
        self.sections = sections
        self.entities: dict[str, Entity] = {}

    # -- helpers ------------------------------------------------------------------------
    def _scope(self, section: str) -> tuple[str, int]:
        """Return (text to search, offset) - the section if found, else the full text."""
        for s in self.sections:
            if s.name == section and s.method == "heading":
                return self.text[s.start:s.end], s.start
        return self.text, 0

    def _set(self, name: str, value, m: re.Match | None, conf: float, method: str, offset: int = 0,
             group: int | str = 0) -> None:
        if m is None:
            self.entities[name] = Entity(name, value, None, conf, method)
        else:
            self.entities[name] = Entity(name, value, m.group(group), conf, method,
                                         m.start(group) + offset, m.end(group) + offset)

    def _first(self, patterns: list[re.Pattern], scope: str, offset: int):
        for i, pattern in enumerate(patterns):
            m = pattern.search(scope)
            if m:
                return m, i
        return None, -1

    # -- field extractors ------------------------------------------------------------------
    PARTY = re.compile(
        r"(?:[Bb]etween|BETWEEN|\band|\bAND)\s+(?!BETWEEN\b)(?P<name>[A-Z][\w&.,'\s-]{2,90}?)"
        r"(?:,\s+a\s+company[^()]{0,160})?\s?"
        r"\((?:hereinafter\s+(?:referred\s+to\s+as\s+)?)?(?:the\s+)?\"(?P<role>[A-Za-z\s]{3,25})\"\)"
    )

    def parties(self) -> None:
        for m in self.PARTY.finditer(self.text):
            role = re.sub(r"\s+", " ", m.group("role")).strip().lower()
            name = re.sub(r"\s+", " ", m.group("name")).strip(" ,")
            field = "vendor_name" if role in VENDOR_ROLES else "customer_name" if role in CUSTOMER_ROLES else None
            if field and field not in self.entities:
                self.entities[field] = Entity(field, name, m.group(0), 0.9, "regex:party_definition",
                                              m.start("name"), m.end("name"))

    def contract_number(self) -> None:
        m = _p(r"(?:contract (?:no\.?|number)|agreement (?:reference|no\.?|number))\s*[:#]?\s*"
               r"(?P<num>[A-Z0-9][A-Z0-9/-]{3,30}\d)").search(self.text)
        if m:
            self._set("contract_number", m.group("num"), m, 0.95, "regex:label", group="num")

    def dates(self) -> None:
        signing = _p(rf"(?:entered into|made) on (?P<d>{DATE})")
        starts = [_p(rf"commence on (?P<d>{DATE})"), _p(rf"effective date: ?(?P<d>{DATE})"),
                  _p(rf"commencing from (?P<d>{DATE})")]
        ends = [_p(rf"in force until (?P<d>{DATE})"), _p(rf"expiry date: ?(?P<d>{DATE})"),
                _p(rf"(?:expire|expires|ending) on (?P<d>{DATE})")]

        m = signing.search(self.text)
        if m and (d := parse_date(m.group("d"))):
            self._set("signing_date", d, m, 0.9, "regex:signing", group="d")

        term_scope, off = self._scope("term")
        m, _ = self._first(starts, term_scope, off)
        if m and (d := parse_date(m.group("d"))):
            self._set("start_date", d, m, 0.9, "regex:start", off, group="d")
        m, _ = self._first(ends, term_scope, off)
        if m and (d := parse_date(m.group("d"))):
            self._set("end_date", d, m, 0.9, "regex:end", off, group="d")
        elif "start_date" in self.entities:
            tm = _p(r"term of this agreement is (?P<n>\d{1,3}) months").search(term_scope)
            if tm:
                end = self.entities["start_date"].value + relativedelta(months=int(tm.group("n"))) - timedelta(days=1)
                self._set("end_date", end, tm, 0.8, "derived:start+term_months", off, group="n")

    def contract_value(self) -> None:
        scope, off = self._scope("payment")
        anchored = _p(rf"(?:total contract value is|total fee of|shall not exceed|contract value of|"
                      rf"consideration of) (?P<m>{MONEY})")
        m = anchored.search(scope)
        method, conf = "regex:anchored_amount", 0.9
        if not m:
            scope, off = self.text, 0
            candidates = list(_p(MONEY).finditer(scope))
            parsed = [(parse_money(c), c) for c in candidates]
            parsed = [(p, c) for p, c in parsed if p]
            if not parsed:
                return
            (_, _), m = max(parsed, key=lambda pc: pc[0][0])
            method, conf = "heuristic:largest_amount", 0.5
        parsed = parse_money(m)
        if parsed:
            amount, cur = parsed
            self._set("contract_value", amount, m, conf, method, off)
            self._set("currency", cur, m, conf, method, off)

    def payment_days(self) -> None:
        scope, off = self._scope("payment")
        patterns = [_p(r"net (?P<n>\d{1,3}) days"),
                    _p(r"(?:invoices?|payment)[^.]{0,80}?within [^.()]{0,40}?\(?(?P<n>\d{1,3})\)? days")]
        m, _ = self._first(patterns, scope, off)
        if m:
            conf = 0.9 if scope is not self.text else 0.7
            self._set("payment_days", int(m.group("n")), m, conf, "regex:payment_days", off, group="n")

    def termination_notice(self) -> None:
        scope, off = self._scope("termination")
        if scope is self.text and not any(s.name == "termination" for s in self.sections):
            return
        m = _p(r"\(?(?P<n>\d{1,3})\)? days'? (?:prior )?(?:written )?notice").search(scope)
        if m:
            self._set("termination_notice_days", int(m.group("n")), m, 0.9, "regex:notice", off, group="n")

    def penalty(self) -> None:
        scope, off = self._scope("penalties")
        has_section = any(s.name == "penalties" for s in self.sections)
        m = _p(r"(?P<n>\d{1,2}(?:\.\d+)?) ?(?:%|percent) of the (?:total )?contract value").search(scope)
        if m:
            self._set("penalty_pct", float(m.group("n")), m, 0.9, "regex:penalty_pct", off, group="n")
            return
        m = _p(rf"penalty of (?P<m>{MONEY})").search(scope)
        value = self.entities.get("contract_value")
        if m and value and value.value:
            parsed = parse_money(m)
            if parsed:
                self._set("penalty_pct", round(parsed[0] / value.value * 100, 4), m, 0.8,
                          "derived:penalty_amount/contract_value", off)
                return
        if not has_section or _p(r"no (?:liquidated damages|penalt)").search(scope):
            self._set("penalty_pct", 0.0, None, 0.7, "absent:no_penalty_clause")

    def renewal(self) -> None:
        scope, off = self._scope("renewal")
        m = _p(r"automatically renew[^.]{0,60}?(?P<n>\d{1,3}) months").search(scope)
        if m:
            self._set("renewal_months", int(m.group("n")), m, 0.9, "regex:auto_renewal", off, group="n")
        else:
            self._set("renewal_months", 0, None, 0.7, "absent:no_auto_renewal")

    def jurisdiction(self) -> None:
        scope, off = self._scope("jurisdiction")
        law = _p(r"governed by the laws of (?P<law>(?:the )?[A-Z][\w ,.]{2,60}?)(?: and the courts|[.;]| shall)",
                 flags=0).search(scope)
        if law:
            value = re.sub(r"\s+", " ", law.group("law")).strip(" ,.")
            value = "India" if "india" in value.lower() else value
            self._set("governing_law", value, law, 0.9, "regex:governing_law", off, group="law")
        court = _p(r"courts? (?:at|of|in) (?P<c>(?:the )?[A-Z][\w ,.]{2,60}?) shall have", flags=0).search(scope)
        if court:
            self._set("jurisdiction", re.sub(r"\s+", " ", court.group("c")).strip(" ,."), court, 0.9,
                      "regex:courts", off, group="c")

    def liability(self) -> None:
        scope, off = self._scope("liability")
        if not any(s.name == "liability" for s in self.sections):
            self._set("liability_cap", "missing", None, 0.7, "absent:no_liability_clause")
            return
        if _p(r"unlimited liability|liability[^.]{0,60}shall be unlimited").search(scope):
            m = _p(r"unlimited").search(scope)
            self._set("liability_cap", "unlimited", m, 0.9, "regex:unlimited", off)
        elif m := _p(r"shall not exceed|limited to|capped at").search(scope):
            self._set("liability_cap", "capped", m, 0.9, "regex:cap", off)
        else:
            self._set("liability_cap", "unspecified", None, 0.5, "heuristic:no_cap_language")

    def run(self) -> dict[str, Entity]:
        self.parties()
        self.contract_number()
        self.dates()
        self.contract_value()
        self.payment_days()
        self.termination_notice()
        self.penalty()
        self.renewal()
        self.jurisdiction()
        self.liability()
        return self.entities


def extract_entities(text: str, sections: list[Section]) -> dict[str, Entity]:
    return ContractEntityExtractor(text, sections).run()


__all__ = ["Entity", "ENTITY_FIELDS", "extract_entities", "normalize_vendor_key", "parse_date", "section_text"]
