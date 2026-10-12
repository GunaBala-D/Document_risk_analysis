"""Document type detection (Vendor Contract / Invoice / Purchase Order / Unknown).

A transparent cue-scoring classifier: each type has weighted lexical cues; the type with
the highest score wins if it clears a minimum score and margin. For V1 this is preferable
to a learned classifier: we have no labelled invoices/POs, the cues are stable, and the
decision is fully explainable ("matched: 'governing law', 'termination', ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CUES: dict[str, list[tuple[str, float]]] = {
    "vendor_contract": [
        (r"\bagreement\b", 1.0), (r"\bhereinafter\b", 1.5), (r"\bgoverning law\b|\bgoverned by the laws\b", 2.0),
        (r"\bterminat(e|ion)\b", 1.0), (r"\bparties\b|\beither party\b", 1.5), (r"\bliabilit(y|ies)\b", 1.0),
        (r"\bconfidential", 1.0), (r"\bin witness whereof\b", 2.0), (r"\beffective date\b", 1.0),
    ],
    "invoice": [
        (r"\b(tax )?invoice\b", 1.5), (r"\binvoice (no|number|date)\b", 2.5), (r"\bbill to\b", 2.0),
        (r"\bamount due\b|\btotal amount due\b", 2.5), (r"\bgstin\b", 1.5), (r"\b[cs]gst\b|\bigst\b", 1.5),
    ],
    "purchase_order": [
        (r"\bpurchase order\b", 3.0), (r"\bpo (no|number|date)\b", 2.5), (r"\bship to\b", 1.5),
        (r"\bunit price\b", 1.5), (r"\bdelivery date\b", 1.5), (r"\bquantity\b", 1.0),
    ],
}
MIN_SCORE = 3.0
MIN_MARGIN = 1.5
SUPPORTED_TYPES = {"vendor_contract"}
PHASE_MESSAGES = {
    "invoice": "Invoice processing is planned for Phase 2. The document was stored but not risk-scored.",
    "purchase_order": "Purchase order processing is planned for Phase 2. The document was stored but not risk-scored.",
    "unknown": "The document could not be identified as a vendor contract, so it was not risk-scored.",
}


@dataclass(frozen=True)
class DocTypeResult:
    doc_type: str
    confidence: float
    scores: dict[str, float]
    matched_cues: list[str]

    @property
    def supported(self) -> bool:
        return self.doc_type in SUPPORTED_TYPES


def detect_document_type(text: str) -> DocTypeResult:
    lower = text.lower()
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for doc_type, cues in CUES.items():
        total, hits = 0.0, []
        for pattern, weight in cues:
            m = re.search(pattern, lower)
            if m:
                total += weight
                hits.append(m.group(0))
        scores[doc_type] = round(total, 2)
        matched[doc_type] = hits

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score < MIN_SCORE or best_score - second_score < MIN_MARGIN:
        return DocTypeResult("unknown", 0.0, scores, [])
    max_possible = sum(w for _, w in CUES[best])
    confidence = round(min(1.0, best_score / max_possible) * (1 - second_score / max(best_score, 1e-9) * 0.5), 3)
    return DocTypeResult(best, confidence, scores, matched[best])
