"""Human-feedback validation.

A reviewer's decision is NOT automatically a training label. Feedback enters the
validated dataset only after a *different* user with the validate permission (risk
manager) approves it (four-eyes principle). Automatic consistency checks raise flags the
validator must look at - they do not silently drop feedback.
"""

from __future__ import annotations

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def consistency_flags(reviewer_label: str, model_level: str, ml_class: str | None, decision: str,
                      violation_severities: list[str], extraction_method: str | None) -> list[str]:
    flags = []
    if decision == "confirm" and reviewer_label != model_level:
        flags.append("confirm_decision_but_label_differs")
    if "CRITICAL" in violation_severities and LEVELS.index(reviewer_label) < LEVELS.index("HIGH"):
        flags.append("label_below_HIGH_despite_CRITICAL_violation")
    if abs(LEVELS.index(reviewer_label) - LEVELS.index(model_level)) >= 2:
        flags.append("large_disagreement_with_system_(2+_levels)")
    if ml_class and abs(LEVELS.index(reviewer_label) - LEVELS.index(ml_class)) >= 2:
        flags.append("large_disagreement_with_model_(2+_levels)")
    if extraction_method in ("ocr", "mixed"):
        flags.append("ocr_document_verify_extracted_values")
    return flags


def can_validate(reviewer_id: int, validator_id: int) -> bool:
    return reviewer_id != validator_id
