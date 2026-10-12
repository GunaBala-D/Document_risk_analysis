"""Rule-based risk labeling policy (see configs/labeling_policy.yaml).

Applied to *ground-truth* contract attributes in time order, so vendor-history factors
(prior violations, contract splitting) only use contracts that existed before the one
being labeled. Every label carries the list of factors that produced it, which makes the
labels auditable and lets a human reviewer validate a sample.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from src.config import load_yaml_config
from src.synthetic.generator import ContractSpec

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _has_policy_violation(spec: ContractSpec, th: dict, approved: list[str]) -> bool:
    """A MEDIUM-or-higher policy violation - same definition as the compliance engine
    (configs/compliance_rules.yaml, type=violation, severity >= MEDIUM)."""
    return (
        not spec.has_termination
        or not spec.has_liability
        or not spec.has_confidentiality
        or spec.unlimited_liability
        or spec.payment_days > th["payment_days_policy"]
        or spec.penalty_pct > th["penalty_pct_high"]
        or spec.governing_law not in approved
        or spec.renewal_months > th["renewal_months_max"]
    )


def points_to_level(points: int, levels: dict[str, int]) -> str:
    label = "LOW"
    for name in RISK_LEVELS:
        if points >= levels[name]:
            label = name
    return label


def label_contracts(specs: list[ContractSpec], policy: dict | None = None) -> list[dict]:
    """Return one label record per spec: {doc_id, risk_label, risk_points, label_reasons}."""
    policy = policy or load_yaml_config("labeling_policy")
    th, pts, approved = policy["thresholds"], policy["points"], policy["approved_jurisdictions"]
    limit = th["approval_limit_inr"]
    lo, hi = th["splitting_value_band"]

    history: dict[str, list[ContractSpec]] = defaultdict(list)
    out = []
    for spec in sorted(specs, key=lambda s: (s.submission_date, s.doc_id)):
        prior = history[spec.vendor_id]
        reasons: list[str] = []

        if not spec.has_termination:
            reasons.append("missing_termination_clause")
        if not spec.has_liability:
            reasons.append("missing_liability_clause")
        if spec.unlimited_liability:
            reasons.append("unlimited_liability")
        if not spec.has_confidentiality:
            reasons.append("missing_confidentiality_clause")
        if spec.payment_days > th["payment_days_excessive"]:
            reasons.append("payment_days_excessive")
        elif spec.payment_days > th["payment_days_policy"]:
            reasons.append("payment_days_over_policy")
        if spec.penalty_pct > th["penalty_pct_excessive"]:
            reasons.append("penalty_excessive")
        elif spec.penalty_pct > th["penalty_pct_high"]:
            reasons.append("penalty_high")
        if spec.contract_value_inr > th["value_vs_vendor_typical"] * spec.vendor_typical_value_inr:
            reasons.append("value_unusual_for_vendor")
        if spec.contract_value_inr > limit:
            reasons.append("value_over_approval_limit")
        days_to_expiry = (spec.end_date - spec.submission_date).days
        if 0 <= days_to_expiry <= th["expiry_warning_days"]:
            reasons.append("expiry_within_warning")
        if spec.renewal_months > th["renewal_months_max"]:
            reasons.append("long_auto_renewal")
        if spec.governing_law not in approved:
            reasons.append("non_approved_jurisdiction")

        if len(prior) >= th["vendor_history_min_contracts"]:
            rate = sum(_has_policy_violation(p, th, approved) for p in prior) / len(prior)
            if rate >= th["vendor_violation_rate_many"]:
                reasons.append("vendor_prior_violations_many")
            elif rate >= th["vendor_violation_rate_some"]:
                reasons.append("vendor_prior_violations_some")

        window_start = spec.submission_date - timedelta(days=th["splitting_window_days"])
        in_band = lambda s: lo * limit <= s.contract_value_inr <= hi * limit  # noqa: E731
        recent_in_band = [p for p in prior if p.submission_date >= window_start and in_band(p)]
        if in_band(spec) and len(recent_in_band) >= 2:
            reasons.append("possible_contract_splitting")

        if len(spec.risk_keywords) >= th["one_sided_terms_min_keywords"]:
            reasons.append("one_sided_terms")

        points = sum(pts[r] for r in reasons)
        out.append({
            "doc_id": spec.doc_id,
            "risk_label": points_to_level(points, policy["levels"]),
            "risk_points": points,
            "label_reasons": "|".join(reasons),
            "label_policy_version": policy["version"],
        })
        history[spec.vendor_id].append(spec)
    return out
