"""Feature engineering (feature version declared in configs/features.yaml).

The same functions are used for training (training/build_dataset.py) and serving
(src/pipeline/analyzer.py) so there is no train/serve skew in feature logic.

Vendor features are computed *point-in-time*: only contracts dated strictly before the
analysis date are visible. Using the full history would leak the future into training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.config import load_yaml_config
from src.extraction.entities import Entity
from src.preprocessing.sections import Section, missing_mandatory

NAN = float("nan")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    group: str
    type: str
    source: str
    definition: str
    transformation: str
    value_format: str = ""  # presence | inr_log10 | percent | ratio | pct_value | delta_days | "" (by type)


def format_feature_value(spec: FeatureSpec, value: float | None) -> str:
    """Human-readable value, so a factor can never be misread (e.g. 'Termination clause: missing')."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    fmt = spec.value_format
    if fmt == "presence":
        return "present" if value >= 0.5 else "missing"
    if fmt == "inr_log10":
        inr = 10 ** value
        return f"INR {inr / 1e7:.2f} Cr" if inr >= 1e7 else f"INR {inr / 1e5:.1f} L" if inr >= 1e5 else f"INR {inr:,.0f}"
    if fmt == "percent":
        return f"{value * 100:.0f}%"
    if fmt == "ratio":
        return f"{value:.1f}x"
    if fmt == "delta_days":
        return f"{value:+.0f} days"
    if fmt == "pct_value":
        return f"{value:g}% of contract value"
    if spec.type == "binary":
        return "yes" if value >= 0.5 else "no"
    return f"{round(value, 1):g}"


@dataclass(frozen=True)
class FeatureRegistry:
    version: str
    specs: tuple[FeatureSpec, ...]
    config: dict = field(compare=False, hash=False, repr=False)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def labels(self) -> dict[str, str]:
        return {s.name: s.label for s in self.specs}

    def describe(self, feature: str, value: float | None) -> str:
        spec = next((s for s in self.specs if s.name == feature), None)
        if spec is None:
            return f"{feature}: {value}"
        return f"{spec.label}: {format_feature_value(spec, value)}"


def load_registry() -> FeatureRegistry:
    cfg = load_yaml_config("features")
    return FeatureRegistry(cfg["version"], tuple(FeatureSpec(**f) for f in cfg["features"]), cfg)


@dataclass
class VendorContract:
    """One historical contract of a vendor (as known at the time)."""

    contract_date: date
    value_inr: float | None
    payment_days: float | None
    violation_count: int
    high_risk: bool | None


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else NAN


def _ent(entities: dict[str, Entity], name: str):
    e = entities.get(name)
    return e.value if e is not None else None


def contract_value_inr(entities: dict[str, Entity], fx: dict[str, float]) -> float | None:
    value, cur = _ent(entities, "contract_value"), _ent(entities, "currency") or "INR"
    if value is None:
        return None
    return float(value) * fx.get(cur, 1.0)


def vendor_features(history: list[VendorContract], as_of: date, value_inr: float | None,
                    payment_days: float | None) -> dict[str, float]:
    prior = [h for h in history if h.contract_date < as_of]
    avg_value = _mean([h.value_inr for h in prior])
    avg_pay = _mean([h.payment_days for h in prior])
    known_risk = [h.high_risk for h in prior if h.high_risk is not None]
    window = as_of - timedelta(days=90)
    n_violating = sum(1 for h in prior if h.violation_count > 0)
    return {
        "vendor_prior_contracts": float(len(prior)),
        "vendor_prior_avg_value_log10": math.log10(avg_value) if avg_value and avg_value > 0 else NAN,
        "vendor_prior_violation_count": float(n_violating),
        "vendor_prior_violation_rate": n_violating / len(prior) if prior else NAN,
        "vendor_prior_high_risk_ratio": (sum(known_risk) / len(known_risk)) if known_risk else NAN,
        "value_vs_vendor_avg": (value_inr / avg_value) if value_inr and avg_value and avg_value > 0 else NAN,
        "payment_days_vs_vendor_avg": (payment_days - avg_pay)
        if payment_days is not None and not math.isnan(avg_pay) else NAN,
        "vendor_contracts_last_90d": float(sum(1 for h in prior if h.contract_date >= window)),
    }


def _keyword_count(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def build_features(
    text: str,
    sections: list[Section],
    entities: dict[str, Entity],
    as_of: date,
    vendor_history: list[VendorContract],
    registry: FeatureRegistry | None = None,
) -> dict[str, float]:
    """Return {feature_name: value} in registry order. Missing values are NaN."""
    registry = registry or load_registry()
    cfg = registry.config
    value_inr = contract_value_inr(entities, cfg["fx_to_inr"])
    start, end = _ent(entities, "start_date"), _ent(entities, "end_date")
    payment = _ent(entities, "payment_days")
    present = {s.name for s in sections}
    governing_law = _ent(entities, "governing_law")
    liability_cap = _ent(entities, "liability_cap")

    feats: dict[str, float] = {
        "contract_value_log10": math.log10(value_inr) if value_inr and value_inr > 0 else NAN,
        "contract_duration_months": ((end - start).days / 30.44) if start and end else NAN,
        "payment_days": float(payment) if payment is not None else NAN,
        "termination_notice_days": float(_ent(entities, "termination_notice_days"))
        if _ent(entities, "termination_notice_days") is not None else NAN,
        "renewal_months": float(_ent(entities, "renewal_months") or 0),
        "penalty_pct": float(_ent(entities, "penalty_pct")) if _ent(entities, "penalty_pct") is not None else NAN,
        "days_to_expiry": float((end - as_of).days) if end else NAN,
        "non_approved_jurisdiction": float(
            governing_law is not None and governing_law not in cfg["approved_jurisdictions"]
        ),
        **vendor_features(vendor_history, as_of, value_inr, float(payment) if payment is not None else None),
        "missing_clause_count": float(len(missing_mandatory(sections))),
        "has_termination_clause": float("termination" in present),
        "has_liability_clause": float("liability" in present),
        "has_confidentiality_clause": float("confidentiality" in present),
        "unlimited_liability": float(liability_cap == "unlimited"),
        "risk_keyword_count": float(_keyword_count(text, cfg["risk_keywords"])),
        "compliance_keyword_count": float(_keyword_count(text, cfg["compliance_keywords"])),
    }
    lim = cfg["policy_limits"]
    pay, pen, dte = feats["payment_days"], feats["penalty_pct"], feats["days_to_expiry"]
    feats.update({
        "payment_above_policy": float(not math.isnan(pay) and pay > lim["payment_days"]),
        "payment_above_hard_limit": float(not math.isnan(pay) and pay > lim["payment_days_hard"]),
        "penalty_above_policy": float(not math.isnan(pen) and pen > lim["penalty_pct"]),
        "expiry_within_warning": float(not math.isnan(dte) and 0 <= dte <= lim["expiry_warning_days"]),
        "value_above_approval_limit": float(value_inr is not None and value_inr > lim["approval_limit_inr"]),
        "renewal_above_max": float(feats["renewal_months"] > lim["renewal_months"]),
    })
    missing = set(registry.names) - set(feats)
    if missing:
        raise RuntimeError(f"Feature builder does not produce registered features: {sorted(missing)}")
    return {name: feats[name] for name in registry.names}


def build_facts(features: dict[str, float], entities: dict[str, Entity], sections: list[Section],
                fx: dict[str, float] | None = None) -> dict:
    """Facts consumed by the compliance rule engine (features + raw entity values)."""
    fx = fx or load_registry().config["fx_to_inr"]
    present = {s.name for s in sections}
    facts = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}
    facts.update({
        "contract_value_inr": contract_value_inr(entities, fx),
        "governing_law": _ent(entities, "governing_law"),
        "liability_cap": _ent(entities, "liability_cap"),
        "has_compliance_clause": "compliance" in present,
        "has_payment_clause": "payment" in present,
        "has_jurisdiction_clause": "jurisdiction" in present,
        "contract_number": _ent(entities, "contract_number"),
        "end_date": _ent(entities, "end_date"),
        "vendor_name": _ent(entities, "vendor_name"),
    })
    return facts
