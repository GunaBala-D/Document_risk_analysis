"""Composite risk scoring engine.

    risk_score = 100 * sum(w_i * c_i) / sum(w_i over available components)

Each component c_i is normalised to [0, 1] before weighting:
  ml          expected severity from class probabilities: sum(p_k * severity_k)
  anomaly     rescaled Isolation-Forest percentile, lifted by vendor-behaviour anomalies
  compliance  noisy-OR of rule severities: 1 - prod(1 - s_i)  (bounded, order-independent)
  business    financial exposure (log-scaled value) and expiry urgency

If a component is unavailable (e.g. the ML model failed to load) its weight is
redistributed over the remaining components and the result is marked ``degraded``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from src.compliance.engine import RuleResult
from src.config import load_yaml_config

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class RiskScore:
    risk_score: int
    risk_level: str
    components: dict[str, float | None]
    contributions: dict[str, float]
    weights_used: dict[str, float]
    degraded: bool
    floors_applied: list[str] = field(default_factory=list)
    config_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compliance_score(results: list[RuleResult], cfg: dict) -> float:
    survive = 1.0
    for r in results:
        s = cfg["severity_weights"][r.severity] * (cfg["warning_factor"] if r.type == "warning" else 1.0)
        survive *= 1 - s
    return 1 - survive


def business_score(value_inr: float | None, days_to_expiry: float | None, cfg: dict) -> float:
    exposure = 0.0
    if value_inr and value_inr > 0:
        lo, hi = math.log10(cfg["exposure_floor_inr"]), math.log10(cfg["exposure_cap_inr"])
        exposure = min(1.0, max(0.0, (math.log10(value_inr) - lo) / (hi - lo)))
    urgency = 0.0
    if days_to_expiry is not None and not math.isnan(days_to_expiry):
        u = cfg["urgency"]
        if days_to_expiry < 0:
            urgency = u["expired"]
        elif days_to_expiry <= 45:
            urgency = u["within_45_days"]
        elif days_to_expiry <= 90:
            urgency = u["within_90_days"]
    return cfg["exposure_weight"] * exposure + cfg["urgency_weight"] * urgency


def level_for(score: float, levels: dict[str, int]) -> str:
    out = "LOW"
    for name in LEVELS:
        if score >= levels[name]:
            out = name
    return out


class RiskScorer:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_yaml_config("risk_scoring")
        w = self.cfg["weights"]
        if abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError(f"Risk weights must sum to 1, got {sum(w.values())}")

    def score(self, ml: float | None, anomaly: float | None, compliance: float | None, business: float | None,
              critical_violation: bool = False, ml_class: str | None = None) -> RiskScore:
        comps = {"ml": ml, "anomaly": anomaly, "compliance": compliance, "business": business}
        for k, v in comps.items():
            if v is not None and not 0.0 <= v <= 1.0 + 1e-9:
                raise ValueError(f"Component {k} must be in [0, 1], got {v}")
        available = {k: v for k, v in comps.items() if v is not None}
        if not available:
            raise ValueError("No risk components available")
        w_total = sum(self.cfg["weights"][k] for k in available)
        weights_used = {k: self.cfg["weights"][k] / w_total for k in available}
        contributions = {k: round(100 * weights_used[k] * v, 2) for k, v in available.items()}
        raw = sum(contributions.values())
        score = int(round(min(100.0, max(0.0, raw))))
        level = level_for(score, self.cfg["levels"])

        floors = []
        conditions = {"critical_violation": critical_violation, "ml_predicts_critical": ml_class == "CRITICAL"}
        for rule in self.cfg.get("level_floors", []):
            if conditions.get(rule["when"]) and LEVELS.index(level) < LEVELS.index(rule["min_level"]):
                level = rule["min_level"]
                floors.append(f"{rule['when']} -> at least {rule['min_level']}")

        return RiskScore(score, level, comps, contributions, weights_used, degraded=len(available) < 4,
                         floors_applied=floors, config_version=self.cfg.get("version", ""))
