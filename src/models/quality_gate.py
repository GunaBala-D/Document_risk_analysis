"""Model promotion quality gate.

Decides whether a candidate may replace production. Deliberately not "higher accuracy
wins": the candidate must clear absolute minimums on the metrics that matter for risk
(macro-F1, PR-AUC, high-risk and CRITICAL recall), must not degrade any of them beyond a
tolerance relative to production, and must use the serving feature schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.config import load_yaml_config


@dataclass
class GateCheck:
    name: str
    passed: bool
    candidate: float | str | None
    threshold: float | str | None
    detail: str


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    policy_version: str = ""

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "policy_version": self.policy_version,
                "checks": [asdict(c) for c in self.checks]}


def evaluate_gate(candidate: dict, production: dict | None, candidate_features: list[str],
                  serving_features: list[str], policy: dict | None = None) -> GateResult:
    """``candidate``/``production`` are test-set metric dicts from ``compute_metrics``."""
    policy = policy or load_yaml_config("quality_gate")
    checks: list[GateCheck] = []

    if policy.get("require_same_feature_schema", True):
        same = list(candidate_features) == list(serving_features)
        diff = sorted(set(candidate_features) ^ set(serving_features))
        checks.append(GateCheck("feature_schema", same, "match" if same else "mismatch", "match",
                                "Feature list identical to serving registry" if same else f"Differs: {diff}"))

    for metric, minimum in policy["absolute"].items():
        value = candidate.get(metric)
        ok = value is not None and value >= minimum
        checks.append(GateCheck(f"min_{metric}", ok, value, minimum, f"{metric} >= {minimum}"))

    if production:
        for metric, tolerance in policy["relative"].items():
            cand, prod = candidate.get(metric), production.get(metric)
            if prod is None:
                continue
            floor = prod - tolerance
            ok = cand is not None and cand >= floor
            checks.append(GateCheck(
                f"no_regression_{metric}", ok, cand, round(floor, 4),
                f"{metric} must not drop more than {tolerance} below production ({prod:.4f})",
            ))

    return GateResult(all(c.passed for c in checks), checks, policy.get("version", ""))
