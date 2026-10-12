"""Deterministic compliance rule engine.

Policy requirements are evaluated by explicit rules rather than learned by the model:
  * a policy is a hard requirement - it must fire 100% of the time, not "usually",
  * policies change by decision (new limit next quarter) - editing YAML is instant,
    retraining a model to learn a new threshold is slow and unreliable,
  * every violation must be traceable to a written rule for audit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from string import Formatter

from src.config import load_yaml_config

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "missing", "present"}


class RuleConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    description: str
    severity: str
    type: str
    condition: dict
    message: str
    remediation: str
    enabled: bool = True
    supersedes: tuple[str, ...] = ()


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    severity: str
    type: str
    description: str
    message: str
    remediation: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _validate_condition(cond: dict, rule_id: str) -> None:
    if not isinstance(cond, dict) or len(cond) == 0:
        raise RuleConfigError(f"{rule_id}: condition must be a non-empty mapping")
    if "all" in cond or "any" in cond:
        for sub in cond.get("all", cond.get("any")):
            _validate_condition(sub, rule_id)
    elif "not" in cond:
        _validate_condition(cond["not"], rule_id)
    else:
        if "fact" not in cond or cond.get("op") not in _OPS:
            raise RuleConfigError(f"{rule_id}: leaf condition needs 'fact' and a valid 'op' ({sorted(_OPS)})")


def _facts_in(cond: dict) -> set[str]:
    if "all" in cond or "any" in cond:
        return set().union(*(_facts_in(c) for c in cond.get("all", cond.get("any"))))
    if "not" in cond:
        return _facts_in(cond["not"])
    return {cond["fact"]}


def evaluate_condition(cond: dict, facts: dict) -> bool:
    if "all" in cond:
        return all(evaluate_condition(c, facts) for c in cond["all"])
    if "any" in cond:
        return any(evaluate_condition(c, facts) for c in cond["any"])
    if "not" in cond:
        return not evaluate_condition(cond["not"], facts)

    op, target = cond["op"], cond.get("value")
    value = facts.get(cond["fact"])
    if op == "missing":
        return _is_missing(value)
    if op == "present":
        return not _is_missing(value)
    if _is_missing(value):
        return False  # unknown facts never trigger a violation; COMPLIANCE-015 flags missing data
    if isinstance(value, bool) or isinstance(target, bool):
        value, target = (bool(value), bool(target)) if op in {"eq", "ne"} else (value, target)
    if op == "eq":
        return value == target
    if op == "ne":
        return value != target
    if op == "gt":
        return value > target
    if op == "gte":
        return value >= target
    if op == "lt":
        return value < target
    if op == "lte":
        return value <= target
    if op == "between":
        lo, hi = target
        return lo <= value <= hi
    if op == "in":
        return value in target
    if op == "not_in":
        return value not in target
    raise RuleConfigError(f"Unknown op {op}")


class _SafeFormatter(Formatter):
    def format_field(self, value, format_spec):
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            return str(value)

    def get_value(self, key, args, kwargs):
        value = kwargs.get(key)
        return "unknown" if _is_missing(value) else value


class ComplianceEngine:
    def __init__(self, config: dict | None = None):
        config = config or load_yaml_config("compliance_rules")
        self.version: str = config["version"]
        self.rules: list[Rule] = []
        seen: set[str] = set()
        for raw in config["rules"]:
            rule = Rule(**{**raw, "supersedes": tuple(raw.get("supersedes", ()))})
            if rule.id in seen:
                raise RuleConfigError(f"Duplicate rule id {rule.id}")
            if rule.severity not in SEVERITY_ORDER:
                raise RuleConfigError(f"{rule.id}: invalid severity {rule.severity}")
            if rule.type not in {"violation", "warning"}:
                raise RuleConfigError(f"{rule.id}: type must be violation or warning")
            _validate_condition(rule.condition, rule.id)
            seen.add(rule.id)
            self.rules.append(rule)

    @property
    def enabled_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.enabled]

    def evaluate(self, facts: dict) -> list[RuleResult]:
        fired = [r for r in self.enabled_rules if evaluate_condition(r.condition, facts)]
        hidden = {sid for r in fired for sid in r.supersedes}
        results = []
        for r in fired:
            if r.id in hidden:
                continue
            evidence = {k: facts.get(k) for k in sorted(_facts_in(r.condition))}
            evidence = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in evidence.items()}
            results.append(RuleResult(
                rule_id=r.id, rule_name=r.name, severity=r.severity, type=r.type,
                description=r.description, message=_SafeFormatter().format(r.message, **facts),
                remediation=r.remediation, evidence=evidence,
            ))
        return sorted(results, key=lambda x: (-SEVERITY_ORDER[x.severity], x.rule_id))


def count_policy_violations(results: list[RuleResult], min_severity: str = "MEDIUM") -> int:
    floor = SEVERITY_ORDER[min_severity]
    return sum(1 for r in results if r.type == "violation" and SEVERITY_ORDER[r.severity] >= floor)
