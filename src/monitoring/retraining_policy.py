"""Retraining trigger policy.

Drift alone never triggers retraining: retraining on unlabeled drifted data teaches the
model nothing new, and drift is often harmless. Retraining needs *new validated labels*,
and is justified when the data moved significantly or measured performance dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.config import load_yaml_config


@dataclass
class RetrainingDecision:
    retrain: bool
    reasons: list[str]
    new_validated_labels: int
    dataset_drift: bool | None
    performance_degraded: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def decide_retraining(new_validated_labels: int, dataset_drift: bool | None, performance_degraded: bool | None,
                      cfg: dict | None = None) -> RetrainingDecision:
    cfg = cfg or load_yaml_config("monitoring")["retraining"]
    reasons = []
    enough = new_validated_labels >= cfg["min_new_validated_labels"]
    reasons.append(f"{new_validated_labels} new validated labels (threshold {cfg['min_new_validated_labels']})")
    if dataset_drift:
        reasons.append("significant data drift detected")
    if performance_degraded:
        reasons.append("model performance on validated feedback has degraded")

    retrain = (enough and bool(dataset_drift or performance_degraded)) or (
        bool(performance_degraded) and new_validated_labels >= cfg["min_labels_for_degradation_only"])
    if not retrain:
        if not enough and not performance_degraded:
            reasons.append("not enough new validated labels -> no retraining")
        elif enough and not (dataset_drift or performance_degraded):
            reasons.append("labels available but no drift or degradation -> retraining not justified")
    return RetrainingDecision(retrain, reasons, new_validated_labels, dataset_drift, performance_degraded)
