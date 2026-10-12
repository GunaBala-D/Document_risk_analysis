"""Feature calculations, compliance rules, risk scoring, anomaly checks, quality gate,
drift statistics, retraining policy and feedback validation."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.anomaly.detector import AnomalyFinding, anomaly_component_score, vendor_behavior_anomalies
from src.compliance.engine import ComplianceEngine, RuleConfigError, RuleResult, evaluate_condition
from src.config import load_yaml_config
from src.extraction.entities import extract_entities
from src.features.builder import VendorContract, build_facts, build_features, load_registry, vendor_features
from src.feedback.validation import can_validate, consistency_flags
from src.models.quality_gate import evaluate_gate
from src.monitoring.drift import feature_drift, performance_on_feedback, prediction_drift, psi
from src.monitoring.retraining_policy import decide_retraining
from src.preprocessing.cleaning import clean_text
from src.preprocessing.sections import detect_sections
from src.scoring.risk_scorer import RiskScorer, business_score, compliance_score
from tests.conftest import make_contract

AS_OF = date(2025, 6, 1)


def _features(forced=None, history=None, seed=3):
    spec, text = make_contract(seed=seed, forced=forced, submission=AS_OF)
    clean = clean_text(text)
    sections = detect_sections(clean)
    ents = extract_entities(clean, sections)
    feats = build_features(clean, sections, ents, AS_OF, history or [])
    return spec, feats, build_facts(feats, ents, sections)


# ---------------------------------------------------------------- features
def test_feature_vector_matches_registry_order_and_values():
    spec, feats, _ = _features()
    assert list(feats) == load_registry().names
    assert feats["payment_days"] == spec.payment_days
    assert feats["renewal_months"] == spec.renewal_months
    assert feats["contract_value_log10"] == pytest.approx(math.log10(spec.contract_value_inr), abs=1e-3)
    assert feats["days_to_expiry"] == (spec.end_date - AS_OF).days
    assert feats["vendor_prior_contracts"] == 0 and math.isnan(feats["vendor_prior_avg_value_log10"])


def test_missing_clause_features():
    _, feats, _ = _features(forced=["missing_termination", "missing_confidentiality"])
    assert feats["has_termination_clause"] == 0 and feats["has_confidentiality_clause"] == 0
    assert feats["missing_clause_count"] == 2
    assert math.isnan(feats["termination_notice_days"])


def test_vendor_features_are_point_in_time():
    hist = [VendorContract(date(2025, 1, 1), 2_000_000, 30, 0, False),
            VendorContract(date(2025, 3, 15), 2_200_000, 45, 1, True),  # 78 days before AS_OF
            VendorContract(date(2025, 7, 1), 99_000_000, 120, 3, True)]  # in the future -> must be ignored
    f = vendor_features(hist, AS_OF, value_inr=12_000_000, payment_days=90)
    assert f["vendor_prior_contracts"] == 2
    assert f["vendor_prior_violation_count"] == 1 and f["vendor_prior_violation_rate"] == 0.5
    assert f["value_vs_vendor_avg"] == pytest.approx(12_000_000 / 2_100_000)
    assert f["payment_days_vs_vendor_avg"] == pytest.approx(90 - 37.5)
    assert f["vendor_contracts_last_90d"] == 1


def test_policy_indicator_features():
    _, feats, _ = _features(forced=["excessive_payment", "high_penalty", "long_renewal"])
    assert feats["payment_above_policy"] == 1 and feats["penalty_above_policy"] == 1 and feats["renewal_above_max"] == 1


# ---------------------------------------------------------------- compliance
def test_compliance_rules_fire_on_violations():
    _, _, facts = _features(forced=["missing_termination", "unlimited_liability", "foreign_jurisdiction"])
    ids = {r.rule_id for r in ComplianceEngine().evaluate(facts)}
    assert {"COMPLIANCE-001", "COMPLIANCE-007", "COMPLIANCE-010"} <= ids


def test_clean_contract_has_no_violations():
    _, _, facts = _features(seed=11)
    results = ComplianceEngine().evaluate(facts)
    assert not [r for r in results if r.type == "violation" and r.severity in ("HIGH", "CRITICAL")]


def test_supersedes_prevents_double_counting():
    facts = {"payment_days": 120}
    ids = [r.rule_id for r in ComplianceEngine().evaluate(facts)]
    assert "COMPLIANCE-003" in ids and "COMPLIANCE-002" not in ids


def test_expiry_warning_and_message_formatting():
    results = ComplianceEngine().evaluate({"days_to_expiry": 20.0})
    r = next(r for r in results if r.rule_id == "COMPLIANCE-004")
    assert r.type == "warning" and "20 days" in r.message


def test_missing_facts_do_not_trigger_rules_but_flag_data_quality():
    ids = {r.rule_id for r in ComplianceEngine().evaluate({"payment_days": None})}
    assert "COMPLIANCE-002" not in ids and "COMPLIANCE-015" in ids


def test_disabled_rule_and_invalid_config():
    cfg = load_yaml_config("compliance_rules")
    cfg = {**cfg, "rules": [{**r, "enabled": r["id"] != "COMPLIANCE-001"} for r in cfg["rules"]]}
    assert "COMPLIANCE-001" not in {r.rule_id for r in ComplianceEngine(cfg).evaluate({"has_termination_clause": 0})}
    bad = {"version": "x", "rules": [{**cfg["rules"][0], "severity": "EXTREME"}]}
    with pytest.raises(RuleConfigError):
        ComplianceEngine(bad)


def test_condition_operators():
    assert evaluate_condition({"all": [{"fact": "a", "op": "gt", "value": 1}, {"fact": "b", "op": "in", "value": [1, 2]}]},
                              {"a": 2, "b": 2})
    assert evaluate_condition({"not": {"fact": "a", "op": "between", "value": [0, 5]}}, {"a": 9})
    assert evaluate_condition({"fact": "x", "op": "missing"}, {})


# ---------------------------------------------------------------- scoring
def _rr(sev, typ="violation"):
    return RuleResult("R", "r", sev, typ, "", "", "")


def test_compliance_score_noisy_or():
    cfg = load_yaml_config("risk_scoring")["compliance"]
    assert compliance_score([], cfg) == 0
    one = compliance_score([_rr("HIGH")], cfg)
    two = compliance_score([_rr("HIGH"), _rr("HIGH")], cfg)
    assert one == pytest.approx(0.5) and 0.5 < two < 1 and two == pytest.approx(0.75)
    assert compliance_score([_rr("HIGH", "warning")], cfg) == pytest.approx(0.25)


def test_business_score_bounds():
    cfg = load_yaml_config("risk_scoring")["business"]
    assert business_score(None, None, cfg) == 0
    assert business_score(1e9, 10, cfg) == pytest.approx(1.0)
    assert 0 < business_score(2_000_000, 400, cfg) < 0.6


def test_risk_score_weights_levels_and_range():
    s = RiskScorer()
    low = s.score(0.0, 0.0, 0.0, 0.0)
    high = s.score(1.0, 1.0, 1.0, 1.0)
    assert low.risk_score == 0 and low.risk_level == "LOW"
    assert high.risk_score == 100 and high.risk_level == "CRITICAL"
    mid = s.score(0.5, 0.5, 0.5, 0.5)
    assert mid.risk_score == 50 and mid.risk_level == "MEDIUM"
    assert sum(mid.contributions.values()) == pytest.approx(50)


def test_level_thresholds_are_inclusive_and_configurable():
    s = RiskScorer()
    assert s.score(0.61, 0.61, 0.61, 0.61).risk_level == "HIGH"
    assert s.score(0.30, 0.30, 0.30, 0.30).risk_level == "LOW"
    assert s.score(0.31, 0.31, 0.31, 0.31).risk_level == "MEDIUM"
    cfg = {**load_yaml_config("risk_scoring"), "levels": {"LOW": 0, "MEDIUM": 10, "HIGH": 20, "CRITICAL": 30}}
    assert RiskScorer(cfg).score(0.35, 0.35, 0.35, 0.35).risk_level == "CRITICAL"


def test_missing_component_redistributes_weight_and_marks_degraded():
    r = RiskScorer().score(None, 0.4, 0.4, 0.4)
    assert r.degraded and "ml" not in r.weights_used and sum(r.weights_used.values()) == pytest.approx(1)
    assert r.risk_score == 40


def test_critical_violation_floor():
    r = RiskScorer().score(0.0, 0.0, 0.8, 0.0, critical_violation=True)
    assert r.risk_level == "HIGH" and r.floors_applied


def test_invalid_component_rejected():
    with pytest.raises(ValueError):
        RiskScorer().score(1.5, 0, 0, 0)


# ---------------------------------------------------------------- anomaly
def test_vendor_value_deviation_example_from_brief():
    """Vendor history INR 18L-25L, new contract INR 1.2 Cr -> anomaly."""
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    hist = [VendorContract(date(2024, m, 1), v, 30, 0, False)
            for m, v in zip(range(1, 7), [1.8e6, 2.0e6, 2.2e6, 2.5e6, 1.9e6, 2.1e6], strict=True)]
    found = vendor_behavior_anomalies(hist, AS_OF, 12_000_000, 30, cfg)
    dev = next(f for f in found if f.anomaly_type == "vendor_value_deviation")
    assert dev.is_anomaly and dev.details["ratio_to_median"] > 5
    normal = vendor_behavior_anomalies(hist, AS_OF, 2_100_000, 30, cfg)
    assert not next(f for f in normal if f.anomaly_type == "vendor_value_deviation").is_anomaly


def test_frequency_burst_and_payment_deviation():
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    hist = [VendorContract(date(2025, 5, d), 9e6, 30, 0, False) for d in (2, 10, 20)]
    types = {f.anomaly_type for f in vendor_behavior_anomalies(hist, AS_OF, 9e6, 90, cfg) if f.is_anomaly}
    assert {"vendor_contract_frequency", "vendor_payment_terms_deviation"} <= types


def test_busy_vendor_normal_cadence_is_not_a_burst():
    """A vendor that always signs ~3 contracts a quarter is not anomalous at 3 in 90 days."""
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    hist = [VendorContract(date(2024, 6, 1) + pd.Timedelta(days=30 * k).to_pytimedelta(), 5e6, 45, 0, False)
            for k in range(12)]  # one contract every 30 days for a year
    flagged = {f.anomaly_type for f in vendor_behavior_anomalies(hist, date(2025, 5, 20), 5e6, 45, cfg) if f.is_anomaly}
    assert "vendor_contract_frequency" not in flagged


def test_burst_relative_to_vendor_rate_is_flagged():
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    older = [VendorContract(date(2023, 1, 1) + pd.Timedelta(days=180 * k).to_pytimedelta(), 5e6, 45, 0, False)
             for k in range(4)]  # ~0.5 contracts per 90 days
    burst = [VendorContract(date(2025, 5, d), 9.5e6, 45, 0, False) for d in (1, 8, 15)]
    found = [f for f in vendor_behavior_anomalies(older + burst, AS_OF, 9.5e6, 45, cfg) if f.is_anomaly]
    freq = next(f for f in found if f.anomaly_type == "vendor_contract_frequency")
    assert freq.details["normal_rate_per_90d"] < 1


def test_payment_deviation_requires_exceeding_vendor_maximum():
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    hist = [VendorContract(date(2024, m, 1), 5e6, p, 0, False) for m, p in zip(range(1, 6), [30, 30, 30, 30, 90], strict=True)]
    flagged = {f.anomaly_type for f in vendor_behavior_anomalies(hist, AS_OF, 5e6, 75, cfg) if f.is_anomaly}
    assert "vendor_payment_terms_deviation" not in flagged   # 75 days was exceeded before (90)


def test_no_history_means_no_vendor_anomalies():
    cfg = load_yaml_config("anomaly")["vendor_behavior"]
    assert vendor_behavior_anomalies([], AS_OF, 1e7, 30, cfg) == []


def test_anomaly_component_scaling():
    iso = lambda p, a=False: AnomalyFinding("multivariate_isolation_forest", p, a, "")  # noqa: E731
    assert anomaly_component_score([iso(0.5)]) == 0
    assert anomaly_component_score([iso(1.0, True)]) == pytest.approx(1.0)
    assert anomaly_component_score([iso(0.6), AnomalyFinding("vendor_value_deviation", 1, True, "")]) == 0.9


# ---------------------------------------------------------------- quality gate
GOOD = {"f1_macro": 0.83, "pr_auc_macro": 0.9, "high_risk_recall": 0.94, "critical_recall": 0.75,
        "escalation_recall": 0.97}


def test_quality_gate_passes_good_candidate():
    names = load_registry().names
    assert evaluate_gate(GOOD, None, names, names).passed


def test_quality_gate_rejects_regression_and_schema_change():
    names = load_registry().names
    worse = {**GOOD, "f1_macro": 0.80}
    r = evaluate_gate(worse, GOOD, names, names)
    assert not r.passed and [c.name for c in r.failures] == ["no_regression_f1_macro"]
    r2 = evaluate_gate(GOOD, GOOD, names[:-1], names)
    assert not r2.passed and r2.failures[0].name == "feature_schema"


def test_high_recall_alone_does_not_pass_gate():
    """A model that flags everything has high recall but must still fail."""
    names = load_registry().names
    flag_all = {**GOOD, "f1_macro": 0.22, "pr_auc_macro": 0.59, "high_risk_recall": 0.99}
    assert not evaluate_gate(flag_all, None, names, names).passed


# ---------------------------------------------------------------- monitoring
def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.5


def test_feature_drift_detects_shift_and_requires_min_samples():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.integers(0, 2, 2000).astype(float)})
    cur = pd.DataFrame({"a": rng.normal(2, 1, 200), "b": rng.integers(0, 2, 200).astype(float)})
    rep = feature_drift(ref, cur, ["a", "b"])
    by = {f["feature"]: f for f in rep["features"]}
    assert by["a"]["drifted"] and not by["b"]["drifted"] and rep["drift_share"] == 0.5
    assert feature_drift(ref, cur.head(10), ["a"])["status"] == "insufficient_data"


def test_prediction_drift_and_performance():
    ref = {"LOW": 0.5, "MEDIUM": 0.3, "HIGH": 0.15, "CRITICAL": 0.05}
    assert prediction_drift(ref, ["CRITICAL"] * 40)["drifted"]
    assert not prediction_drift(ref, ["LOW"] * 20 + ["MEDIUM"] * 12 + ["HIGH"] * 6 + ["CRITICAL"] * 2)["drifted"]
    perf = performance_on_feedback(["LOW"] * 30, ["HIGH"] * 30, 0.8)
    assert perf["degraded"] and perf["f1_macro"] == 0
    assert performance_on_feedback(["LOW"], ["LOW"], 0.8)["status"] == "insufficient_labels"


@pytest.mark.parametrize("labels,drift,degraded,expected", [
    (10, True, False, False),     # drift without labels never triggers retraining
    (60, False, False, False),    # labels but nothing changed
    (60, True, False, True),      # labels + drift
    (60, False, True, True),      # labels + degradation
    (35, False, True, True),      # degradation with enough labels for degradation-only path
    (10, False, True, False),
])
def test_retraining_policy(labels, drift, degraded, expected):
    assert decide_retraining(labels, drift, degraded).retrain is expected


# ---------------------------------------------------------------- feedback validation
def test_feedback_consistency_flags_and_four_eyes():
    flags = consistency_flags("LOW", "HIGH", "CRITICAL", "correct", ["CRITICAL"], "ocr")
    assert "label_below_HIGH_despite_CRITICAL_violation" in flags
    assert any("large_disagreement_with_model" in f for f in flags)
    assert "ocr_document_verify_extracted_values" in flags
    assert consistency_flags("HIGH", "HIGH", "HIGH", "confirm", [], "native") == []
    assert not can_validate(5, 5) and can_validate(5, 6)
