"""Validate the composite risk-score weights against project data.

The brief's starting weights (ML 40 / anomaly 25 / compliance 25 / business 10) are an
assumption. This script computes the four normalised components for every contract in the
frozen TEST split (production models from the registry, point-in-time vendor history) and
measures, for the default weights, single-component baselines and a full weight grid
(step 0.05 on the simplex):

  * Spearman rank correlation of the 0-100 score with the labeling-policy risk points
  * ROC-AUC of the score for identifying HIGH+CRITICAL contracts
  * macro-F1 of the resulting LOW/MEDIUM/HIGH/CRITICAL level (configured thresholds + floors)

It REPORTS; it does not change configs/risk_scoring.yaml - weights are a policy decision,
and choosing them on the test set would overfit it. Caveat: labels come from a policy that
shares factors with the compliance rules, so the compliance component is favoured by
construction on this synthetic data.

Usage:
    python -m training.validate_weights
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from src.anomaly.detector import (
    AnomalyDetector,
    AnomalyFinding,
    anomaly_component_score,
    vendor_behavior_anomalies,
)
from src.compliance.engine import ComplianceEngine, RuleResult
from src.config import ARTIFACTS_DIR, DATA_DIR, get_settings, load_yaml_config
from src.features.builder import VendorContract
from src.models.evaluation import CLASSES
from src.models.registry import load_model_by_alias
from src.models.risk_model import CLASS_SEVERITY
from src.scoring.risk_scorer import business_score, compliance_score
from training.common import frozen_split, load_dataset

COMPONENTS = ["ml", "anomaly", "compliance", "business"]


def components_for_test_set() -> pd.DataFrame:
    settings = get_settings()
    tcfg = load_yaml_config("training")
    scfg = load_yaml_config("risk_scoring")
    acfg = load_yaml_config("anomaly")
    df, X, y, meta = load_dataset()
    _, te = frozen_split(df, y, meta["dataset_version"], tcfg["test_size"], tcfg["seed"])
    d, Xt = df.iloc[te].reset_index(drop=True), X.iloc[te].reset_index(drop=True)

    clf, cmeta, cver = load_model_by_alias(settings.classifier_model_name, settings=settings)
    proba = clf.predict_proba(Xt[cmeta["feature_names"]])
    pipe, ameta, aver = load_model_by_alias(settings.anomaly_model_name, settings=settings)
    det = AnomalyDetector(pipe, ameta["features"], ameta["quantiles"], float(ameta["flag_percentile"]), config=acfg)
    iso = det.score(Xt)

    vh = pd.read_parquet(DATA_DIR / "processed" / "vendor_history.parquet")
    history = defaultdict(list)
    for r in vh.itertuples():
        history[r.vendor_key].append(VendorContract(pd.Timestamp(r.contract_date).date(), r.value_inr, r.payment_days,
                                                    r.violation_count, r.high_risk))
    rules = {r.id: r for r in ComplianceEngine().rules}

    rows = []
    for i, r in d.iterrows():
        value_inr = 10 ** r.contract_value_log10 if not np.isnan(r.contract_value_log10) else None
        findings = [AnomalyFinding("multivariate_isolation_forest", float(iso[i]), iso[i] >= det.flag_percentile, "")]
        findings += vendor_behavior_anomalies(history.get(r.vendor_key, []), date.fromisoformat(r.submission_date),
                                              value_inr, r.payment_days, acfg["vendor_behavior"])
        results = [RuleResult(rid, rules[rid].name, rules[rid].severity, rules[rid].type, "", "", "")
                   for rid in str(r.compliance_violations or "").split("|") if rid in rules]
        rows.append({
            "ml": float(proba[i] @ CLASS_SEVERITY),
            "anomaly": anomaly_component_score(findings),
            "compliance": compliance_score(results, scfg["compliance"]),
            "business": business_score(value_inr, r.days_to_expiry, scfg["business"]),
            "critical_violation": any(x.severity == "CRITICAL" and x.type == "violation" for x in results),
            "ml_critical": CLASSES[int(np.argmax(proba[i] * np.asarray(cmeta.get("decision_weights") or [1, 1, 1, 1])))]
            == "CRITICAL",
            "label": r.risk_label, "points": r.risk_points,
        })
    out = pd.DataFrame(rows)
    out.attrs.update({"classifier_version": cver, "anomaly_version": aver})
    return out


def evaluate(comp: pd.DataFrame, w: dict[str, float], levels: dict[str, int]) -> dict:
    score = np.clip(np.round(100 * sum(w[k] * comp[k] for k in COMPONENTS)), 0, 100)
    lvl = np.select([score >= levels["CRITICAL"], score >= levels["HIGH"], score >= levels["MEDIUM"]],
                    [3, 2, 1], default=0)
    lvl = np.where((comp.critical_violation | comp.ml_critical) & (lvl < 2), 2, lvl)   # configured floors
    y = comp.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return {
        "spearman_vs_points": float(spearmanr(score, comp.points).statistic),
        "roc_auc_high_risk": float(roc_auc_score(y >= 2, score)),
        "level_f1_macro": float(f1_score(y, lvl, average="macro")),
        "level_distribution": {CLASSES[k]: int((lvl == k).sum()) for k in range(4)},
    }


def main() -> None:
    cfg = load_yaml_config("risk_scoring")
    comp = components_for_test_set()
    default = cfg["weights"]
    result = {"n_test": len(comp), "classifier_version": comp.attrs["classifier_version"],
              "anomaly_version": comp.attrs["anomaly_version"],
              "component_correlation_with_points": {k: float(spearmanr(comp[k], comp.points).statistic)
                                                    for k in COMPONENTS},
              "default": {"weights": default, **evaluate(comp, default, cfg["levels"])},
              "single_component": {k: evaluate(comp, {c: float(c == k) for c in COMPONENTS}, cfg["levels"])
                                   for k in COMPONENTS}}
    grid = []
    steps = np.round(np.arange(0, 1.0001, 0.05), 2)
    for a, b, c in itertools.product(steps, steps, steps):
        d = round(1 - a - b - c, 2)
        if d < -1e-9:
            continue
        w = {"ml": float(a), "anomaly": float(b), "compliance": float(c), "business": max(0.0, float(d))}
        grid.append({"weights": w, **evaluate(comp, w, cfg["levels"])})
    g = pd.DataFrame(grid)
    for metric in ("spearman_vs_points", "roc_auc_high_risk", "level_f1_macro"):
        best = g.loc[g[metric].idxmax()]
        result[f"best_by_{metric}"] = {"weights": best.weights, metric: float(best[metric]),
                                       "default_rank_percentile": float((g[metric] < result["default"][metric]).mean())}
    result["n_grid"] = len(g)
    out = ARTIFACTS_DIR / "reports" / "weight_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Test contracts: {len(comp)} (classifier v{result['classifier_version']}, anomaly v{result['anomaly_version']})")
    print("Component Spearman vs policy points:", {k: round(v, 3) for k, v in result["component_correlation_with_points"].items()})
    dflt = result["default"]
    print(f"DEFAULT {default}: spearman={dflt['spearman_vs_points']:.3f} auc={dflt['roc_auc_high_risk']:.3f} "
          f"level_f1={dflt['level_f1_macro']:.3f} levels={dflt['level_distribution']}")
    for k, v in result["single_component"].items():
        print(f"  only {k:10s}: spearman={v['spearman_vs_points']:.3f} auc={v['roc_auc_high_risk']:.3f} "
              f"level_f1={v['level_f1_macro']:.3f}")
    for metric in ("spearman_vs_points", "roc_auc_high_risk", "level_f1_macro"):
        b = result[f"best_by_{metric}"]
        print(f"BEST by {metric}: {b['weights']} -> {b[metric]:.3f} (default beats {b['default_rank_percentile']:.0%} of grid)")


if __name__ == "__main__":
    main()
