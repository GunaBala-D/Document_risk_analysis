"""Model monitoring: four distinct questions, four distinct checks.

  Data drift         P(X) changed       PSI + KS test (numeric) / chi-square (binary) per feature
  Prediction drift   P(y_hat) changed   PSI of predicted-class distribution vs training reference
  Concept drift /    P(y|X) changed     only observable with labels: macro-F1 of the model on
  performance                           *validated* reviewer labels vs its test macro-F1
  Service health     the API itself     latency percentiles, error rate, request volume

Data drift alone does not mean the model is wrong (it may generalise fine), and concept
drift can happen with no data drift at all - which is why retraining is triggered by a
combination of signals (see retraining_policy.py), not by drift alone.

Implemented directly with numpy/scipy (instead of Evidently) so the statistics are
transparent, unit-tested and stored in our own database for the dashboard.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from src.config import load_yaml_config

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference distribution."""
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, EPS, None), np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(ref_dist: dict[str, float], cur_dist: dict[str, float]) -> float:
    keys = set(ref_dist) | set(cur_dist)
    total = 0.0
    for k in keys:
        r, c = max(ref_dist.get(k, 0.0), EPS), max(cur_dist.get(k, 0.0), EPS)
        total += (c - r) * math.log(c / r)
    return float(total)


def _is_binary(s: pd.Series) -> bool:
    return set(pd.unique(s.dropna())).issubset({0.0, 1.0})


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], cfg: dict | None = None) -> dict:
    cfg = cfg or load_yaml_config("monitoring")["drift"]
    if len(current) < cfg["min_samples"]:
        return {"status": "insufficient_data", "n_current": len(current), "min_samples": cfg["min_samples"],
                "features": [], "drift_share": None, "dataset_drift": None}
    rows = []
    for f in features:
        ref, cur = reference[f].astype(float), current[f].astype(float)
        p = psi(ref.to_numpy(), cur.to_numpy(), cfg["psi_bins"])
        if _is_binary(ref) and _is_binary(cur):
            table = np.array([[(ref == 0).sum(), (ref == 1).sum()], [(cur == 0).sum(), (cur == 1).sum()]]) + 0.5
            pval = float(stats.chi2_contingency(table)[1])
            test = "chi_square"
        else:
            r, c = ref.dropna(), cur.dropna()
            pval = float(stats.ks_2samp(r, c).pvalue) if len(r) and len(c) else float("nan")
            test = "ks"
        drifted = bool((not math.isnan(p) and p >= cfg["psi_drift"]) and (math.isnan(pval) or pval < cfg["p_value"]))
        rows.append({
            "feature": f, "psi": None if math.isnan(p) else round(p, 4), "test": test,
            "p_value": None if math.isnan(pval) else round(pval, 6), "drifted": drifted,
            "severity": "drift" if drifted else "warn" if (not math.isnan(p) and p >= cfg["psi_warn"]) else "ok",
            "reference_mean": None if ref.dropna().empty else round(float(ref.mean()), 4),
            "current_mean": None if cur.dropna().empty else round(float(cur.mean()), 4),
            "current_missing_share": round(float(cur.isna().mean()), 4),
        })
    share = sum(r["drifted"] for r in rows) / len(rows)
    return {"status": "ok", "n_reference": len(reference), "n_current": len(current),
            "features": sorted(rows, key=lambda r: -(r["psi"] or 0)), "drift_share": round(share, 4),
            "dataset_drift": share >= cfg["dataset_drift_share"]}


def prediction_drift(reference_dist: dict[str, float], current_classes: list[str], cfg: dict | None = None) -> dict:
    cfg = cfg or load_yaml_config("monitoring")["drift"]
    if len(current_classes) < cfg["min_samples"]:
        return {"status": "insufficient_data", "n_current": len(current_classes)}
    cur = pd.Series(current_classes).value_counts(normalize=True).to_dict()
    p = categorical_psi(reference_dist, cur)
    return {"status": "ok", "psi": round(p, 4), "drifted": p >= cfg["psi_drift"],
            "reference_distribution": {k: round(reference_dist.get(k, 0.0), 4) for k in LEVELS},
            "current_distribution": {k: round(cur.get(k, 0.0), 4) for k in LEVELS},
            "n_current": len(current_classes)}


def performance_on_feedback(model_classes: list[str], reviewer_labels: list[str], test_f1_macro: float | None,
                            cfg: dict | None = None) -> dict:
    cfg = cfg or load_yaml_config("monitoring")["performance"]
    n = len(reviewer_labels)
    if n < cfg["min_labels"]:
        return {"status": "insufficient_labels", "n_labels": n, "min_labels": cfg["min_labels"]}
    f1 = float(f1_score(reviewer_labels, model_classes, labels=LEVELS, average="macro", zero_division=0))
    acc = float(np.mean(np.array(model_classes) == np.array(reviewer_labels)))
    hr_true = np.isin(reviewer_labels, ["HIGH", "CRITICAL"])
    hr_pred = np.isin(model_classes, ["HIGH", "CRITICAL"])
    hr_recall = float((hr_true & hr_pred).sum() / hr_true.sum()) if hr_true.sum() else None
    degraded = test_f1_macro is not None and f1 < test_f1_macro - cfg["degradation_tolerance"]
    return {"status": "ok", "n_labels": n, "f1_macro": round(f1, 4), "agreement": round(acc, 4),
            "high_risk_recall": None if hr_recall is None else round(hr_recall, 4),
            "baseline_test_f1_macro": test_f1_macro, "degraded": bool(degraded)}


def service_health(metrics: pd.DataFrame, cfg: dict | None = None) -> dict:
    """``metrics`` columns: route, status_code, latency_ms, ts."""
    cfg = cfg or load_yaml_config("monitoring")["service"]
    if metrics.empty:
        return {"status": "no_traffic", "requests": 0}
    err = metrics["status_code"] >= 500
    by_route = []
    for route, g in metrics.groupby("route"):
        by_route.append({"route": route, "requests": len(g), "p50_ms": round(float(g.latency_ms.quantile(0.5)), 1),
                         "p95_ms": round(float(g.latency_ms.quantile(0.95)), 1),
                         "error_rate": round(float((g.status_code >= 500).mean()), 4),
                         "client_error_rate": round(float(((g.status_code >= 400) & (g.status_code < 500)).mean()), 4)})
    p95 = float(metrics.latency_ms.quantile(0.95))
    error_rate = float(err.mean())
    hourly = (metrics.assign(hour=pd.to_datetime(metrics.ts, utc=True).dt.floor("h"))
              .groupby("hour").agg(requests=("route", "size"), p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
                                   errors=("status_code", lambda s: int((s >= 500).sum())))
              .reset_index())
    return {
        "status": "ok" if p95 <= cfg["latency_p95_slo_ms"] and error_rate <= cfg["error_rate_slo"] else "degraded",
        "requests": len(metrics), "error_rate": round(error_rate, 4),
        "p50_ms": round(float(metrics.latency_ms.quantile(0.5)), 1), "p95_ms": round(p95, 1),
        "slo": {"latency_p95_ms": cfg["latency_p95_slo_ms"], "error_rate": cfg["error_rate_slo"]},
        "by_route": sorted(by_route, key=lambda r: -r["requests"]),
        "hourly": [{"hour": h.isoformat(), "requests": int(r), "p95_ms": round(float(p), 1), "errors": int(e)}
                   for h, r, p, e in hourly.itertuples(index=False)],
    }
