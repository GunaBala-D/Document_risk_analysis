"""Anomaly detection - kept separate from risk classification.

Classification answers "does this look like contracts we labelled risky?". Anomaly
detection answers "is this unusual compared with what we normally see?" - it needs no
labels and can surface new patterns the labelling policy never anticipated.

Two complementary detectors:
  1. Multivariate Isolation Forest over contract + behaviour features. The raw
     ``score_samples`` output is converted to a 0..1 score via its percentile among the
     training contracts, so 0.98 means "more isolated than 98% of training contracts".
     Per-feature attributions come from SHAP's TreeExplainer on the forest.
  2. Vendor-behaviour checks vs the vendor's own history (robust z-score of contract
     value using median/MAD, payment-term increase, contract-frequency bursts). These give
     precise, explainable statements such as "6.1x the vendor's median value".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import load_yaml_config
from src.features.builder import VendorContract


@dataclass
class AnomalyFinding:
    anomaly_type: str
    score: float
    is_anomaly: bool
    description: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def anomaly_frame(features: pd.DataFrame | dict) -> pd.DataFrame:
    df = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
    ratio = df["value_vs_vendor_avg"].astype(float)
    df["value_vs_vendor_avg_log2"] = np.where(ratio > 0, np.log2(ratio.where(ratio > 0, 1.0)), 0.0)
    df["value_vs_vendor_avg_log2"] = df["value_vs_vendor_avg_log2"].fillna(0.0)
    return df


def build_isolation_forest(cfg: dict | None = None) -> Pipeline:
    cfg = (cfg or load_yaml_config("anomaly"))["isolation_forest"]
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("iforest", IsolationForest(n_estimators=cfg["n_estimators"], max_samples=cfg["max_samples"],
                                    contamination=cfg["contamination"], random_state=cfg["random_state"])),
    ])


def calibration_quantiles(raw_scores: np.ndarray, n: int = 1001) -> list[float]:
    """Quantiles of the training 'abnormality' (= -score_samples) used for percentile scoring."""
    return np.quantile(raw_scores, np.linspace(0, 1, n)).tolist()


def _fmt_inr(v: float) -> str:
    if v >= 1e7:
        return f"INR {v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"INR {v / 1e5:.1f} L"
    return f"INR {v:,.0f}"


@dataclass
class AnomalyDetector:
    pipeline: Pipeline
    features: list[str]
    quantiles: list[float]
    flag_percentile: float
    model_version: str = "local"
    config: dict = field(default_factory=dict)
    _explainer: object = field(default=None, repr=False)

    def raw_abnormality(self, X: pd.DataFrame) -> np.ndarray:
        return -self.pipeline.score_samples(anomaly_frame(X)[self.features])

    def percentile(self, raw: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.quantiles), raw, side="right") / len(self.quantiles)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.percentile(self.raw_abnormality(X))

    def _attributions(self, X_row: pd.DataFrame, top_k: int = 3) -> list[dict]:
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.pipeline.named_steps["iforest"])
            imputed = self.pipeline.named_steps["impute"].transform(X_row[self.features])
            sv = np.asarray(self._explainer.shap_values(imputed))[0]
        except Exception:  # attribution is a nice-to-have; never fail the analysis for it
            return []
        # IsolationForest SHAP explains the decision score: negative = pushes towards anomalous.
        order = np.argsort(sv)[:top_k]
        return [{"feature": self.features[j], "value": float(imputed[0, j]), "contribution": float(-sv[j])}
                for j in order if sv[j] < 0]

    def detect(self, features: dict, vendor_history: list[VendorContract], as_of: date,
               value_inr: float | None) -> list[AnomalyFinding]:
        X = anomaly_frame(features)
        pct = float(self.score(X)[0])
        flagged = pct >= self.flag_percentile
        findings = [AnomalyFinding(
            anomaly_type="multivariate_isolation_forest", score=round(pct, 4), is_anomaly=flagged,
            description=(f"Contract profile is more unusual than {pct:.0%} of historical contracts."
                         if flagged else f"Contract profile is within the normal range (percentile {pct:.0%})."),
            details={"percentile": pct, "threshold": self.flag_percentile,
                     "top_contributors": self._attributions(X) if flagged else []},
        )]
        findings.extend(vendor_behavior_anomalies(vendor_history, as_of, value_inr, features.get("payment_days"),
                                                  self.config.get("vendor_behavior") or
                                                  load_yaml_config("anomaly")["vendor_behavior"]))
        return findings


def vendor_behavior_anomalies(history: list[VendorContract], as_of: date, value_inr: float | None,
                              payment_days: float | None, cfg: dict) -> list[AnomalyFinding]:
    prior = [h for h in history if h.contract_date < as_of]
    findings: list[AnomalyFinding] = []

    window_start = as_of - timedelta(days=90)
    recent = [h for h in prior if h.contract_date >= window_start]
    older = [h for h in prior if h.contract_date < window_start]
    # The vendor's own normal rate: busy vendors routinely sign several contracts a quarter,
    # so "3 in 90 days" is only unusual relative to that vendor's history.
    normal_rate = None
    if len(older) >= cfg.get("burst_baseline_min_contracts", 3):
        span_days = max(90, (window_start - min(h.contract_date for h in older)).days)
        normal_rate = len(older) / (span_days / 90)
    is_burst = len(recent) >= cfg["burst_contracts_90d"] and (
        normal_rate is None or len(recent) >= cfg.get("burst_rate_multiple", 2.5) * normal_rate)
    if is_burst:
        baseline = f"vs a normal {normal_rate:.1f} per 90 days" if normal_rate is not None else "with no longer history"
        findings.append(AnomalyFinding(
            "vendor_contract_frequency", 1.0, True,
            f"{len(recent)} earlier contracts with this vendor in the last 90 days {baseline} - unusual "
            f"frequency (possible contract splitting).",
            {"contracts_last_90d": len(recent), "normal_rate_per_90d": normal_rate,
             "threshold": cfg["burst_contracts_90d"]},
        ))

    values = [h.value_inr for h in prior if h.value_inr]
    if value_inr and len(values) >= cfg["min_history"]:
        logs = np.log10(values)
        med = float(np.median(logs))
        mad = max(float(np.median(np.abs(logs - med))), cfg["mad_floor_log10"])
        z = 0.6745 * (math.log10(value_inr) - med) / mad
        ratio = value_inr / 10 ** med
        is_anom = abs(z) >= cfg["value_robust_z"]
        findings.append(AnomalyFinding(
            "vendor_value_deviation", round(min(1.0, abs(z) / (2 * cfg["value_robust_z"])), 4), is_anom,
            (f"Contract value {_fmt_inr(value_inr)} is {ratio:.1f}x this vendor's historical median "
             f"{_fmt_inr(10 ** med)} (robust z = {z:.1f}; history range {_fmt_inr(min(values))}-"
             f"{_fmt_inr(max(values))})." if is_anom else
             f"Contract value is consistent with the vendor's history (robust z = {z:.1f})."),
            {"robust_z": z, "ratio_to_median": ratio, "vendor_median_inr": 10 ** med,
             "history_min_inr": min(values), "history_max_inr": max(values), "n_history": len(values)},
        ))

    pays = [h.payment_days for h in prior if h.payment_days is not None]
    if payment_days is not None and not math.isnan(payment_days) and len(pays) >= cfg["min_history"]:
        med_pay, max_pay = float(np.median(pays)), float(max(pays))
        if payment_days - med_pay >= cfg["payment_days_increase"] and payment_days > max_pay:
            findings.append(AnomalyFinding(
                "vendor_payment_terms_deviation", 1.0, True,
                f"Payment period of {payment_days:.0f} days is {payment_days - med_pay:.0f} days longer than this "
                f"vendor's usual {med_pay:.0f} days and longer than any earlier contract with them "
                f"(max {max_pay:.0f}).",
                {"payment_days": payment_days, "vendor_median_payment_days": med_pay, "vendor_max_payment_days": max_pay},
            ))
    return findings


def anomaly_component_score(findings: list[AnomalyFinding]) -> float:
    """0..1 anomaly component for the risk score.

    The Isolation-Forest percentile is rescaled so a median contract contributes 0 and the
    most isolated contract contributes 1; a flagged vendor-behaviour anomaly lifts it to >= 0.9.
    """
    scores = [f.score for f in findings if f.anomaly_type == "multivariate_isolation_forest"]
    base = max(0.0, (scores[0] - 0.5) * 2) if scores else 0.0
    if any(f.is_anomaly and f.anomaly_type != "multivariate_isolation_forest" for f in findings):
        base = max(base, 0.9)
    return float(base)
