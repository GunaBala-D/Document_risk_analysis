"""Label-observability diagnostic.

For each labeling-policy factor, check whether it can be reproduced from the *extracted*
model features. Factors that cannot be reproduced bound the achievable model quality:
no classifier can learn information the features do not contain. Used to decide between
"tune the model harder" and "fix the features".

Usage:
    python -m training.diagnose_labels
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, load_yaml_config
from src.features.builder import load_registry


def observed_factors(r: pd.Series, th: dict) -> dict[str, bool]:
    def gt(v, x):
        return bool(not np.isnan(v) and v > x)

    rate, n_prior = r.vendor_prior_violation_rate, r.vendor_prior_contracts
    has_hist = n_prior >= th["vendor_history_min_contracts"] and not np.isnan(rate)
    return {
        "missing_termination_clause": r.has_termination_clause == 0,
        "missing_liability_clause": r.has_liability_clause == 0,
        "unlimited_liability": r.unlimited_liability == 1,
        "missing_confidentiality_clause": r.has_confidentiality_clause == 0,
        "payment_days_excessive": gt(r.payment_days, th["payment_days_excessive"]),
        "payment_days_over_policy": gt(r.payment_days, th["payment_days_policy"])
        and not gt(r.payment_days, th["payment_days_excessive"]),
        "penalty_excessive": gt(r.penalty_pct, th["penalty_pct_excessive"]),
        "penalty_high": gt(r.penalty_pct, th["penalty_pct_high"]) and not gt(r.penalty_pct, th["penalty_pct_excessive"]),
        "value_over_approval_limit": gt(r.contract_value_log10, np.log10(th["approval_limit_inr"])),
        "expiry_within_warning": bool(0 <= r.days_to_expiry <= th["expiry_warning_days"]),
        "long_auto_renewal": gt(r.renewal_months, th["renewal_months_max"]),
        "non_approved_jurisdiction": r.non_approved_jurisdiction == 1,
        "one_sided_terms": r.risk_keyword_count >= th["one_sided_terms_min_keywords"],
        "value_unusual_for_vendor": gt(r.value_vs_vendor_avg, th["value_vs_vendor_typical"]),
        "vendor_prior_violations_many": has_hist and rate >= th["vendor_violation_rate_many"],
        "vendor_prior_violations_some": has_hist and th["vendor_violation_rate_some"] <= rate
        < th["vendor_violation_rate_many"],
    }


def main() -> None:
    policy = load_yaml_config("labeling_policy")
    th = policy["thresholds"]
    ds = pd.read_parquet(DATA_DIR / "processed" / f"dataset_{load_registry().version}.parquet")
    reasons = ds["label_reasons"].fillna("").str.split("|")
    obs = pd.DataFrame([observed_factors(r, th) for _, r in ds.iterrows()])

    rows = []
    for f in policy["points"]:
        truth = reasons.apply(lambda rs, f=f: f in rs)
        if f not in obs:
            rows.append({"factor": f, "fired": int(truth.sum()), "agreement": None,
                         "missed": int(truth.sum()), "spurious": 0, "observable": False})
            continue
        pred = obs[f].astype(bool)
        rows.append({"factor": f, "fired": int(truth.sum()), "agreement": round(float((pred == truth).mean()), 4),
                     "missed": int((truth & ~pred).sum()), "spurious": int((~truth & pred).sum()),
                     "observable": True})

    # Reconstruct the label from observable factors only.
    pts = policy["points"]
    obs_points = obs.apply(lambda r: sum(pts[k] for k, v in r.items() if v), axis=1)
    levels = policy["levels"]
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    obs_label = obs_points.apply(lambda p: [c for c in order if p >= levels[c]][-1])
    agreement = float((obs_label == ds["risk_label"]).mean())

    report = {"factors": rows, "label_reconstruction_agreement": round(agreement, 4),
              "confusion": pd.crosstab(ds["risk_label"], obs_label).reindex(index=order, columns=order).fillna(0)
              .astype(int).to_dict()}
    out = ARTIFACTS_DIR / "reports" / "label_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nLabel reconstructed from extracted features alone agrees with the true label: {agreement:.4f}")


if __name__ == "__main__":
    main()
