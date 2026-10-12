"""Train, evaluate and register the anomaly detector (Isolation Forest).

Evaluation uses ground truth that the detector never sees: the generator's injected
"unusual" scenarios (unusual value, excessive payment terms, high penalty, long renewal,
contract splitting). Anomaly detection is unsupervised, so this measures whether the
score separates genuinely unusual contracts on the hold-out split.

Usage:
    python -m training.train_anomaly
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, date, datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.anomaly.detector import (
    AnomalyDetector,
    anomaly_frame,
    build_isolation_forest,
    calibration_quantiles,
    vendor_behavior_anomalies,
)
from src.config import ARTIFACTS_DIR, DATA_DIR, get_settings, load_yaml_config
from src.features.builder import VendorContract
from src.models.registry import (
    STAGING,
    git_commit,
    log_sklearn_model,
    promote_to_production,
    set_alias,
    setup_mlflow,
)
from training.common import frozen_split, load_dataset

ANOMALY_SCENARIOS = {"unusual_value", "excessive_payment", "high_penalty", "long_renewal", "contract_splitting"}


def main() -> None:
    cfg = load_yaml_config("anomaly")
    tcfg = load_yaml_config("training")
    settings = get_settings()
    df, X, y, meta = load_dataset()
    tr, te = frozen_split(df, y, meta["dataset_version"], tcfg["test_size"], tcfg["seed"])
    feats = cfg["isolation_forest"]["features"]

    gt = pd.read_csv(DATA_DIR / "synthetic" / "ground_truth.csv").set_index("doc_id")
    scen = gt.loc[df["doc_id"], "scenarios"].fillna("").str.split("|")
    truth = scen.apply(lambda s: bool(ANOMALY_SCENARIOS & set(s))).to_numpy()
    truth_value = scen.apply(lambda s: "unusual_value" in s).to_numpy()
    truth_split = scen.apply(lambda s: "contract_splitting" in s).to_numpy()
    truth_pay = scen.apply(lambda s: "excessive_payment" in s).to_numpy()

    pipe = build_isolation_forest(cfg)
    A_tr = anomaly_frame(X.iloc[tr])[feats]
    pipe.fit(A_tr)
    raw_tr = -pipe.score_samples(A_tr)
    detector = AnomalyDetector(pipe, feats, calibration_quantiles(raw_tr), cfg["isolation_forest"]["flag_percentile"],
                               config=cfg)

    score_te = detector.score(X.iloc[te])
    flag_te = score_te >= detector.flag_percentile
    metrics = {
        "if_roc_auc": roc_auc_score(truth[te], score_te),
        "if_pr_auc": average_precision_score(truth[te], score_te),
        "if_flag_precision": precision_score(truth[te], flag_te, zero_division=0),
        "if_flag_recall": recall_score(truth[te], flag_te, zero_division=0),
        "if_flag_rate": float(flag_te.mean()),
        "anomalous_share_test": float(truth[te].mean()),
    }

    # Vendor value-deviation check on the test split (point-in-time vendor history)
    vh = pd.read_parquet(DATA_DIR / "processed" / "vendor_history.parquet")
    history = defaultdict(list)
    for r in vh.itertuples():
        history[r.vendor_key].append(VendorContract(pd.Timestamp(r.contract_date).date(), r.value_inr,
                                                    r.payment_days, r.violation_count, r.high_risk))
    value_flags, freq_flags, pay_flags = [], [], []
    for i in te:
        row = df.iloc[i]
        value_inr = 10 ** row.contract_value_log10 if not np.isnan(row.contract_value_log10) else None
        f = vendor_behavior_anomalies(history.get(row.vendor_key, []), date.fromisoformat(row.submission_date),
                                      value_inr, row.payment_days, cfg["vendor_behavior"])
        flagged = {x.anomaly_type for x in f if x.is_anomaly}
        value_flags.append("vendor_value_deviation" in flagged)
        freq_flags.append("vendor_contract_frequency" in flagged)
        pay_flags.append("vendor_payment_terms_deviation" in flagged)
    value_flags, freq_flags, pay_flags = np.array(value_flags), np.array(freq_flags), np.array(pay_flags)
    metrics.update({
        "vendor_value_check_precision": precision_score(truth_value[te], value_flags, zero_division=0),
        "vendor_value_check_recall": recall_score(truth_value[te], value_flags, zero_division=0),
        "unusual_value_share_test": float(truth_value[te].mean()),
        "vendor_value_check_flag_rate": float(value_flags.mean()),
        "frequency_check_precision": precision_score(truth_split[te], freq_flags, zero_division=0),
        "frequency_check_recall": recall_score(truth_split[te], freq_flags, zero_division=0),
        "frequency_check_flag_rate": float(freq_flags.mean()),
        "payment_check_precision": precision_score(truth_pay[te], pay_flags, zero_division=0),
        "payment_check_recall": recall_score(truth_pay[te], pay_flags, zero_division=0),
        "payment_check_flag_rate": float(pay_flags.mean()),
    })
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))

    metadata = {
        "model_name": settings.anomaly_model_name, "features": feats, "quantiles": detector.quantiles,
        "flag_percentile": detector.flag_percentile, "config_version": cfg["version"],
        "dataset_version": meta["dataset_version"], "feature_version": meta["feature_version"],
        "test_metrics": metrics, "trained_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "family": "isolation_forest", "synthetic_data": True,
    }
    setup_mlflow(settings)
    with mlflow.start_run(run_name=f"anomaly-{datetime.now(UTC):%Y%m%d-%H%M%S}"):
        mlflow.set_tags({"pipeline": "anomaly", "dataset_version": meta["dataset_version"],
                         "feature_version": meta["feature_version"]})
        mlflow.log_params({k: v for k, v in cfg["isolation_forest"].items() if k != "features"})
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_metrics(metrics)
        version = log_sklearn_model(pipe, settings.anomaly_model_name, metadata, input_example=A_tr.head(3))
        set_alias(settings.anomaly_model_name, STAGING, version)
        passed = metrics["if_roc_auc"] >= cfg["quality_gate"]["min_roc_auc"]
        mlflow.set_tag("quality_gate", "PASS" if passed else "FAIL")
        if passed:
            promote_to_production(settings.anomaly_model_name, version)
        print(f"Anomaly detector v{version}: gate {'PASS -> @production' if passed else 'FAIL'}")

    out = ARTIFACTS_DIR / "reports" / "anomaly_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": version, "promoted": passed, "metrics": metrics,
                               "gate_min_roc_auc": cfg["quality_gate"]["min_roc_auc"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
