"""Feedback-driven retraining job.

    New validated feedback -> trigger policy -> candidate training -> evaluation ->
    comparison with production -> quality gate -> MLflow registry (@production or rejected)

* Runs only if the retraining policy says so (enough NEW validated labels AND drift or
  measured degradation), unless --force is given.
* Training data = original training split + validated reviewer feedback (features as stored
  at analysis time, label = reviewer's validated label). Unvalidated feedback is never used.
* The candidate keeps the production model family/hyper-parameters (a refit, not a new model
  search) and re-tunes decision thresholds on out-of-fold predictions.
* Evaluated on the SAME frozen test set as production, so the gate compares like with like.
  It is also scored on a held-out slice of recent validated feedback (reported, not gated):
  under concept drift the frozen test set reflects the old concept - see docs.
* --simulate-bad-candidate corrupts training labels on purpose to demonstrate that the
  quality gate rejects a poor model. Such runs are tagged simulation=true.

Usage:
    python -m training.retrain [--force] [--simulate-bad-candidate]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sqlalchemy import select

from api.monitoring_service import drift_report, performance_report, unused_validated_labels
from database.models import DocumentFeatures, ModelVersion, MonitoringReport, ReviewFeedback
from database.session import get_sessionmaker
from src.config import ARTIFACTS_DIR, get_settings, load_yaml_config
from src.explainability.shap_explainer import RiskExplainer
from src.features.builder import load_registry
from src.models.evaluation import (
    CLASSES,
    apply_decision_weights,
    compute_metrics,
    tune_decision_weights,
    tune_escalation_threshold,
)
from src.models.quality_gate import evaluate_gate
from src.models.registry import (
    STAGING,
    git_commit,
    log_sklearn_model,
    promote_to_production,
    set_alias,
    setup_mlflow,
)
from src.models.risk_model import RiskModelBundle, fit_pipeline, make_pipeline
from src.monitoring.retraining_policy import decide_retraining
from src.pipeline.model_service import ModelService
from training.common import frozen_split, load_dataset, oof_predict

LEVEL_IDX = {c: i for i, c in enumerate(CLASSES)}


def validated_feedback_frame(session, feature_names: list[str], feature_version: str) -> pd.DataFrame:
    """One row per validated, not-yet-used feedback item, with features stored at analysis time."""
    fb = session.scalars(select(ReviewFeedback).where(
        ReviewFeedback.validation_status == "validated", ReviewFeedback.used_in_training_run.is_(None))).all()
    rows = []
    for f in fb:
        feat = session.scalars(select(DocumentFeatures).where(
            DocumentFeatures.document_id == f.document_id, DocumentFeatures.feature_version == feature_version)
            .order_by(DocumentFeatures.created_at.desc())).first()
        if feat is None:
            continue
        rows.append({"feedback_id": f.id, "document_id": f.document_id, "label": f.reviewer_label,
                     **{k: feat.features.get(k) for k in feature_names}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df[feature_names] = df[feature_names].astype(float)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="skip the trigger policy")
    ap.add_argument("--simulate-bad-candidate", action="store_true",
                    help="corrupt 60%% of training labels to demonstrate the quality gate rejecting a model")
    args = ap.parse_args()

    settings = get_settings()
    tcfg = load_yaml_config("training")
    registry = load_registry()
    models = ModelService.load(settings)
    if models.classifier is None:
        raise SystemExit(f"No production classifier available: {models.errors}")
    prod_meta = models.classifier_metadata

    Session = get_sessionmaker()
    with Session() as session:
        n_new = unused_validated_labels(session)
        drift = drift_report(session, settings, models)
        perf = performance_report(session, models)
        decision = decide_retraining(n_new, drift["data_drift"].get("dataset_drift"), perf.get("degraded"))
        print("Retraining decision:", json.dumps(decision.to_dict(), indent=1))
        if not decision.retrain and not args.force:
            session.add(MonitoringReport(report_type="retraining_decision", model_version=models.classifier.model_version,
                                         summary={"decision": decision.to_dict(), "action": "skipped"}))
            session.commit()
            print("No retraining.")
            return
        fb = validated_feedback_frame(session, registry.names, registry.version)

    df, X, y, meta = load_dataset()
    tr, te = frozen_split(df, y, meta["dataset_version"], tcfg["test_size"], tcfg["seed"])
    X_tr, y_tr, X_te, y_te = X.iloc[tr], y[tr], X.iloc[te], y[te]

    rng = np.random.default_rng(tcfg["seed"])
    fb_train, fb_hold = fb, fb.iloc[0:0]
    if len(fb) >= 20:
        mask = rng.random(len(fb)) < 0.7
        fb_train, fb_hold = fb[mask], fb[~mask]
    if not fb_train.empty:
        X_tr = pd.concat([X_tr, fb_train[registry.names]], ignore_index=True)
        y_tr = np.concatenate([y_tr, fb_train["label"].map(LEVEL_IDX).to_numpy()])
    print(f"Candidate training rows: {len(y_tr)} (base {len(tr)} + feedback {len(fb_train)}); "
          f"feedback hold-out: {len(fb_hold)}")

    if args.simulate_bad_candidate:
        flip = rng.random(len(y_tr)) < 0.6
        y_tr = y_tr.copy()
        y_tr[flip] = rng.integers(0, len(CLASSES), flip.sum())
        print(f"SIMULATION: corrupted {flip.sum()} training labels")

    family, imbalance = prod_meta["family"], prod_meta["imbalance"]
    params = {f"clf__{k}": v for k, v in prod_meta["params"].items()}
    pipe = make_pipeline(family, imbalance, random_state=tcfg["seed"]).set_params(**params)
    oof = oof_predict(pipe, family, imbalance, X_tr.reset_index(drop=True), y_tr, tcfg["cv_folds"], tcfg["seed"])
    gate_abs = load_yaml_config("quality_gate")["absolute"]
    dw = tune_decision_weights(y_tr, oof, gate_abs["high_risk_recall"] + tcfg["selection_recall_margin"],
                               gate_abs["critical_recall"] + tcfg["selection_recall_margin"])
    thr = tune_escalation_threshold(y_tr, oof, tcfg["escalation_min_recall"])["chosen"]["threshold"]
    fit_pipeline(pipe, family, imbalance, X_tr, y_tr)

    proba = pipe.predict_proba(X_te)
    test_metrics = compute_metrics(y_te, proba, y_pred=apply_decision_weights(proba, dw["weights"]),
                                   escalation_threshold=thr)
    recent = {}
    if not fb_hold.empty:
        y_fb = fb_hold["label"].map(LEVEL_IDX).to_numpy()
        cand_pred = apply_decision_weights(pipe.predict_proba(fb_hold[registry.names]), dw["weights"])
        prod_pred = apply_decision_weights(models.classifier.predict_proba(fb_hold[registry.names]),
                                           models.classifier.decision_weights)
        recent = {"n": int(len(fb_hold)),
                  "candidate_f1_macro": float(f1_score(y_fb, cand_pred, average="macro", zero_division=0)),
                  "production_f1_macro": float(f1_score(y_fb, prod_pred, average="macro", zero_division=0))}

    gate = evaluate_gate(test_metrics, prod_meta["test_metrics"], registry.names, registry.names)
    print(f"Candidate test: f1_macro={test_metrics['f1_macro']:.4f} pr_auc={test_metrics['pr_auc_macro']:.4f} "
          f"hr_recall={test_metrics['high_risk_recall']:.4f} crit_recall={test_metrics['critical_recall']:.4f}")
    print(f"Production test: f1_macro={prod_meta['test_metrics']['f1_macro']:.4f} "
          f"hr_recall={prod_meta['test_metrics']['high_risk_recall']:.4f}")
    if recent:
        print(f"Recent validated feedback hold-out (n={recent['n']}): candidate f1={recent['candidate_f1_macro']:.4f} "
              f"production f1={recent['production_f1_macro']:.4f}")

    bundle = RiskModelBundle(pipe, registry.names, registry.version, thr, settings.classifier_model_name, family=family,
                             decision_weights=dw["weights"])
    explainer = RiskExplainer(bundle, background=X_tr.sample(min(200, len(X_tr)), random_state=0),
                              feature_labels=registry.labels)
    test_pred = apply_decision_weights(proba, dw["weights"])
    metadata = {
        **{k: prod_meta[k] for k in ("model_name", "family", "imbalance", "params", "feature_labels", "classes",
                                     "label_policy_version") if k in prod_meta},
        "feature_names": registry.names, "feature_version": registry.version,
        "dataset_version": f"{meta['dataset_version']}+fb{len(fb_train)}",
        "escalation_threshold": thr, "decision_weights": dw["weights"],
        "selection_reason": f"Retrain of production v{models.classifier.model_version} ({family}) with "
                            f"{len(fb_train)} validated feedback rows",
        "cv_metrics": compute_metrics(y_tr, oof, y_pred=apply_decision_weights(oof, dw["weights"])),
        "test_metrics": test_metrics, "recent_feedback_holdout": recent,
        "global_shap": explainer.global_importance(X_tr.sample(min(500, len(X_tr)), random_state=1)),
        "reference_prediction_distribution":
            pd.Series(test_pred).map(dict(enumerate(CLASSES))).value_counts(normalize=True).to_dict(),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_feedback_rows": int(len(fb_train)),
        "retrained_from_version": models.classifier.model_version, "simulation": args.simulate_bad_candidate,
        "trained_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "synthetic_data": True,
        "shap_background": X_tr.sample(min(100, len(X_tr)), random_state=2).to_dict(orient="list"),
    }

    # Training-time feature distribution of this candidate = drift reference if it is promoted.
    ref_dir = ARTIFACTS_DIR / "reports" / "retrain"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / "reference_features.parquet"
    X_tr.assign(risk_label=pd.Series(y_tr).map(dict(enumerate(CLASSES))).to_numpy()).to_parquet(ref_path)

    setup_mlflow(settings)
    with mlflow.start_run(run_name=f"retrain-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags({"pipeline": "retraining", "simulation": str(args.simulate_bad_candidate).lower(),
                         "retrained_from": models.classifier.model_version, "feature_version": registry.version,
                         "dataset_version": metadata["dataset_version"]})
        mlflow.log_params({"family": family, "imbalance": imbalance, "n_feedback_rows": len(fb_train),
                           "decision_weights": str(dw["weights"]), "escalation_threshold": thr})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(ref_path))
        mlflow.log_dict(decision.to_dict(), "retraining_decision.json")
        mlflow.log_dict(gate.to_dict(), "quality_gate.json")
        version = log_sklearn_model(pipe, settings.classifier_model_name, metadata, input_example=X_tr.head(3))
        set_alias(settings.classifier_model_name, STAGING, version)
        mlflow.set_tag("quality_gate", "PASS" if gate.passed else "FAIL")
        promoted = gate.passed
        if promoted:
            promote_to_production(settings.classifier_model_name, version)
        run_id = run.info.run_id

    with Session() as session:
        if promoted and not fb_train.empty:
            for fid in fb_train["feedback_id"]:
                session.get(ReviewFeedback, int(fid)).used_in_training_run = run_id
        session.add(ModelVersion(model_name=settings.classifier_model_name, version=str(version),
                                 alias="production" if promoted else "rejected", family=family, run_id=run_id,
                                 feature_version=registry.version, dataset_version=metadata["dataset_version"],
                                 metrics=test_metrics, quality_gate=gate.to_dict(),
                                 status="production" if promoted else "rejected"))
        session.add(MonitoringReport(report_type="retraining_decision", model_version=str(version), summary={
            "decision": decision.to_dict(), "forced": args.force, "simulation": args.simulate_bad_candidate,
            "candidate_version": str(version), "promoted": promoted, "quality_gate": gate.to_dict(),
            "candidate_test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, int | float)},
            "production_test_metrics": {k: v for k, v in prod_meta["test_metrics"].items()
                                        if isinstance(v, int | float)},
            "recent_feedback_holdout": recent, "n_feedback_rows": int(len(fb_train)),
        }))
        session.commit()

    if promoted:
        print(f"Quality gate PASSED -> candidate v{version} promoted to @production. Restart the API to load it.")
    else:
        print(f"Quality gate FAILED -> candidate v{version} rejected:")
        for c in gate.failures:
            print(f"   - {c.name}: candidate={c.candidate} threshold={c.threshold}")


if __name__ == "__main__":
    main()
