"""Risk-classifier training pipeline with experiment tracking.

Steps (all tracked in MLflow under one parent run):
  1. Frozen stratified hold-out split (test set never used for selection or tuning).
  2. Experiment grid on the training set with 5-fold StratifiedKFold:
     {Logistic Regression, Random Forest, XGBoost} x {no rebalancing, class weights, SMOTE}.
  3. RandomizedSearchCV per family using that family's best imbalance strategy.
  4. Evidence-based selection: best CV macro-F1; within a small margin the simpler family wins.
  5. Escalation-threshold tuning on out-of-fold probabilities (target HIGH+CRITICAL recall).
  6. Final fit on the full training set, evaluation on the hold-out test set,
     a temporal robustness check (train on older contracts, test on newer).
  7. Global SHAP importance, model registration, @staging alias, quality gate, promotion.

Usage:
    python -m training.train
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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
    PRODUCTION,
    STAGING,
    get_alias_version,
    git_commit,
    load_version_metadata,
    log_sklearn_model,
    promote_to_production,
    set_alias,
    setup_mlflow,
)
from src.models.risk_model import (
    IMBALANCE_STRATEGIES,
    MODEL_FAMILIES,
    RiskModelBundle,
    fit_pipeline,
    fit_sample_weight,
    make_pipeline,
)
from training.common import (
    flat_metrics,
    frozen_split,
    load_dataset,
    oof_predict,
    plot_confusion,
    plot_importance,
)

REPORT_DIR = ARTIFACTS_DIR / "reports"


def run_grid(X, y, cfg) -> list[dict]:
    results = []
    for family in MODEL_FAMILIES:
        for imbalance in IMBALANCE_STRATEGIES:
            t0 = time.perf_counter()
            pipe = make_pipeline(family, imbalance, random_state=cfg["seed"])
            proba = oof_predict(pipe, family, imbalance, X, y, cfg["cv_folds"], cfg["seed"])
            m = compute_metrics(y, proba)
            with mlflow.start_run(run_name=f"grid-{family}-{imbalance}", nested=True):
                mlflow.set_tags({"stage": "imbalance_grid", "family": family, "imbalance": imbalance})
                mlflow.log_params({"family": family, "imbalance": imbalance, "cv_folds": cfg["cv_folds"]})
                mlflow.log_metrics(flat_metrics(m, "cv_"))
            results.append({"family": family, "imbalance": imbalance, "cv": m,
                            "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  grid {family:20s} {imbalance:13s} f1_macro={m['f1_macro']:.4f} "
                  f"pr_auc={m['pr_auc_macro']:.4f} hr_recall={m['high_risk_recall']:.4f} "
                  f"crit_recall={m['critical_recall']:.4f}")
    return results


def tune_family(family, imbalance, X, y, cfg) -> dict:
    space = {k: v for k, v in cfg["search_spaces"][family].items()}
    pipe = make_pipeline(family, imbalance, random_state=cfg["seed"])
    n_iter = min(cfg["search_iterations"], int(np.prod([len(v) for v in space.values()])))
    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter, scoring="f1_macro", random_state=cfg["seed"], n_jobs=1,
        cv=StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"]), refit=False,
    )
    w = fit_sample_weight(family, imbalance, y)
    t0 = time.perf_counter()
    search.fit(X, y, **({"clf__sample_weight": w} if w is not None else {}))
    best_params = search.best_params_
    tuned = make_pipeline(family, imbalance, random_state=cfg["seed"]).set_params(**best_params)
    proba = oof_predict(tuned, family, imbalance, X, y, cfg["cv_folds"], cfg["seed"])
    m_argmax = compute_metrics(y, proba)

    # Multi-class threshold moving, tuned on out-of-fold predictions only.
    gate = load_yaml_config("quality_gate")["absolute"]
    dw = tune_decision_weights(y, proba, gate["high_risk_recall"] + cfg["selection_recall_margin"],
                               gate["critical_recall"] + cfg["selection_recall_margin"])
    m = compute_metrics(y, proba, y_pred=apply_decision_weights(proba, dw["weights"]))
    with mlflow.start_run(run_name=f"tuned-{family}", nested=True):
        mlflow.set_tags({"stage": "hyperparameter_search", "family": family, "imbalance": imbalance})
        mlflow.log_params({"family": family, "imbalance": imbalance, "n_iter": n_iter,
                           "decision_weights": str(dw["weights"]),
                           **{k.replace("clf__", ""): str(v) for k, v in best_params.items()}})
        mlflow.log_metrics({**flat_metrics(m_argmax, "cv_argmax_"), **flat_metrics(m, "cv_"),
                            "search_best_f1_macro": float(search.best_score_)})
    print(f"  tuned {family:20s} ({imbalance}) argmax f1={m_argmax['f1_macro']:.4f} "
          f"crit_rec={m_argmax['critical_recall']:.3f} | thresholded f1={m['f1_macro']:.4f} "
          f"hr_rec={m['high_risk_recall']:.3f} crit_rec={m['critical_recall']:.3f} feasible={dw['feasible']} "
          f"w={dw['weights']} params={best_params}")
    return {"family": family, "imbalance": imbalance, "params": best_params, "cv": m, "cv_argmax": m_argmax,
            "decision_weights": dw, "oof_proba": proba, "search_best_score": float(search.best_score_),
            "n_iter": n_iter, "seconds": round(time.perf_counter() - t0, 1)}


def select(tuned: list[dict], cfg) -> tuple[dict, str]:
    """Constraint-aware selection: among models whose CV decisions meet the recall constraints,
    take the best macro-F1; within ``simplicity_tie_margin`` prefer the simpler family."""
    metric, margin = cfg["selection_metric"], cfg["simplicity_tie_margin"]
    feasible = [t for t in tuned if t["decision_weights"]["feasible"]]
    pool = feasible or tuned
    best_score = max(t["cv"][metric] for t in pool)
    contenders = [t for t in pool if t["cv"][metric] >= best_score - margin]
    order = cfg["family_complexity"]
    chosen = min(contenders, key=lambda t: order.index(t["family"]))
    reason = (f"{len(feasible)}/{len(tuned)} tuned families meet the CV recall constraints "
              f"(HIGH+CRITICAL and CRITICAL recall). Best CV {metric} among them = {best_score:.4f}. "
              f"Within {margin}: {[t['family'] for t in contenders]}. Chose the simplest: {chosen['family']} "
              f"({metric} = {chosen['cv'][metric]:.4f}).")
    if not feasible:
        reason = "WARNING: no family met the recall constraints in CV. " + reason
    return chosen, reason


def temporal_check(df, X, y, chosen, cfg) -> dict:
    """Train on the oldest 80% of contracts, test on the newest 20% (time-ordered robustness)."""
    order = np.argsort(df["submission_date"].to_numpy(), kind="stable")
    cut = int(len(order) * 0.8)
    tr, te = order[:cut], order[cut:]
    pipe = make_pipeline(chosen["family"], chosen["imbalance"], random_state=cfg["seed"]).set_params(**chosen["params"])
    fit_pipeline(pipe, chosen["family"], chosen["imbalance"], X.iloc[tr], y[tr])
    proba = pipe.predict_proba(X.iloc[te])
    m = compute_metrics(y[te], proba, y_pred=apply_decision_weights(proba, chosen["decision_weights"]["weights"]))
    m["train_until"] = str(df["submission_date"].iloc[order[cut - 1]])
    m["test_from"] = str(df["submission_date"].iloc[order[cut]])
    return m


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="free-text context recorded as an MLflow tag (e.g. why this run exists)")
    args = ap.parse_args()
    cfg = load_yaml_config("training")
    settings = get_settings()
    registry = load_registry()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y, meta = load_dataset()
    tr, te = frozen_split(df, y, meta["dataset_version"], cfg["test_size"], cfg["seed"])
    X_tr, y_tr, X_te, y_te = X.iloc[tr], y[tr], X.iloc[te], y[te]
    print(f"Train {len(tr)} / test {len(te)}; train labels {np.bincount(y_tr).tolist()} "
          f"test labels {np.bincount(y_te).tolist()}")

    setup_mlflow(settings)
    with mlflow.start_run(run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}") as parent:
        mlflow.set_tags({"pipeline": "training", "dataset_version": meta["dataset_version"],
                         "feature_version": registry.version, "git_commit": git_commit() or "uncommitted",
                         "synthetic_data": "true", "run_note": args.note or "-"})
        mlflow.log_params({"seed": cfg["seed"], "test_size": cfg["test_size"], "cv_folds": cfg["cv_folds"],
                           "n_train": len(tr), "n_test": len(te), "n_features": len(registry.names)})

        print("Experiment 1: model family x imbalance strategy (5-fold CV on train)")
        grid = run_grid(X_tr, y_tr, cfg)

        print("Experiment 2: hyperparameter search per family")
        tuned = []
        for family in MODEL_FAMILIES:
            fam_rows = [g for g in grid if g["family"] == family]
            best_imb = max(fam_rows, key=lambda g: g["cv"][cfg["selection_metric"]])["imbalance"]
            tuned.append(tune_family(family, best_imb, X_tr, y_tr, cfg))

        chosen, reason = select(tuned, cfg)
        print("Selection:", reason)

        thr = tune_escalation_threshold(y_tr, chosen["oof_proba"], cfg["escalation_min_recall"])
        threshold = thr["chosen"]["threshold"]
        print(f"Escalation threshold {threshold} (OOF recall {thr['chosen']['recall']:.3f}, "
              f"precision {thr['chosen']['precision']:.3f}, review rate {thr['chosen']['review_rate']:.3f})")

        final = make_pipeline(chosen["family"], chosen["imbalance"], random_state=cfg["seed"])
        final.set_params(**chosen["params"])
        fit_pipeline(final, chosen["family"], chosen["imbalance"], X_tr, y_tr)
        weights = chosen["decision_weights"]["weights"]
        test_proba = final.predict_proba(X_te)
        test_pred = apply_decision_weights(test_proba, weights)
        test_metrics = compute_metrics(y_te, test_proba, y_pred=test_pred, escalation_threshold=threshold)
        test_metrics_argmax = compute_metrics(y_te, test_proba)
        temporal = temporal_check(df, X, y, chosen, cfg)
        print(f"Test: f1_macro={test_metrics['f1_macro']:.4f} pr_auc={test_metrics['pr_auc_macro']:.4f} "
              f"roc_auc={test_metrics['roc_auc_ovr_macro']:.4f} hr_recall={test_metrics['high_risk_recall']:.4f} "
              f"crit_recall={test_metrics['critical_recall']:.4f} esc_recall={test_metrics['escalation_recall']:.4f}")
        print(f"Temporal check: f1_macro={temporal['f1_macro']:.4f} hr_recall={temporal['high_risk_recall']:.4f}")

        bundle = RiskModelBundle(final, registry.names, registry.version, threshold,
                                 model_name=settings.classifier_model_name, family=chosen["family"],
                                 decision_weights=weights)
        explainer = RiskExplainer(bundle, background=X_tr.sample(min(200, len(X_tr)), random_state=0),
                                  feature_labels=registry.labels)
        global_shap = explainer.global_importance(X_tr.sample(min(500, len(X_tr)), random_state=1))

        # Comparison of the best untuned and tuned configuration of every family on the TEST set,
        # reported for transparency only (selection above used CV only).
        test_comparison = []
        for t in tuned:
            p = make_pipeline(t["family"], t["imbalance"], random_state=cfg["seed"]).set_params(**t["params"])
            fit_pipeline(p, t["family"], t["imbalance"], X_tr, y_tr)
            pp = p.predict_proba(X_te)
            test_comparison.append({
                "family": t["family"], "imbalance": t["imbalance"], "decision_weights": t["decision_weights"]["weights"],
                "test_argmax": compute_metrics(y_te, pp),
                "test": compute_metrics(y_te, pp, y_pred=apply_decision_weights(pp, t["decision_weights"]["weights"])),
            })

        pred_dist = pd.Series(test_pred).map(dict(enumerate(CLASSES))).value_counts(normalize=True)
        metadata = {
            "model_name": settings.classifier_model_name,
            "family": chosen["family"], "imbalance": chosen["imbalance"],
            "params": {k.replace("clf__", ""): v for k, v in chosen["params"].items()},
            "feature_names": registry.names, "feature_version": registry.version,
            "feature_labels": registry.labels, "classes": CLASSES,
            "dataset_version": meta["dataset_version"],
            "label_policy_version": load_yaml_config("labeling_policy")["version"],
            "escalation_threshold": threshold, "decision_weights": weights, "selection_reason": reason,
            "cv_metrics": chosen["cv"], "cv_metrics_argmax": chosen["cv_argmax"],
            "test_metrics": test_metrics, "test_metrics_argmax": test_metrics_argmax, "temporal_check": temporal,
            "global_shap": global_shap, "reference_prediction_distribution": pred_dist.to_dict(),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "trained_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
            "synthetic_data": True, "run_note": args.note,
            # background sample for SHAP LinearExplainer (used if a linear model is ever selected)
            "shap_background": X_tr.sample(min(100, len(X_tr)), random_state=2).to_dict(orient="list"),
        }

        plot_confusion(test_metrics["confusion_matrix"], f"Test confusion matrix - {chosen['family']}",
                       REPORT_DIR / "confusion_matrix_test.png")
        plot_importance(global_shap, "Global SHAP importance", REPORT_DIR / "global_shap.png")
        X_tr.assign(risk_label=pd.Series(y_tr, index=X_tr.index).map(dict(enumerate(CLASSES)))).to_parquet(
            REPORT_DIR / "reference_features.parquet")
        mlflow.log_artifact(str(REPORT_DIR / "confusion_matrix_test.png"))
        mlflow.log_artifact(str(REPORT_DIR / "global_shap.png"))
        mlflow.log_artifact(str(REPORT_DIR / "reference_features.parquet"))
        mlflow.log_dict(thr, "threshold_tuning.json")
        mlflow.log_params({"selected_family": chosen["family"], "selected_imbalance": chosen["imbalance"],
                           "escalation_threshold": threshold})
        mlflow.log_metrics({**flat_metrics(chosen["cv"], "selected_cv_"), **flat_metrics(test_metrics, "test_"),
                            **flat_metrics(temporal, "temporal_")})

        version = log_sklearn_model(final, settings.classifier_model_name, metadata,
                                    input_example=X_tr.head(3))
        set_alias(settings.classifier_model_name, STAGING, version)
        print(f"Registered {settings.classifier_model_name} v{version} -> @staging")

        prod_version = get_alias_version(settings.classifier_model_name, PRODUCTION)
        prod_metrics = load_version_metadata(settings.classifier_model_name, prod_version)["test_metrics"] \
            if prod_version else None
        gate = evaluate_gate(test_metrics, prod_metrics, registry.names, registry.names)
        mlflow.log_dict(gate.to_dict(), "quality_gate.json")
        mlflow.set_tag("quality_gate", "PASS" if gate.passed else "FAIL")
        if gate.passed:
            old = promote_to_production(settings.classifier_model_name, version)
            print(f"Quality gate PASSED -> v{version} is @production (previous: {old})")
        else:
            print("Quality gate FAILED -> not promoted:", [c.name for c in gate.failures])

        report = {
            "run_id": parent.info.run_id, "registered_version": version, "promoted": gate.passed,
            "dataset": meta, "split": {"n_train": int(len(tr)), "n_test": int(len(te)),
                                       "train_labels": np.bincount(y_tr).tolist(),
                                       "test_labels": np.bincount(y_te).tolist()},
            "grid": [{k: v for k, v in g.items()} for g in grid],
            "tuned": [{k: v for k, v in t.items() if k != "oof_proba"} for t in tuned],
            "selection_reason": reason, "threshold_tuning": thr, "test_metrics": test_metrics,
            "test_comparison": test_comparison, "temporal_check": temporal, "global_shap": global_shap,
            "quality_gate": gate.to_dict(), "metadata": {k: v for k, v in metadata.items() if k != "global_shap"},
        }
        (REPORT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("Wrote", REPORT_DIR / "training_report.json")


if __name__ == "__main__":
    main()
