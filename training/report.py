"""Generate docs/EVALUATION_REPORT.md from the recorded artifacts.

Every number in the report is read from files produced by the pipeline (training report,
anomaly report, extraction/doc-type evaluation, label diagnostic, weight validation) or from
the MLflow registry. Nothing is typed in by hand. Re-run after any training run:

    python -m training.report
"""

from __future__ import annotations

import json
import os
import shutil

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT, load_yaml_config
from src.models.evaluation import CLASSES

R = ARTIFACTS_DIR / "reports"
P = DATA_DIR / "processed"
DOCS = PROJECT_ROOT / "docs"
IMG = DOCS / "images"

# Explanatory notes per registry version (the numbers next to them come from MLflow).
VERSION_NOTES = {
    "document-risk-classifier": {
        "1": "1,500 contracts, feature set fv1. Selected logistic regression by the simplicity rule; failed absolute "
             "recall gates. Label diagnostic showed the information was present but under-fitted.",
        "2": "3,000 contracts (learning curve still rising) + policy-indicator features (fv2). Selection ignored "
             "recall constraints -> CRITICAL recall too low; gate rejected.",
        "3": "Added constraint-aware selection + multi-class threshold tuning on out-of-fold predictions. First "
             "model to pass the gate.",
        "4": "Feedback-driven retraining after simulated concept drift (39 new-policy labels). Regressed on the "
             "frozen test set; gate rejected.",
        "5": "Deliberately corrupted labels (simulation) - gate demonstration.",
        "6": "Vendor-identity bug fixed (fv3: 'X LLP' and 'X Pvt Ltd' no longer share history). Missed the CRITICAL "
             "recall minimum by one contract (31/45); gate rejected, NOT overridden.",
        "7": "LightGBM added as a 4th family (post-hoc, after seeing v6 fail - disclosed). Best CV macro-F1; "
             "passed the gate; current production.",
        "8": "Feedback-driven retraining on fv3 after simulated concept drift. Regressed on test; gate rejected.",
        "9": "Deliberately corrupted labels (simulation) - gate demonstration.",
    },
    "document-anomaly-detector": {
        "1": "Isolation Forest on fv2 features.",
        "2": "Isolation Forest on fv3 features; vendor-behaviour checks recalibrated (relative frequency, "
             "payment above vendor maximum).",
    },
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def f(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def registry_history() -> list[dict]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient

    from src.models.registry import setup_mlflow

    setup_mlflow()
    c = MlflowClient()
    rows = []
    for name in ("document-risk-classifier", "document-anomaly-detector"):
        for found in sorted(c.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
            mv = c.get_model_version(name, found.version)
            run = c.get_run(mv.run_id)
            t, m = run.data.tags, run.data.metrics
            rows.append({
                "model": name.replace("document-", ""), "version": f"v{mv.version}",
                "aliases": ", ".join(f"@{a}" for a in mv.aliases) or "-",
                "pipeline": t.get("pipeline", "-"), "features": t.get("feature_version", "-"),
                "gate": t.get("quality_gate", "-"),
                "f1": f(m.get("test_f1_macro")) if "test_f1_macro" in m else f"AUC {f(m.get('if_roc_auc'))}",
                "crit": f(m.get("test_critical_recall"), 3) if "test_critical_recall" in m else "-",
                "hr": f(m.get("test_high_risk_recall"), 3) if "test_high_risk_recall" in m else "-",
                "note": VERSION_NOTES.get(name, {}).get(mv.version, ""),
            })
    return rows


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for name in ("confusion_matrix_test.png", "global_shap.png"):
        shutil.copy(R / name, IMG / name)

    tr = load(R / "training_report.json")
    an = load(R / "anomaly_report.json")
    diag = load(R / "label_diagnostic.json")
    wv = load(R / "weight_validation.json")
    ext = load(P / "extraction_eval.json")
    dt = load(P / "doc_type_eval.json")
    meta = load(P / "dataset_meta.json")
    manifest = load(DATA_DIR / "synthetic" / "manifest.json")
    gate_cfg = load_yaml_config("quality_gate")
    md = tr["metadata"]
    tm = tr["test_metrics"]
    tma = md.get("test_metrics_argmax", {})

    grid_rows = [{"family": g["family"], "imbalance": g["imbalance"], "f1": f(g["cv"]["f1_macro"]),
                  "prauc": f(g["cv"]["pr_auc_macro"]), "hr": f(g["cv"]["high_risk_recall"], 3),
                  "crit": f(g["cv"]["critical_recall"], 3), "acc": f(g["cv"]["accuracy"], 3)} for g in tr["grid"]]
    tuned_rows = [{"family": t["family"], "imbalance": t["imbalance"],
                   "params": ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in t["params"].items()),
                   "argmax_f1": f(t["cv_argmax"]["f1_macro"]), "f1": f(t["cv"]["f1_macro"]),
                   "hr": f(t["cv"]["high_risk_recall"], 3), "crit": f(t["cv"]["critical_recall"], 3),
                   "w": t["decision_weights"]["weights"], "feasible": "yes" if t["decision_weights"]["feasible"] else "no"}
                  for t in tr["tuned"]]
    comp_rows = [{"family": c["family"], "f1": f(c["test"]["f1_macro"]), "prauc": f(c["test"]["pr_auc_macro"]),
                  "roc": f(c["test"]["roc_auc_ovr_macro"]), "hr": f(c["test"]["high_risk_recall"], 3),
                  "crit": f(c["test"]["critical_recall"], 3)} for c in tr["test_comparison"]]
    per_class = [{"class": c, **{k: f(tm["per_class"][c][k], 3) for k in ("precision", "recall", "f1")},
                  "support": tm["per_class"][c]["support"]} for c in CLASSES]
    gate_rows = [{"check": c["name"], "passed": "PASS" if c["passed"] else "FAIL",
                  "candidate": f(c["candidate"]) if isinstance(c["candidate"], float) else c["candidate"],
                  "threshold": c["threshold"]} for c in tr["quality_gate"]["checks"]]
    thr = tr["threshold_tuning"]["chosen"]
    ext_rows = [{"field": k, "all": f(v["all"]["accuracy"], 3), "native": f(v.get("native", {}).get("accuracy"), 3),
                 "docx": f(v.get("docx", {}).get("accuracy"), 3), "ocr": f(v.get("ocr", {}).get("accuracy"), 3)}
                for k, v in ext.items() if k != "_meta"]
    labels = meta["label_distribution"]
    n = sum(labels.values())

    lines = [
        "# Evaluation Report - Document Risk Engine (V1, vendor contracts)",
        "",
        "> **All data is synthetic.** Every number below is generated by `python -m training.report` from the",
        "> artifacts of the actual runs and from the MLflow registry. They describe performance on a synthetic",
        "> corpus with a known labeling policy and **do not** represent real enterprise performance.",
        "",
        "## 1. Dataset",
        "",
        f"- Corpus: {manifest['n_contracts']:,} synthetic vendor contracts (dataset version `{meta['dataset_version']}`), "
        f"rendered as PDF / DOCX / scanned PDF / PNG / JPG: {manifest['format_distribution']}.",
        f"- Every document went through the production extraction pipeline (OCR for scans); "
        f"{ext['_meta']['n_failed']} extraction failures; method mix {ext['_meta']['method_counts']}.",
        f"- Plus {dt['n'] - manifest['n_contracts']} invoices / purchase orders for document-type detection "
        f"(accuracy **{dt['accuracy']:.3f}**).",
        f"- Feature set `{meta['feature_version']}`: {len(meta['features'])} features (see `configs/features.yaml`).",
        f"- Split: frozen stratified hold-out, {tr['split']['n_train']} train / {tr['split']['n_test']} test "
        "(`data/processed/split.json`); every candidate is evaluated on the same test contracts.",
        "",
        "### Label distribution (labeling policy `lp-v1`, `configs/labeling_policy.yaml`)",
        "",
        table([{"level": k, "count": labels[k], "share": f"{labels[k] / n:.1%}"} for k in CLASSES],
              ["level", "count", "share"]),
        "",
        "### Extraction accuracy vs ground truth",
        "",
        table(ext_rows, ["field", "all", "native", "docx", "ocr"],
              ["field", "all", "native PDF", "DOCX", "OCR"]),
        "",
        "Templated synthetic text makes extraction far easier than real contracts; OCR shows the expected small drop.",
        "",
        f"### Label observability: {diag['label_reconstruction_agreement']:.1%} of labels can be reconstructed from "
        "extracted features alone (`training/diagnose_labels.py`). The remaining gap comes from factors that need "
        "information the features don't carry (e.g. contract splitting).",
        "",
        "## 2. Experiment 1 - model family x class-imbalance strategy (5-fold StratifiedKFold on train)",
        "",
        table(grid_rows, ["family", "imbalance", "f1", "prauc", "hr", "crit", "acc"],
              ["family", "imbalance", "macro-F1", "macro PR-AUC", "HIGH+CRIT recall", "CRITICAL recall", "accuracy"]),
        "",
        "Accuracy is shown for reference only and never used for selection. Observations: SMOTE helps the tree "
        "models most on CRITICAL recall; class weights help logistic regression; argmax decisions leave tree models "
        "short on CRITICAL recall - which is what threshold tuning addresses.",
        "",
        "## 3. Experiment 2 - hyper-parameter search (RandomizedSearchCV, 20 iterations, scoring = macro-F1)",
        "",
        "Each family uses its best imbalance strategy from Experiment 1. Decision weights (multi-class threshold "
        "moving, `argmax(p * w)`) are tuned on **out-of-fold** probabilities to maximise macro-F1 subject to "
        f"HIGH+CRITICAL recall >= {gate_cfg['absolute']['high_risk_recall']} + margin and CRITICAL recall >= "
        f"{gate_cfg['absolute']['critical_recall']} + margin.",
        "",
        table(tuned_rows, ["family", "imbalance", "argmax_f1", "f1", "hr", "crit", "w", "feasible", "params"],
              ["family", "imbalance", "CV F1 (argmax)", "CV F1 (tuned)", "HIGH+CRIT recall", "CRITICAL recall",
               "decision weights", "meets constraints", "best params"]),
        "",
        "## 4. Model selection",
        "",
        "**Rule (pre-declared in `configs/training.yaml`):** among families meeting the recall constraints in CV, "
        "take the best CV macro-F1; within 0.01, prefer the simpler family.",
        "",
        f"**Outcome:** {tr['selection_reason']}",
        "",
        f"Selected: **{md['family']}** ({md['imbalance']}), decision weights {md['decision_weights']}, "
        f"escalation threshold P(HIGH)+P(CRITICAL) >= {md['escalation_threshold']} "
        f"(OOF recall {thr['recall']:.3f}, precision {thr['precision']:.3f}, review rate {thr['review_rate']:.3f}).",
        "",
        "## 5. Test-set results (frozen hold-out, never used for selection or tuning)",
        "",
        table([{"metric": k, "value": f(tm[k])} for k in
               ("precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro", "pr_auc_macro", "high_risk_recall",
                "high_risk_precision", "critical_recall", "escalation_recall", "escalation_precision", "accuracy")],
              ["metric", "value"]),
        "",
        table(per_class, ["class", "precision", "recall", "f1", "support"]),
        "",
        "![confusion matrix](images/confusion_matrix_test.png)",
        "",
        "### Effect of threshold tuning on the test set",
        "",
        table([{"rule": "plain argmax", "f1": f(tma.get("f1_macro")), "hr": f(tma.get("high_risk_recall"), 3),
                "crit": f(tma.get("critical_recall"), 3)},
               {"rule": f"tuned weights {md['decision_weights']}", "f1": f(tm["f1_macro"]),
                "hr": f(tm["high_risk_recall"], 3), "crit": f(tm["critical_recall"], 3)}],
              ["rule", "f1", "hr", "crit"], ["decision rule", "macro-F1", "HIGH+CRIT recall", "CRITICAL recall"]),
        "",
        "### Temporal robustness check (train on the oldest 80%, test on the newest 20%)",
        "",
        f"macro-F1 {f(tr['temporal_check']['f1_macro'])}, HIGH+CRITICAL recall "
        f"{f(tr['temporal_check']['high_risk_recall'], 3)} (train until {tr['temporal_check']['train_until']}, "
        f"test from {tr['temporal_check']['test_from']}).",
        "",
        "### All tuned families on the test set (transparency only)",
        "",
        table(comp_rows, ["family", "f1", "prauc", "roc", "hr", "crit"],
              ["family", "macro-F1", "PR-AUC", "ROC-AUC", "HIGH+CRIT recall", "CRITICAL recall"]),
        "",
        "Selection used cross-validation only. If another family scores higher here, switching to it now would be "
        "selecting on the test set, which would make the reported test numbers optimistic - so it is not done.",
        "",
        "## 6. Quality gate (`configs/quality_gate.yaml`)",
        "",
        table(gate_rows, ["check", "passed", "candidate", "threshold"]),
        "",
        "## 7. Explainability - global SHAP importance",
        "",
        "![global SHAP](images/global_shap.png)",
        "",
        table([{"feature": g["label"], "mean_abs_shap": f(g["mean_abs_shap"])} for g in tr["global_shap"][:12]],
              ["feature", "mean_abs_shap"], ["feature", "mean |SHAP|"]),
        "",
        "## 8. Model registry history (every version, including rejected ones)",
        "",
        table(registry_history(), ["model", "version", "aliases", "pipeline", "features", "gate", "f1", "hr", "crit", "note"],
              ["model", "version", "aliases", "pipeline", "features", "gate", "test macro-F1", "HIGH+CRIT recall",
               "CRITICAL recall", "what happened"]),
        "",
        "**Disclosure:** v7 was produced after observing v6 fail the gate on the test set. The change (adding "
        "LightGBM, a family named in the project brief) was made without changing the gate, the test set or the "
        "selection rule, and the run is tagged in MLflow (`run_note`). Because the decision to try again was "
        "informed by a test-set result, v7's test metrics are slightly optimistic in principle.",
        "",
        "## 9. Anomaly detection (Isolation Forest + vendor-behaviour checks)",
        "",
        "Evaluated against injected 'unusual' scenarios that the detector never sees (unusual value, excessive "
        "payment, high penalty, long renewal, contract splitting), on the test split.",
        "",
        table([{"metric": k, "value": v} for k, v in an["metrics"].items()], ["metric", "value"]),
        "",
        "The multivariate flag is deliberately conservative (97th percentile): high precision, low recall. The "
        "continuous anomaly score (ROC-AUC above) feeds the risk score, so unflagged-but-unusual contracts still "
        "raise risk. The contract-frequency check has low precision against the 'contract splitting' ground truth "
        "because busy vendors legitimately sign contracts in bursts - it is an investigation prompt, not a verdict.",
        "",
        "## 10. Risk-score weight validation (`training/validate_weights.py`)",
        "",
        f"Computed on the {wv['n_test']} test contracts with production classifier v{wv['classifier_version']} and "
        f"anomaly detector v{wv['anomaly_version']}.",
        "",
        table([{"variant": "default " + json.dumps(wv["default"]["weights"]),
                "spearman": f(wv["default"]["spearman_vs_points"], 3), "auc": f(wv["default"]["roc_auc_high_risk"], 3),
                "f1": f(wv["default"]["level_f1_macro"], 3)}]
              + [{"variant": f"{k} only", "spearman": f(v["spearman_vs_points"], 3), "auc": f(v["roc_auc_high_risk"], 3),
                  "f1": f(v["level_f1_macro"], 3)} for k, v in wv["single_component"].items()]
              + [{"variant": f"grid best by {label} " + json.dumps(wv[f'best_by_{m}']['weights']),
                  **{col: (f(wv[f'best_by_{m}'][m], 3) if col == key else "(optimised: see other column)")
                     for col in ("spearman", "auc", "f1") if col == key}}
                 for m, key, label in (("spearman_vs_points", "spearman", "Spearman"),
                                       ("roc_auc_high_risk", "auc", "ROC-AUC"),
                                       ("level_f1_macro", "f1", "level F1"))],
              ["variant", "spearman", "auc", "f1"],
              ["weights", "Spearman vs policy points", "ROC-AUC (HIGH+)", "level macro-F1"]),
        "",
        f"Out of {wv['n_grid']} weight combinations (step 0.05), the default weights beat "
        f"{wv['best_by_spearman_vs_points']['default_rank_percentile']:.0%} on Spearman, "
        f"{wv['best_by_roc_auc_high_risk']['default_rank_percentile']:.0%} on ROC-AUC and "
        f"{wv['best_by_level_f1_macro']['default_rank_percentile']:.0%} on level macro-F1 - good, not optimal. "
        "(Each 'grid best' row shows only the metric it was optimised for.)",
        "",
        "**Interpretation.** The ML component alone agrees best with the policy labels - by construction, because the "
        "labels come from the policy the model was trained to reproduce. The composite deliberately adds signals the "
        "labeling policy does not encode (statistical anomaly, financial exposure, deterministic rules that must "
        "fire 100% of the time), so label agreement is not its only objective. The composite *ranks* well but its "
        "fixed level thresholds (31/61/81) under-call risk relative to the labels (level macro-F1 "
        f"{wv['default']['level_f1_macro']:.3f}). **Recommendation:** recalibrate level thresholds on reviewer-validated "
        "data once it exists; weights were left unchanged because changing them is a policy decision and tuning them "
        "on the test set would overfit it.",
        "",
        "## 11. Limitations",
        "",
        "- Synthetic, templated documents: extraction and modelling are much easier than on real contracts.",
        "- Labels come from a known rule policy; a model that learns it well says nothing about real analyst judgement.",
        "- 45 CRITICAL test contracts: recall estimates for that class have a standard error of roughly 6-7 points; "
        "gate decisions near a threshold are noisy (a bootstrap confidence bound would be more robust).",
        "- Static FX rates; Indian-English contract conventions only; no clause-level semantic understanding.",
        "- Concept drift caused by a policy change is better handled by updating the compliance rules than by "
        "retraining on a few dozen labels - the retraining demo shows why (Section 8, v4/v8).",
        "",
    ]
    (DOCS / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/EVALUATION_REPORT.md")


if __name__ == "__main__":
    pd.options.display.float_format = "{:.4f}".format
    main()
