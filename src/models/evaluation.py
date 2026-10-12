"""Model evaluation metrics.

Accuracy is reported but never used for selection: with ~10% CRITICAL contracts a model
that never predicts CRITICAL can still look accurate. Selection and promotion use
macro-F1, macro PR-AUC and recall on the HIGH+CRITICAL ("high-risk") group, because
missing a high-risk contract costs far more than an extra review.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
HIGH_RISK_IDX = [2, 3]


def high_risk_score(proba: np.ndarray) -> np.ndarray:
    return proba[:, HIGH_RISK_IDX].sum(axis=1)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray | None = None,
                    escalation_threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true)
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    y_bin = label_binarize(y_true, classes=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    hr_true = np.isin(y_true, HIGH_RISK_IDX).astype(int)
    hr_score = high_risk_score(proba)
    hr_pred = np.isin(y_pred, HIGH_RISK_IDX).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"),
        "pr_auc_macro": average_precision_score(y_bin, proba, average="macro"),
        "high_risk_recall": recall_score(hr_true, hr_pred, zero_division=0),
        "high_risk_precision": precision_score(hr_true, hr_pred, zero_division=0),
        "high_risk_pr_auc": average_precision_score(hr_true, hr_score),
        "critical_recall": float(r[3]),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if escalation_threshold is not None:
        esc = (hr_score >= escalation_threshold).astype(int)
        out["escalation_threshold"] = escalation_threshold
        out["escalation_recall"] = recall_score(hr_true, esc, zero_division=0)
        out["escalation_precision"] = precision_score(hr_true, esc, zero_division=0)
        out["escalation_rate"] = float(esc.mean())
    return {k: (float(v) if isinstance(v, np.floating | float) else v) for k, v in out.items()}


def apply_decision_weights(proba: np.ndarray, weights: list[float] | None) -> np.ndarray:
    """Multi-class threshold moving: predict argmax(p_k * w_k). weights=None -> plain argmax."""
    if weights is None:
        return proba.argmax(axis=1)
    return (proba * np.asarray(weights)[None, :]).argmax(axis=1)


def tune_decision_weights(y_true: np.ndarray, proba: np.ndarray, min_high_risk_recall: float,
                          min_critical_recall: float) -> dict:
    """Grid-search per-class decision weights on OUT-OF-FOLD probabilities.

    Objective: maximise macro-F1 subject to HIGH+CRITICAL recall and CRITICAL recall
    constraints (the business cost of a missed high-risk contract is much higher than an
    extra review). If no combination meets the constraints, maximise the summed recalls.
    """
    y_true = np.asarray(y_true)
    hr_true = np.isin(y_true, HIGH_RISK_IDX)
    best, best_key, candidates = None, None, 0
    for w_med in (1.0, 1.25, 1.5):
        for w_high in (1.0, 1.25, 1.5, 2.0, 2.5):
            for w_crit in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
                w = [1.0, w_med, w_high, w_crit]
                pred = apply_decision_weights(proba, w)
                f1 = f1_score(y_true, pred, average="macro", zero_division=0)
                hr_rec = recall_score(hr_true, np.isin(pred, HIGH_RISK_IDX), zero_division=0)
                crit_rec = recall_score(y_true == 3, pred == 3, zero_division=0)
                feasible = hr_rec >= min_high_risk_recall and crit_rec >= min_critical_recall
                candidates += feasible
                key = (feasible, f1 if feasible else hr_rec + crit_rec)
                if best_key is None or key > best_key:
                    best_key = key
                    best = {"weights": w, "f1_macro": float(f1), "high_risk_recall": float(hr_rec),
                            "critical_recall": float(crit_rec), "feasible": bool(feasible)}
    best["n_feasible"] = int(candidates)
    best["constraints"] = {"min_high_risk_recall": min_high_risk_recall, "min_critical_recall": min_critical_recall}
    return best


def tune_escalation_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float = 0.95) -> dict:
    """Pick the highest threshold on P(HIGH)+P(CRITICAL) whose high-risk recall >= min_recall.

    Highest qualifying threshold = fewest documents sent to mandatory review while still
    catching the required share of genuinely high-risk contracts. Tuned on out-of-fold
    predictions only (never on the test set).
    """
    hr_true = np.isin(np.asarray(y_true), HIGH_RISK_IDX).astype(int)
    score = high_risk_score(proba)
    curve = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (score >= t).astype(int)
        curve.append({
            "threshold": float(t),
            "recall": recall_score(hr_true, pred, zero_division=0),
            "precision": precision_score(hr_true, pred, zero_division=0),
            "review_rate": float(pred.mean()),
        })
    ok = [c for c in curve if c["recall"] >= min_recall]
    chosen = max(ok, key=lambda c: c["threshold"]) if ok else max(curve, key=lambda c: c["recall"])
    return {"min_recall_target": min_recall, "chosen": chosen, "curve": curve}
