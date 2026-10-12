"""Shared training utilities: dataset loading, the frozen split, CV and plotting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from src.config import DATA_DIR  # noqa: E402
from src.features.builder import load_registry  # noqa: E402
from src.models.evaluation import CLASSES  # noqa: E402
from src.models.risk_model import fit_sample_weight  # noqa: E402

PROCESSED = DATA_DIR / "processed"
SPLIT_PATH = PROCESSED / "split.json"


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    path = path or PROCESSED / f"dataset_{load_registry().version}.parquet"
    df = pd.read_parquet(path)
    meta = json.loads((PROCESSED / "dataset_meta.json").read_text(encoding="utf-8"))
    names = load_registry().names
    X = df[names].astype(float)
    y = df["risk_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return df, X, y, meta


def frozen_split(df: pd.DataFrame, y: np.ndarray, dataset_version: str, test_size: float, seed: int):
    """Stratified hold-out split, persisted so every future candidate uses the same test set."""
    if SPLIT_PATH.exists():
        saved = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if saved["dataset_version"] == dataset_version:
            test_ids = set(saved["test_doc_ids"])
            is_test = df["doc_id"].isin(test_ids).to_numpy()
            return np.where(~is_test)[0], np.where(is_test)[0]
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    SPLIT_PATH.write_text(json.dumps({
        "dataset_version": dataset_version, "seed": seed, "test_size": test_size,
        "test_doc_ids": sorted(df.iloc[test_idx]["doc_id"].tolist()),
    }, indent=1), encoding="utf-8")
    return np.sort(train_idx), np.sort(test_idx)


def oof_predict(pipe, family: str, imbalance: str, X: pd.DataFrame, y: np.ndarray, folds: int,
                seed: int) -> np.ndarray:
    """Out-of-fold class probabilities from StratifiedKFold."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), len(CLASSES)))
    for tr, va in skf.split(X, y):
        model = clone(pipe)
        w = fit_sample_weight(family, imbalance, y[tr])
        if w is not None:
            model.fit(X.iloc[tr], y[tr], clf__sample_weight=w)
        else:
            model.fit(X.iloc[tr], y[tr])
        proba[va] = model.predict_proba(X.iloc[va])
    return proba


def plot_confusion(cm: list[list[int]], title: str, path: Path) -> None:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="#fcfcfb")
    ax.imshow(cm_arr, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"]))
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9, color="#52514e")
    ax.set_ylabel("Actual", fontsize=9, color="#52514e")
    thresh = cm_arr.max() * 0.55
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=10,
                    color="#ffffff" if cm_arr[i, j] > thresh else "#0b0b0b")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(rows: list[dict], title: str, path: Path, top: int = 15) -> None:
    rows = rows[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(rows) + 1.2), facecolor="#fcfcfb")
    ax.barh([r["label"] for r in rows], [r["mean_abs_shap"] for r in rows], color="#2a78d6", height=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#898781", labelsize=9)
    ax.grid(axis="x", color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.set_xlabel("mean |SHAP value| (averaged over classes)", fontsize=9, color="#52514e")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def flat_metrics(metrics: dict, prefix: str = "") -> dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, int | float)}
