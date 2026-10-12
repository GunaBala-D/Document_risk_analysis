"""Test helpers: train small but real models in-process (no MLflow, no dataset files needed),
so ML/API/integration tests run in CI from a clean checkout."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from src.anomaly.detector import AnomalyDetector, anomaly_frame, build_isolation_forest, calibration_quantiles
from src.compliance.engine import ComplianceEngine, count_policy_violations
from src.config import load_yaml_config
from src.explainability.shap_explainer import RiskExplainer
from src.extraction.entities import extract_entities, normalize_vendor_key
from src.features.builder import (
    VendorContract,
    build_facts,
    build_features,
    contract_value_inr,
    load_registry,
)
from src.models.evaluation import CLASSES, compute_metrics
from src.models.risk_model import RiskModelBundle, make_pipeline
from src.pipeline.model_service import ModelService
from src.preprocessing.cleaning import clean_text
from src.preprocessing.sections import detect_sections
from src.synthetic.generator import generate_contract_specs, render_contract_text
from src.synthetic.labeling import label_contracts


@lru_cache(maxsize=2)
def synthetic_feature_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    specs, _ = generate_contract_specs(n, 40, seed=seed, n_splitting_events=3)
    labels = {r["doc_id"]: r["risk_label"] for r in label_contracts(specs)}
    rng = np.random.default_rng(seed)
    registry, engine = load_registry(), ComplianceEngine()
    history: dict[str, list[VendorContract]] = defaultdict(list)
    rows, ys = [], []
    for s in sorted(specs, key=lambda s: (s.submission_date, s.doc_id)):
        text = clean_text(render_contract_text(s, rng))
        sections = detect_sections(text)
        ents = extract_entities(text, sections)
        vkey = normalize_vendor_key(ents["vendor_name"].value) if "vendor_name" in ents else None
        feats = build_features(text, sections, ents, s.submission_date, history[vkey], registry)
        n_viol = count_policy_violations(engine.evaluate(build_facts(feats, ents, sections)))
        pay = feats["payment_days"]
        history[vkey].append(VendorContract(s.submission_date, contract_value_inr(ents, registry.config["fx_to_inr"]),
                                            None if np.isnan(pay) else pay, n_viol,
                                            labels[s.doc_id] in ("HIGH", "CRITICAL")))
        rows.append(feats)
        ys.append(CLASSES.index(labels[s.doc_id]))
    return pd.DataFrame(rows)[registry.names], np.array(ys)


@lru_cache(maxsize=1)
def quick_model_service() -> ModelService:
    registry = load_registry()
    X, y = synthetic_feature_frame()
    pipe = make_pipeline("random_forest", "none", params={"n_estimators": 60}, random_state=0)
    pipe.fit(X, y)
    bundle = RiskModelBundle(pipe, registry.names, registry.version, escalation_threshold=0.3,
                             model_name="document-risk-classifier", model_version="test-1", family="random_forest")
    explainer = RiskExplainer(bundle, feature_labels=registry.labels)

    cfg = load_yaml_config("anomaly")
    feats = cfg["isolation_forest"]["features"]
    iforest = build_isolation_forest(cfg)
    A = anomaly_frame(X)[feats]
    iforest.fit(A)
    detector = AnomalyDetector(iforest, feats, calibration_quantiles(-iforest.score_samples(A)),
                               cfg["isolation_forest"]["flag_percentile"], model_version="test-1", config=cfg)
    proba = pipe.predict_proba(X)
    meta = {"family": "random_forest", "feature_names": registry.names, "feature_version": registry.version,
            "test_metrics": compute_metrics(y, proba),
            "reference_prediction_distribution":
                pd.Series(proba.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(normalize=True).to_dict()}
    return ModelService(classifier=bundle, explainer=explainer, anomaly=detector, classifier_metadata=meta,
                        anomaly_metadata={"features": feats})


def fake_api_key(name: str) -> str:
    return "test_" + hashlib.sha256(name.encode()).hexdigest()[:32]
