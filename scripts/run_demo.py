"""Reproducible end-to-end MLOps demo (SYNTHETIC data, real API, real models).

Steps:
  1. --reset: clear operational data (users are kept) and uploaded files
  2. seed point-in-time vendor history
  3. upload + analyse the hand-made demo contracts (data/demo/)
  4. baseline traffic (40 contracts)            -> monitoring snapshot (expect: no drift)
  5. drifted traffic (60 contracts)             -> monitoring snapshot (expect: feature + prediction drift,
                                                   but NO retraining: there are no new labels)
  6. reviewer feedback under a changed policy   -> monitoring snapshot (expect: performance degraded ->
     (concept drift), validated by a risk manager   retraining recommended)
Then run `python -m training.retrain` to see the candidate go through the quality gate.

Usage:
    python -m scripts.run_demo --reset
"""

from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace

from fastapi.testclient import TestClient
from sqlalchemy import delete

from database.models import (
    AnomalyRecord,
    AuditLog,
    ComplianceViolation,
    Document,
    DocumentEntity,
    DocumentFeatures,
    ModelVersion,
    MonitoringReport,
    Prediction,
    RequestMetric,
    ReviewFeedback,
    RiskFactor,
    VendorContractRecord,
)
from database.session import get_sessionmaker
from scripts import make_demo_documents, monitoring_snapshot, simulate_traffic
from scripts.init_db import migrate, seed_users, seed_vendor_history
from src.config import ARTIFACTS_DIR, DATA_DIR, get_settings


def reset() -> None:
    with get_sessionmaker()() as s:
        for model in (ReviewFeedback, RiskFactor, AnomalyRecord, ComplianceViolation, Prediction, DocumentFeatures,
                      DocumentEntity, VendorContractRecord, AuditLog, Document, RequestMetric, MonitoringReport,
                      ModelVersion):
            s.execute(delete(model))
        s.commit()
    shutil.rmtree(get_settings().upload_dir, ignore_errors=True)
    shutil.rmtree(DATA_DIR / "demo" / "batches", ignore_errors=True)
    print("Operational data cleared (users kept).")


def upload_demo_documents() -> None:
    from api.main import create_app

    key = {"X-API-Key": json.loads((ARTIFACTS_DIR / "demo_api_keys.json").read_text())["asha.compliance"]["api_key"]}
    with TestClient(create_app()) as c:
        for f in sorted((DATA_DIR / "demo").glob("0*")):
            doc_id = c.post("/documents/upload", headers=key, files={"file": (f.name, f.read_bytes())}).json()["document_id"]
            r = c.post(f"/documents/{doc_id}/analyze", headers=key, params={"explain": "false"}).json()
            print(f"  {f.name:42s} -> {r.get('risk_level', r.get('processing_status'))} {r.get('risk_score', '')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    migrate()
    with get_sessionmaker()() as s:
        seed_users(s)
    if args.reset:
        reset()
    with get_sessionmaker()() as s:
        seed_vendor_history(s)

    print("\n== Demo contracts")
    make_demo_documents.main()
    upload_demo_documents()

    print("\n== Baseline traffic")
    simulate_traffic.cmd_upload(Namespace(profile="baseline", n=40, spread_days=20, seed=11))
    monitoring_snapshot.main()

    print("\n== Drifted traffic (data drift)")
    simulate_traffic.cmd_upload(Namespace(profile="drift", n=60, spread_days=20, seed=12))
    monitoring_snapshot.main()

    batch = sorted(p.name for p in (DATA_DIR / "demo" / "batches").glob("drift-*"))[-1]
    print(f"\n== Reviewer feedback under a changed policy (concept drift) on {batch}")
    simulate_traffic.cmd_feedback(Namespace(batch=batch, policy="shifted", n=0))
    monitoring_snapshot.main()
    print("\nNext: python -m training.retrain   (and: python -m training.retrain --force --simulate-bad-candidate)")


if __name__ == "__main__":
    main()
