"""Initialise the operational database.

  1. alembic upgrade head
  2. create one demo user per role (API keys from SEED_*_KEY env vars, or generated)
  3. seed point-in-time vendor history from the processed synthetic dataset
     (vendor features and vendor-behaviour anomaly checks need history to compare against)

Usage:
    python -m scripts.init_db [--skip-vendor-history]
Generated API keys are written to artifacts/demo_api_keys.json (git-ignored), never printed.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from api.auth import generate_key, hash_key
from database.models import User, VendorContractRecord
from database.session import get_sessionmaker
from src.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT

DEMO_USERS = [
    ("admin", "Platform Administrator", "admin", "SEED_ADMIN_KEY"),
    ("asha.compliance", "Asha Rao (Compliance Analyst)", "compliance_analyst", "SEED_ANALYST_KEY"),
    ("vikram.finance", "Vikram Shah (Finance Analyst)", "finance_analyst", None),
    ("neha.procurement", "Neha Iyer (Procurement Analyst)", "procurement_analyst", None),
    ("rahul.risk", "Rahul Menon (Risk Manager)", "risk_manager", "SEED_RISK_MANAGER_KEY"),
]


def migrate() -> None:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
    command.upgrade(cfg, "head")


def seed_users(session) -> dict[str, str]:
    created = {}
    for username, name, role, env in DEMO_USERS:
        if session.scalar(select(User).where(User.username == username)):
            continue
        key = (os.environ.get(env) if env else None) or generate_key()
        session.add(User(username=username, full_name=name, role=role, api_key_hash=hash_key(key)))
        created[username] = {"role": role, "api_key": key}
    session.commit()
    if created:
        path = ARTIFACTS_DIR / "demo_api_keys.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**existing, **created}, indent=2), encoding="utf-8")
        print(f"Created users {list(created)}; API keys written to {path}")
    return created


def seed_vendor_history(session) -> None:
    if session.scalar(select(func.count()).select_from(VendorContractRecord)
                      .where(VendorContractRecord.source == "historical_seed")):
        print("Vendor history already seeded")
        return
    path = DATA_DIR / "processed" / "vendor_history.parquet"
    if not path.exists():
        print(f"{path} not found - run the dataset pipeline first; skipping vendor history")
        return
    df = pd.read_parquet(path)
    session.bulk_save_objects([
        VendorContractRecord(vendor_key=r.vendor_key, vendor_name=r.vendor_name,
                             contract_date=pd.Timestamp(r.contract_date).date(),
                             value_inr=None if pd.isna(r.value_inr) else float(r.value_inr),
                             payment_days=None if pd.isna(r.payment_days) else float(r.payment_days),
                             violation_count=int(r.violation_count), high_risk=bool(r.high_risk),
                             source="historical_seed")
        for r in df.itertuples()])
    session.commit()
    print(f"Seeded {len(df)} historical vendor contracts for {df.vendor_key.nunique()} vendors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vendor-history", action="store_true")
    args = ap.parse_args()
    migrate()
    with get_sessionmaker()() as session:
        seed_users(session)
        if not args.skip_vendor_history:
            seed_vendor_history(session)


if __name__ == "__main__":
    main()
