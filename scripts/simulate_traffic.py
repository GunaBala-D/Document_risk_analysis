"""Simulate production traffic and reviewer feedback (SYNTHETIC) to demonstrate monitoring,
drift detection and the retraining loop end-to-end through the real API.

Subcommands
-----------
upload    Generate a batch of contracts for known vendors, upload + analyze them via the API.
          --profile baseline : same distribution as training
          --profile drift    : DATA DRIFT - contract values x2.2, payment terms +30 days,
                               1.6x more risky clauses (e.g. a new, more aggressive supplier market)
feedback  Reviewers label a batch; a risk manager validates (four-eyes) the feedback.
          --policy current   : labels follow the policy the model was trained on
          --policy shifted   : CONCEPT DRIFT - a new treasury policy: payment terms above 45 days
                               are now a violation (+2) and auto-renewal above 12 months counts (+1).
                               Same inputs, different correct answer -> P(y|X) changed.

Every batch writes a manifest (ground truth + labels) to data/demo/batches/<batch>/.

Examples:
    python -m scripts.simulate_traffic upload --profile baseline --n 40
    python -m scripts.simulate_traffic upload --profile drift --n 60
    python -m scripts.simulate_traffic feedback --batch drift-<id> --policy shifted
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
from datetime import date, timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.config import ARTIFACTS_DIR, DATA_DIR, load_yaml_config
from src.synthetic.generator import (
    ContractSpec,
    VendorProfile,
    render_contract_text,
    sample_contract,
)
from src.synthetic.labeling import label_contracts
from src.synthetic.render import render_document

BATCHES = DATA_DIR / "demo" / "batches"
PROFILES = {
    "baseline": {"drift": {}, "prob_scale": 1.0},
    "drift": {"drift": {"value_multiplier": 2.2, "payment_days_shift": 30}, "prob_scale": 1.6},
}


def shifted_policy() -> dict:
    policy = copy.deepcopy(load_yaml_config("labeling_policy"))
    policy["version"] = "lp-v2-simulated-concept-shift"
    policy["thresholds"]["payment_days_policy"] = 45
    policy["points"]["payment_days_over_policy"] = 2
    policy["thresholds"]["renewal_months_max"] = 12
    return policy


def api_keys() -> dict:
    return json.loads((ARTIFACTS_DIR / "demo_api_keys.json").read_text(encoding="utf-8"))


def client() -> TestClient:
    from api.main import create_app

    return TestClient(create_app())


def cmd_upload(args) -> None:
    rng = np.random.default_rng(args.seed)
    prof = PROFILES[args.profile]
    vendors_df = pd.read_csv(DATA_DIR / "synthetic" / "vendors.csv")
    vendors = [VendorProfile(r.vendor_id, r.name, r.category, float(r.typical_value_inr), int(r.typical_payment_days),
                             bool(r.risky), float(r.activity)) for r in vendors_df.itertuples()]
    w = np.array([v.activity for v in vendors])
    w /= w.sum()

    batch = f"{args.profile}-{secrets.token_hex(3)}"
    out = BATCHES / batch
    out.mkdir(parents=True)
    today = date.today()
    specs = []
    for i in range(args.n):
        v = vendors[int(rng.choice(len(vendors), p=w))]
        d = today - timedelta(days=int(rng.integers(0, args.spread_days)))
        spec = sample_contract(f"{batch}-{i:03d}", v, d, rng, seq=50_000 + i, prob_scale=prof["prob_scale"],
                               drift=prof["drift"])
        spec.render_format = str(rng.choice(["pdf", "pdf", "pdf", "docx"]))
        specs.append(spec)

    # Labels under both policies, with the full point-in-time vendor history
    history = [ContractSpec.from_record(r) for r in
               pd.read_csv(DATA_DIR / "synthetic" / "ground_truth.csv").to_dict(orient="records")]
    current = {r["doc_id"]: r for r in label_contracts(history + specs)}
    shifted = {r["doc_id"]: r for r in label_contracts(history + specs, shifted_policy())}

    rows = []
    for spec in specs:
        path = render_document(render_contract_text(spec, rng), spec.render_format, out, spec.doc_id)
        rows.append({**spec.to_record(), "file_name": path.name,
                     "label_current_policy": current[spec.doc_id]["risk_label"],
                     "label_shifted_policy": shifted[spec.doc_id]["risk_label"]})
    manifest = pd.DataFrame(rows)

    key = {"X-API-Key": api_keys()["asha.compliance"]["api_key"]}
    doc_ids, levels = [], []
    with client() as c:
        for r in manifest.itertuples():
            data = (out / r.file_name).read_bytes()
            up = c.post("/documents/upload", headers=key, files={"file": (r.file_name, data)})
            if up.status_code != 201:
                raise SystemExit(f"Upload failed for {r.file_name}: {up.status_code} {up.text}")
            doc_id = up.json()["document_id"]
            res = c.post(f"/documents/{doc_id}/analyze", headers=key,
                         params={"explain": "false", "as_of": r.submission_date})
            if res.status_code != 200:
                raise SystemExit(f"Analyze failed for {r.file_name}: {res.status_code} {res.text}")
            doc_ids.append(doc_id)
            levels.append(res.json().get("risk_level"))
    manifest["document_id"], manifest["system_risk_level"] = doc_ids, levels
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"Batch {batch}: uploaded and analyzed {len(manifest)} contracts")
    print("System risk levels:", pd.Series(levels).value_counts().to_dict())
    print("Labels (current policy):", manifest.label_current_policy.value_counts().to_dict())
    print("Labels (shifted policy):", manifest.label_shifted_policy.value_counts().to_dict())


def cmd_feedback(args) -> None:
    manifest = pd.read_csv(BATCHES / args.batch / "manifest.csv")
    if args.n:
        manifest = manifest.head(args.n)
    keys = api_keys()
    reviewer = {"X-API-Key": keys["asha.compliance"]["api_key"]}
    validator = {"X-API-Key": keys["rahul.risk"]["api_key"]}
    label_col = "label_shifted_policy" if args.policy == "shifted" else "label_current_policy"
    stats = {"confirm": 0, "correct": 0, "validated": 0, "skipped": 0}
    with client() as c:
        for r in manifest.itertuples():
            risk = c.get(f"/documents/{r.document_id}/risk", headers=reviewer)
            if risk.status_code != 200:
                stats["skipped"] += 1
                continue
            truth = getattr(r, label_col)
            if truth == risk.json()["risk_level"]:
                body = {"decision": "confirm", "comment": "Reviewed; agree with the assessment."}
            else:
                body = {"decision": "correct", "corrected_label": truth,
                        "comment": f"Reviewer assessment under {args.policy} policy: {truth}."}
            fb = c.post(f"/documents/{r.document_id}/feedback", headers=reviewer, json=body)
            if fb.status_code != 201:
                stats["skipped"] += 1
                continue
            stats[body["decision"]] += 1
            v = c.post(f"/feedback/{fb.json()['id']}/validate", headers=validator,
                       json={"approve": True, "comment": "Validated by risk manager (simulation)."})
            stats["validated"] += v.status_code == 200
    print(f"Feedback for batch {args.batch} under '{args.policy}' policy:", stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--profile", choices=list(PROFILES), default="baseline")
    up.add_argument("--n", type=int, default=40)
    up.add_argument("--spread-days", type=int, default=20)
    up.add_argument("--seed", type=int, default=None)
    fb = sub.add_parser("feedback")
    fb.add_argument("--batch", required=True)
    fb.add_argument("--policy", choices=["current", "shifted"], default="current")
    fb.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cmd_upload(args) if args.cmd == "upload" else cmd_feedback(args)


if __name__ == "__main__":
    main()
