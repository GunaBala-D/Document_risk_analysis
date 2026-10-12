"""Generate the SYNTHETIC vendor-contract corpus (+ a few invoices/POs for doc-type tests).

Usage:
    python -m scripts.generate_dataset --n-contracts 1500 --seed 42

Outputs (data/synthetic/):
    contracts/<doc_id>.<pdf|docx|png|jpg>   rendered documents
    other/<id>.pdf                          invoices & purchase orders (doc-type detection only)
    ground_truth.csv                        true attributes of every contract
    labels.csv                              rule-based risk labels + reasons (labeling policy)
    vendors.csv                             synthetic vendor master
    manifest.json                           generation parameters + dataset version hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.synthetic.generator import (
    generate_contract_specs,
    render_contract_text,
    render_invoice_text,
    render_po_text,
)
from src.synthetic.labeling import label_contracts
from src.synthetic.render import render_document, write_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-contracts", type=int, default=1500)
    ap.add_argument("--n-vendors", type=int, default=150)
    ap.add_argument("--n-other", type=int, default=40, help="invoices and POs each")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "synthetic")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "contracts").mkdir(parents=True)
    (out / "other").mkdir(parents=True)

    specs, vendors = generate_contract_specs(args.n_contracts, args.n_vendors, seed=args.seed,
                                             n_splitting_events=max(4, args.n_contracts // 125))
    labels = pd.DataFrame(label_contracts(specs))
    text_rng = np.random.default_rng(args.seed + 1)

    records = []
    for spec in specs:
        text = render_contract_text(spec, text_rng)
        path = render_document(text, spec.render_format, out / "contracts", spec.doc_id)
        rec = spec.to_record()
        rec["file_name"] = path.name
        records.append(rec)

    for i in range(args.n_other):
        v = vendors[i % len(vendors)]
        write_pdf(render_invoice_text(text_rng, v, i + 1), out / "other" / f"INV-{i + 1:04d}.pdf")
        write_pdf(render_po_text(text_rng, v, i + 1), out / "other" / f"PO-{i + 1:04d}.pdf")

    gt = pd.DataFrame(records)
    gt.to_csv(out / "ground_truth.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    pd.DataFrame([v.__dict__ for v in vendors]).to_csv(out / "vendors.csv", index=False)

    digest = hashlib.sha256(
        (gt.to_csv(index=False) + labels.to_csv(index=False)).encode("utf-8")
    ).hexdigest()[:12]
    manifest = {
        "dataset": "synthetic-vendor-contracts",
        "synthetic": True,
        "dataset_version": f"syn-{digest}",
        "generated_at": datetime.now(UTC).isoformat(),
        "params": {k: str(v) for k, v in vars(args).items()},
        "n_contracts": len(gt),
        "label_distribution": labels["risk_label"].value_counts().to_dict(),
        "format_distribution": gt["render_format"].value_counts().to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
