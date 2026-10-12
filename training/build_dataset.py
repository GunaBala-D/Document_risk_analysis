"""Build the model-ready dataset by running every synthetic document through the real
production pipeline (validation -> extraction/OCR -> cleaning -> sections -> entities).

Then, in time order, compute point-in-time vendor features and compliance results, and
measure extraction quality against ground truth.

Usage:
    python -m training.build_dataset [--workers 6]

Outputs (data/processed/):
    dataset_<feature_version>.parquet  features + label + metadata, one row per contract
    vendor_history.parquet      per-contract vendor history rows (seed for the serving DB)
    extraction_eval.json        per-field extraction accuracy vs ground truth
    doc_type_eval.json          document-type detection accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from src.compliance.engine import ComplianceEngine, count_policy_violations
from src.config import DATA_DIR, get_settings
from src.extraction.entities import normalize_vendor_key
from src.features.builder import (
    VendorContract,
    build_facts,
    build_features,
    contract_value_inr,
    load_registry,
)
from src.pipeline.document_processor import process_document

SYN = DATA_DIR / "synthetic"
OUT = DATA_DIR / "processed"


def _process_file(path_str: str) -> dict:
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    settings = get_settings()
    path = Path(path_str)
    doc = process_document(path.name, path.read_bytes(), max_bytes=50 * 1024 * 1024,
                           tesseract_cmd=settings.tesseract_cmd or None)
    return {
        "file_name": path.name,
        "ok": doc.ok,
        "error": doc.extraction.error,
        "method": doc.extraction.method,
        "page_count": doc.extraction.page_count,
        "ocr_pages": len(doc.extraction.ocr_pages),
        "doc_type": doc.doc_type.doc_type if doc.doc_type else None,
        "text": doc.text,
        "sections": doc.sections,
        "entities": doc.entities,
        "timings": doc.timings_ms,
    }


def _close(a, b, rel=0.005) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= rel * max(abs(float(b)), 1e-9)


def _evaluate_extraction(rows: list[dict], gt: pd.DataFrame) -> dict:
    gt = gt.set_index("file_name")
    checks = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # field -> method -> [correct, total]

    def check(field, method, ok):
        checks[field][method][0] += int(ok)
        checks[field][method][1] += 1
        checks[field]["all"][0] += int(ok)
        checks[field]["all"][1] += 1

    for r in rows:
        if r["file_name"] not in gt.index:
            continue
        t, ents, m = gt.loc[r["file_name"]], r["entities"], r["method"]
        val = lambda k, ents=ents: ents[k].value if k in ents else None  # noqa: E731
        present = {s.name for s in r["sections"]}
        check("vendor_name", m, val("vendor_name") == t.vendor_name)
        check("customer_name", m, val("customer_name") == t.customer_name)
        check("contract_number", m, val("contract_number") == t.contract_number)
        check("start_date", m, val("start_date") == date.fromisoformat(t.start_date))
        check("end_date", m, val("end_date") == date.fromisoformat(t.end_date))
        check("contract_value", m, _close(val("contract_value"), t.contract_value))
        check("currency", m, val("currency") == t.currency)
        check("payment_days", m, val("payment_days") == t.payment_days)
        tn = None if pd.isna(t.termination_notice_days) else int(t.termination_notice_days)
        check("termination_notice_days", m, val("termination_notice_days") == tn)
        check("penalty_pct", m, val("penalty_pct") is not None and abs(val("penalty_pct") - t.penalty_pct) < 0.05)
        check("renewal_months", m, val("renewal_months") == t.renewal_months)
        check("governing_law", m, val("governing_law") == t.governing_law)
        check("termination_clause_presence", m, ("termination" in present) == bool(t.has_termination))
        check("liability_clause_presence", m, ("liability" in present) == bool(t.has_liability))
        check("confidentiality_clause_presence", m, ("confidentiality" in present) == bool(t.has_confidentiality))
        check("compliance_clause_presence", m, ("compliance" in present) == bool(t.has_compliance_clause))
        lc = val("liability_cap")
        check("unlimited_liability", m, (lc == "unlimited") == bool(t.unlimited_liability))

    return {
        f: {m: {"accuracy": round(c / n, 4), "n": n} for m, (c, n) in by_m.items()}
        for f, by_m in checks.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(SYN / "ground_truth.csv")
    labels = pd.read_csv(SYN / "labels.csv")
    manifest = json.loads((SYN / "manifest.json").read_text(encoding="utf-8"))
    contract_files = [str(SYN / "contracts" / f) for f in gt["file_name"]]
    other_files = sorted(str(p) for p in (SYN / "other").glob("*.pdf"))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_process_file, contract_files + other_files, chunksize=8))
    elapsed = time.perf_counter() - t0
    print(f"Processed {len(rows)} documents in {elapsed:.1f}s with {args.workers} workers")

    # ---- document-type detection evaluation ----
    expected = {}
    for r in rows:
        name = r["file_name"]
        expected[name] = "invoice" if name.startswith("INV-") else "purchase_order" if name.startswith("PO-") \
            else "vendor_contract"
    dt = pd.DataFrame([{"expected": expected[r["file_name"]], "predicted": r["doc_type"]} for r in rows])
    doc_type_eval = {
        "accuracy": round(float((dt.expected == dt.predicted).mean()), 4),
        "confusion": pd.crosstab(dt.expected, dt.predicted).to_dict(),
        "n": len(dt),
    }
    (OUT / "doc_type_eval.json").write_text(json.dumps(doc_type_eval, indent=2), encoding="utf-8")

    contract_rows = [r for r in rows if expected[r["file_name"]] == "vendor_contract"]
    failed = [r for r in contract_rows if not r["ok"] or r["doc_type"] != "vendor_contract"]
    print(f"Contracts failing extraction or doc-type: {len(failed)}")

    extraction_eval = _evaluate_extraction([r for r in contract_rows if r["ok"]], gt)
    extraction_eval["_meta"] = {
        "n_contracts": len(contract_rows), "n_failed": len(failed),
        "method_counts": pd.Series([r["method"] for r in contract_rows]).value_counts().to_dict(),
        "processing_seconds": round(elapsed, 1), "workers": args.workers,
        "mean_timings_ms": pd.DataFrame([r["timings"] for r in contract_rows]).mean().round(2).to_dict(),
    }
    (OUT / "extraction_eval.json").write_text(json.dumps(extraction_eval, indent=2), encoding="utf-8")

    # ---- point-in-time features (time order) ----
    registry = load_registry()
    engine = ComplianceEngine()
    meta = gt.merge(labels, on="doc_id").set_index("file_name")
    by_name = {r["file_name"]: r for r in contract_rows if r["ok"] and r["doc_type"] == "vendor_contract"}
    ordered = meta.loc[[f for f in meta.index if f in by_name]].sort_values(["submission_date", "doc_id"])

    history: dict[str, list[VendorContract]] = defaultdict(list)
    records, vendor_rows = [], []
    for file_name, m in ordered.iterrows():
        r = by_name[file_name]
        as_of = date.fromisoformat(m.submission_date)
        vkey = normalize_vendor_key(r["entities"]["vendor_name"].value) if "vendor_name" in r["entities"] else None
        feats = build_features(r["text"], r["sections"], r["entities"], as_of, history.get(vkey, []), registry)
        facts = build_facts(feats, r["entities"], r["sections"], registry.config["fx_to_inr"])
        results = engine.evaluate(facts)
        n_viol = count_policy_violations(results)
        value_inr = contract_value_inr(r["entities"], registry.config["fx_to_inr"])
        pay = feats["payment_days"]
        high_risk = m.risk_label in ("HIGH", "CRITICAL")

        records.append({
            "doc_id": m.doc_id, "file_name": file_name, "submission_date": m.submission_date,
            "vendor_key": vkey, "vendor_id": m.vendor_id, "extraction_method": r["method"],
            **feats,
            "compliance_violations": "|".join(x.rule_id for x in results),
            "n_policy_violations": n_viol,
            "risk_label": m.risk_label, "risk_points": m.risk_points, "label_reasons": m.label_reasons,
        })
        if vkey:
            hc = VendorContract(as_of, value_inr, None if math.isnan(pay) else pay, n_viol, high_risk)
            history[vkey].append(hc)
            vendor_rows.append({
                "vendor_key": vkey, "vendor_name": r["entities"]["vendor_name"].value, "contract_date": as_of,
                "value_inr": value_inr, "payment_days": hc.payment_days, "violation_count": n_viol,
                "high_risk": high_risk, "source_doc_id": m.doc_id,
            })

    ds = pd.DataFrame(records)
    ds.attrs["dataset_version"] = manifest["dataset_version"]
    ds.to_parquet(OUT / f"dataset_{registry.version}.parquet", index=False)
    pd.DataFrame(vendor_rows).to_parquet(OUT / "vendor_history.parquet", index=False)
    (OUT / "dataset_meta.json").write_text(json.dumps({
        "dataset_version": manifest["dataset_version"], "feature_version": registry.version,
        "rows": len(ds), "features": registry.names,
        "label_distribution": ds["risk_label"].value_counts().to_dict(),
    }, indent=2), encoding="utf-8")

    print(f"Dataset rows: {len(ds)}  label distribution: {ds['risk_label'].value_counts().to_dict()}")
    print("Doc-type accuracy:", doc_type_eval["accuracy"])
    for f, v in extraction_eval.items():
        if f != "_meta":
            print(f"  {f:34s} {v['all']['accuracy']:.3f}  (n={v['all']['n']})")


if __name__ == "__main__":
    main()
