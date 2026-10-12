"""Create a small set of SYNTHETIC demo documents dated relative to today.

They use vendors that exist in the seeded vendor history, so vendor-behaviour features and
anomaly checks have real history to compare against. Outputs go to data/demo/.

    python -m scripts.make_demo_documents
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.synthetic.generator import (
    VendorProfile,
    render_contract_text,
    render_invoice_text,
    sample_contract,
)
from src.synthetic.render import render_document, write_pdf

OUT = DATA_DIR / "demo"

SCENARIOS = [
    ("01_standard_contract", "pdf", [], 1.0),
    ("02_missing_termination_long_payment", "pdf", ["missing_termination", "excessive_payment"], 1.0),
    ("03_unlimited_liability_foreign_law", "docx",
     ["unlimited_liability", "foreign_jurisdiction", "high_penalty", "missing_confidentiality", "one_sided_terms"], 1.0),
    ("04_unusual_value_for_vendor", "pdf", [], 6.0),
    ("05_scanned_contract_short_expiry", "png", ["short_expiry", "long_renewal"], 1.0),
]


def pick_vendors(n: int) -> list[VendorProfile]:
    vendors = pd.read_csv(DATA_DIR / "synthetic" / "vendors.csv")
    hist = pd.read_parquet(DATA_DIR / "processed" / "vendor_history.parquet")
    counts = hist.groupby("vendor_name").size()
    # established, mostly-clean vendors with >= 6 contracts of history
    clean = hist.groupby("vendor_name")["violation_count"].apply(lambda s: (s > 0).mean())
    names = [v for v in counts[counts >= 6].index if clean.get(v, 1) < 0.3][:n]
    rows = vendors.set_index("name").loc[names].reset_index()
    return [VendorProfile(r.vendor_id, r.name, r.category, float(r.typical_value_inr), int(r.typical_payment_days),
                          False, float(r.activity)) for r in rows.itertuples()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("0*_*.*"):  # only our own demo files - never the simulation batches
        old.unlink()
    rng = np.random.default_rng(2026)
    vendors = pick_vendors(len(SCENARIOS))
    hist = pd.read_parquet(DATA_DIR / "processed" / "vendor_history.parquet")
    today = date.today()
    for i, ((stem, fmt, forced, value_mult), vendor) in enumerate(zip(SCENARIOS, vendors, strict=True), start=1):
        value = None
        if value_mult != 1.0:
            median = hist[hist.vendor_name == vendor.name].value_inr.median()
            value = float(median * value_mult)
        spec = sample_contract(f"DEMO-{i}", vendor, today, rng, seq=9000 + i, forced=forced, value_override_inr=value)
        spec.render_format = fmt
        path = render_document(render_contract_text(spec, rng), fmt, OUT, stem)
        print(f"{path.name:45s} vendor={vendor.name}  scenarios={forced or ['none']}")
    write_pdf(render_invoice_text(rng, vendors[0], 77), OUT / "06_invoice_not_supported_in_v1.pdf")
    print("06_invoice_not_supported_in_v1.pdf")


if __name__ == "__main__":
    main()
