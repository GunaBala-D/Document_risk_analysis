"""Synthetic enterprise vendor-contract generator.

ALL DATA PRODUCED HERE IS SYNTHETIC. Vendor and customer names are fictional combinations.
Nothing here represents real enterprise documents or real enterprise performance.

Design:
  * A vendor universe with category-specific spend profiles and a small set of "risky"
    vendors whose contracts are more likely to contain problematic terms. This gives the
    vendor-history features a genuine (not leaked) signal.
  * Contracts are generated in time order so vendor history can be computed point-in-time.
  * Risk scenarios (missing clauses, excessive payment terms, high penalties, unusual
    values, short expiry, long auto-renewal, foreign jurisdiction, contract splitting) are
    injected with configurable probabilities and recorded as ground truth.
  * The rendered text uses several heading styles and phrasings per clause so that the
    extraction layer must be robust, and a share of documents is rendered as scanned
    images so the OCR path is exercised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta

FX_TO_INR = {"INR": 1.0, "USD": 83.0, "EUR": 90.0}

CATEGORIES: dict[str, tuple[float, int]] = {
    # category: (median contract value in INR, typical payment days)
    "IT Services": (4_000_000, 45),
    "Logistics": (2_500_000, 30),
    "Facility Management": (1_500_000, 30),
    "Raw Materials": (6_000_000, 60),
    "Consulting": (3_000_000, 45),
    "Marketing Services": (1_200_000, 30),
    "Packaging": (2_000_000, 45),
    "Engineering": (5_000_000, 60),
}
CATEGORY_SUFFIX = {
    "IT Services": ["Technologies", "Systems", "Infotech", "Digital Solutions"],
    "Logistics": ["Logistics", "Freight Movers", "Supply Chain"],
    "Facility Management": ["Facility Services", "Property Care", "Workplace Solutions"],
    "Raw Materials": ["Industrial Supplies", "Metals", "Polymers", "Chemicals"],
    "Consulting": ["Consulting", "Advisory", "Management Partners"],
    "Marketing Services": ["Marketing Services", "Media Works", "Brand Labs"],
    "Packaging": ["Packaging", "Containers", "Packworks"],
    "Engineering": ["Engineering", "Infra Projects", "Projects & Controls"],
}
NAME_PREFIX = [
    "Apex", "Bluestone", "Crescent", "Deccan", "Evergreen", "Falcon", "Granite", "Horizon",
    "Indus", "Jadeleaf", "Kaveri", "Lotus", "Meridian", "Novalink", "Orion", "Pinnacle",
    "Quantix", "Redwood", "Sapphire", "Trident", "Unity", "Vertex", "Westbridge", "Zenith",
    "Coastal", "Summit", "Harborline", "Silverline", "Northstar", "Brightpath", "Cedar",
    "Monsoon", "Saffron", "Tarang", "Vistara", "Aravali", "Nilgiri", "Godavari", "Kestrel",
]
LEGAL_SUFFIX = ["Pvt. Ltd.", "Private Limited", "LLP", "Ltd."]
CUSTOMERS = ["Kestrel Manufacturing Ltd.", "Arcadia Retail Pvt. Ltd.", "Bluewater Pharmaceuticals Ltd."]
INDIAN_CITIES = ["Mumbai", "Bengaluru", "Chennai", "New Delhi", "Pune", "Hyderabad", "Kolkata"]
FOREIGN_JURISDICTIONS = ["Singapore", "England and Wales", "the State of Delaware, USA"]

DAY_WORDS = {
    7: "seven", 10: "ten", 15: "fifteen", 30: "thirty", 45: "forty-five", 60: "sixty",
    75: "seventy-five", 90: "ninety", 120: "one hundred and twenty", 180: "one hundred and eighty",
}

# Base probability of each risk scenario for a normal vendor (risky vendors x RISKY_MULT, capped).
SCENARIO_PROBS = {
    "missing_termination": 0.04,
    "excessive_payment": 0.045,
    "high_penalty": 0.045,
    "unusual_value": 0.035,
    "short_expiry": 0.06,
    "long_renewal": 0.035,
    "missing_liability": 0.03,
    "unlimited_liability": 0.025,
    "missing_confidentiality": 0.045,
    "foreign_jurisdiction": 0.035,
    "one_sided_terms": 0.045,
    "missing_compliance_clause": 0.08,
}
RISKY_MULT = 4.0
RISK_KEYWORD_PHRASES = [
    "at its sole discretion",
    "the advance payment shall be non-refundable",
    "the Vendor shall be the exclusive supplier",
    "this commitment is irrevocable",
    "the Customer hereby waives all claims",
]


@dataclass
class VendorProfile:
    vendor_id: str
    name: str
    category: str
    typical_value_inr: float
    typical_payment_days: int
    risky: bool
    activity: float


@dataclass
class ContractSpec:
    """Ground truth for one generated contract."""

    doc_id: str
    vendor_id: str
    vendor_name: str
    vendor_category: str
    vendor_typical_value_inr: float
    vendor_risky: bool
    customer_name: str
    contract_number: str
    submission_date: date
    signing_date: date
    start_date: date
    end_date: date
    term_months: int
    currency: str
    contract_value: float
    contract_value_inr: float
    payment_days: int
    has_termination: bool
    termination_notice_days: int | None
    has_liability: bool
    unlimited_liability: bool
    has_confidentiality: bool
    has_compliance_clause: bool
    jurisdiction: str
    governing_law: str
    penalty_pct: float
    penalty_as_amount: bool
    auto_renewal: bool
    renewal_months: int
    risk_keywords: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    render_format: str = "pdf"
    heading_style: int = 0

    @classmethod
    def from_record(cls, rec: dict) -> ContractSpec:
        """Inverse of to_record (e.g. rows of ground_truth.csv)."""
        import math
        from dataclasses import fields

        kw = {f.name: rec[f.name] for f in fields(cls) if f.name in rec}
        for key in ("submission_date", "signing_date", "start_date", "end_date"):
            kw[key] = date.fromisoformat(str(kw[key]))
        for key in ("risk_keywords", "scenarios"):
            raw = kw.get(key)
            kw[key] = [] if raw is None or (isinstance(raw, float) and math.isnan(raw)) else \
                [x for x in str(raw).split("|") if x]
        tn = kw.get("termination_notice_days")
        kw["termination_notice_days"] = None if tn is None or (isinstance(tn, float) and math.isnan(tn)) else int(tn)
        for key in ("term_months", "payment_days", "renewal_months", "heading_style"):
            kw[key] = int(kw[key])
        return cls(**kw)

    def to_record(self) -> dict:
        rec = asdict(self)
        for key in ("submission_date", "signing_date", "start_date", "end_date"):
            rec[key] = rec[key].isoformat()
        rec["risk_keywords"] = "|".join(self.risk_keywords)
        rec["scenarios"] = "|".join(self.scenarios)
        return rec


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def format_indian_grouping(amount: float) -> str:
    """18500000 -> '1,85,00,000' (Indian digit grouping)."""
    s = str(int(round(amount)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def money_text(value: float, currency: str, rng: np.random.Generator, allow_rupee_symbol: bool) -> str:
    if currency == "USD":
        return rng.choice([f"USD {value:,.0f}", f"US$ {value:,.2f}"])
    if currency == "EUR":
        return f"EUR {value:,.0f}"
    options = [f"INR {format_indian_grouping(value)}", f"Rs. {format_indian_grouping(value)}/-"]
    if value >= 10_000_000:
        options.append(f"INR {value / 10_000_000:.2f} Crore")
    elif value >= 100_000:
        options.append(f"Rs. {value / 100_000:.2f} Lakh")
    if allow_rupee_symbol:
        options.append(f"₹{format_indian_grouping(value)}")
    return str(rng.choice(options))


def date_text(d: date, rng: np.random.Generator) -> str:
    fmt = int(rng.integers(0, 4))
    if fmt == 0:
        return d.strftime("%d %B %Y").lstrip("0")
    if fmt == 1:
        return d.strftime("%B %d, %Y")
    if fmt == 2:
        return d.strftime("%d/%m/%Y")
    return d.strftime("%Y-%m-%d")


def days_text(n: int, rng: np.random.Generator) -> str:
    if n in DAY_WORDS and rng.random() < 0.6:
        return f"{DAY_WORDS[n]} ({n}) days"
    return f"{n} days"


# --------------------------------------------------------------------------------------
# Vendor universe & contract sampling
# --------------------------------------------------------------------------------------
def build_vendors(n_vendors: int, rng: np.random.Generator) -> list[VendorProfile]:
    vendors: list[VendorProfile] = []
    used: set[str] = set()
    cats = list(CATEGORIES)
    while len(vendors) < n_vendors:
        cat = cats[int(rng.integers(0, len(cats)))]
        name = (
            f"{rng.choice(NAME_PREFIX)} {rng.choice(CATEGORY_SUFFIX[cat])} {rng.choice(LEGAL_SUFFIX)}"
        )
        if name in used:
            continue
        used.add(name)
        median, pay = CATEGORIES[cat]
        rank = len(vendors) + 1
        vendors.append(
            VendorProfile(
                vendor_id=f"V{rank:04d}",
                name=name,
                category=cat,
                typical_value_inr=float(round(median * rng.lognormal(0, 0.5), -3)),
                typical_payment_days=int(rng.choice([pay, pay, pay, 30, 45])),
                risky=bool(rng.random() < 0.15),
                activity=1.0 / rank**0.7,
            )
        )
    return vendors


def _scenario_flags(vendor: VendorProfile, rng: np.random.Generator, prob_scale: float) -> list[str]:
    mult = RISKY_MULT if vendor.risky else 1.0
    flags = []
    for name, p in SCENARIO_PROBS.items():
        if rng.random() < min(0.5, p * mult * prob_scale):
            flags.append(name)
    if "missing_liability" in flags and "unlimited_liability" in flags:
        flags.remove("unlimited_liability")
    return flags


def sample_contract(
    doc_id: str,
    vendor: VendorProfile,
    submission: date,
    rng: np.random.Generator,
    seq: int,
    forced: list[str] | None = None,
    value_override_inr: float | None = None,
    prob_scale: float = 1.0,
    drift: dict | None = None,
) -> ContractSpec:
    drift = drift or {}
    scenarios = list(forced) if forced is not None else _scenario_flags(vendor, rng, prob_scale)

    term_months = int(rng.choice([12, 12, 24, 24, 36, 6]))
    signing = submission - timedelta(days=int(rng.integers(0, 8)))
    if "short_expiry" in scenarios:
        end = submission + timedelta(days=int(rng.integers(5, 45)))
        start = end - relativedelta(months=term_months) + timedelta(days=1)
    else:
        start = signing + timedelta(days=int(rng.integers(0, 31)))
        end = start + relativedelta(months=term_months) - timedelta(days=1)

    value_inr = vendor.typical_value_inr * rng.lognormal(0, 0.25) * drift.get("value_multiplier", 1.0)
    if "unusual_value" in scenarios:
        value_inr = vendor.typical_value_inr * rng.uniform(4.0, 8.0)
    if value_override_inr is not None:
        value_inr = value_override_inr

    currency = str(rng.choice(["INR"] * 18 + ["USD", "EUR"]))
    value = round(value_inr / FX_TO_INR[currency], -2 if currency == "INR" else 0)
    value_inr = value * FX_TO_INR[currency]

    if "excessive_payment" in scenarios:
        payment_days = int(rng.choice([90, 120, 120, 180]))
    else:
        payment_days = int(rng.choice([vendor.typical_payment_days] * 9 + [30, 45, 75]))
    payment_days = int(payment_days + drift.get("payment_days_shift", 0))

    has_termination = "missing_termination" not in scenarios
    notice = int(rng.choice([15, 30, 30, 60, 90])) if has_termination else None

    if "high_penalty" in scenarios:
        penalty_pct = float(rng.choice([12.0, 15.0, 20.0, 25.0, 30.0]))
    elif rng.random() < 0.35:
        penalty_pct = 0.0
    else:
        penalty_pct = float(rng.choice([2.0, 5.0, 5.0, 7.5, 10.0]))
    penalty_as_amount = penalty_pct > 0 and rng.random() < 0.3
    if penalty_as_amount:
        amount = round(value * penalty_pct / 100, -2 if currency == "INR" else 0)
        penalty_pct = round(amount / value * 100, 4)

    if "long_renewal" in scenarios:
        auto_renewal, renewal_months = True, int(rng.choice([36, 48, 60]))
    elif rng.random() < 0.45:
        auto_renewal, renewal_months = True, 12
    else:
        auto_renewal, renewal_months = False, 0

    if "foreign_jurisdiction" in scenarios:
        governing_law = str(rng.choice(FOREIGN_JURISDICTIONS))
        jurisdiction = governing_law
    else:
        governing_law = "India"
        jurisdiction = str(rng.choice(INDIAN_CITIES))

    keywords: list[str] = []
    if "one_sided_terms" in scenarios:
        idx = rng.choice(len(RISK_KEYWORD_PHRASES), size=int(rng.integers(2, 4)), replace=False)
        keywords = [RISK_KEYWORD_PHRASES[i] for i in sorted(idx)]
    elif rng.random() < 0.15:
        keywords = [RISK_KEYWORD_PHRASES[int(rng.integers(0, len(RISK_KEYWORD_PHRASES)))]]

    prefix = str(rng.choice(["VSA", "MSA", "SUP", "PSC"]))
    return ContractSpec(
        doc_id=doc_id,
        vendor_id=vendor.vendor_id,
        vendor_name=vendor.name,
        vendor_category=vendor.category,
        vendor_typical_value_inr=vendor.typical_value_inr,
        vendor_risky=vendor.risky,
        customer_name=str(rng.choice(CUSTOMERS)),
        contract_number=f"{prefix}-{submission.year}-{seq:05d}",
        submission_date=submission,
        signing_date=signing,
        start_date=start,
        end_date=end,
        term_months=term_months,
        currency=currency,
        contract_value=float(value),
        contract_value_inr=float(value_inr),
        payment_days=payment_days,
        has_termination=has_termination,
        termination_notice_days=notice,
        has_liability="missing_liability" not in scenarios,
        unlimited_liability="unlimited_liability" in scenarios,
        has_confidentiality="missing_confidentiality" not in scenarios,
        has_compliance_clause="missing_compliance_clause" not in scenarios,
        jurisdiction=jurisdiction,
        governing_law=governing_law,
        penalty_pct=float(penalty_pct),
        penalty_as_amount=bool(penalty_as_amount),
        auto_renewal=auto_renewal,
        renewal_months=renewal_months,
        risk_keywords=keywords,
        scenarios=scenarios,
        heading_style=int(rng.integers(0, 4)),
    )


def generate_contract_specs(
    n_contracts: int,
    n_vendors: int = 150,
    seed: int = 42,
    start: date = date(2023, 1, 2),
    end: date = date(2026, 9, 30),
    n_splitting_events: int = 12,
    approval_limit_inr: float = 10_000_000,
    prob_scale: float = 1.0,
    drift: dict | None = None,
) -> tuple[list[ContractSpec], list[VendorProfile]]:
    rng = np.random.default_rng(seed)
    vendors = build_vendors(n_vendors, rng)
    weights = np.array([v.activity for v in vendors])
    weights /= weights.sum()
    span = (end - start).days

    rows: list[tuple[date, VendorProfile, dict]] = []
    n_regular = n_contracts - 3 * n_splitting_events
    for _ in range(n_regular):
        v = vendors[int(rng.choice(len(vendors), p=weights))]
        rows.append((start + timedelta(days=int(rng.integers(0, span))), v, {}))

    # "Unusual vendor behaviour": a burst of contracts priced just under the approval limit
    # (classic contract splitting to avoid CFO approval).
    risky = [v for v in vendors if v.risky] or vendors
    for _ in range(n_splitting_events):
        v = risky[int(rng.integers(0, len(risky)))]
        base = start + timedelta(days=int(rng.integers(30, span - 90)))
        for k in range(3):
            d = base + timedelta(days=int(k * rng.integers(10, 30)))
            rows.append((d, v, {"split": approval_limit_inr * rng.uniform(0.82, 0.98)}))

    rows.sort(key=lambda r: r[0])
    specs = []
    for i, (d, v, extra) in enumerate(rows, start=1):
        spec = sample_contract(
            doc_id=f"SYN-{i:05d}", vendor=v, submission=d, rng=rng, seq=i,
            value_override_inr=extra.get("split"), prob_scale=prob_scale, drift=drift,
        )
        if "split" in extra:
            spec.scenarios.append("contract_splitting")
        specs.append(spec)
    _assign_formats(specs, rng)
    return specs, vendors


def _assign_formats(specs: list[ContractSpec], rng: np.random.Generator) -> None:
    formats = ["pdf", "docx", "scan_pdf", "png", "jpg"]
    probs = [0.70, 0.18, 0.06, 0.04, 0.02]
    for s in specs:
        s.render_format = str(rng.choice(formats, p=probs))


# --------------------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------------------
SECTION_TITLES = {
    "parties": ["Parties", "Recitals", "Parties to the Agreement"],
    "term": ["Term", "Term of Agreement", "Duration", "Contract Period"],
    "payment": ["Payment Terms", "Fees and Payment", "Consideration and Payment", "Invoicing and Payment"],
    "termination": ["Termination", "Termination Rights", "Termination of Agreement"],
    "renewal": ["Renewal", "Renewal and Extension", "Auto-Renewal"],
    "penalties": ["Penalties", "Liquidated Damages", "Service Credits and Penalties"],
    "liability": ["Limitation of Liability", "Liability", "Indemnity and Liability"],
    "confidentiality": ["Confidentiality", "Confidential Information", "Non-Disclosure"],
    "compliance": ["Compliance with Laws", "Anti-Bribery and Compliance", "Regulatory Compliance"],
    "jurisdiction": ["Governing Law and Jurisdiction", "Governing Law", "Dispute Resolution and Jurisdiction"],
    "general": ["General Provisions", "Miscellaneous"],
}


def _heading(style: int, idx: int, title: str) -> str:
    if style == 0:
        return f"{idx}. {title.upper()}"
    if style == 1:
        return f"ARTICLE {idx} - {title}"
    if style == 2:
        return f"Clause {idx}: {title}"
    return f"{title.upper()}:"


def render_contract_text(spec: ContractSpec, rng: np.random.Generator) -> str:
    allow_symbol = spec.render_format == "docx"
    money = money_text(spec.contract_value, spec.currency, rng, allow_symbol)
    party_v = str(rng.choice(["Vendor", "Supplier", "Service Provider"]))
    party_c = str(rng.choice(["Customer", "Company", "Client"]))
    title = str(rng.choice(["VENDOR SERVICES AGREEMENT", "MASTER SERVICES AGREEMENT", "SUPPLY AGREEMENT",
                            "PROFESSIONAL SERVICES CONTRACT"]))
    ref_label = str(rng.choice(["Contract No.:", "Agreement Reference:", "Contract Number:"]))

    lines: list[str] = [title, f"{ref_label} {spec.contract_number}", ""]
    sections: list[tuple[str, str]] = []

    if rng.random() < 0.5:
        parties = (
            f"This Agreement is entered into on {date_text(spec.signing_date, rng)} between "
            f"{spec.customer_name} (\"{party_c}\") and {spec.vendor_name} (\"{party_v}\")."
        )
    else:
        parties = (
            f"This Agreement is made on {date_text(spec.signing_date, rng)} BY AND BETWEEN "
            f"{spec.customer_name}, a company incorporated under the Companies Act, 2013 (hereinafter the "
            f"\"{party_c}\") AND {spec.vendor_name} (hereinafter the \"{party_v}\")."
        )
    parties += f" The {party_v} provides {spec.vendor_category.lower()} to the {party_c}."
    sections.append(("parties", parties))

    tv = int(rng.integers(0, 3))
    if tv == 0:
        term = (
            f"This Agreement shall commence on {date_text(spec.start_date, rng)} (the \"Effective Date\") and "
            f"shall remain in force until {date_text(spec.end_date, rng)}, unless terminated earlier."
        )
    elif tv == 1:
        term = (
            f"The term of this Agreement is {spec.term_months} months commencing from "
            f"{date_text(spec.start_date, rng)}."
        )
    else:
        term = (
            f"Effective Date: {date_text(spec.start_date, rng)}. Expiry Date: {date_text(spec.end_date, rng)}. "
            f"The {party_v} shall perform the services throughout this period."
        )
    sections.append(("term", term))

    pv = int(rng.integers(0, 3))
    if pv == 0:
        pay = f"The {party_c} shall pay all undisputed invoices within {days_text(spec.payment_days, rng)} of receipt."
    elif pv == 1:
        pay = f"Payment terms: Net {spec.payment_days} days from the date of invoice."
    else:
        pay = f"Invoices raised by the {party_v} shall be settled within {days_text(spec.payment_days, rng)}."
    value_phrase = str(rng.choice([
        f"The total contract value is {money}.",
        f"In consideration of the services, the {party_c} shall pay a total fee of {money}.",
        f"The aggregate value of this Agreement shall not exceed {money}.",
    ]))
    sections.append(("payment", f"{value_phrase} {pay} All amounts are exclusive of applicable taxes."))

    if spec.has_termination:
        n = days_text(spec.termination_notice_days, rng)
        sections.append(("termination", str(rng.choice([
            f"Either party may terminate this Agreement by giving {n} prior written notice to the other party.",
            f"The {party_c} may terminate this Agreement for convenience upon {n} written notice. "
            f"Either party may terminate this Agreement immediately for material breach.",
        ]))))

    if spec.auto_renewal:
        renewal = (
            f"This Agreement shall automatically renew for successive periods of {spec.renewal_months} months "
            f"unless either party gives written notice of non-renewal."
        )
    else:
        renewal = "Any renewal of this Agreement shall be subject to the mutual written agreement of the parties."
    sections.append(("renewal", renewal))

    if spec.penalty_pct > 0:
        if spec.penalty_as_amount:
            amt = spec.contract_value * spec.penalty_pct / 100
            pen = (
                f"In the event of delay or service failure, the {party_v} shall pay a penalty of "
                f"{money_text(amt, spec.currency, rng, allow_symbol)}."
            )
        else:
            pct = f"{spec.penalty_pct:g}"
            pen = str(rng.choice([
                f"In the event of delay, the {party_v} shall pay liquidated damages of {pct}% of the contract value.",
                f"Penalties for service-level breaches shall be capped at {pct} percent of the total contract value.",
            ]))
        sections.append(("penalties", pen))
    elif rng.random() < 0.5:
        sections.append(("penalties", "No liquidated damages or penalties shall apply under this Agreement."))

    if spec.has_liability:
        if spec.unlimited_liability:
            liab = str(rng.choice([
                f"The {party_v}'s liability under this Agreement shall be unlimited.",
                f"The {party_c} accepts unlimited liability for all claims arising under this Agreement.",
            ]))
        else:
            liab = (
                "Each party's aggregate liability under this Agreement shall not exceed the total fees paid "
                "in the twelve months preceding the claim."
            )
        sections.append(("liability", liab))

    if spec.has_confidentiality:
        sections.append((
            "confidentiality",
            "Each party shall keep confidential all non-public information received from the other party "
            "and shall not disclose it to any third party without prior written consent.",
        ))

    if spec.has_compliance_clause:
        sections.append((
            "compliance",
            f"The {party_v} shall comply with all applicable laws, including anti-bribery and anti-corruption "
            f"laws and data protection requirements. The {party_c} shall have audit rights over relevant records.",
        ))

    if spec.governing_law == "India":
        jur = (
            f"This Agreement shall be governed by the laws of India and the courts at {spec.jurisdiction} "
            f"shall have exclusive jurisdiction."
        )
    else:
        jur = (
            f"This Agreement shall be governed by the laws of {spec.governing_law} and the courts of "
            f"{spec.jurisdiction} shall have exclusive jurisdiction."
        )
    sections.append(("jurisdiction", jur))

    general = (
        "Neither party shall be liable for failure to perform due to force majeure events. "
        "All notices shall be in writing. This Agreement constitutes the entire agreement between the parties."
    )
    if spec.risk_keywords:
        general += " " + " ".join(f"It is agreed that {k}." for k in spec.risk_keywords)
    sections.append(("general", general))

    middle = sections[3:-2]
    order = sections[:3] + [middle[i] for i in rng.permutation(len(middle))] + sections[-2:]
    for idx, (key, body) in enumerate(order, start=1):
        titles = SECTION_TITLES[key]
        lines.append(_heading(spec.heading_style, idx, titles[int(rng.integers(0, len(titles)))]))
        lines.append(body)
        lines.append("")

    lines += [
        "IN WITNESS WHEREOF, the parties have executed this Agreement by their authorised signatories.",
        f"For {spec.customer_name}: Authorised Signatory",
        f"For {spec.vendor_name}: Authorised Signatory",
    ]
    return "\n".join(lines)


def render_invoice_text(rng: np.random.Generator, vendor: VendorProfile, n: int) -> str:
    d = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 600)))
    qty, rate = int(rng.integers(1, 50)), float(rng.integers(1000, 90000))
    total = qty * rate * 1.18
    return "\n".join([
        "TAX INVOICE",
        f"Invoice Number: INV-{d.year}-{n:05d}",
        f"Invoice Date: {date_text(d, rng)}",
        f"Seller: {vendor.name}   GSTIN: 27AAACX{n:04d}Z1",
        f"Bill To: {CUSTOMERS[n % len(CUSTOMERS)]}",
        "Description            Qty     Rate      Amount",
        f"Services rendered      {qty}      {rate:,.2f}   {qty * rate:,.2f}",
        f"CGST 9%  SGST 9%   Total Amount Due: INR {total:,.2f}",
        f"Payment due within {int(rng.choice([15, 30, 45]))} days. Bank: HDFC Bank, A/c details on request.",
    ])


def render_po_text(rng: np.random.Generator, vendor: VendorProfile, n: int) -> str:
    d = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 600)))
    return "\n".join([
        "PURCHASE ORDER",
        f"PO Number: PO-{d.year}-{n:05d}     PO Date: {date_text(d, rng)}",
        f"Supplier: {vendor.name}",
        f"Ship To: {CUSTOMERS[n % len(CUSTOMERS)]}, Warehouse 3",
        "Item   Description          Quantity   Unit Price   Line Total",
        f"1      {vendor.category} items   {int(rng.integers(5, 500))}        {int(rng.integers(100, 5000))}",
        f"Delivery Date: {date_text(d + timedelta(days=30), rng)}. Terms: as per the approved vendor agreement.",
        "Please quote the PO number on all invoices and delivery challans.",
    ])
