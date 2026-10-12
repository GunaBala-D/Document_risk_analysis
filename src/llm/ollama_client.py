"""Ollama (local LLM) explanation layer.

The LLM never decides risk. It receives the *structured evidence* the ML/rules/anomaly
pipeline already produced and turns it into readable language. Hallucination controls:
  1. System instruction forbids inventing facts, policies or risk factors.
  2. Only structured evidence is sent - never the raw contract text.
  3. Output is constrained to a JSON schema (Ollama structured outputs), low temperature.
  4. Post-hoc grounding check: every factor the LLM explains must exist in the evidence and
     every number it states must appear in the evidence; violations are reported as
     ``grounding_warnings`` and ungrounded factor explanations are dropped.
If Ollama is down, slow or returns invalid output, callers get ``LLMUnavailable`` and the
API still returns the full ML/rule-based result.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import asdict, dataclass, field

import httpx

SYSTEM_PROMPT = (
    "You are a contract-risk assistant for procurement and compliance analysts. "
    "Only explain the structured evidence supplied by the application. Do not invent contract facts, "
    "policies, or risk factors. If the evidence does not support a statement, do not make it. "
    "Use plain business English, be concise, and never change the risk level or score. "
    "Each model risk factor is written as '<name>: <value>'. A clause marked 'missing' is NOT in the contract; "
    "a clause marked 'present' IS in the contract - never state the opposite. The 'effect' field only describes "
    "the direction of the machine-learning model's prediction."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "factor_explanations": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"factor": {"type": "string"}, "explanation": {"type": "string"}},
                      "required": ["factor", "explanation"]},
        },
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "factor_explanations", "recommended_actions"],
}


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMExplanation:
    summary: str
    factor_explanations: list[dict]
    recommended_actions: list[str]
    model: str
    latency_ms: float
    grounded: bool
    grounding_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers(text: str) -> set[float]:
    out = set()
    for m in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text):
        with contextlib.suppress(ValueError):
            out.add(round(float(m.replace(",", "")), 2))
    return out


def _evidence_numbers(evidence: dict) -> set[float]:
    nums: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, int | float):
            for v in (x, round(x), round(x, 1), round(x, 2), x * 100, round(x * 100), round(x * 100, 1)):
                nums.add(round(float(v), 2))
        elif isinstance(x, str):
            nums.update(_numbers(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(evidence)
    return nums


def evidence_factor_names(evidence: dict) -> list[str]:
    names = [f["factor"] for f in evidence.get("model_risk_factors", [])]
    names += [v["rule"] for v in evidence.get("compliance_violations", [])]
    names += [a["type"] for a in evidence.get("anomalies", [])]
    return names


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def clause_states(evidence: dict) -> dict[str, str]:
    """{'termination': 'missing' | 'present', ...} as stated by the evidence."""
    states: dict[str, str] = {}
    for f in evidence.get("model_risk_factors", []):
        m = re.match(r"(\w+) clause: (present|missing)", str(f.get("factor", "")), re.IGNORECASE)
        if m:
            states[m.group(1).lower()] = m.group(2).lower()
    for v in evidence.get("compliance_violations", []):
        m = re.match(r"missing (\w+) clause", str(v.get("rule", "")), re.IGNORECASE)
        if m:
            states[m.group(1).lower()] = "missing"
    return states


def contradictions(text: str, states: dict[str, str]) -> list[str]:
    """Statements that flip a clause's presence relative to the evidence."""
    found = []
    for clause, state in states.items():
        says_present = re.search(
            rf"\b(includes?|has|contains?|provides?|with)\s+(a|an|the)?\s*{clause}\s+clause|"
            rf"\b{clause}\s+clause\s+(is|was)\s+(present|included|in place)", text, re.IGNORECASE)
        says_missing = re.search(
            rf"\b(missing|lacks?|no|without|absent)\s+(a|an|the)?\s*{clause}\s+clause|"
            rf"\b{clause}\s+clause\s+(is|was)\s+(missing|absent|not present)", text, re.IGNORECASE)
        if state == "missing" and says_present:
            found.append(f"Contradiction: says the {clause} clause is present, evidence says it is missing")
        if state == "present" and says_missing:
            found.append(f"Contradiction: says the {clause} clause is missing, evidence says it is present")
    return found


def check_grounding(output: dict, evidence: dict) -> tuple[list[dict], list[str]]:
    """Return (grounded factor explanations, warnings)."""
    warnings: list[str] = []
    allowed = {_norm(n) for n in evidence_factor_names(evidence)}
    states = clause_states(evidence)
    kept = []
    for fe in output.get("factor_explanations", []):
        name = _norm(str(fe.get("factor", "")))
        if not any(name == a or name in a or a in name for a in allowed if a):
            warnings.append(f"Dropped explanation for factor not in evidence: '{fe.get('factor')}'")
            continue
        bad = contradictions(f"{fe.get('factor', '')}. {fe.get('explanation', '')}", states)
        if bad:
            warnings.extend(f"Dropped explanation for '{fe.get('factor')}': {b}" for b in bad)
            continue
        kept.append(fe)
    warnings.extend(contradictions(output.get("summary", ""), states))
    allowed_nums = _evidence_numbers(evidence) | {float(i) for i in range(0, 11)}
    text = " ".join([output.get("summary", "")] + [fe.get("explanation", "") for fe in kept]
                    + list(output.get("recommended_actions", [])))
    for n in sorted(_numbers(text)):
        if n not in allowed_nums:
            warnings.append(f"Number {n:g} does not appear in the evidence")
    return kept, warnings


class OllamaExplainer:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport)

    def health(self) -> dict:
        try:
            with self._client() as c:
                r = c.get("/api/tags", timeout=3.0)
                r.raise_for_status()
                models = [m.get("name") for m in r.json().get("models", [])]
            return {"status": "up" if self.model in models else "model_missing", "model": self.model}
        except Exception as exc:
            return {"status": "down", "model": self.model, "error": type(exc).__name__}

    def explain(self, evidence: dict) -> LLMExplanation:
        user = (
            "Explain this contract risk assessment to a reviewer.\n"
            "Return JSON with: summary (3-4 sentences: why the contract has this risk level), "
            "factor_explanations (one entry per item in model_risk_factors, compliance_violations and anomalies, "
            "using the exact factor/rule/type name as 'factor'), recommended_actions (rephrase the supplied "
            "recommended_actions; do not add new ones).\n\nEVIDENCE:\n" + json.dumps(evidence, indent=1, default=str)
        )
        payload = {
            "model": self.model, "stream": False, "format": OUTPUT_SCHEMA,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "options": {"temperature": 0.1, "num_predict": 700},
        }
        start = time.perf_counter()
        try:
            with self._client() as c:
                r = c.post("/api/chat", json=payload)
                r.raise_for_status()
                content = r.json()["message"]["content"]
            output = json.loads(content)
        except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        latency = round((time.perf_counter() - start) * 1000, 1)
        if not isinstance(output, dict) or not isinstance(output.get("summary"), str):
            raise LLMUnavailable("LLM returned output that does not match the schema")
        kept, warnings = check_grounding(output, evidence)
        actions = [a for a in output.get("recommended_actions", []) if isinstance(a, str)][:8]
        return LLMExplanation(
            summary=output["summary"].strip(), factor_explanations=kept, recommended_actions=actions,
            model=self.model, latency_ms=latency,
            grounded=not any(w.startswith(("Number", "Contradiction")) for w in warnings),
            grounding_warnings=warnings,
        )
