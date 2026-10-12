import json

import httpx
import pytest

from src.features.builder import load_registry
from src.llm.ollama_client import LLMUnavailable, OllamaExplainer, check_grounding, clause_states

EVIDENCE = {
    "risk_score": 70, "risk_level": "HIGH",
    "model_risk_factors": [
        {"factor": "Missing mandatory clauses: 1", "effect": "pushes the model towards CRITICAL"},
        {"factor": "Termination clause: missing", "effect": "pushes the model towards CRITICAL"},
        {"factor": "Payment period (days): 90", "effect": "pushes the model towards CRITICAL"},
    ],
    "compliance_violations": [{"rule": "Missing Termination Clause", "severity": "HIGH",
                               "finding": "No termination clause was found in the contract."}],
    "anomalies": [],
    "recommended_actions": ["Negotiate payment terms down to 60 days or obtain treasury approval."],
}


def test_regression_llm_claimed_missing_clause_was_present():
    """Replays the faulty output observed from llama3.2:3b during development."""
    output = {
        "summary": "This contract has a high risk level. The contract lacks a termination clause.",
        "factor_explanations": [
            {"factor": "Termination clause present",
             "explanation": "The contract includes a termination clause, but its presence does not mitigate the risk."},
            {"factor": "Payment period (days)", "explanation": "The payment period of 90 days increases the risk."},
        ],
        "recommended_actions": [],
    }
    kept, warnings = check_grounding(output, EVIDENCE)
    assert [k["factor"] for k in kept] == ["Payment period (days)"]
    assert any("Termination clause present" in w for w in warnings)


def test_correct_factor_name_with_contradictory_explanation_is_dropped():
    output = {"summary": "High risk.", "recommended_actions": [], "factor_explanations": [
        {"factor": "Termination clause: missing", "explanation": "The contract includes a termination clause."}]}
    kept, warnings = check_grounding(output, EVIDENCE)
    assert kept == []
    assert any("Contradiction" in w for w in warnings)


def test_summary_contradiction_is_flagged():
    output = {"summary": "The termination clause is present, so exit risk is low.", "factor_explanations": [],
              "recommended_actions": []}
    _, warnings = check_grounding(output, EVIDENCE)
    assert any(w.startswith("Contradiction") for w in warnings)


def test_invented_numbers_and_factors_are_flagged():
    output = {"summary": "Penalty of 37% is far above policy.",
              "factor_explanations": [{"factor": "Vendor bankruptcy history", "explanation": "Vendor went bankrupt."}],
              "recommended_actions": []}
    kept, warnings = check_grounding(output, EVIDENCE)
    assert kept == []
    assert any("not in evidence" in w for w in warnings)
    assert any("Number 37" in w for w in warnings)


def test_clause_states_from_factors_and_rules():
    assert clause_states(EVIDENCE) == {"termination": "missing"}


def test_binary_factors_are_self_describing():
    reg = load_registry()
    assert reg.describe("has_termination_clause", 0.0) == "Termination clause: missing"
    assert reg.describe("has_liability_clause", 1.0) == "Liability clause: present"
    assert reg.describe("contract_value_log10", 7.0) == "Contract value: INR 1.00 Cr"
    assert reg.describe("payment_above_policy", 1.0) == "Payment terms above policy: yes"
    assert reg.describe("payment_days_vs_vendor_avg", 37.5) ==         "Payment days vs vendor average: +38 days"
    assert reg.describe("value_vs_vendor_avg", float("nan")) == "Value vs vendor average: not available"


def _explainer(handler) -> OllamaExplainer:
    return OllamaExplainer("http://ollama.test", "test-model", timeout=2, transport=httpx.MockTransport(handler))


def test_ollama_down_raises_llm_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailable):
        _explainer(handler).explain(EVIDENCE)
    assert _explainer(handler).health()["status"] == "down"


def test_invalid_json_from_llm_raises_llm_unavailable():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    with pytest.raises(LLMUnavailable):
        _explainer(handler).explain(EVIDENCE)


def test_valid_llm_response_is_parsed_and_grounded():
    content = {"summary": "Risk is HIGH with a score of 70 because the termination clause is missing.",
               "factor_explanations": [{"factor": "Termination clause: missing",
                                        "explanation": "Without a termination clause the company cannot exit."}],
               "recommended_actions": ["Negotiate payment terms down to 60 days."]}

    def handler(request):
        body = json.loads(request.content)
        assert body["format"]["required"] == ["summary", "factor_explanations", "recommended_actions"]
        assert "Do not invent" in body["messages"][0]["content"]
        return httpx.Response(200, json={"message": {"content": json.dumps(content)}})

    out = _explainer(handler).explain(EVIDENCE)
    assert out.grounded and not out.grounding_warnings
    assert out.factor_explanations[0]["factor"] == "Termination clause: missing"
