"""Unit tests for intent JSON validation, confidence scoring, and escalation rules."""
import json
from app.agent import (
    compute_confidence,
    parse_intent_json,
    should_escalate,
)
from app.schemas import IntentType


def test_parse_intent_json_valid():
    raw = json.dumps({
        "intent": "returns",
        "is_sensitive": False,
        "sensitivity_reason": None,
    })
    res = parse_intent_json(raw)
    assert res.intent == IntentType.returns
    assert res.is_sensitive is False


def test_parse_intent_json_with_markdown_fences():
    raw = '```json\n{"intent": "billing", "is_sensitive": true, "sensitivity_reason": "fraud"}\n```'
    res = parse_intent_json(raw)
    assert res.intent == IntentType.billing
    assert res.is_sensitive is True
    assert res.sensitivity_reason == "fraud"


def test_parse_intent_json_malformed_falls_back_safely():
    res = parse_intent_json("not valid json at all")
    assert res.intent == IntentType.other
    assert res.is_sensitive is False


def test_parse_intent_json_unknown_intent_falls_back_to_other():
    raw = json.dumps({"intent": "unknown_category", "is_sensitive": False})
    res = parse_intent_json(raw)
    assert res.intent == IntentType.other


def test_compute_confidence_supported_vs_unsupported():
    high = compute_confidence(top_similarity=0.85, self_check_supported=True)
    low = compute_confidence(top_similarity=0.85, self_check_supported=False)
    assert high > 0.8
    assert low < 0.55
    assert 0.0 <= high <= 1.0
    assert 0.0 <= low <= 1.0


def test_compute_confidence_clamps_similarity():
    assert compute_confidence(1.5, True) == 1.0
    assert compute_confidence(-0.3, False) == 0.0


def test_escalation_rule_triggers_on_low_confidence():
    assert should_escalate(confidence=0.40, threshold=0.55, is_sensitive=False) is True
    assert should_escalate(confidence=0.54, threshold=0.55, is_sensitive=False) is True


def test_escalation_rule_triggers_on_sensitive_even_if_high_confidence():
    assert should_escalate(confidence=0.95, threshold=0.55, is_sensitive=True) is True


def test_escalation_rule_auto_replies_when_confident_and_safe():
    assert should_escalate(confidence=0.75, threshold=0.55, is_sensitive=False) is False
