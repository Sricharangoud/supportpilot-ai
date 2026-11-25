"""Safety guardrails: prompt-injection detection, context sanitization, and sensitivity rules."""
import re

# Patterns commonly used in direct or indirect prompt-injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+(rules|instructions)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|hidden\s+rules|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(dan|an?\s+unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|<<SYS>>", re.IGNORECASE),
    re.compile(r"override\s+(system|safety|policy)\s+(rules|constraints)", re.IGNORECASE),
]

# Deterministic keywords that always warrant human escalation (legal/fraud/formal complaint)
_SENSITIVE_PATTERNS = [
    (re.compile(r"\b(sue|lawsuit|attorney|lawyer|legal\s+action|subpoena|court)\b", re.IGNORECASE), "legal"),
    (re.compile(r"\b(fraud|fraudulent|unauthorized\s+charge|stolen\s+card|identity\s+theft|chargeback)\b", re.IGNORECASE), "fraud"),
    (re.compile(r"\b(bbb|better\s+business\s+bureau|ftc|attorney\s+general|formal\s+complaint)\b", re.IGNORECASE), "complaint"),
]


def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """Return (True, reason) if user text contains prompt-injection patterns."""
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Matched injection pattern: {match.group(0)}"
    return False, ""


def sanitize_context(text: str) -> str:
    """Neutralize instruction-like directives inside retrieved KB chunks."""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered-instruction]", cleaned)
    return cleaned


def is_sensitive_message(text: str) -> tuple[bool, str]:
    """Detect high-risk legal, fraud, or formal complaint inquiries for mandatory escalation."""
    for pattern, category in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True, category
    return False, ""
