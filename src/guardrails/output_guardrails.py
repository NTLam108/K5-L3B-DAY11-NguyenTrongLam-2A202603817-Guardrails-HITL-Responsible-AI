"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

REDACTED = "[REDACTED]"

# Official public contacts (ground truth) — allowed to appear in replies.
ALLOWED_CONTACTS = {"support@vinbank.example"}

# (issue_type, pattern, keep_group)
#   keep_group = index of a capture group to keep (label such as "password is ")
#   so only the *value* gets redacted; None → redact the whole match.
# Order matters: specific secrets first, generic numbers last, so a secret
# like "sk-…" isn't half-eaten by a looser pattern.
PII_PATTERNS: list[tuple[str, str, int | None]] = [
    # API keys: "sk-…" tokens and "api key: xyz"
    ("api_key", r"\bsk-[A-Za-z0-9_\-]{6,}", None),
    ("api_key", r"(\bapi[\s_-]?key\s*(?:is|=|:)\s*)[^\s,;]+", 1),
    # Passwords: "password is X", "password=X", "mật khẩu: X"
    (
        "password",
        r"(\b(?:password|passwd|pwd|passcode|m[aậ]t kh[aẩ]u)\s*(?:is|=|:|l[aà])\s*)[^\s,;]+",
        1,
    ),
    # Internal hostnames / DB endpoints (db.vinbank.internal:5432)
    ("internal_host", r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|corp)(?::\d{2,5})?\b", None),
    # Email
    ("email", r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b", None),
    # VN phone: 0xxxxxxxxx (10 số di động) / 0xxxxxxxxxx (11 số cố định) / +84…,
    # allowing spaces, dots or dashes between digits. Hotline 1900… is not PII.
    ("phone", r"(?<![\w])(?:\+84|0)(?:[\s.-]?\d){9,10}(?!\d)", None),
    # National ID: CCCD 12 digits / CMND 9 digits (not money amounts)
    ("national_id", r"(?<!\d)(?:\d{12}|\d{9})(?!\d)(?!\s*(?:vnd|vnđ|đ|dong|usd))", None),
]
_COMPILED = [(n, re.compile(p, re.IGNORECASE), g) for n, p, g in PII_PATTERNS]


def _secret_needles() -> tuple[list[str], list[str]]:
    """Exact protected values + simple encodings of them (base64/hex/reversed).

    Regex alone misses a leaked secret written without a label ("it's admin123")
    or lightly encoded — the classic red-team bypasses.
    """
    import base64

    try:
        from core.config import DEMO_SECRETS
    except Exception:  # pragma: no cover - config missing
        DEMO_SECRETS = []
    plain = [s for s in DEMO_SECRETS if s]
    encoded: list[str] = []
    for s in plain:
        raw = s.encode()
        encoded += [
            s[::-1],
            base64.b64encode(raw).decode().rstrip("="),
            raw.hex(),
        ]
    return plain, [e for e in encoded if len(e) >= 6]


_SECRET_PLAIN, _SECRET_ENCODED = _secret_needles()


def _redact_literal(text: str, needle: str, suffix: str = "") -> tuple[str, int]:
    rx = re.compile(re.escape(needle) + suffix, re.IGNORECASE)
    return rx.subn(REDACTED, text)


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues: list[str] = []
    redacted = response or ""
    counts: dict[str, int] = {}

    def bump(kind: str, n: int) -> None:
        if n:
            counts[kind] = counts.get(kind, 0) + n

    # 1. Generic PII / secret patterns (gives precise issue types)
    for name, rx, keep in _COMPILED:
        def _sub(m: re.Match, keep=keep, name=name) -> str:
            value = m.group(0)
            if name == "email" and value.lower() in ALLOWED_CONTACTS:
                return value  # official support address is public
            if keep and m.group(0)[len(m.group(keep)):].startswith(REDACTED):
                return value  # already redacted
            bump(name, 1)
            return (m.group(keep) + REDACTED) if keep else REDACTED

        redacted = rx.sub(_sub, redacted)

    # 2. Safety net: protected values that slipped past the patterns
    #    (unlabelled "it's admin123", or base64 / hex / reversed encodings)
    for needle in _SECRET_PLAIN:
        redacted, n = _redact_literal(redacted, needle)
        bump("secret", n)
    for needle in _SECRET_ENCODED:
        redacted, n = _redact_literal(redacted, needle, suffix="=*")  # eat base64 padding
        bump("encoded_secret", n)

    for kind, n in counts.items():
        issues.append(f"{kind}: {n} found")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []  # issues of the most recent response (for audit)

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        self.last_issues = []

        # 1. Deterministic filter: redact PII / secrets
        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            self.last_issues = result["issues"]
            response_text = result["redacted"]
            llm_response.content = self._text_content(response_text)

        # 2. Optional LLM-as-Judge on the (already redacted) text
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_issues.append(f"judge: {verdict['verdict'][:80]}")
                llm_response.content = self._text_content(
                    "I'm sorry, I can't share that information. "
                    "Please contact VinBank support for help with your request."
                )

        # 3. Return (possibly modified) response
        return llm_response

    @staticmethod
    def _text_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
