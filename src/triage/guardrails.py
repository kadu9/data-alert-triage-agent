"""Pre-LLM (redact / cap / delimit untrusted text) and post-LLM (validate extracted features) guardrails. Pure functions."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|[A-Za-z0-9+/_-]{32,})\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TAG = re.compile(r"</?\s*untrusted_data[^>]*>", re.IGNORECASE)
TRUNCATED = "...[truncated]"
MIN_QUOTE_CHARS = 8


class AlertFeatures(BaseModel):
    """LLM-extracted labels from alert free text (title, description, notes).

    Each non-default label (stated_noise, recurrence, impact) must have a verbatim
    quote >= MIN_QUOTE_CHARS from the text we sent. Validated by validate_features;
    used by axes.context_signal to add the optional 7th scoring axis.
    """

    stated_noise: Literal["none", "expected_or_known", "retired_or_draft", "historical_backlog"] = "none"
    noise_quote: str = ""
    recurrence: Literal["unknown", "first", "repeated"] = "unknown"
    recurrence_quote: str = ""
    impact: Literal["unknown", "low", "medium", "high"] = "unknown"
    impact_quote: str = ""


class GuardrailViolation(ValueError):
    """Raised when LLM output fails a post-call guard (quote missing, wrong length, etc.).

    Triggers a bounded re-ask with fixed feedback text; if retries are exhausted the
    labels are discarded and the alert gets a fallback verdict. Never echoes model
    output back to avoid prompt injection.
    """


def sanitize_untrusted(text: str | None, max_chars: int) -> tuple[str | None, bool]:
    """Redact secrets/emails, strip fake delimiter tags, truncate the body to max_chars (then append TRUNCATED); return (clean, was_truncated).

    In plain words: prepares free text before sending to the LLM by removing potentially sensitive
    patterns (API keys, emails), stripping any fake <untrusted_data> tags that could confuse the
    prompt structure, and truncating if too long. Returns the cleaned text and whether it was cut.
    """
    if text is None:
        return None, False
    clean = _TAG.sub("", _EMAIL.sub("[email]", _SECRET.sub("[redacted]", text)))
    if len(clean) > max_chars:
        return clean[:max_chars] + TRUNCATED, True
    return clean, False


def wrap_untrusted(name: str, text: str) -> str:
    """Wrap text in <untrusted_data field="name"> delimiters for the LLM prompt."""
    return f'<untrusted_data field="{name}">\n{text}\n</untrusted_data>'


def norm(s: str) -> str:
    """Whitespace-insensitive form used to compare quotes with their source text.

    In plain words: collapses all whitespace runs to single spaces so that quote matching
    tolerates minor formatting differences (tabs, multiple spaces, newlines).
    """
    return " ".join(s.split())


def find_quote(quote: str, sources: dict[str, str]) -> str | None:
    """Return the name of the first source containing the quote (whitespace-insensitive).

    In plain words: searches each source text to find where a quote came from. Uses whitespace-
    insensitive matching so "12th consecutive  daily alert" matches "12th consecutive daily alert".

    Example:
        sources = {'description': 'This is the 12th consecutive  daily alert'}
        find_quote('12th consecutive daily alert', sources) -> 'description'
        find_quote('not there', sources) -> None
    """
    q = norm(quote)
    return next((k for k, v in sources.items() if q in norm(v)), None)


def validate_features(features: object, sources: dict[str, str]) -> None:
    """Validate LLM features: every non-default label must have a quote >= MIN_QUOTE_CHARS found in the sanitized text (whitespace-normalised).

    In plain words: when the LLM claims a label (like "retired_or_draft"), it must provide a quote
    proving that text actually says so. This function checks: (1) the normalised quote is at least MIN_QUOTE_CHARS characters,
    (2) quote is not a redaction placeholder, (3) quote appears in the sanitized text we sent (whitespace-insensitive).
    Raises GuardrailViolation (triggers re-ask) if any check fails.

    Raises GuardrailViolation with a fixed message (never echoing model output) on failure.
    """
    if not isinstance(features, AlertFeatures):
        raise GuardrailViolation("output was not a valid features object")
    for label, default, quote in (
        (features.stated_noise, "none", features.noise_quote),
        (features.recurrence, "unknown", features.recurrence_quote),
        (features.impact, "unknown", features.impact_quote),
    ):
        if label == default:
            continue
        if len(norm(quote)) < MIN_QUOTE_CHARS or "[redacted]" in quote or "[email]" in quote or find_quote(quote, sources) is None:
            raise GuardrailViolation("a non-default label has no verbatim quote from the provided text")
