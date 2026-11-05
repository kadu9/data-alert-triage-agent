"""LLM extraction stage per alert (transport-retried, re-asked on guardrail rejection): read the free text (title, description, monitor notes) and extract a few labels.

The model never sets a tier. Labels need verbatim quotes (post-guard); the tier is computed in code from them.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar, cast

import httpx
import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from triage.axes import inner_config
from triage.config import LlmConfig
from triage.guardrails import (
    AlertFeatures,
    GuardrailViolation,
    sanitize_untrusted,
    validate_features,
    wrap_untrusted,
)
from triage.prompts import chat_prompt
from triage.schema import Alert, DataGap, Monitor

log = logging.getLogger(__name__)

TRANSPORT_ERRORS = (openai.OpenAIError, httpx.HTTPError, TimeoutError)
BAD_OUTPUT_ERRORS = (OutputParserException, ValidationError)

class Chain(Protocol):
    """Protocol for an async LLM chain that takes a dict and returns structured output.

    Used as the type hint for the prompt|llm.with_structured_output chain in
    FeatureExtractor and Summarizer. The ainvoke method mirrors LangChain's interface.
    """

    async def ainvoke(self, input: dict[str, str], /) -> Any: ...


class Extraction(BaseModel):
    """Result of the label-extraction LLM call for one alert.

    Holds features (AlertFeatures or None), sources dict of sanitized text actually
    sent, error_type and error_message if the call failed, and gaps list. Returned
    by FeatureExtractor.extract; used by service._score to add context_signal.
    """

    features: AlertFeatures | None = None
    sources: dict[str, str] = Field(default_factory=dict)  # sanitized text actually sent, by field name
    error_type: str | None = None
    error_message: str | None = None
    gaps: list[DataGap] = Field(default_factory=list)


def build_prompt() -> ChatPromptTemplate:
    """Load the Jinja2 prompt template for feature extraction."""
    return chat_prompt("features")


def build_llm(cfg: LlmConfig, model_env: str, default_model: str | None = None) -> ChatOpenAI:
    """Chat client from env vars named in config. Fails loudly if a required one is unset.

    In plain words: creates the LangChain OpenAI client using environment variables for
    credentials (never hard-coded). The actual values are read from env vars whose NAMES
    are in the config (e.g. LLM_BASE_URL, LLM_API_KEY). Raises RuntimeError if required
    vars are missing.
    """
    needed = [cfg.base_url_env, cfg.api_key_env, *([] if default_model else [model_env])]
    missing = [n for n in needed if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"LLM is required; set environment variables: {', '.join(missing)}")
    return ChatOpenAI(
        base_url=os.environ[cfg.base_url_env],
        api_key=os.environ[cfg.api_key_env],  # pyright: ignore[reportArgumentType] - str accepted, wrapped as SecretStr
        model=os.environ.get(model_env) or default_model or "",
        temperature=cfg.temperature,
        timeout=cfg.timeout_seconds,
        max_retries=0,  # retries are ours (tenacity), so they are logged and bounded by config
    )


def build_chain(cfg: LlmConfig) -> Chain:
    """Build the LLM chain for feature extraction (prompt | llm.with_structured_output)."""
    return build_prompt() | build_llm(cfg, cfg.model_env).with_structured_output(AlertFeatures)


def _sources(alert: Alert, monitor: Monitor | None, max_chars: int) -> tuple[dict[str, str], list[DataGap]]:
    """Gather and sanitize free-text sources (title, description, notes) for the LLM."""
    raw = {"title": alert.title, "description": alert.description}
    if monitor is not None:
        notes = inner_config(monitor).get("notes")
        raw["notes"] = notes if isinstance(notes, str) else None
    out: dict[str, str] = {}
    gaps: list[DataGap] = []
    for name, text in raw.items():
        clean, truncated = sanitize_untrusted(text, max_chars)
        if clean:
            out[name] = clean
        if truncated:
            gaps.append(DataGap(field=f"free_text.{name}", reason=f"text longer than {max_chars} chars; truncated before the LLM"))
    return out, gaps


def _log_retry(state: RetryCallState) -> None:
    """Log a retry event without exposing exception text (may contain request data)."""
    exc = state.outcome.exception() if state.outcome else None
    log.warning("llm_call status=retrying attempt=%d error_type=%s", state.attempt_number, type(exc).__name__)


def log_failure(logger: logging.Logger, event: str, exc: BaseException) -> None:
    """Log error type and stack frames without the exception text (may hold request data).

    In plain words: logs enough detail for debugging without leaking sensitive information
    that might be in the exception message (like request bodies or API keys). Logs the
    error type and full traceback, but not the exception's text.
    """
    frames = "".join(traceback.format_tb(exc.__traceback__)).rstrip()
    logger.error("%s error_type=%s traceback (most recent call last):\n%s", event, type(exc).__name__, frames)


_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def with_retry(cfg: LlmConfig, fn: _F) -> _F:
    """Wrap an async function with exponential-backoff retry on transport errors.

    In plain words: adds automatic retry with exponential backoff for TRANSPORT_ERRORS
    (openai.OpenAIError, httpx.HTTPError, TimeoutError). Does NOT retry on bad model output (schema
    violations, guardrail failures) - those use a separate feedback-retry loop.
    """
    return cast(_F, retry(
        stop=stop_after_attempt(cfg.max_attempts),
        wait=wait_exponential_jitter(multiplier=cfg.backoff_initial_seconds, max=cfg.backoff_max_seconds, jitter=cfg.backoff_jitter_seconds),
        retry=retry_if_exception_type(TRANSPORT_ERRORS),
        before_sleep=_log_retry,
        reraise=True,
    )(fn))


class FeatureExtractor:
    """Async extractor that makes one LLM call per alert to read labels from free text.

    Initialized with a Chain and LlmConfig. The extract method sanitizes text, calls
    the chain with tenacity retry, validates features via guardrails, and returns an
    Extraction. Used by TriageService.triage_one in the features step of the pipeline.
    """

    def __init__(self, chain: Chain, cfg: LlmConfig):
        self._chain, self._cfg = chain, cfg
        self._invoke = with_retry(cfg, self._ainvoke)

    async def _ainvoke(self, inputs: dict[str, str]) -> Any:
        return await self._chain.ainvoke(inputs)

    async def extract(self, alert: Alert, monitor: Monitor | None) -> Extraction:
        """Extract features from alert free text; return Extraction with features or error info."""
        sources, gaps = _sources(alert, monitor, self._cfg.max_text_chars)
        text = "\n".join(wrap_untrusted(k, v) for k, v in sources.items())
        feedback, err, msg = "", "", ""
        start = time.monotonic()
        for attempt in range(1 + self._cfg.max_feedback_retries):
            try:
                features = await self._invoke({"text": text, "feedback": feedback})
                validate_features(features, sources)
            except GuardrailViolation as exc:
                err, msg = "GuardrailViolation", str(exc)
                feedback = f"Your previous answer was rejected: {msg}"
            except BAD_OUTPUT_ERRORS:
                err, msg = "BadModelOutput", "model output did not match the required schema"
                feedback = f"Your previous answer was rejected: {msg}"
            except TRANSPORT_ERRORS as exc:
                err, msg = type(exc).__name__, "LLM request failed"
                break
            else:
                log.info("llm_features alert_id=%s status=ok attempt=%d latency_ms=%d", alert.alert_id, attempt + 1, 1000 * (time.monotonic() - start))
                return Extraction(features=features, sources=sources, gaps=gaps)
            log.warning("llm_features alert_id=%s status=rejected attempt=%d error_type=%s", alert.alert_id, attempt + 1, err)
        log.warning("llm_features alert_id=%s status=fallback error_type=%s", alert.alert_id, err)
        return Extraction(
            sources=sources,
            error_type=err,
            error_message=msg,
            gaps=[*gaps, DataGap(field="llm_features", reason=f"LLM features unavailable ({msg}); free text was not used")],
        )
