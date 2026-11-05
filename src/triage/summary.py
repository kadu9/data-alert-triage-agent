"""Second LLM stage (transport-retried, re-asked on guardrail rejection): write a plain-language explanation from a finished verdict's facts.

The model never sees raw alerts and never changes the tier. Guardrails reject any number the facts do not support, a missing exact
quote of a cited LLM-label source, and a missing or misplaced "I can't tell from this data:" sentence; if the text is still rejected
after the re-ask, the deterministic explanation stays and error_type/error_message record why (status is unchanged).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from triage.config import ScoringConfig
from triage.explain import GAP_PHRASE
from triage.features import BAD_OUTPUT_ERRORS, TRANSPORT_ERRORS, Chain, build_llm, log_failure, with_retry
from triage.guardrails import GuardrailViolation, sanitize_untrusted, wrap_untrusted
from triage.prompts import chat_prompt
from triage.verdict import Verdict

log = logging.getLogger(__name__)

LABEL_NOTE = "LLM label:"  # note on evidence lines that cite a free-text quote behind a context_signal label
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class Summary(BaseModel):
    """Structured output from the summary LLM call.

    Contains a sentences list with one string per sentence. Parsed by
    llm.with_structured_output; validated by validate_summary before use.
    """

    sentences: list[str] = Field(description="The explanation, one string per sentence")


class SummaryResult(BaseModel):
    """Result of the summary LLM call for one verdict.

    Holds text (the joined explanation, or None on failure), error_type, and
    error_message. Returned by Summarizer.summarize; None text means the caller
    keeps the deterministic explanation and records error_type/error_message.
    """

    text: str | None = None  # None -> caller keeps the deterministic explanation
    error_type: str | None = None
    error_message: str | None = None


def build_facts(v: Verdict, max_chars: int) -> str:
    """Build the facts block sent to the summary LLM from evidence and data_gaps.

    In plain words: assembles all the evidence citations and data gaps into a text block
    that the summary LLM will use to write the explanation. Wraps it in <untrusted_data>
    tags and applies the same sanitization as other LLM inputs.
    """
    lines = [f"- {e.source} {e.field} = {e.value}" + (f" ({e.note})" if e.note else "") for e in v.evidence]
    lines += [f"- data gap: {g.field} ({g.reason})" for g in v.data_gaps]
    clean, _ = sanitize_untrusted("\n".join(lines), max_chars)
    return wrap_untrusted("facts", clean or "")


def validate_summary(summary: object, facts: str, v: Verdict) -> str:
    """Validate the LLM summary against guardrail rules; return text or raise GuardrailViolation.

    In plain words: ensures the summary meets all formatting and accuracy requirements before
    accepting it. The number check prevents the LLM from inventing statistics - every number
    in the explanation must appear somewhere in the facts we gave it.

    Checks: numbers must appear in facts, must quote an LLM-label source exactly when one is cited,
    must include GAP_PHRASE when v.cannot_tell, and must not use it otherwise.

    Example (number guard):
        facts = '- derived downstream_tables.count = 387\\n- bi_reports.count = 6\\n- score = 0.81'
        text = 'This alert affects 387 downstream tables and 6 BI reports.'
        Numbers in facts: {'387', '6', '0.81'}
        Numbers in text: {'387', '6'}
        Check passes: {'387', '6'} is subset of {'387', '6', '0.81'}

        text_bad = 'This affects 500 tables.'
        Numbers: {'500'} - NOT in facts -> GuardrailViolation
    """
    if not isinstance(summary, Summary):
        raise GuardrailViolation("output was not a valid summary object")
    text = " ".join(s.strip() for s in summary.sentences if s.strip())
    if not set(_NUMBER.findall(text)) <= set(_NUMBER.findall(facts)):
        raise GuardrailViolation("the summary contains a number that is not in the facts")
    quotes = [e.value for e in v.evidence if (e.note or "").startswith(LABEL_NOTE) and e.value in facts]
    if quotes and not any(q in text for q in quotes):
        raise GuardrailViolation("the summary must quote the free-text source of an LLM label exactly")
    if v.cannot_tell:
        if f"{GAP_PHRASE}:" not in text:
            raise GuardrailViolation(f"this verdict cannot be told from the data, so the summary must say \"{GAP_PHRASE}:\" and name what is missing")
    elif GAP_PHRASE in text:
        raise GuardrailViolation(f"this verdict can be told from the data, so the summary must not say \"{GAP_PHRASE}\"")
    return text


def build_summary_chain(cfg: ScoringConfig) -> Chain:
    """Build the LLM chain for summary generation from config."""
    llm = build_llm(cfg.llm, cfg.summary.model_env, cfg.summary.default_model)
    return chat_prompt("summary") | llm.with_structured_output(Summary)


class Summarizer:
    """Async summarizer that writes a plain-language explanation from verdict facts.

    Initialized with a Chain and ScoringConfig. The summarize method builds facts,
    calls the chain with tenacity retry, validates output via guardrails, and returns
    a SummaryResult. Used by TriageService.triage_one in the summary step of the pipeline.
    """

    def __init__(self, chain: Chain, cfg: ScoringConfig):
        self._chain, self._cfg = chain, cfg
        self._invoke = with_retry(cfg.llm, self._ainvoke)

    async def _ainvoke(self, inputs: dict[str, Any]) -> Any:
        return await self._chain.ainvoke(inputs)

    async def summarize(self, v: Verdict) -> SummaryResult:
        """Generate explanation text for a verdict; return result with fallback on failure."""
        sc = self._cfg.summary
        facts = build_facts(v, sc.max_facts_chars)
        base = {"facts": facts}
        feedback, err, msg, start = "", "", "", time.monotonic()
        for attempt in range(1 + self._cfg.llm.max_feedback_retries):
            try:
                text = validate_summary(await self._invoke({**base, "feedback": feedback}), facts, v)
            except GuardrailViolation as exc:
                err, msg = "GuardrailViolation", str(exc)
                feedback = f"Your previous answer was rejected: {msg}"
                log_failure(log, f"llm_summary alert_id={v.alert_id} status=rejected attempt={attempt + 1}", exc)
            except BAD_OUTPUT_ERRORS as exc:
                err, msg = "BadModelOutput", "model output did not match the required schema"
                feedback = f"Your previous answer was rejected: {msg}"
                log_failure(log, f"llm_summary alert_id={v.alert_id} status=bad_output attempt={attempt + 1}", exc)
            except TRANSPORT_ERRORS as exc:
                err, msg = type(exc).__name__, "summary LLM request failed"
                log_failure(log, f"llm_summary alert_id={v.alert_id} status=failed", exc)
                break
            except Exception as exc:  # noqa: BLE001 - one alert's summary must never stop the batch; traceback is logged
                err, msg = type(exc).__name__, "unexpected error while writing the summary; see logs for the traceback"
                log_failure(log, f"llm_summary alert_id={v.alert_id} status=failed", exc)
                break
            else:
                log.info("llm_summary alert_id=%s status=ok attempt=%d latency_ms=%d", v.alert_id, attempt + 1, 1000 * (time.monotonic() - start))
                return SummaryResult(text=text)
        log.warning("llm_summary alert_id=%s status=fallback error_type=%s", v.alert_id, err)
        return SummaryResult(error_type=err, error_message=msg)
