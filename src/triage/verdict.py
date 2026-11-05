"""Output contract: one Verdict per input alert (PLAN section 4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from triage.config import Level, Tier
from triage.schema import DataGap


class EvidenceCitation(BaseModel):
    """One traceable citation backing a claim in the verdict's explanation.

    Holds source (file name or "derived"), field path, stringified value, and optional
    note. Built by explain.build_evidence and features._quote_evidence; checked against
    raw data by eval/verify_claims.py.
    """

    source: str  # file the value came from, or "derived" for counts computed from joins
    field: str
    value: str  # stringified value as it appears in the data
    note: str | None = None


class Verdict(BaseModel):
    """Final triage output for one alert; the full record goes to verdicts_audit.jsonl.

    Contains alert_id, triage_score (HIGH/MEDIUM/LOW), confidence_support, cannot_tell, explanation,
    evidence list, axis_summary, score, deterministic_tier, data_gaps, and status
    (ok/llm_fallback/failed). Produced by service.TriageService._score
    and written by cli.py: verdicts.jsonl gets only alert_id, triage_score and explanation.
    """

    alert_id: str
    triage_score: Tier
    confidence_support: Level
    cannot_tell: bool = False  # LOW tier resting on <= confidence.cannot_tell_max_axes axes with real data; explanation must say so
    explanation: str
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    axis_summary: dict[str, float | None] = Field(default_factory=dict)  # None = could not be computed
    score: float | None = None  # weighted sum before overrides; None if no axis had a value
    tier_override_reason: str | None = None
    deterministic_tier: Tier
    data_gaps: list[DataGap] = Field(default_factory=list)
    status: Literal["ok", "llm_fallback", "failed"] = "ok"
    error_type: str | None = None
    error_message: str | None = None
