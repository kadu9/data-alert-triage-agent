"""Typed view of config/scoring.yaml. Sections are added as the code that consumes them lands."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Unit = Field(ge=0, le=1)
Cap = Field(gt=0)


class _Strict(BaseModel):
    """Base pydantic model with extra="forbid" so typos in config/scoring.yaml fail loudly.

    All config section classes inherit from this. The forbid setting means any unknown
    key in the YAML produces a validation error, catching mistakes early.
    """

    model_config = ConfigDict(extra="forbid")


def _check_map(v: dict[str, float], *, keys: set[str] | None = None, unit: bool = False, positive: bool = False) -> dict[str, float]:
    """Validate a config map: check keys, range (unit or positive), and lower-case key names."""
    if keys is not None and set(v) != keys:
        raise ValueError(f"keys must be exactly {sorted(keys)}, got {sorted(v)}")
    for k, x in v.items():
        if k != k.lower():
            raise ValueError(f"key {k!r} must be lower-case (lookups lower-case the data side)")
        if unit and not 0 <= x <= 1:
            raise ValueError(f"{k}={x} outside [0,1]")
        if positive and x <= 0:
            raise ValueError(f"{k}={x} must be > 0")
    return v


class HistoryConfig(_Strict):
    """Settings for labeled-history lookups from the history section of scoring.yaml.

    Defines laplace_alpha for smoothing, acted_statuses and not_acted_statuses for
    outcome classification, excluded_alert_types to skip plumbing alerts, and type_map
    bridging alerts.jsonl types to history types. Used by HistoryIndex and history_stats.
    """

    laplace_alpha: float = Field(gt=0)
    acted_statuses: frozenset[str]
    not_acted_statuses: frozenset[str]
    excluded_alert_types: frozenset[str]
    type_map: dict[str, tuple[str, ...]]


class CriticalityAxis(_Strict):
    """Config for axes.business_criticality from scoring.yaml axes.business_criticality.

    Contains component_weights (importance/level/usage blend), level_values mapping
    criticality strings to scores, usage_caps for log-scaling activity metrics,
    missing_value fallback, and golden_bonus. Used by axes.business_criticality.
    """

    component_weights: dict[str, float]
    level_values: dict[str, float]
    usage_caps: dict[str, float]
    missing_value: float = Unit
    golden_bonus: float = Unit

    @field_validator("component_weights")
    @classmethod
    def _cw(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"importance", "level", "usage"}, unit=True)

    @field_validator("level_values")
    @classmethod
    def _lv(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, unit=True)

    @field_validator("usage_caps")
    @classmethod
    def _uc(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"reads_30d", "writes_30d", "active_users_30d"}, positive=True)


class BlastRadiusAxis(_Strict):
    """Config for axes.blast_radius from scoring.yaml axes.blast_radius.

    Contains downstream_tables_cap and bi_reports_cap for log-scaling counts, and
    tables_share to blend the two components. Used by axes.blast_radius to score
    potential downstream impact of alert tables.
    """

    downstream_tables_cap: float = Cap
    bi_reports_cap: float = Cap
    tables_share: float = Unit


class HistoryAxis(_Strict):
    """Config for axes.historical_action_rate from scoring.yaml axes.historical_action_rate.

    Contains cold_start_weight_factor (shrink when using pool fallback), n_labeled_min
    threshold for low-support detection, and low_support_weight_factor. Used by
    axes.historical_action_rate to decide weight and low_support flags.
    """

    cold_start_weight_factor: float = Unit
    n_labeled_min: int = Field(ge=1)
    low_support_weight_factor: float = Unit


class SeverityAxis(_Strict):
    """Config for axes.anomaly_severity from scoring.yaml axes.anomaly_severity.

    Contains event_rules mapping event_type to scoring rule (band/breach/schema/none),
    caps and floors for magnitude scaling, schema_weights for schema-change scoring,
    and bump settings for multiple events. Used by axes.anomaly_severity.
    """

    event_rules: dict[str, Literal["band", "breach", "schema", "none"]]
    band_multiple_cap: float = Cap
    breach_excess_cap: float = Cap
    breach_floor: float = Unit
    schema_weights: dict[str, float]
    schema_cap: float = Cap
    no_magnitude_value: float = Unit
    extra_event_bump: float = Field(ge=0)
    max_event_bump: float = Field(ge=0)

    @field_validator("schema_weights")
    @classmethod
    def _sw(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"fields_added", "fields_removed", "field_type_changes"}, positive=True)


class SignalQualityAxis(_Strict):
    """Config for axes.monitor_signal_quality from scoring.yaml axes.monitor_signal_quality.

    Contains sensitivity_values and priority_values dicts mapping string levels to [0,1]
    scores. Used by axes.monitor_signal_quality to score monitor credibility based on
    its sensitivity setting and inner-config priority.
    """

    sensitivity_values: dict[str, float]
    priority_values: dict[str, float]

    @field_validator("sensitivity_values", "priority_values")
    @classmethod
    def _unit_map(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, unit=True)


class RecencyAxis(_Strict):
    """Config for axes.recency_pressure from scoring.yaml axes.recency_pressure.

    Contains window_hours defining the cluster window, base_value for isolated alerts,
    and cluster_value for alerts that re-fired within the window. Used by
    axes.recency_pressure to detect alert clustering on a monitor.
    """

    window_hours: float = Cap
    base_value: float = Unit
    cluster_value: float = Unit


class ContextSignalAxis(_Strict):
    """Config for axes.context_signal from scoring.yaml axes.context_signal.

    Maps LLM-extracted label values (noise, recurrence, impact) to [0,1] scores. The
    label "none"/"unknown" is never scored. Used by axes.context_signal to convert
    validated LLM features into the optional 7th axis.
    """

    noise_values: dict[str, float]
    recurrence_values: dict[str, float]
    impact_values: dict[str, float]

    @field_validator("noise_values")
    @classmethod
    def _nv(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"expected_or_known", "retired_or_draft", "historical_backlog"}, unit=True)

    @field_validator("recurrence_values")
    @classmethod
    def _rv(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"first", "repeated"}, unit=True)

    @field_validator("impact_values")
    @classmethod
    def _iv(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, keys={"low", "medium", "high"}, unit=True)


class AxesConfig(_Strict):
    """Container for all per-axis config sections from scoring.yaml axes.

    Groups the seven axis configs: business_criticality, blast_radius,
    historical_action_rate, anomaly_severity, monitor_signal_quality, recency_pressure,
    and context_signal. Used by axes.py functions and combine.py for scoring.
    """

    business_criticality: CriticalityAxis
    blast_radius: BlastRadiusAxis
    historical_action_rate: HistoryAxis
    anomaly_severity: SeverityAxis
    monitor_signal_quality: SignalQualityAxis
    recency_pressure: RecencyAxis
    context_signal: ContextSignalAxis


Tier = Literal["HIGH", "MEDIUM", "LOW"]
Level = Literal["high", "medium", "low"]
AXIS_NAMES = frozenset(AxesConfig.model_fields)


class RuntimeConfig(_Strict):
    """Runtime settings from scoring.yaml runtime section.

    Currently holds max_concurrency controlling the semaphore limit in
    TriageService.triage_all. Can be overridden via constructor or env var.
    """

    max_concurrency: int = Field(ge=1)


class Thresholds(_Strict):
    """Score thresholds for tier mapping from scoring.yaml thresholds section.

    Contains HIGH and MEDIUM cutoffs; a score >= HIGH maps to HIGH tier, >= MEDIUM
    maps to MEDIUM, else LOW. Validated to ensure HIGH > MEDIUM. Used by combine._tier_for.
    """

    HIGH: float = Unit
    MEDIUM: float = Unit

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if self.HIGH <= self.MEDIUM:
            raise ValueError("thresholds.HIGH must be > thresholds.MEDIUM")
        return self


class PausedMonitor(_Strict):
    """Override rule for paused monitors from scoring.yaml overrides.paused_monitor.

    Defines action (downgrade_one_tier or none) and confidence_penalty applied when
    monitor.status is "paused". Used by combine.combine to adjust tier and confidence.
    """

    action: Literal["downgrade_one_tier", "none"]
    confidence_penalty: Level


class MissingCriticality(_Strict):
    """Override rule for missing criticality from scoring.yaml overrides.missing_criticality.

    Defines confidence_penalty applied when no alert table has a criticality level.
    Used by combine.combine to cap confidence when table importance is unknown.
    """

    confidence_penalty: Level


class Overrides(_Strict):
    """Global override rules from scoring.yaml overrides section.

    Contains paused_monitor and missing_criticality sub-rules, pii_boost amount,
    and no_signal_tier fallback. Used by combine.combine to adjust tier, confidence,
    and handle edge cases after the weighted score is computed.
    """

    paused_monitor: PausedMonitor
    pii_boost: float = Unit
    missing_criticality: MissingCriticality
    no_signal_tier: Tier  # tier when no axis has a value, or the alert could not be processed (status flags it)


class ConfidenceConfig(_Strict):
    """Thresholds for confidence_support from scoring.yaml confidence section.

    Defines medium_at and low_at counts: if the number of unknown or low-support axes
    is >= low_at, confidence is low; elif >= medium_at, medium; else high. Used by
    combine.combine to set Verdict.confidence_support. cannot_tell_max_axes is the most axes with
    real data a LOW verdict may have and still be flagged cannot_tell (see combine.combine).
    """

    medium_at: int = Field(ge=1)
    low_at: int = Field(ge=1)
    cannot_tell_max_axes: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceConfig:
        if self.low_at < self.medium_at:
            raise ValueError("confidence.low_at must be >= confidence.medium_at")
        return self


class EvidenceConfig(_Strict):
    """Evidence output settings from scoring.yaml evidence section.

    Contains max_events_cited limiting how many event details appear in the evidence
    list. Used by explain.build_evidence to keep output size bounded.
    """

    max_events_cited: int = Field(ge=1)


class LlmConfig(_Strict):
    """LLM connection and call settings from scoring.yaml llm section.

    Holds env var names for base_url, api_key, and model (never the values themselves),
    plus temperature, retry/backoff settings, max_feedback_retries for guardrail
    re-asks, max_text_chars for truncation, and timeout. Used by features.build_llm
    and summary.build_summary_chain.
    """

    base_url_env: str
    api_key_env: str
    model_env: str
    temperature: float = Field(ge=0)
    max_attempts: int = Field(ge=1, le=8)  # transport-level tries (tenacity), per model call
    backoff_initial_seconds: float = Field(ge=0)
    backoff_max_seconds: float = Field(ge=0)
    backoff_jitter_seconds: float = Field(ge=0)
    max_feedback_retries: int = Field(ge=0, le=5)  # re-asks after a schema/guardrail rejection
    max_text_chars: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)


class SummaryConfig(_Strict):
    """Config for the second LLM call from scoring.yaml summary section.

    Holds model_env and default_model for the summary model, max_facts_chars. Used by
    summary.Summarizer to write and validate the plain-language explanation.
    """

    model_env: str  # env var naming the summary model; falls back to `default_model` when unset
    default_model: str
    max_facts_chars: int = Field(ge=100)  # cap on the facts block sent to the model


class ScoringConfig(_Strict):
    """Root config model validated from scoring.yaml by load_config.

    Contains all config sections: runtime, weights dict, thresholds, min_axes_weight,
    overrides, confidence, history, axes, evidence, llm, and summary. This is the single
    config object passed through TriageService, axes, combine, and features.
    """

    runtime: RuntimeConfig
    weights: dict[str, float]
    thresholds: Thresholds
    min_axes_weight: float = Unit
    overrides: Overrides
    confidence: ConfidenceConfig
    history: HistoryConfig
    axes: AxesConfig
    evidence: EvidenceConfig
    llm: LlmConfig
    summary: SummaryConfig

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        _check_map(v, keys=set(AXIS_NAMES))
        if any(x < 0 for x in v.values()) or sum(v.values()) <= 0:
            raise ValueError("weights must be >= 0 with a positive sum")
        return v


def load_config(path: str | Path) -> ScoringConfig:
    """Load and validate scoring config from a YAML file; extra='forbid' rejects typos."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)  # safe_load: no arbitrary object construction (CWE-502)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return ScoringConfig.model_validate(raw)  # extra='forbid': unknown/typo'd sections fail loudly
