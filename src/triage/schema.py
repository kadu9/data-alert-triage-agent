"""Typed records. Null stays None; unparseable inputs become DataGap entries (never 0/False)."""
from __future__ import annotations

from datetime import timezone
from typing import Annotated, Any

from pydantic import AfterValidator, AwareDatetime, BaseModel, Field

# Aware input is required (naive -> error); accepted values are normalized to UTC.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda d: d.astimezone(timezone.utc))]


class DataGap(BaseModel):
    """Record of a missing or unparseable value encountered during loading or scoring.

    Holds the field path (e.g. "events[0].actual_value") and a reason string explaining
    what went wrong. DataGaps propagate through loader, context, axes, and combine steps
    into the final Verdict, where explain.gaps_sentence renders them as "I can't tell
    from this data".
    """

    field: str   # e.g. "events[0].actual_value"
    reason: str  # e.g. "unparseable numeric: 'abc'"


class Event(BaseModel):
    """A single data-quality event within an alert, parsed from alerts.jsonl events[].

    Contains the event_type (e.g. "validation_breach", "schema_change"), numeric fields
    (actual_value, lower_threshold, upper_threshold), and parsed schema_changes dict.
    Used by axes.anomaly_severity to compute breach/band distance or schema-change size.
    """

    event_id: str
    created_at: UtcDatetime
    event_type: str
    field: str | None = None
    actual_value: float | None = None
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    schema_changes: dict[str, Any] | None = None
    data_gaps: list[DataGap] = Field(default_factory=list)


class Alert(BaseModel):
    """One data-quality alert from alerts.jsonl, the primary input to the triage pipeline.

    Contains alert_id, monitor_id, created_at, alert_type, title, description, status,
    tables list, and nested events. Loaded by loader._alert, joined to context in
    ContextIndex.resolve, and scored by service.TriageService to produce a Verdict.
    Fields severity/owner/priority are always null in the dataset and never invented.
    """

    alert_id: str
    monitor_id: str
    created_at: UtcDatetime
    alert_type: str
    title: str
    description: str | None = None
    status: str | None = None
    severity: str | None = None  # null on every input; never fabricated
    owner: str | None = None
    priority: str | None = None
    tables: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)


class Monitor(BaseModel):
    """Monitor configuration from monitors.jsonl, keyed by monitor_id.

    Holds name, type, status (e.g. "paused"), tables list, sensitivity, and a parsed
    config dict with type-specific settings. Joined to alerts in ContextIndex.resolve;
    status "paused" triggers a tier override in combine; sensitivity/priority drive
    axes.monitor_signal_quality; config.notes are sent to the LLM in features.
    """

    monitor_id: str
    name: str
    type: str
    status: str | None = None
    tables: list[str] = Field(default_factory=list)  # EMPTY for `table` monitors (G1)
    sensitivity: str | None = None
    config: dict[str, Any] | None = None  # parsed JSON; None if missing/malformed
    created_at: UtcDatetime | None = None
    data_gaps: list[DataGap] = Field(default_factory=list)


class HistoryRow(BaseModel):
    """A past alert from alert_history.jsonl, used for action-rate priors.

    Contains alert_id, monitor_id, table_id, created_at, alert_type, and status
    (the outcome label: fixed, acknowledged, expected, no action needed, etc.).
    Indexed by HistoryIndex and scored in history.history_stats; leakage controls
    (as_of, exclude_alert_id) ensure only prior rows are used.
    """

    alert_id: str
    monitor_id: str
    table_id: str | None = None
    created_at: UtcDatetime
    alert_type: str
    status: str | None = None  # None = unlabeled, NOT dismissed
    data_gaps: list[DataGap] = Field(default_factory=list)


class TableCriticality(BaseModel):
    """Business importance and usage metrics for one table from table_criticality.jsonl.

    Holds criticality level, importance_score, is_golden, is_pii, and 30-day activity
    counts (reads, writes, active_users). Joined by ContextIndex.resolve; max across
    alert tables drives axes.business_criticality; is_pii triggers a score boost in combine.
    """

    table_id: str
    criticality: str | None = None
    importance_score: float | None = None
    is_golden: bool | None = None
    is_pii: bool | None = None
    reads_30d: float | None = None
    writes_30d: float | None = None
    active_users_30d: float | None = None
    data_gaps: list[DataGap] = Field(default_factory=list)


class BiReport(BaseModel):
    """A BI report from bi_reports.json with its upstream table dependencies.

    Contains report_id, name, and upstream_tables list. Used by ContextIndex to find
    reports hit by an alert's tables or their downstream; the count feeds the
    axes.blast_radius score via the bi_reports_cap log-scale.
    """

    report_id: str
    name: str
    upstream_tables: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)


class Dataset(BaseModel):
    """The complete typed dataset loaded from a zip or directory by loader.load_files.

    Contains alerts list, monitors dict (by monitor_id), history list, criticality dict
    (by table_id), bi_reports list, and lineage_edges as (source, target) pairs. This is
    the input passed to TriageService, ContextIndex and HistoryIndex (read-only by convention,
    not frozen).
    """

    alerts: list[Alert]
    monitors: dict[str, Monitor]
    history: list[HistoryRow]
    criticality: dict[str, TableCriticality]
    bi_reports: list[BiReport]
    lineage_edges: list[tuple[str, str]]

