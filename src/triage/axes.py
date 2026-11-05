"""Seven scoring axes: six deterministic plus the LLM-label-derived context_signal. Pure, sync, config-driven. Each returns AxisResult: value in [0,1] or None (+ DataGap)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from triage.config import (
    BlastRadiusAxis,
    ContextSignalAxis,
    CriticalityAxis,
    HistoryAxis,
    RecencyAxis,
    SeverityAxis,
    SignalQualityAxis,
)
from triage.context import AlertContext
from triage.guardrails import AlertFeatures
from triage.history import HistoryStats
from triage.schema import DataGap, Event, Monitor, TableCriticality


class AxisResult(BaseModel):
    """Result of scoring one axis, returned by each axis function.

    Holds value in [0,1] (None if unknown), weight_factor for low-support shrink,
    low_support flag, a human-readable note, and gaps list. Used by combine.combine
    to compute the weighted score; the note appears in evidence citations.
    """

    value: float | None  # None = can't tell; combine drops it and renormalises
    weight_factor: float = 1.0  # combine multiplies the configured weight by this (low-support shrink)
    low_support: bool = False  # value exists but rests on thin/no-magnitude evidence -> lowers confidence_support
    note: str = ""  # human-readable basis, used for citations/explanations
    gaps: list[DataGap] = Field(default_factory=list)


def _unknown(axis: str, reason: str) -> AxisResult:
    """Return an AxisResult with no value and a DataGap explaining why."""
    return AxisResult(value=None, gaps=[DataGap(field=axis, reason=reason)])


def _log_scale(x: float, cap: float) -> float:
    """Map x to [0, 1] via log1p, capped at 1 when x >= cap.

    In plain words: takes a raw count (like downstream tables) and compresses it to [0, 1]
    using a logarithmic curve so that the first few items matter more than later ones. Values
    at or above the cap saturate to 1.0.

    Logic: result = min(log(1 + max(x, 0)) / log(1 + cap), 1.0).

    Example (config cap 2000):
        _log_scale(387, 2000) = log1p(387) / log1p(2000) = 5.96 / 7.60 = 0.7842
        _log_scale(6, 50) = log1p(6) / log1p(50) = 1.95 / 3.93 = 0.4949
    """
    return min(math.log1p(max(x, 0.0)) / math.log1p(cap), 1.0)


def _mean_weighted(parts: dict[str, tuple[float, float]]) -> float | None:
    """Compute weighted mean, renormalising over the parts supplied (skip missing).

    In plain words: given a dict of {name: (value, weight)}, sum up value*weight for each part
    and divide by the sum of weights. Returns None if total weight is zero. Used throughout
    scoring to blend multiple components with potentially missing values.

    Logic: total_w = sum(w); result = sum(v*w) / total_w if total_w > 0, else None.

    Example:
        parts = {'importance': (0.983, 0.4), 'level': (1.0, 0.3), 'usage': (0.91, 0.3)}
        numer = 0.983*0.4 + 1.0*0.3 + 0.91*0.3 = 0.966
        denom = 0.4 + 0.3 + 0.3 = 1.0
        result = 0.966 / 1.0 = 0.966
    """
    total = sum(w for _, w in parts.values())
    return sum(v * w for v, w in parts.values()) / total if total > 0 else None



def _table_criticality_score(c: TableCriticality, cfg: CriticalityAxis) -> float | None:
    """Score one table's criticality by blending importance, level, and usage.

    In plain words: combines three signals (importance_score from the data, criticality level
    mapped to a number, and log-scaled usage metrics) into one [0,1] score. Each component is
    optional; missing ones are skipped and weights are renormalized. Golden tables get a bonus.

    Logic: weighted mean of (importance_score, level_values[criticality], usage_mean)
    using component_weights from config. Golden tables get +golden_bonus. Returns None
    if no usable component. Uses _log_scale for usage fields (usage_caps: reads_30d,
    writes_30d, active_users_30d).

    Example (config: component_weights importance=0.4, level=0.3, usage=0.3;
             level_values high=1.0; usage_caps reads=10000, writes=2000, users=20):
        importance_score=0.983, criticality='HIGH', reads_30d=12000, writes_30d=500, active_users_30d=15
        reads_scaled = 1.0 (capped), writes_scaled = 0.8178, users_scaled = 0.9107
        usage_mean = (1.0 + 0.8178 + 0.9107)/3 = 0.9095
        base = (0.983*0.4 + 1.0*0.3 + 0.9095*0.3) / 1.0 = 0.9661
    """
    parts: dict[str, tuple[float, float]] = {}
    w = cfg.component_weights
    if c.importance_score is not None:
        parts["importance"] = (min(max(c.importance_score, 0.0), 1.0), w.get("importance", 0.0))
    if c.criticality is not None and c.criticality.lower() in cfg.level_values:
        parts["level"] = (cfg.level_values[c.criticality.lower()], w.get("level", 0.0))
    caps = cfg.usage_caps
    usage = [
        _log_scale(v, cap)
        for v, cap in (
            (c.reads_30d, caps["reads_30d"]),
            (c.writes_30d, caps["writes_30d"]),
            (c.active_users_30d, caps["active_users_30d"]),
        )
        if v is not None
    ]
    if usage:
        parts["usage"] = (sum(usage) / len(usage), w.get("usage", 0.0))
    base = _mean_weighted(parts)
    if base is None:
        return None
    return min(base + (cfg.golden_bonus if c.is_golden else 0.0), 1.0)


def business_criticality(ctx: AlertContext, cfg: CriticalityAxis) -> AxisResult:
    """Return the highest table-criticality score among the alert's tables.

    Logic: score each table via _table_criticality_score, take max. If no table
    has usable criticality data, return axes.business_criticality.missing_value
    with low_support=True.
    """
    scored = {t: s for t, c in ctx.criticality.items() if c is not None and (s := _table_criticality_score(c, cfg)) is not None}
    if not scored:
        # missing criticality is the documented exception: neutral mid value + confidence penalty, never imputed
        return AxisResult(
            value=cfg.missing_value,
            low_support=True,
            note="no usable table_criticality for any alert table",
            gaps=[DataGap(field="business_criticality", reason="no alert table has usable table_criticality data; neutral value used")],
        )
    top = max(scored, key=lambda t: (scored[t], t))
    return AxisResult(value=scored[top], note=f"max over {len(scored)} table(s); driver {top}")



def blast_radius(ctx: AlertContext, cfg: BlastRadiusAxis) -> AxisResult:
    """Score the potential downstream impact of the alert's tables.

    In plain words: measures how far data-quality issues could propagate. Counts downstream
    tables (via lineage graph) and BI reports that consume the affected tables, then blends
    these two signals using tables_share. Tables not in lineage are flagged as unknown.

    Logic:
      1. n_down = count of transitive downstream tables from lineage_edges.csv.
      2. n_bi = count of BI reports whose upstream_tables hit alert tables or downstream.
      3. value = tables_share * log_scale(n_down, downstream_tables_cap)
                 + (1 - tables_share) * log_scale(n_bi, bi_reports_cap).
      4. If no alert table is in lineage, only the BI score is used (value = bi score), low_support=True.
         If only some tables are missing from lineage, the value is kept but flagged as a lower bound.

    Example (config: downstream_tables_cap=2000, bi_reports_cap=50, tables_share=0.5):
        n_down=387, n_bi=6
        tables_score = log_scale(387, 2000) = 0.7842
        bi_score = log_scale(6, 50) = 0.4949
        value = 0.5 * 0.7842 + 0.5 * 0.4949 = 0.6396
    """
    known = [d for d in ctx.downstream_tables.values() if d is not None]
    missing = sorted(t for t, d in ctx.downstream_tables.items() if d is None)
    n_bi = len(ctx.bi_reports)
    bi_score = _log_scale(n_bi, cfg.bi_reports_cap)
    if not known:
        # lineage unknown for every alert table: BI hits (direct upstream matches) are a lower bound only
        return AxisResult(
            value=bi_score,
            low_support=True,
            note=f"lineage unknown; {n_bi} BI report(s) directly upstream",
            gaps=[DataGap(field="blast_radius", reason="no alert table found in lineage; downstream tables unknown")],
        )
    n_down = len(frozenset().union(*known))
    value = cfg.tables_share * _log_scale(n_down, cfg.downstream_tables_cap) + (1 - cfg.tables_share) * bi_score
    if missing:
        return AxisResult(
            value=value,
            low_support=True,
            note=f"{n_down} downstream table(s), {n_bi} BI report(s); lower bound (no lineage for {', '.join(missing)})",
            gaps=[DataGap(field="blast_radius", reason=f"partial lineage coverage; score is a lower bound; not in lineage: {', '.join(missing)}")],
        )
    return AxisResult(value=value, note=f"{n_down} downstream table(s), {n_bi} BI report(s)")


def historical_action_rate(monitor_stats: HistoryStats, pool_stats: HistoryStats, cfg: HistoryAxis) -> AxisResult:
    """Return Laplace-smoothed action rate from monitor history, else alert-type pool.

    In plain words: looks at past alerts from the same monitor and computes what fraction
    the team acted on (fixed/acknowledged). Uses Laplace smoothing to handle sparse data.
    Falls back to a pool of similar alert types if this monitor has no history.

    Logic:
      1. If monitor has labeled history: use monitor_stats.action_rate.
         - If n_labeled < n_labeled_min: weight_factor = low_support_weight_factor.
      2. Else if pool has labeled history: use pool_stats.action_rate with
         weight_factor = cold_start_weight_factor, low_support=True.
      3. Else return None (unknown).

    Example (config: laplace_alpha=1, n_labeled_min=5):
        monitor has 11 acted out of 11 labeled
        action_rate = (11+1) / (11+2) = 12/13 = 0.9231

        monitor has 2 acted out of 30 labeled
        action_rate = (2+1) / (30+2) = 3/32 = 0.0938
    """
    gaps = [*monitor_stats.data_gaps, *pool_stats.data_gaps]
    if monitor_stats.action_rate is not None:
        thin = monitor_stats.n_labeled < cfg.n_labeled_min
        return AxisResult(
            value=monitor_stats.action_rate,
            weight_factor=cfg.low_support_weight_factor if thin else 1.0,
            low_support=thin,
            note=f"monitor: {monitor_stats.n_acted} acted / {monitor_stats.n_labeled} labeled of {monitor_stats.n_total} rows",
            gaps=gaps,
        )
    if pool_stats.action_rate is not None:
        gaps.append(DataGap(field="historical_action_rate", reason="monitor has no labeled history; alert-type pool prior used"))
        return AxisResult(
            value=pool_stats.action_rate,
            weight_factor=cfg.cold_start_weight_factor,
            low_support=True,
            note=f"pool {pool_stats.scope}: {pool_stats.n_acted} acted / {pool_stats.n_labeled} labeled",
            gaps=gaps,
        )
    res = _unknown("historical_action_rate", "no labeled history for monitor or alert-type pool")
    res.gaps.extend(gaps)
    return res



def _breach_excess(ev: Event) -> float | None:
    """Return how far the actual value is above the allowed threshold (same unit as the value).

    In plain words: for breach-type events (like validation_breach), this computes the overage.
    If actual=150 and threshold=100, the excess is 50. Returns None if values are missing.

    Logic: allowed = upper_threshold ?? lower_threshold; excess = max(actual - allowed, 0).
    Returns None if actual_value or allowed is missing.
    """
    allowed = ev.upper_threshold if ev.upper_threshold is not None else ev.lower_threshold
    if ev.actual_value is None or allowed is None:
        return None
    return max(ev.actual_value - allowed, 0.0)


def _band_distance(ev: Event) -> float | None:
    """Return distance outside the [lower, upper] threshold band.

    In plain words: for band-type events (like metric_anomaly), this computes how far outside
    the acceptable range the value is. If actual=80 and band is [40,60], the distance is 20.

    Logic: distance = max(actual - upper, lower - actual, 0).
    Returns None if any of actual_value, lower_threshold, upper_threshold is missing.
    """
    if ev.actual_value is None or ev.lower_threshold is None or ev.upper_threshold is None:
        return None
    return max(ev.actual_value - ev.upper_threshold, ev.lower_threshold - ev.actual_value, 0.0)


def _event_severity(ev: Event, rule: str, cfg: SeverityAxis, idx: int) -> tuple[float | None, bool, list[DataGap]]:
    """Score one event's severity based on its rule type.

    In plain words: each event type has a rule (band, breach, schema, or none) that determines
    how to measure its severity. This function applies the appropriate formula and returns
    a [0,1] score.

    Logic (by rule):
      - none: return no_magnitude_value, low_support=True.
      - schema: sum(len(changes[k]) * schema_weights[k]) / schema_cap.
      - band: distance / band_width / band_multiple_cap, then breach_floor + (1 - breach_floor) * mag.
      - breach: log_scale(excess, breach_excess_cap), then breach_floor + (1 - breach_floor) * mag.
    Returns (value, low_support, gaps).

    Example (config: breach_excess_cap=1000, breach_floor=0.3):
        breach rule, actual=150, threshold=100
        excess = 50
        mag = log_scale(50, 1000) = 0.5691
        score = 0.3 + 0.7 * 0.5691 = 0.6984

    Example (config: band_multiple_cap=2.0, breach_floor=0.3):
        band rule, actual=80, lower=40, upper=60
        distance = 20, width = 20
        mag = 20/20/2.0 = 0.5
        score = 0.3 + 0.7 * 0.5 = 0.65
    """
    f = f"events[{idx}]"
    if rule == "none":
        return cfg.no_magnitude_value, True, []
    if rule == "schema":
        sc = ev.schema_changes
        if sc is None:
            return None, False, [DataGap(field=f"{f}.schema_changes", reason="schema_changes missing; cannot size the change")]
        score = sum(len(sc.get(k) or []) * w for k, w in cfg.schema_weights.items())
        return min(score / cfg.schema_cap, 1.0), False, []
    if rule == "band":
        dist = _band_distance(ev)
        width = None if ev.lower_threshold is None or ev.upper_threshold is None else ev.upper_threshold - ev.lower_threshold
        if dist is None or width is None or width <= 0:
            return None, False, [DataGap(field=f"{f}.thresholds", reason="actual/band bounds missing or zero-width; cannot normalise")]
        mag = min(dist / width / cfg.band_multiple_cap, 1.0)
    else:
        excess = _breach_excess(ev)
        if excess is None:
            return None, False, [DataGap(field=f"{f}.actual_value", reason="actual_value or threshold missing; cannot measure breach size")]
        mag = _log_scale(excess, cfg.breach_excess_cap)
    return cfg.breach_floor + (1 - cfg.breach_floor) * mag, False, []


def anomaly_severity(alert_events: Sequence[Event], cfg: SeverityAxis) -> AxisResult:
    """Score the alert's events by max severity plus a bump for multiple events.

    In plain words: takes all events in the alert, scores each one according to its type
    (band, breach, schema, or none), then returns the maximum. Multiple scored events add
    a small bonus to reflect that more anomalies = more serious.

    Logic:
      1. For each event, look up event_rules[event_type] to get a rule (band/breach/schema/none).
      2. Score each event via _event_severity.
      3. value = min(max(event scores) + min((n_scored - 1) * extra_event_bump, max_event_bump), 1.0),
         where n_scored counts only events that produced a score; events without a rule or data are gaps.
    """
    vals: list[float] = []
    low = False
    gaps: list[DataGap] = []
    for i, ev in enumerate(alert_events):
        rule = cfg.event_rules.get(ev.event_type)
        if rule is None:
            gaps.append(DataGap(field=f"events[{i}].event_type", reason=f"no severity rule for event_type {ev.event_type!r}"))
            continue
        v, lo, g = _event_severity(ev, rule, cfg, i)
        gaps.extend(g)
        if v is not None:
            vals.append(v)
            low = low or lo
    if not vals:
        return AxisResult(value=None, gaps=gaps or [DataGap(field="anomaly_severity", reason="alert has no events")])
    bump = min((len(vals) - 1) * cfg.extra_event_bump, cfg.max_event_bump)
    return AxisResult(
        value=min(max(vals) + bump, 1.0),
        low_support=low and max(vals) == cfg.no_magnitude_value,
        note=f"{len(vals)} scored event(s)",
        gaps=gaps,
    )



def inner_config(monitor: Monitor) -> dict[str, Any]:
    """Return the type-specific config dict from monitor.config[monitor.type][0]."""
    section = (monitor.config or {}).get(monitor.type)
    if isinstance(section, list) and section and isinstance(section[0], dict):
        return section[0]
    return {}


def monitor_signal_quality(monitor: Monitor | None, cfg: SignalQualityAxis) -> AxisResult:
    """Score the monitor's credibility from sensitivity and inner-config priority.

    In plain words: combines monitor settings (sensitivity level and priority) to estimate
    how reliable this monitor's alerts are. Lower sensitivity (fewer alerts) and higher
    priority both indicate more credible signals.

    Logic: plain mean of sensitivity_values[sensitivity] (monitor.sensitivity, else inner config) and
    priority_values[priority] (inner config only). Returns None if neither is present.
    Does NOT use paused status (that's a tier override in combine).
    """
    if monitor is None:
        return _unknown("monitor_signal_quality", "monitor not found")
    inner = inner_config(monitor)
    parts: dict[str, tuple[float, float]] = {}
    sens = monitor.sensitivity or inner.get("sensitivity")
    if isinstance(sens, str) and sens.lower() in cfg.sensitivity_values:
        parts["sensitivity"] = (cfg.sensitivity_values[sens.lower()], 1.0)
    prio = inner.get("priority")
    if isinstance(prio, str) and prio.lower() in cfg.priority_values:
        parts["priority"] = (cfg.priority_values[prio.lower()], 1.0)
    value = _mean_weighted(parts)
    if value is None:
        return _unknown("monitor_signal_quality", "monitor has neither sensitivity nor inner-config priority")
    return AxisResult(value=value, note=", ".join(sorted(parts)))



def recency_pressure(alert_time: datetime, peer_times: Sequence[datetime], cfg: RecencyAxis) -> AxisResult:
    """Score whether this monitor fired multiple times within a window.

    In plain words: detects alert clustering - if the same monitor fired multiple times
    recently, that suggests a real ongoing problem rather than a one-off noise event.

    Logic: if any peer alert from the same monitor is within window_hours of this alert,
    return cluster_value; else return base_value (both from axes.recency_pressure).
    """
    window = timedelta(hours=cfg.window_hours)
    clustered = any(abs(alert_time - t) <= window for t in peer_times)
    return AxisResult(
        value=cfg.cluster_value if clustered else cfg.base_value,
        note=f"same-monitor re-fire within {cfg.window_hours:g}h" if clustered else "no same-monitor re-fire in window",
    )


def context_signal(features: AlertFeatures | None, cfg: ContextSignalAxis) -> AxisResult:
    """Score LLM-extracted labels from free text (noise, recurrence, impact).

    In plain words: the LLM reads the alert's title, description, and monitor notes and
    extracts structured labels. This function converts those labels to numbers and averages
    them. If nothing was stated (all labels are none/unknown), the axis is skipped entirely.

    Logic: for each stated label (not none/unknown), look up its value in noise_values,
    recurrence_values, or impact_values; return the mean. If nothing is stated, return None
    (axis is skipped, does not hurt confidence).

    Example (config: noise_values retired_or_draft=0.0, recurrence_values repeated=0.8):
        stated_noise=retired_or_draft, recurrence=repeated, impact=unknown
        context_signal = mean([0.0, 0.8]) = 0.40

    Example (config: noise_values expected_or_known=0.1, recurrence_values repeated=0.8,
             impact_values high=0.9):
        all three stated: context_signal = mean([0.1, 0.8, 0.9]) = 0.60
    """
    if features is None:
        return AxisResult(value=None, note="no LLM features")
    parts: dict[str, tuple[float, float]] = {}
    if features.stated_noise in cfg.noise_values:
        parts[f"stated_noise={features.stated_noise}"] = (cfg.noise_values[features.stated_noise], 1.0)
    if features.recurrence in cfg.recurrence_values:
        parts[f"recurrence={features.recurrence}"] = (cfg.recurrence_values[features.recurrence], 1.0)
    if features.impact in cfg.impact_values:
        parts[f"impact={features.impact}"] = (cfg.impact_values[features.impact], 1.0)
    return AxisResult(value=_mean_weighted(parts), note=f"{len(parts)} label(s) the LLM read from cited free-text quotes" if parts else "nothing stated")
