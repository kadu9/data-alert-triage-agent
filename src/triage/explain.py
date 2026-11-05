"""Deterministic evidence + prose. Every sentence is backed by a citation whose value is copied from the data."""
from __future__ import annotations

from triage.axes import AxisResult
from triage.combine import Combined
from triage.context import AlertContext
from triage.history import HistoryStats
from triage.schema import DataGap
from triage.verdict import EvidenceCitation

GAP_PHRASE = "I can't tell from this data"  # said only when Verdict.cannot_tell; other gaps are listed as "Data gaps"


def _ev(source: str, field: str, value: object, note: str | None = None) -> EvidenceCitation:
    """Create an EvidenceCitation with stringified value."""
    return EvidenceCitation(source=source, field=field, value=str(value), note=note)


def build_evidence(
    ctx: AlertContext, monitor_stats: HistoryStats, axes: dict[str, AxisResult], comb: Combined, *, max_events: int
) -> list[EvidenceCitation]:
    """Build citations for raw and derived values used in scoring (traceable to source files)."""
    a = ctx.alert
    out = [_ev("alerts.jsonl", "alert_type", a.alert_type), _ev("alerts.jsonl", "events.length", len(a.events))]
    for i, e in enumerate(a.events[:max_events]):
        out.append(_ev("alerts.jsonl", f"events[{i}].event_type", e.event_type))
        for f in ("actual_value", "lower_threshold", "upper_threshold"):
            if (v := getattr(e, f)) is not None:
                out.append(_ev("alerts.jsonl", f"events[{i}].{f}", v))
    if ctx.monitor is not None:
        out.append(_ev("monitors.jsonl", "status", ctx.monitor.status))
    for t, c in ctx.criticality.items():
        if c is None:
            continue
        for f in ("criticality", "importance_score", "is_golden", "is_pii"):
            if (v := getattr(c, f)) is not None:
                out.append(_ev("table_criticality.jsonl", f"{t}.{f}", v))
    known = [d for d in ctx.downstream_tables.values() if d is not None]
    if known:
        out.append(_ev("derived", "downstream_tables.count", len(frozenset().union(*known)), "lineage_edges.csv transitive closure"))
    out.append(_ev("derived", "bi_reports.count", len(ctx.bi_reports), "bi_reports.json upstream_tables hit"))
    out.append(_ev("derived", "history.n_total", monitor_stats.n_total, "alert_history.jsonl rows for monitor_id"))
    out.append(_ev("derived", "history.n_labeled", monitor_stats.n_labeled))
    if monitor_stats.n_labeled:
        out.append(_ev("derived", "history.n_acted", monitor_stats.n_acted, "status in fixed/acknowledged"))
    # every axis value / adjustment quoted in the explanation gets its own citation (explanation == subset of evidence)
    out.append(_ev("derived", "combined.tier", comb.tier))
    out.append(_ev("derived", "combined.score", _fmt(comb.score)))
    out.append(_ev("derived", "combined.confidence_support", comb.confidence))
    out.append(_ev("derived", "combined.cannot_tell", comb.cannot_tell))
    for k, r in axes.items():
        out.append(_ev("derived", f"axis.{k}", _fmt(r.value), r.note or ("unknown: see data_gaps" if r.value is None else None)))
    for reason in comb.reasons:
        out.append(_ev("config/scoring.yaml", "adjustment", reason))
    return out


def _fmt(v: float | None) -> str:
    """Format a float as 2 decimals or 'n/a' for None."""
    return "n/a" if v is None else f"{v:.2f}"


def build_explanation(ctx: AlertContext, axes: dict[str, AxisResult], comb: Combined, gaps: list[DataGap]) -> str:
    """Build the deterministic explanation string from tier, axes, adjustments, and gaps."""
    parts = [
        f"Score {_fmt(comb.score)}, confidence {comb.confidence}: "
        f"{ctx.alert.alert_type} alert with {len(ctx.alert.events)} event(s).",
        "Axes: " + "; ".join(f"{k}={_fmt(r.value)}" + (f" [{r.note}]" if r.note else "") for k, r in axes.items()) + ".",
    ]
    if comb.reasons:
        parts.append("Adjustments: " + "; ".join(comb.reasons) + ".")
    if gaps:
        parts.append(gaps_sentence(gaps, cannot_tell=comb.cannot_tell))
    return " ".join(parts)


def gaps_sentence(gaps: list[DataGap], *, cannot_tell: bool) -> str:
    """List the data gaps; lead with GAP_PHRASE only when the verdict cannot be told from the data, otherwise with 'Data gaps'."""
    shown = "; ".join(f"{g.field} ({g.reason})" for g in gaps[:5])
    more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
    return f"{GAP_PHRASE if cannot_tell else 'Data gaps'}: {shown}{more}."
