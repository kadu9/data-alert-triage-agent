"""Scoring: the six deterministic axes, the LLM-fed context_signal axis, combine (tiers/overrides) and config validation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from triage.axes import (
    AxisResult,
    anomaly_severity,
    blast_radius,
    business_criticality,
    context_signal,
    historical_action_rate,
    monitor_signal_quality,
    recency_pressure,
)
from triage.combine import combine
from triage.config import load_config
from triage.context import AlertContext
from triage.guardrails import AlertFeatures
from triage.history import HistoryStats
from triage.schema import Alert, BiReport, Event, Monitor, TableCriticality

YAML_PATH = Path(__file__).parent.parent / "config" / "scoring.yaml"
CFG = load_config(YAML_PATH)
AX = CFG.axes
T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)
NAMES = list(CFG.weights)


def val(r) -> float:
    assert r.value is not None
    return r.value


def _ev(etype, actual=None, lo=None, hi=None, sc=None):
    return Event(event_id="e", created_at=T0, event_type=etype, actual_value=actual,
                 lower_threshold=lo, upper_threshold=hi, schema_changes=sc)


def _ctx(crit: dict[str, TableCriticality | None] | None = None, down=None, bi=(), status="enabled", pii=False, level: str | None = "HIGH"):
    if crit is None:
        crit = {"t": TableCriticality(table_id="t", criticality=level, is_pii=pii)}
    return AlertContext(
        alert=Alert(alert_id="a", monitor_id="m", created_at=T0, alert_type="volume", title="t", tables=list(crit)),
        monitor=Monitor(monitor_id="m", name="n", type="table", status=status),
        criticality=crit, downstream_tables=down or {}, bi_reports=list(bi), data_gaps=[],
    )


def _tc(**kw):
    return TableCriticality(table_id="t", **kw)


def _stats(rate, n_labeled, n_acted=0, scope="monitor"):
    return HistoryStats(scope=scope, n_total=n_labeled, n_labeled=n_labeled, n_acted=n_acted, action_rate=rate, data_gaps=[])


def _axes(v, **over):
    return {**{k: AxisResult(value=v) for k in NAMES}, **over}


def test_business_criticality_and_blast_radius_axes():
    hi = _tc(criticality="HIGH", importance_score=0.95, reads_30d=5000, writes_30d=500, active_users_30d=15)
    lo = _tc(criticality="LOW", importance_score=0.1, reads_30d=0, writes_30d=0, active_users_30d=2)
    c = AX.business_criticality
    r_hi, r_lo = business_criticality(_ctx({"a": hi}), c), business_criticality(_ctx({"b": lo}), c)
    assert val(r_lo) < val(r_hi) <= 1.0
    assert business_criticality(_ctx({"a": hi, "b": lo}), c).value == r_hi.value  # max over tables
    miss = business_criticality(_ctx({"a": None}), c)  # unknown table -> neutral + flagged, never imputed
    assert miss.value == c.missing_value and miss.low_support and miss.gaps
    assert business_criticality(_ctx({"a": _tc(criticality="HIGH")}), c).value == c.level_values["high"]  # partial nulls renormalise
    allnull = business_criticality(_ctx({"a": _tc()}), c)  # row present but every field null -> neutral, flagged
    assert allnull.value == c.missing_value and allnull.low_support
    golden = _tc(criticality="HIGH", importance_score=1.0, reads_30d=1e9, writes_30d=1e9, active_users_30d=1e9, is_golden=True)
    assert business_criticality(_ctx({"a": golden}), c).value == 1.0  # capped

    b = AX.blast_radius
    zero = blast_radius(_ctx(down={"a": frozenset()}), b)
    big = blast_radius(_ctx(down={"a": frozenset(f"t{i}" for i in range(2000))},
                            bi=[BiReport(report_id=str(i), name="r") for i in range(50)]), b)
    assert zero.value == 0.0 and big.value == pytest.approx(1.0)
    unk = blast_radius(_ctx(down={"a": None}), b)
    assert unk.low_support and unk.gaps
    partial = blast_radius(_ctx(down={"a": frozenset({"x", "y"}), "b": None}), b)  # lower bound, flagged
    full = blast_radius(_ctx(down={"a": frozenset({"x", "y"})}), b)
    assert partial.low_support and partial.gaps and not full.low_support and partial.value == full.value
    assert "3 downstream" in blast_radius(_ctx(down={"a": frozenset({"x", "y"}), "b": frozenset({"y", "z"})}), b).note


def test_history_and_anomaly_severity_axes():
    h = AX.historical_action_rate
    full = historical_action_rate(_stats(0.75, 8, 6), _stats(0.1, 50), h)
    assert full.value == 0.75 and full.weight_factor == 1.0 and not full.low_support
    thin = historical_action_rate(_stats(0.75, 2, 1), _stats(0.1, 50), h)
    assert thin.low_support and thin.weight_factor == h.low_support_weight_factor
    cold = historical_action_rate(_stats(None, 0), _stats(0.1, 50, scope="pool"), h)
    assert cold.value == 0.1 and cold.weight_factor == h.cold_start_weight_factor and cold.gaps
    none = historical_action_rate(_stats(None, 0), _stats(None, 0, scope="pool"), h)
    assert none.value is None and none.gaps  # no labels anywhere -> unknown, not "dismissed"

    s = AX.anomaly_severity
    assert anomaly_severity([_ev("metric_anomaly", 5, 0, 10)], s).value == s.breach_floor
    assert anomaly_severity([_ev("metric_anomaly", 40, 0, 10)], s).value == 1.0
    for bad in (_ev("row_count_anomaly", 5, 3, 3), _ev("metric_anomaly", None, 0, 10), _ev("metric_anomaly", 5, None, 10)):
        assert anomaly_severity([bad], s).value is None  # zero-width band / null -> gap, not a guess
    small, large = (anomaly_severity([_ev("validation_breach", x, 0, None)], s) for x in (1, 3018))
    assert val(small) < val(large) <= 1.0
    assert anomaly_severity([_ev("custom_rule_breach", 0, 0, 0)], s).value == s.breach_floor
    sc = lambda **k: {"fields_added": [], "fields_removed": [], "field_type_changes": [], **k}  # noqa: E731
    added = anomaly_severity([_ev("schema_change", sc=sc(fields_added=["a"]))], s)
    removed = anomaly_severity([_ev("schema_change", sc=sc(fields_removed=["a"]))], s)
    assert val(removed) > val(added)
    assert anomaly_severity([_ev("schema_change")], s).gaps
    nm = anomaly_severity([_ev("freshness_anomaly")], s)
    assert nm.value == s.no_magnitude_value and nm.low_support
    many = anomaly_severity([_ev("custom_rule_breach", 1, 0, None)] * 30, s)
    one = anomaly_severity([_ev("custom_rule_breach", 1, 0, None)], s)
    assert many.value == pytest.approx(val(one) + s.max_event_bump)  # repeat bump is capped
    assert anomaly_severity([_ev("brand_new_type", 1, 0, 1)], s).gaps and anomaly_severity([], s).gaps


def test_signal_quality_recency_and_context_signal_axes():
    q = AX.monitor_signal_quality
    m = Monitor(monitor_id="m", name="n", type="custom_sql",
                config={"custom_sql": [{"priority": "P3", "sensitivity": "low"}]})
    assert val(monitor_signal_quality(m, q)) == pytest.approx((q.sensitivity_values["low"] + q.priority_values["p3"]) / 2)
    assert monitor_signal_quality(None, q).value is None
    assert monitor_signal_quality(Monitor(monitor_id="m", name="n", type="custom_sql", config={"custom_sql": [{}]}), q).value is None

    r = AX.recency_pressure
    near = recency_pressure(T0, [T0 + timedelta(hours=3)], r)
    far = recency_pressure(T0, [T0 + timedelta(hours=r.window_hours + 1)], r)
    assert near.value == r.cluster_value and far.value == recency_pressure(T0, [], r).value == r.base_value

    c = AX.context_signal
    for f in (None, AlertFeatures()):
        x = context_signal(f, c)
        assert x.value is None and not x.low_support and not x.gaps  # nothing stated -> axis skipped
    assert val(context_signal(AlertFeatures(stated_noise="retired_or_draft", noise_quote="x" * 8), c)) == c.noise_values["retired_or_draft"]
    both = context_signal(AlertFeatures(recurrence="repeated", recurrence_quote="x" * 8, impact="high", impact_quote="y" * 8), c)
    assert val(both) == pytest.approx((c.recurrence_values["repeated"] + c.impact_values["high"]) / 2)


def test_combine_score_tiers_and_overrides():
    c = combine(_axes(0.5), _ctx(), CFG)
    assert c.score == pytest.approx(0.5) and c.tier == "MEDIUM" and c.confidence == "high"
    for v, tier in [(0.65, "HIGH"), (0.6499, "MEDIUM"), (0.35, "MEDIUM"), (0.3499, "LOW")]:
        assert combine(_axes(v), _ctx(), CFG).tier == tier
    none_axis = combine(_axes(0.8, recency_pressure=AxisResult(value=None)), _ctx(), CFG)
    assert none_axis.score == pytest.approx(0.8) and none_axis.confidence == "medium"  # renormalised, not zero
    assert combine(_axes(1.0, historical_action_rate=AxisResult(value=0.0, weight_factor=0.0)), _ctx(), CFG).score == pytest.approx(1.0)
    half = combine(_axes(0.0, blast_radius=AxisResult(value=1.0, weight_factor=0.5)), _ctx(), CFG)
    w = CFG.weights
    assert half.score == pytest.approx(w["blast_radius"] * 0.5 / (sum(w.values()) - w["blast_radius"] * 0.5))

    assert combine(_axes(0.5), _ctx(pii=True), CFG).score == pytest.approx(0.5 + CFG.overrides.pii_boost)
    assert combine(_axes(0.98), _ctx(pii=True), CFG).score == 1.0

    thin = {k: AxisResult(value=None) for k in NAMES} | {"business_criticality": AxisResult(value=1.0)}
    capped = combine(thin, _ctx(), CFG)  # too little evidence: HIGH capped to MEDIUM, confidence low
    assert capped.tier == "MEDIUM" and capped.confidence == "low"
    one = {k: AxisResult(value=None) for k in NAMES} | {"recency_pressure": AxisResult(value=1.0)}
    assert combine(one, _ctx(), CFG).confidence == "low"  # a single axis with data: low confidence (any one axis is below min_axes_weight)
    lone = combine(_axes(0.0, **{k: AxisResult(value=None) for k in NAMES if k != "recency_pressure"}), _ctx(), CFG)
    assert lone.tier == "LOW" and lone.cannot_tell  # LOW and only one axis with real data -> cannot tell
    assert not combine(one, _ctx(), CFG).cannot_tell  # same single axis but tier is not LOW
    neutral = {k: AxisResult(value=0.0, low_support=True) for k in NAMES} | {"recency_pressure": AxisResult(value=0.0)}
    assert combine(neutral, _ctx(), CFG).cannot_tell  # neutral defaults (low_support) do not count as data
    assert not combine(_axes(0.0), _ctx(), CFG).cannot_tell  # LOW but several axes with real data
    zero = {k: AxisResult(value=1.0, weight_factor=0.0) for k in NAMES} | {"business_criticality": AxisResult(value=1.0)}
    z = combine(zero, _ctx(), CFG)  # zero-weight axes do not count as evidence (effective weight)
    assert not z.cannot_tell or z.tier == "LOW"  # zero-weight axes are not real data for cannot_tell either
    zw = combine({k: AxisResult(value=0.0, weight_factor=0.0) for k in NAMES} | {"business_criticality": AxisResult(value=0.0), "recency_pressure": AxisResult(value=0.0, weight_factor=0.0)}, _ctx(), CFG)
    assert zw.tier == "LOW" and zw.cannot_tell  # one weighted axis only
    assert z.tier == "MEDIUM" and z.confidence == "low"
    paused = combine(_axes(0.9), _ctx(status="paused"), CFG)
    assert paused.tier == "MEDIUM" and combine(_axes(0.1), _ctx(status="paused"), CFG).tier == "LOW"
    assert any("paused" in r for r in combine(_axes(0.1), _ctx(status="paused"), CFG).reasons)
    nothing = combine({k: AxisResult(value=None) for k in NAMES}, _ctx(), CFG)
    assert nothing.score is None and nothing.tier == CFG.overrides.no_signal_tier and nothing.confidence == "low"
    weak = _axes(0.5, blast_radius=AxisResult(value=0.5, low_support=True), anomaly_severity=AxisResult(value=0.5, low_support=True))
    assert combine(weak, _ctx(), CFG).confidence == "low"
    assert combine(_axes(0.5), _ctx(level=None), CFG).confidence == "medium"  # missing criticality level caps confidence
    base = {k: AxisResult(value=0.5) for k in NAMES if k != "context_signal"}  # optional axis absent: no effect
    absent = combine(base, _ctx(), CFG)
    assert absent.score == pytest.approx(0.5) and absent.confidence == "high"
    with_sig = combine({**base, "context_signal": AxisResult(value=0.0)}, _ctx(), CFG)
    assert with_sig.score is not None and absent.score is not None and with_sig.score < absent.score


def test_config_rejects_invalid_values(tmp_path: Path):
    text = YAML_PATH.read_text()
    for old, new in [
        ("HIGH: 0.65", "HIGH: 0.30"),                                   # HIGH must exceed MEDIUM
        ("business_criticality: 0.28", "bogus: 0.28"),                  # weights must name exactly the axes
        ("tables_share: 0.5", "tables_share: 1.5"),                     # out of [0,1]
        ("tables_share: 0.5", "tables_share: 0.5\n    surprise: 1"),   # unknown key
        ("{reads_30d: 10000,", "{reads_x: 10000,"),                     # wrong map key
    ]:
        assert old in text, old
        p = tmp_path / "c.yaml"
        p.write_text(text.replace(old, new, 1))
        with pytest.raises(Exception):
            load_config(p)
    p = tmp_path / "c.yaml"  # history section is required and must be a mapping
    p.write_text("weights: {}\n")
    with pytest.raises(Exception):
        load_config(p)
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(p)
    assert "fixed" in CFG.history.acted_statuses and "rule_run_execution_error" in CFG.history.excluded_alert_types
