from datetime import datetime, timezone

from triage.context import ContextIndex
from triage.loader import load_zip
from triage.schema import Alert, BiReport, DataGap, Dataset, Monitor, TableCriticality

T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _alert(tables, monitor_id="m1"):
    return Alert(alert_id="a1", monitor_id=monitor_id, created_at=T0, alert_type="volume", title="t", tables=tables)


def _ds(edges=(), reports=(), crit=(), monitors=()):
    return Dataset(
        alerts=[], monitors={m.monitor_id: m for m in monitors}, history=[],
        criticality={c.table_id: c for c in crit}, bi_reports=list(reports), lineage_edges=list(edges),
    )


def _mon(mid="m1", tables=()):
    return Monitor(monitor_id=mid, name="n", type="table", tables=list(tables))


def test_context_lineage_scope_gaps_and_bi_blast_radius():
    idx = ContextIndex(_ds(edges=[("a", "b"), ("b", "c")]))
    assert idx.downstream("a") == frozenset({"b", "c"}) and idx.downstream("c") == frozenset()
    assert ContextIndex(_ds(edges=[("a", "a"), ("a", "b"), ("b", "a")])).downstream("a") == frozenset({"b"})  # cycles terminate
    assert idx.downstream("zzz") is None  # absent from lineage is unknown, not empty

    ctx = ContextIndex(_ds(monitors=[_mon(tables=[])], edges=[("a", "b")])).resolve(_alert(["a"]))
    assert set(ctx.criticality) == {"a"} and ctx.monitor is not None  # scope = alert.tables (G1)

    bare = ContextIndex(_ds()).resolve(_alert(["a"]))
    assert {"monitor", "criticality[a]", "lineage[a]"} <= {g.field for g in bare.data_gaps}
    assert bare.criticality["a"] is None and bare.downstream_tables["a"] is None
    nullc = ContextIndex(_ds(crit=[TableCriticality(table_id="a", criticality=None)], monitors=[_mon()])).resolve(_alert(["a"]))
    assert any(g.field == "criticality[a].criticality" for g in nullc.data_gaps)

    a = _alert(["a"]).model_copy(update={"data_gaps": [DataGap(field="alerts.x", reason="r")]})
    ds = _ds(crit=[TableCriticality(table_id="a", data_gaps=[DataGap(field="crit.y", reason="r")])], monitors=[_mon()])
    assert {"alerts.x", "crit.y"} <= {g.field for g in ContextIndex(ds).resolve(a).data_gaps}  # upstream gaps propagate

    reps = [
        BiReport(report_id="r2", name="x", upstream_tables=["c"]),
        BiReport(report_id="r1", name="y", upstream_tables=["a", "c"]),
        BiReport(report_id="r3", name="z", upstream_tables=["unrelated"]),
        BiReport(report_id="r0", name="w", upstream_tables=["missing"]),
    ]
    ctx = ContextIndex(_ds(edges=[("a", "b"), ("b", "c")], reports=reps, monitors=[_mon()])).resolve(_alert(["missing", "a"]))
    assert [r.report_id for r in ctx.bi_reports] == ["r0", "r1", "r2"]  # direct + via downstream, sorted, deduped
