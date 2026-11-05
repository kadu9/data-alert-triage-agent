from datetime import timedelta
import json

import pytest

from triage.loader import LoaderError, load_files, load_zip, parse_float


def jl(*rows):
    return "\n".join(json.dumps(r) for r in rows).encode()


def minimal(**over):
    alert = {
        "alert_id": "a1", "monitor_id": "m1", "created_at": "2025-01-01T00:00:00.000Z", "alert_type": "volume",
        "title": "t", "description": None, "status": "not_acknowledged", "severity": None, "owner": None, "priority": None,
        "tables": ["t:s.x"],
        "events": [{"event_id": "e1", "created_at": "2025-01-01T00:00:01.000Z", "event_type": "row_count_anomaly",
                    "actual_value": "abc", "lower_threshold": "1.5", "upper_threshold": None, "schema_changes": '{"fields_added":[]}'}],
    }
    mon = {"monitor_id": "m1", "name": "n", "type": "table", "status": "paused", "tables": [], "config": "{bad",
           "created_at": "2025-01-01 00:00:00.000000+00:00", "updated_at": None}
    hist = {"alert_id": "h1", "monitor_id": "m1", "table_id": None, "created_at": "2025-01-01T00:00:00.000Z",
            "alert_type": "anomalies", "status": None, "ticket_created": False}
    files = {
        "alerts.jsonl": jl({**alert, **over}), "monitors.jsonl": jl(mon), "alert_history.jsonl": jl(hist),
        "table_criticality.jsonl": jl({"table_id": "t:s.x", "criticality": None}),
        "bi_reports.json": b"[]", "lineage_edges.csv": b"source,target\na,b\n",
    }
    return files


def test_loader_null_policy_and_strict_inputs():
    for v, want in [("1.5", 1.5), (3, 3.0), ("1e3", 1000.0), (None, None)]:
        assert parse_float(v, "x") == (want, None)
    for v in ["", "abc", "nan", "inf", "1,000", True]:
        f, g = parse_float(v, "x")
        assert f is None and g is not None and g.field == "x"

    ds = load_files(minimal())  # nulls / bad numbers become gaps, never zero
    a, e, m, h = ds.alerts[0], ds.alerts[0].events[0], ds.monitors["m1"], ds.history[0]
    assert a.severity is None and a.owner is None and a.priority is None
    assert e.actual_value is None and e.lower_threshold == 1.5 and e.upper_threshold is None
    assert [g.field for g in e.data_gaps] == ["alerts[0].events[0].actual_value"]
    assert e.schema_changes == {"fields_added": []}
    assert m.config is None and m.data_gaps and m.tables == []
    assert h.status is None  # unlabeled stays None
    assert ds.criticality["t:s.x"].criticality is None and ds.lineage_edges == [("a", "b")]

    nullt = load_files(minimal(tables=None)).alerts[0]
    assert nullt.tables == [] and any(g.field == "alerts[0].tables" for g in nullt.data_gaps)
    assert load_files(minimal(events=[])).alerts[0].events == []

    with pytest.raises(LoaderError):  # naive timestamp is rejected, aware ones normalised to UTC
        load_files(minimal(created_at="2025-01-01T00:00:00"))
    shifted = load_files(minimal(created_at="2025-01-01T05:00:00+05:00")).alerts[0].created_at
    assert shifted.utcoffset() == timedelta(0) and shifted.hour == 0

    bi = minimal()
    bi["bi_reports.json"] = b"{}"
    with pytest.raises(LoaderError, match="expected array"):
        load_files(bi)
    gone = minimal()
    del gone["monitors.jsonl"]
    with pytest.raises(LoaderError, match="missing"):
        load_files(gone)
    bad = minimal()
    bad["alerts.jsonl"] = b"{oops"
    with pytest.raises(LoaderError, match="invalid JSON"):
        load_files(bad)
