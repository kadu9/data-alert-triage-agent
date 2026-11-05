"""Gold cases: hand-built synthetic worlds (tests/gold/cases.json) with reviewer-intended outcomes, end to end through the service."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException

from triage.config import load_config
from triage.features import FeatureExtractor
from triage.guardrails import AlertFeatures
from triage.schema import Alert, BiReport, Dataset, Event, HistoryRow, Monitor, TableCriticality
from triage.service import TriageService
from triage.loader import parse_float
from tests.fakes import FakeChain, fast

ROOT = Path(__file__).parent.parent
CFG = load_config(ROOT / "config" / "scoring.yaml")
CASES = {c["name"]: c for c in json.loads((Path(__file__).parent / "gold" / "cases.json").read_text())["cases"]}
T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)
ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
TABLE = "w:s.t"


def _event(i, e):
    d = {"event_id": f"e{i}", "created_at": T0, **e}
    gaps = []
    for k in ("actual_value", "lower_threshold", "upper_threshold"):
        d[k], g = parse_float(e.get(k), f"events[{i}].{k}")
        if g:
            gaps.append(g)
    d["schema_changes"] = json.loads(e["schema_changes"]) if e.get("schema_changes") else None
    return Event.model_validate({**d, "data_gaps": gaps})


def build(case) -> Dataset:
    mon = {"monitor_id": "m", "name": "n", "type": "custom_sql", "status": "enabled", "tables": [TABLE], **case.get("monitor", {})}
    alert = {"alert_id": "a", "monitor_id": "m", "created_at": T0, "alert_type": "custom_sql", "title": "t", "tables": [TABLE], **case.get("alert", {})}
    alert["events"] = [_event(i, e) for i, e in enumerate(case.get("events", []))]
    crit = case.get("criticality", [{"table_id": TABLE, "criticality": "MEDIUM", "importance_score": 0.5}])
    hist_type = "anomalies" if alert["alert_type"] in {"volume", "freshness", "schema_change"} else "custom_rule_anomalies"
    hist = [
        HistoryRow(alert_id=f"h{i}", monitor_id="m", created_at=T0 - timedelta(days=i + 1), alert_type=hist_type, status=s)
        for i, s in enumerate(case.get("history_statuses", []))
    ]
    return Dataset(
        alerts=[Alert.model_validate(alert)], monitors={"m": Monitor.model_validate(mon)}, history=hist,
        criticality={c["table_id"]: TableCriticality.model_validate(c) for c in crit},
        bi_reports=[BiReport(report_id=b["report_id"], name=b["report_id"], upstream_tables=b["upstream"]) for b in case.get("bi", [])],
        lineage_edges=[tuple(e) for e in case.get("lineage", [])],
    )


def det(name):
    ds = build(CASES[name])
    return TriageService(ds, CFG, FeatureExtractor(FakeChain(), fast(CFG))).triage_deterministic(ds.alerts[0])


def gap_text(v):
    return " ".join(f"{g.field} {g.reason}" for g in v.data_gaps).lower()


async def _run(name, fn):
    chain = FakeChain(fn)
    ds = build(CASES[name])
    return chain, (await TriageService(ds, CFG, FeatureExtractor(chain, fast(CFG))).triage_all())[0]


def test_gold_expected_outcomes_and_null_policy():
    for name, case in CASES.items():
        v, exp = det(name), case["expected"]
        if "tier" in exp:
            assert v.triage_score == exp["tier"], name
        if "tier_at_least" in exp:
            assert ORDER[v.triage_score] >= ORDER[exp["tier_at_least"]], name
        if "confidence_not" in exp:
            assert v.confidence_support != exp["confidence_not"], name
        for needle in exp.get("gap_contains", []):
            assert needle in gap_text(v), (name, needle)
        for axis, want in exp.get("axis_equals", {}).items():
            assert v.axis_summary[axis] == pytest.approx(want), (name, axis)
        assert ("I can't tell from this data" in v.explanation) == v.cannot_tell, name  # phrase only when cannot_tell
        assert not v.cannot_tell or v.triage_score == "LOW", name  # cannot_tell only ever flags a LOW verdict
        assert not v.data_gaps or "Data gaps:" in v.explanation or v.cannot_tell, name  # other gaps are still listed
        assert v.confidence_support != "low" or v.data_gaps, name  # low confidence always names a reason
    cold = det("G-cold")  # no history is not "dismissed"
    assert cold.axis_summary["historical_action_rate"] is None and cold.confidence_support == "low"
    ev = {e.field: e.value for e in det("G-unlabeled").evidence}  # unlabeled rows are not zero votes
    assert ev["history.n_total"] == "10" and ev["history.n_labeled"] == "0"
    base, paused = det("G-high-crit"), det("G-paused")
    assert ORDER[paused.triage_score] == ORDER[base.triage_score] - 1 and paused.confidence_support != "high"
    tm = det("G-table-monitor")  # table monitors: scope comes from alert.tables
    assert any(TABLE in e.field for e in tm.evidence) and tm.axis_summary["business_criticality"] is not None
    sn = det("G-string-num")  # string-numeric garbage -> gap, no crash
    assert sn.status == "ok" and sn.axis_summary["anomaly_severity"] is None and sn.confidence_support == "low"


async def test_gold_llm_paths_injection_citation_and_fallback():
    chain, out = await _run("G-injection", lambda i: AlertFeatures())
    text = chain.inputs[0]["text"]
    assert '<untrusted_data field="description">' in text
    assert "root@evil.com" not in text and "sk-abcdefghijklmnopqrstuvwx" not in text
    assert out.triage_score == det("G-injection-control").triage_score  # text alone cannot move the tier

    quote = "retired and kept as a draft"
    case = {**CASES["G-high-crit"], "alert": {**CASES["G-high-crit"].get("alert", {}), "description": "This monitor is retired and kept as a draft."}}
    feats = AlertFeatures(stated_noise="retired_or_draft", noise_quote=quote)
    base = TriageService(build(case), CFG, FeatureExtractor(FakeChain(), fast(CFG))).triage_deterministic(build(case).alerts[0])
    out = (await TriageService(build(case), CFG, FeatureExtractor(FakeChain(lambda i: feats), fast(CFG))).triage_all())[0]
    assert out.status == "ok" and out.axis_summary["context_signal"] == 0.0
    assert out.score is not None and base.score is not None and out.score < base.score
    assert out.deterministic_tier == base.triage_score
    assert any(e.value == quote and e.source == "alerts.jsonl" and e.field == "description" for e in out.evidence)

    def fake(i):
        return AlertFeatures(stated_noise="expected_or_known", noise_quote="invented words not in the alert")

    def garbage(i):
        raise OutputParserException("garbage")

    for fn in (fake, garbage):  # bad model output -> flagged fallback with the structured-only tier
        _, v = await _run("G-high-crit", fn)
        assert v.status == "llm_fallback" and "context_signal" not in v.axis_summary
        assert v.triage_score == det("G-high-crit").triage_score and any(g.field == "llm_features" for g in v.data_gaps)
