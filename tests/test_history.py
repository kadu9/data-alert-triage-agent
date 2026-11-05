from datetime import datetime, timezone

import pytest

from triage.config import HistoryConfig
from triage.history import HistoryIndex, history_stats
from triage.schema import HistoryRow

CFG = HistoryConfig(
    laplace_alpha=1,
    acted_statuses=frozenset({"fixed", "acknowledged"}),
    not_acted_statuses=frozenset({"no action needed", "false positive", "expected"}),
    excluded_alert_types=frozenset({"rule_run_execution_error"}),
    type_map={"custom_sql": ("custom_rule_anomalies",), "volume": ("anomalies",)},
)


def _row(i, status=None, monitor="m1", atype="custom_rule_anomalies", day=1):
    return HistoryRow(alert_id=f"h{i}", monitor_id=monitor, created_at=datetime(2025, 6, day, tzinfo=timezone.utc),
                      alert_type=atype, status=status)


def test_history_labels_are_priors_with_leakage_controls():
    s = history_stats([_row(1), _row(2), _row(3, "fixed")], CFG, "monitor")  # unlabeled rows are not votes
    assert (s.n_total, s.n_labeled, s.n_acted) == (3, 1, 1) and s.action_rate == pytest.approx(2 / 3)  # (1+1)/(1+2)
    assert history_stats([_row(1), _row(2)], CFG, "monitor").action_rate is None  # no labels -> unknown + gap
    assert history_stats([_row(1), _row(2)], CFG, "monitor").data_gaps
    assert history_stats([], CFG, "monitor").action_rate is None
    plumbing = history_stats([_row(1, "fixed", atype="rule_run_execution_error"), _row(2, "expected")], CFG, "monitor")
    assert plumbing.n_total == 1 and plumbing.n_acted == 0  # plumbing errors are not data signal
    odd = history_stats([_row(1, "weird"), _row(2, "fixed")], CFG, "monitor")
    assert odd.n_labeled == 1 and any("unrecognised" in g.reason for g in odd.data_gaps)

    idx = HistoryIndex([_row(1, "fixed", day=1), _row(2, "expected", day=5), _row(3, "fixed", monitor="m2")], CFG)
    assert idx.monitor_stats("m1").n_total == 2 and idx.monitor_stats("nope").n_total == 0
    assert idx.monitor_stats("m1", as_of=datetime(2025, 6, 5, tzinfo=timezone.utc)).n_total == 1  # strict <
    assert idx.monitor_stats("m1", exclude_alert_id="h1").n_total == 1

    pool = HistoryIndex([_row(1, "fixed"), _row(2, "expected", atype="anomalies", day=5)], CFG)
    assert pool.pool_stats("custom_sql").n_acted == 1 and pool.pool_stats("volume").n_total == 1
    assert pool.pool_stats("volume", as_of=datetime(2025, 6, 5, tzinfo=timezone.utc)).n_total == 0
    unmapped = pool.pool_stats("metric")
    assert unmapped.action_rate is None and "type_map" in unmapped.data_gaps[0].field
