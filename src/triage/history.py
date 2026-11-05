"""Monitor-keyed history priors. Unlabeled rows (status=None) are NOT votes (PLAN section 4)."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel

from triage.config import HistoryConfig
from triage.schema import DataGap, HistoryRow


class HistoryStats(BaseModel):
    """Aggregated action-rate statistics from history rows.

    Holds scope (monitor or alert_type_pool), counts (n_total, n_labeled, n_acted),
    Laplace-smoothed action_rate (None if no labeled rows), and data_gaps. Computed
    by history_stats and returned by HistoryIndex methods; used by axes.historical_action_rate.
    """

    scope: str                         # "monitor" | "alert_type_pool"
    n_total: int                       # rows considered (excluded plumbing types removed)
    n_labeled: int
    n_acted: int
    action_rate: float | None          # Laplace-smoothed; None when n_labeled == 0
    data_gaps: list[DataGap]


def history_stats(rows: Iterable[HistoryRow], cfg: HistoryConfig, scope: str) -> HistoryStats:
    """Compute Laplace-smoothed action rate from labeled history rows.

    In plain words: counts how many past alerts were acted on (fixed/acknowledged) vs how many
    were labeled at all. Uses Laplace smoothing to handle sparse data - adds a pseudo-count
    to avoid extreme rates when history is thin. Unlabeled rows (status=None) are NOT counted
    as "not acted"; they are simply ignored.

    Logic:
      1. Exclude rows whose alert_type is in excluded_alert_types (plumbing, not signal).
      2. Count acted (status in acted_statuses), not_acted (status in not_acted_statuses),
         unlabeled (status is None).
      3. labeled = acted + not_acted. Rows with unrecognised status are ignored + recorded.
      4. action_rate = (acted + laplace_alpha) / (labeled + 2 * laplace_alpha) if labeled > 0,
         else None.

    Example (config: laplace_alpha=1):
        11 rows with acted status, 0 with not_acted -> labeled=11
        action_rate = (11+1) / (11+2) = 12/13 = 0.9231

        2 rows acted, 28 rows not_acted -> labeled=30
        action_rate = (2+1) / (30+2) = 3/32 = 0.0938
    """
    rows = [r for r in rows if r.alert_type not in cfg.excluded_alert_types]
    acted = sum(r.status in cfg.acted_statuses for r in rows)
    not_acted = sum(r.status in cfg.not_acted_statuses for r in rows)
    unlabeled = sum(r.status is None for r in rows)
    unknown = len(rows) - acted - not_acted - unlabeled
    labeled = acted + not_acted
    a = cfg.laplace_alpha
    gaps = [DataGap(field=f"history[{scope}]", reason="no labeled rows; action rate unknown")] if labeled == 0 else []
    if unknown:
        gaps.append(DataGap(field=f"history[{scope}].status", reason=f"{unknown} rows with unrecognised label ignored"))
    return HistoryStats(
        scope=scope,
        n_total=len(rows),
        n_labeled=labeled,
        n_acted=acted,
        action_rate=(acted + a) / (labeled + 2 * a) if labeled else None,
        data_gaps=gaps,
    )


class HistoryIndex:
    """Pre-built index for history lookups by monitor_id and alert_type.

    Constructed once per Dataset with HistoryConfig. Provides monitor_stats and pool_stats
    methods with leakage controls (as_of, exclude_alert_id) so only prior rows are used.
    Used by TriageService to feed axes.historical_action_rate.
    """

    def __init__(self, rows: Iterable[HistoryRow], cfg: HistoryConfig):
        """Group history rows by monitor and by alert type."""
        self._cfg = cfg
        self._by_monitor: dict[str, list[HistoryRow]] = defaultdict(list)
        self._by_type: dict[str, list[HistoryRow]] = defaultdict(list)
        for r in rows:
            self._by_monitor[r.monitor_id].append(r)
            self._by_type[r.alert_type].append(r)

    def _filter(self, rows: list[HistoryRow], as_of: datetime | None, exclude_alert_id: str | None) -> list[HistoryRow]:
        """Filter rows to those before as_of and excluding a specific alert_id (leakage control)."""
        return [
            r for r in rows
            if (as_of is None or r.created_at < as_of) and r.alert_id != exclude_alert_id
        ]

    def monitor_stats(
        self, monitor_id: str, *, as_of: datetime | None = None, exclude_alert_id: str | None = None
    ) -> HistoryStats:
        """Return action-rate stats for one monitor's history.

        In plain words: fetches all past alerts for a specific monitor, applies leakage controls
        (only rows before as_of, excluding a specific alert_id), and computes the Laplace-smoothed
        action rate. This is the primary input to the historical_action_rate axis.
        """
        rows = self._filter(self._by_monitor.get(monitor_id, []), as_of, exclude_alert_id)
        return history_stats(rows, self._cfg, "monitor")

    def pool_stats(
        self, alert_type: str, *, as_of: datetime | None = None, exclude_alert_id: str | None = None
    ) -> HistoryStats:
        """Return pooled action-rate stats across history alert_types mapped from an alerts.jsonl type.

        In plain words: when a monitor has no labeled history, this provides a fallback by pooling
        history from all monitors with similar alert types. Uses history.type_map to bridge the
        taxonomy difference between alerts.jsonl and alert_history.jsonl.

        Uses history.type_map[alert_type] to find matching history alert_types.
        """
        mapped = self._cfg.type_map.get(alert_type)
        if mapped is None:
            return HistoryStats(
                scope="alert_type_pool", n_total=0, n_labeled=0, n_acted=0, action_rate=None,
                data_gaps=[DataGap(field="history.type_map", reason=f"no history type mapping for alert_type {alert_type!r}")],
            )
        rows = [r for t in mapped for r in self._by_type.get(t, [])]
        return history_stats(self._filter(rows, as_of, exclude_alert_id), self._cfg, "alert_type_pool")
