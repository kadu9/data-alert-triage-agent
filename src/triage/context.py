"""Per-alert context: monitor, criticality, downstream lineage, BI blast radius. Null-safe (gaps, not guesses)."""
from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel

from triage.schema import Alert, BiReport, DataGap, Dataset, Monitor, TableCriticality


class AlertContext(BaseModel):
    """All context joined to one alert: monitor, criticality, lineage, BI reports, gaps.

    Built by ContextIndex.resolve; holds the alert itself, its Monitor (if found),
    criticality dict per table, downstream_tables dict (None = unknown), bi_reports
    list, and accumulated data_gaps. Used by axes and combine steps.
    """

    alert: Alert
    monitor: Monitor | None
    # alert.tables is authoritative: monitors.tables[] is empty for `table` monitors (G1).
    criticality: dict[str, TableCriticality | None]
    downstream_tables: dict[str, frozenset[str] | None]  # None = table absent from lineage (can't tell)
    bi_reports: list[BiReport]                            # reports whose upstream_tables hit alert tables or their downstream
    data_gaps: list[DataGap]


class ContextIndex:
    """Pre-built index for fast alert context resolution.

    Constructed once per Dataset with monitors, criticality, lineage adjacency, and
    BI report mappings. The downstream method returns the transitive closure (memoised).
    The resolve method joins an alert to its full AlertContext. Used by TriageService.
    """

    def __init__(self, ds: Dataset):
        """Index lineage adjacency and BI reports by upstream table for fast lookups."""
        self._ds = ds
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._in_lineage: set[str] = set()
        for src, dst in ds.lineage_edges:
            self._adj[src].add(dst)
            self._in_lineage.update((src, dst))
        self._bi_by_table: dict[str, list[BiReport]] = defaultdict(list)
        for rep in ds.bi_reports:
            for t in set(rep.upstream_tables):
                self._bi_by_table[t].append(rep)
        self._down_cache: dict[str, frozenset[str]] = {}

    def downstream(self, table: str) -> frozenset[str] | None:
        """Return transitive downstream tables (excluding self); None if table is not in lineage.

        In plain words: given a table name, finds all tables that depend on it directly or
        transitively (the "downstream closure"). Uses BFS with cycle detection (G5 self-loops
        are handled). Returns None if the table does not appear in lineage_edges.csv at all,
        meaning downstream impact is unknown.
        """
        if table not in self._in_lineage:
            return None
        if table not in self._down_cache:
            seen, queue = {table}, deque([table])
            while queue:
                for nxt in self._adj.get(queue.popleft(), ()):
                    if nxt not in seen:  # also terminates on cycles / self-loops (G5)
                        seen.add(nxt)
                        queue.append(nxt)
            self._down_cache[table] = frozenset(seen - {table})
        return self._down_cache[table]

    def resolve(self, alert: Alert) -> AlertContext:
        """Resolve full context for an alert: monitor, criticality, downstream, BI reports, gaps.

        In plain words: gathers all the joined data needed to score one alert. Looks up the
        monitor, fetches criticality for each affected table, computes downstream closure,
        finds BI reports that could be affected, and collects all data gaps along the way.
        This is the input to the axes scoring functions.
        """
        gaps: list[DataGap] = list(alert.data_gaps)
        monitor = self._ds.monitors.get(alert.monitor_id)
        if monitor is None:
            gaps.append(DataGap(field="monitor", reason=f"monitor_id {alert.monitor_id!r} not in monitors.jsonl"))
        else:
            gaps.extend(monitor.data_gaps)
        crit: dict[str, TableCriticality | None] = {}
        down: dict[str, frozenset[str] | None] = {}
        reach: set[str] = set(alert.tables)
        for t in alert.tables:
            c = crit[t] = self._ds.criticality.get(t)
            if c is None:
                gaps.append(DataGap(field=f"criticality[{t}]", reason="table not in table_criticality.jsonl"))
            else:
                gaps.extend(c.data_gaps)
                if c.criticality is None:
                    gaps.append(DataGap(field=f"criticality[{t}].criticality", reason="criticality is null"))
            d = down[t] = self.downstream(t)
            if d is None:
                gaps.append(DataGap(field=f"lineage[{t}]", reason="table not in lineage_edges.csv; downstream unknown"))
            else:
                reach |= d
        seen: dict[str, BiReport] = {}
        for t in reach:
            for rep in self._bi_by_table.get(t, ()):
                seen[rep.report_id] = rep
        return AlertContext(
            alert=alert,
            monitor=monitor,
            criticality=crit,
            downstream_tables=down,
            bi_reports=sorted(seen.values(), key=lambda r: r.report_id),
            data_gaps=gaps,
        )
