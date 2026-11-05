"""Generate the final golden dataset from the real zip. Joins only: nothing is imputed, nulls stay null.

Usage: python eval/build_golden.py <zip> [config]
Run once (needs the zip), re-run only to accept an intended scoring change. Writes eval/reports/ (gitignored, real data):
  alert_history_enriched.jsonl  every history row + joined monitor / table / lineage fields + join flags
  golden_labeled.jsonl          one JSON line per labeled alert: metadata + every input the 5 axes read + acted, axes, score, tier
  golden_run.json               build time, config file path + sha256, zip sha256, golden_labeled.jsonl sha256 and every config value used
The stored axes / score / tier are computed by golden.rescore, the same function eval/golden.py re-runs later (no zip needed).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from golden import OUT, rescore, sha256  # pyright: ignore[reportImplicitRelativeImport]  # run as a script: eval/ is on sys.path
from triage import axes as ax
from triage.config import load_config
from triage.context import ContextIndex
from triage.loader import load_files
from triage.schema import Alert, Dataset, HistoryRow
from triage.zipio import read_zip


def enrich(raw: dict[str, Any], h: HistoryRow, ds: Dataset, ctx_index: ContextIndex) -> dict[str, Any]:
    """Original row + fields joined by key. A missing join gives null and a False flag, never a default value."""
    mon, crit = ds.monitors.get(h.monitor_id), ds.criticality.get(h.table_id or "")
    down = ctx_index.downstream(h.table_id) if h.table_id else None
    return {
        **raw,
        "joined_monitor": mon is not None, "joined_table": crit is not None, "joined_lineage": down is not None,
        "monitor_type": mon and mon.type, "monitor_status": mon and mon.status, "monitor_sensitivity": mon and mon.sensitivity,
        "monitor_priority": mon and ax.inner_config(mon).get("priority"),
        "monitor_inner_sensitivity": mon and ax.inner_config(mon).get("sensitivity"),
        "table_criticality": crit and crit.criticality, "table_importance_score": crit and crit.importance_score,
        "table_is_golden": crit and crit.is_golden, "table_is_pii": crit and crit.is_pii,
        "table_reads_30d": crit and crit.reads_30d, "table_writes_30d": crit and crit.writes_30d, "table_active_users_30d": crit and crit.active_users_30d,
        "downstream_tables": None if down is None else len(down),
    }


def golden_row(e: dict[str, Any], h: HistoryRow, ds: Dataset, ctx_index: ContextIndex, by_monitor: dict[str, list[HistoryRow]], cfg: Any) -> dict[str, Any]:
    """Enriched row + the history / BI inputs the axes need, then the stored result of the shared golden.rescore.

    Only rows created BEFORE this one count as history (a live system cannot see later ones). The pool is the row's own
    alert_type (not the production type_map, which maps live alert types to history types).
    """
    def prior(rows: list[HistoryRow]) -> list[HistoryRow]:
        return [r for r in rows if r.created_at < h.created_at and r.alert_id != h.alert_id]

    alert = Alert(alert_id=h.alert_id, monitor_id=h.monitor_id, created_at=h.created_at, alert_type=h.alert_type, title="", tables=[h.table_id] if h.table_id else [])
    base = {
        **e,
        "bi_reports": len(ctx_index.resolve(alert).bi_reports),
        "prior_monitor": [[r.created_at.isoformat(), r.alert_type, r.status] for r in prior(by_monitor[h.monitor_id])],
        "prior_pool": sorted(Counter(r.status for r in prior([r for r in ds.history if r.alert_type == h.alert_type])).items(), key=str),
    }
    return {**base, **rescore(base, cfg)}


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip")
    p.add_argument("config", nargs="?", default="config/scoring.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    files = read_zip(args.zip)
    ds = load_files(files)
    raw = [json.loads(x) for x in files["alert_history.jsonl"].decode().splitlines() if x.strip()]
    if len(raw) != len(ds.history) or any(r["alert_id"] != h.alert_id for r, h in zip(raw, ds.history)):
        sys.exit("loader rows do not line up with the raw history file")
    ctx_index = ContextIndex(ds)
    by_monitor = defaultdict(list)
    for h in ds.history:
        by_monitor[h.monitor_id].append(h)
    enriched = [enrich(r, h, ds, ctx_index) for r, h in zip(raw, ds.history)]
    gold = [golden_row(e, h, ds, ctx_index, by_monitor, cfg) for e, h in zip(enriched, ds.history) if e["status"] is not None]
    OUT.mkdir(exist_ok=True)
    write(OUT / "alert_history_enriched.jsonl", enriched)
    write(OUT / "golden_labeled.jsonl", gold)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(), "config_file": str(Path(args.config).resolve()), "config_sha256": sha256(args.config),
        "zip_sha256": sha256(args.zip), "golden_labeled_sha256": sha256(str(OUT / "golden_labeled.jsonl")),
        "config": cfg.model_dump(mode="json"),
    }
    (OUT / "golden_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(enriched)} enriched rows and {len(gold)} golden rows ({sum(g['acted'] is True for g in gold)} acted) to {OUT}")
    print("next: python eval/golden.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
