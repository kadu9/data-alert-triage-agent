"""Golden regression run: re-scores the saved golden set with the production axes and compares with the stored result.

Usage: python eval/golden.py [config]       (no zip, no flags; default config/scoring.yaml, the file production uses)
Reads eval/reports/golden_labeled.jsonl + golden_run.json, both created once by eval/build_golden.py from the zip.
Prints AUC / tier tables, when the baseline was built and every config value that differs from it (the likely cause), the
headline metrics saved baseline -> now, then for each changed row its monitor / table / type / time / label and old -> new
per axis, score and tier (exit 1). To accept an intended change re-run build_golden.py.
`rescore` is the single scoring function: build_golden.py imports it, so both use triage.axes / history_stats / combine.
"""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from hashlib import sha256 as _sha
from pathlib import Path
from typing import Any

from triage import axes as ax
from triage.combine import combine
from triage.config import ScoringConfig, load_config
from triage.context import AlertContext
from triage.history import history_stats
from triage.schema import Alert, BiReport, HistoryRow, Monitor, TableCriticality

OUT = Path(__file__).parent / "reports"
BOOTSTRAPS = 2000
SEED = 7


def sha256(path: str) -> str:
    return _sha(Path(path).read_bytes()).hexdigest()


def acted_flag(status: str | None, cfg: ScoringConfig) -> bool | None:
    """True/False for a configured label, None for unlabeled or unrecognised status (never guessed)."""
    if status in cfg.history.acted_statuses:
        return True
    return False if status in cfg.history.not_acted_statuses else None


REQUIRED_KEYS = (
    "alert_id", "monitor_id", "created_at", "alert_type", "status", "table_id", "joined_monitor", "joined_table",
    "monitor_type", "monitor_status", "monitor_sensitivity", "monitor_priority", "monitor_inner_sensitivity",
    "table_criticality", "table_importance_score", "table_is_golden", "table_is_pii", "table_reads_30d", "table_writes_30d",
    "table_active_users_30d", "downstream_tables", "bi_reports", "prior_monitor", "prior_pool", "acted", "score", "tier", "reasons", "axes",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read the saved golden set; stop with a clear message on bad JSON, a missing key, an empty file or a duplicate alert_id."""
    rebuild = "re-run `python eval/build_golden.py <zip>`"
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path.name} line {n} is not valid JSON ({exc.msg}); {rebuild}")
            missing = [k for k in REQUIRED_KEYS if not isinstance(row, dict) or k not in row]
            if missing:
                sys.exit(f"{path.name} line {n} lacks {missing}; {rebuild}")
            rows.append(row)
    if not rows:
        sys.exit(f"{path.name} is empty; {rebuild}")
    if len({r["alert_id"] for r in rows}) != len(rows):
        sys.exit(f"{path.name} has duplicate alert_ids; {rebuild}")
    return rows


def rescore(row: dict[str, Any], cfg: ScoringConfig) -> dict[str, Any]:
    """5-axis partial score of one golden row, rebuilt from its stored inputs with the production axes and combine.

    anomaly_severity and context_signal need the alert's events / free text, which history rows do not have, so they are
    left out and the remaining weights renormalise. Objects are rebuilt from stored counts: only the SIZE of the downstream
    set and of the BI list is read by blast_radius, so placeholder members are used.
    """
    tid = row["table_id"]
    monitor = Monitor(
        monitor_id=row["monitor_id"], name="", type=row["monitor_type"], status=row["monitor_status"], sensitivity=row["monitor_sensitivity"],
        config={row["monitor_type"]: [{"priority": row["monitor_priority"], "sensitivity": row["monitor_inner_sensitivity"]}]},
    ) if row["joined_monitor"] else None
    crit = TableCriticality(
        table_id=tid, criticality=row["table_criticality"], importance_score=row["table_importance_score"], is_golden=row["table_is_golden"],
        is_pii=row["table_is_pii"], reads_30d=row["table_reads_30d"], writes_30d=row["table_writes_30d"], active_users_30d=row["table_active_users_30d"],
    ) if row["joined_table"] else None
    n_down = row["downstream_tables"]
    alert = Alert(alert_id=row["alert_id"], monitor_id=row["monitor_id"], created_at=row["created_at"], alert_type=row["alert_type"], title="", tables=[tid] if tid else [])
    ctx = AlertContext(
        alert=alert, monitor=monitor,
        criticality={tid: crit} if tid else {},
        downstream_tables={tid: None if n_down is None else frozenset(f"t{i}" for i in range(n_down))} if tid else {},
        bi_reports=[BiReport(report_id=f"r{i}", name="", upstream_tables=[]) for i in range(row["bi_reports"])],
        data_gaps=[],
    )

    def rows(items: list[list[Any]]) -> list[HistoryRow]:
        return [HistoryRow(alert_id=f"p{i}", monitor_id=row["monitor_id"], created_at=c, alert_type=t, status=s) for i, (c, t, s) in enumerate(items)]

    monitor_rows = rows(row["prior_monitor"])
    pool_rows = rows([[row["created_at"], row["alert_type"], s] for s, n in row["prior_pool"] for _ in range(n)])
    axes = {
        "business_criticality": ax.business_criticality(ctx, cfg.axes.business_criticality),
        "blast_radius": ax.blast_radius(ctx, cfg.axes.blast_radius),
        "historical_action_rate": ax.historical_action_rate(
            history_stats(monitor_rows, cfg.history, "monitor"), history_stats(pool_rows, cfg.history, "alert_type_pool"), cfg.axes.historical_action_rate
        ),
        "monitor_signal_quality": ax.monitor_signal_quality(monitor, cfg.axes.monitor_signal_quality),
        "recency_pressure": ax.recency_pressure(alert.created_at, [r.created_at for r in monitor_rows], cfg.axes.recency_pressure),
    }
    c = combine(axes, ctx, cfg)
    return {"acted": acted_flag(row["status"], cfg), "score": c.score, "tier": c.tier, "reasons": c.reasons, "axes": {k: r.value for k, r in axes.items()}}


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    """Probability a random acted row scores above a random not-acted row (ties count half); None if one class is empty."""
    pos, neg = [s for s, y in pairs if y], [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def bootstrap(rows: list[dict[str, Any]], stat: Callable[[list[dict[str, Any]]], float | None]) -> tuple[float, float] | None:
    """95% interval of stat(sample) when whole monitors are resampled (rows of one monitor are not independent)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["monitor_id"]].append(r)
    keys, rng, vals = sorted(groups), random.Random(SEED), []
    for _ in range(BOOTSTRAPS):
        v = stat([r for k in rng.choices(keys, k=len(keys)) for r in groups[k]])
        if v is not None:
            vals.append(v)
    print(f"bootstrap: {BOOTSTRAPS - len(vals)} of {BOOTSTRAPS} resamples had a single class and are excluded from the interval")
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if len(vals) > 20 else None


def tier_for(score: float, high: float, medium: float) -> str:
    return "HIGH" if score >= high else "MEDIUM" if score >= medium else "LOW"


def score_pairs(rows: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    return [(r["score"], r["acted"]) for r in rows]


def report(gold: list[dict[str, Any]], cfg: ScoringConfig) -> None:
    """Print: AUC of the score (and of each axis) against `acted`, then acted rate per tier for the configured and a few alternative thresholds."""
    scored = [g for g in gold if g["score"] is not None and g["acted"] is not None]
    m = metrics(gold)
    a, ci = m["score_auc"], bootstrap(scored, lambda rs: auc(score_pairs(rs)))
    print(f"labeled rows {len(gold)}, scored {len(scored)}, acted {sum(g['acted'] for g in scored)}, monitors {len({g['monitor_id'] for g in scored})}")
    print(f"AUC (score vs acted) = {a:.3f}, 95% CI over monitors = {ci if ci is None else tuple(round(x, 3) for x in ci)}  (0.5 = no better than chance)")
    for axis, axis_auc in m["axis_auc"].items():
        print(f"  axis {axis}: AUC {_fmt(axis_auc)} on {sum(r['axes'][axis] is not None for r in scored)} rows")
    for high, medium in [(cfg.thresholds.HIGH, cfg.thresholds.MEDIUM), (0.55, 0.30), (0.60, 0.40), (0.70, 0.45)]:
        tag = " <- config" if (high, medium) == (cfg.thresholds.HIGH, cfg.thresholds.MEDIUM) else ""
        cells = []
        for t in ("HIGH", "MEDIUM", "LOW"):
            rows = [r for r in scored if tier_for(r["score"], high, medium) == t]
            cells.append(f"{t} n={len(rows)} acted={sum(r['acted'] for r in rows)}")
        print(f"HIGH>={high:.2f} MEDIUM>={medium:.2f}: " + " | ".join(cells) + tag)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers of one golden result set: AUC of the score and of each axis (labeled rows only) and the tier counts."""
    scored = [g for g in rows if g["score"] is not None and g["acted"] is not None]
    axis_auc = {k: auc([(r["axes"][k], r["acted"]) for r in scored if r["axes"][k] is not None]) for k in scored[0]["axes"]} if scored else {}
    return {"score_auc": auc(score_pairs(scored)), "axis_auc": axis_auc, "tiers": dict(Counter(g["tier"] for g in rows))}


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def print_metric_changes(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    """Print each headline metric as saved baseline -> now, marking the ones that moved."""
    o, n = metrics(old), metrics(new)
    pairs = [("score AUC", o["score_auc"], n["score_auc"])]
    pairs += [(f"axis AUC {k}", o["axis_auc"].get(k), n["axis_auc"].get(k)) for k in sorted(o["axis_auc"].keys() | n["axis_auc"].keys())]
    pairs += [(f"final tier count {t} (after overrides)", o["tiers"].get(t, 0), n["tiers"].get(t, 0)) for t in ("HIGH", "MEDIUM", "LOW")]
    print("metrics, saved baseline -> now:")
    for name, a, b in pairs:
        print(f"  {name}: {_fmt(a)} -> {_fmt(b)}{'' if same(a, b) else '  <- CHANGED'}")


def same(a: float | None, b: float | None) -> bool:
    return a == b if a is None or b is None else math.isclose(a, b, abs_tol=1e-9)


def check(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> int:
    """Print old -> new for every golden row whose tier, score, acted flag or any axis value changed (or is missing); return the count."""
    old_by_id, bad, axis_hits = {g["alert_id"]: g for g in old}, 0, Counter()
    for g in new:
        o = old_by_id.get(g["alert_id"])
        old_axes = o["axes"] if o else {}
        moved = {k: (old_axes.get(k), g["axes"].get(k)) for k in old_axes.keys() | g["axes"].keys() if not same(old_axes.get(k), g["axes"].get(k))}
        if o is None or moved or o["tier"] != g["tier"] or o["acted"] != g["acted"] or not same(o["score"], g["score"]):
            bad += 1
            axis_hits.update(moved.keys())
            who = f"monitor {g['monitor_id']}, table {g['table_id']}, type {g['alert_type']}, created {g['created_at']}, label {g['status']}"
            print(f"CHANGED {g['alert_id']} ({who}): tier {o and o['tier']} -> {g['tier']}, score {o and o['score']} -> {g['score']}; axes (old, new): {moved}; reasons now: {g['reasons']}")
    for m in sorted(set(old_by_id) - {g["alert_id"] for g in new}):
        bad += 1
        print(f"MISSING {m}: in the saved golden set but not produced now")
    print(f"golden check: {bad} change(s) in {len(new)} rows; rows with a moved axis: {dict(axis_hits) or 'none'}")
    if bad:
        print("If the change is intended, re-run eval/build_golden.py to save the new baseline.")
    return bad


def flat(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to {'weights.blast_radius': 0.22, ...} so two configs can be compared key by key (lists sorted)."""
    if not isinstance(d, dict):
        return {prefix: sorted(d, key=str) if isinstance(d, list) else d}  # statuses are sets in config: order is not a change
    return {k: v for key, val in d.items() for k, v in flat(val, f"{prefix}.{key}" if prefix else key).items()}


def print_run_changes(saved: dict[str, Any], cfg: ScoringConfig, config_path: str) -> None:
    """Print who built the baseline and when, whether the config file changed, and each config value that differs."""
    now = sha256(config_path)
    print(f"baseline built {saved['built_at']} from zip sha256 {saved['zip_sha256'][:12]}; config file sha256 {saved['config_sha256'][:12]} -> {now[:12]} ({'same' if now == saved['config_sha256'] else 'CHANGED'})")
    o, n = flat(saved["config"]), flat(cfg.model_dump(mode="json"))
    diffs = [f"  config {k}: {o.get(k)} -> {n.get(k)}" for k in sorted(o.keys() | n.keys()) if o.get(k) != n.get(k)]
    print(f"config {config_path} vs the run that built the golden set: {len(diffs) or 'no value differs'}", *diffs, sep="\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="config/scoring.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    saved, meta_path = OUT / "golden_labeled.jsonl", OUT / "golden_run.json"
    if not saved.exists() or not meta_path.exists():
        sys.exit("no golden set: run `python eval/build_golden.py <zip>` once first")
    try:
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            sys.exit("golden_run.json is not a JSON object; re-run `python eval/build_golden.py <zip>`")
        missing = {"built_at", "zip_sha256", "config_sha256", "golden_labeled_sha256", "config"} - meta.keys()
        if missing:
            sys.exit(f"golden_run.json lacks {sorted(missing)}; re-run `python eval/build_golden.py <zip>`")
        if meta["golden_labeled_sha256"] != sha256(str(saved)):
            sys.exit("golden_labeled.jsonl differs from the file golden_run.json describes; re-run `python eval/build_golden.py <zip>`")
    except json.JSONDecodeError as exc:
        sys.exit(f"golden_run.json is unreadable ({type(exc).__name__}); re-run `python eval/build_golden.py <zip>`")
    old = load_rows(saved)
    new = [{**r, **rescore(r, cfg)} for r in old]
    report(new, cfg)
    print_run_changes(meta, cfg, args.config)
    print_metric_changes(old, new)
    return 1 if check(old, new) else 0


if __name__ == "__main__":
    sys.exit(main())
