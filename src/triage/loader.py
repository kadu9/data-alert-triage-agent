"""Parse raw dataset bytes into typed records. Defensive: bad numerics/JSON -> None + DataGap."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from triage.schema import (
    Alert,
    BiReport,
    Dataset,
    DataGap,
    HistoryRow,
    Monitor,
    TableCriticality,
)
from triage.zipio import read_zip

REQUIRED_FILES = (
    "alerts.jsonl",
    "alert_history.jsonl",
    "monitors.jsonl",
    "table_criticality.jsonl",
    "bi_reports.json",
    "lineage_edges.csv",
)


class LoaderError(ValueError):
    """Error raised when input data is structurally invalid.

    Covers missing files, bad JSON lines, naive timestamps, missing required keys,
    and wrong CSV headers. Raised by loader functions (_jsonl, _build, _lineage,
    load_files) and propagates to the caller; not a recoverable per-alert error.
    """


def parse_float(value: Any, field: str) -> tuple[float | None, DataGap | None]:
    """Parse a value to float; None stays None, bad value becomes None + DataGap.

    In plain words: safely converts strings or numbers to float. If the value cannot be
    parsed (e.g. "abc", a boolean, NaN, Inf), returns None plus a DataGap record explaining
    what went wrong. This implements the "never coerce bad data to 0" policy.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, DataGap(field=field, reason=f"bool is not numeric: {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None, DataGap(field=field, reason=f"unparseable numeric: {value!r}")
    if not math.isfinite(f):
        return None, DataGap(field=field, reason=f"non-finite numeric: {value!r}")
    return f, None


def parse_json_field(value: Any, field: str) -> tuple[Any, DataGap | None]:
    """Parse a JSON-encoded string; None stays None, malformed becomes None + DataGap.

    In plain words: handles fields stored as JSON strings within JSON (like monitors.config).
    If parsing fails, records a DataGap rather than crashing.
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return value, None
    try:
        return json.loads(value), None
    except json.JSONDecodeError as e:
        return None, DataGap(field=field, reason=f"malformed JSON: {e.msg}")


def _list(d: dict[str, Any], key: str, field: str) -> tuple[list[Any], list[DataGap]]:
    """Extract a list field; null becomes [] plus a DataGap (null != known-empty).

    In plain words: when a list field (like events or tables) is null in the data, returns
    an empty list BUT records a DataGap to distinguish "no events" from "events unknown".
    """
    v = d.get(key)
    if v is None:
        return [], [DataGap(field=f"{field}.{key}", reason="null collection treated as empty")]
    return v, []


def _jsonl(raw: bytes, name: str) -> list[dict[str, Any]]:
    """Parse a JSONL file (one JSON object per line) to a list of dicts.

    In plain words: reads newline-delimited JSON, common for streaming data. Blank lines
    are skipped; malformed lines raise LoaderError with the line number.
    """
    rows = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise LoaderError(f"{name}:{n}: invalid JSON ({e.msg})") from e
    return rows


def _build(model: type, src: str, idx: int, /, **kw: Any):
    """Construct a model; wrap validation errors as LoaderError naming file and row index."""
    try:
        return model(**kw)
    except (ValidationError, TypeError) as e:
        raise LoaderError(f"{src}[{idx}]: {e}") from e


def _event(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Parse numeric event fields defensively; bad values become None plus a DataGap."""
    gaps: list[DataGap] = []
    out = dict(d)
    for k in ("actual_value", "lower_threshold", "upper_threshold"):
        out[k], g = parse_float(d.get(k), f"{prefix}.{k}")
        gaps += [g] if g else []
    out["schema_changes"], g = parse_json_field(d.get("schema_changes"), f"{prefix}.schema_changes")
    gaps += [g] if g else []
    out["data_gaps"] = gaps
    return out


def _alert(d: dict[str, Any], i: int) -> Alert:
    """Build an Alert from a raw row, parsing events and tables defensively."""
    f = f"alerts[{i}]"
    raw_events, g1 = _list(d, "events", f)
    tables, g2 = _list(d, "tables", f)
    events = [_event(e, f"{f}.events[{j}]") for j, e in enumerate(raw_events)]
    # severity/owner/priority are deliberately passed through as-is (null in the dataset).
    return _build(Alert, "alerts.jsonl", i, **{**d, "events": events, "tables": tables, "data_gaps": g1 + g2})


def _monitor(d: dict[str, Any], i: int) -> Monitor:
    """Build a Monitor from a raw row; the config JSON string is parsed defensively."""
    f = f"monitors[{i}]"
    cfg, gap = parse_json_field(d.get("config"), f"{f}.config")
    tables, g1 = _list(d, "tables", f)
    return _build(Monitor, "monitors.jsonl", i, **{**d, "config": cfg, "tables": tables, "data_gaps": ([gap] if gap else []) + g1})


def _history(d: dict[str, Any], i: int) -> HistoryRow:
    """Build a HistoryRow from a raw row, keeping only the modelled fields."""
    # severity/owner/ticket_created/minutes_to_* are unused by scoring and intentionally not modelled.
    row = {k: d.get(k) for k in HistoryRow.model_fields if k in d}
    return _build(HistoryRow, "alert_history.jsonl", i, **row)


def _criticality(d: dict[str, Any], i: int) -> TableCriticality:
    """Build a TableCriticality from a raw row, keeping only the modelled fields."""
    # is_key_asset is False on every row (zero signal) and intentionally not modelled.
    row = {k: d.get(k) for k in TableCriticality.model_fields if k in d}
    return _build(TableCriticality, "table_criticality.jsonl", i, **row)


def _lineage(raw: bytes) -> list[tuple[str, str]]:
    """Parse lineage_edges.csv into (source, target) pairs; raise LoaderError on an unexpected header."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != ["source", "target"]:
        raise LoaderError(f"lineage_edges.csv: unexpected header {reader.fieldnames}")
    return [(r["source"], r["target"]) for r in reader]


def _bi_report(d: dict[str, Any], i: int) -> BiReport:
    """Build a BiReport from a raw entry, parsing upstream_tables defensively."""
    upstream, gaps = _list(d, "upstream_tables", f"bi_reports[{i}]")
    return _build(BiReport, "bi_reports.json", i, **{**d, "upstream_tables": upstream, "data_gaps": gaps})


def load_files(files: dict[str, bytes]) -> Dataset:
    """Parse raw file bytes into a typed Dataset; raises LoaderError on structural or schema problems.

    In plain words: takes the raw bytes from a zip or directory and converts them to typed
    Python objects. Field-level problems (bad numerics, malformed nested JSON, null lists) become
    None/[] + DataGap. Missing files, wrong headers, malformed JSONL lines and model
    validation failures raise LoaderError.
    """
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        raise LoaderError(f"missing files: {missing}")
    try:
        bi_raw = json.loads(files["bi_reports.json"].decode("utf-8"))
    except json.JSONDecodeError as e:
        raise LoaderError(f"bi_reports.json: invalid JSON ({e.msg})") from e
    if not isinstance(bi_raw, list):
        raise LoaderError(f"bi_reports.json: expected array, got {type(bi_raw).__name__}")
    monitors = [_monitor(d, i) for i, d in enumerate(_jsonl(files["monitors.jsonl"], "monitors.jsonl"))]
    return Dataset(
        alerts=[_alert(d, i) for i, d in enumerate(_jsonl(files["alerts.jsonl"], "alerts.jsonl"))],
        monitors={m.monitor_id: m for m in monitors},
        history=[_history(d, i) for i, d in enumerate(_jsonl(files["alert_history.jsonl"], "alert_history.jsonl"))],
        criticality={
            c.table_id: c
            for c in (_criticality(d, i) for i, d in enumerate(_jsonl(files["table_criticality.jsonl"], "table_criticality.jsonl")))
        },
        bi_reports=[_bi_report(d, i) for i, d in enumerate(bi_raw)],
        lineage_edges=_lineage(files["lineage_edges.csv"]),
    )


def load_zip(path: str | Path) -> Dataset:
    """Load a Dataset from a zip file via read_zip and load_files."""
    return load_files(read_zip(path))


def load_dir(path: str | Path) -> Dataset:
    """Load a Dataset from an unzipped directory of data files."""
    base = Path(path)
    return load_files({name: (base / name).read_bytes() for name in REQUIRED_FILES if (base / name).is_file()})
