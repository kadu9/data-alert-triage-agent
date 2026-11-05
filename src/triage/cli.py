"""Thin CLI: uv run python -m triage.cli run --data-zip ... --config ... --out verdicts.jsonl --audit-out verdicts_audit.jsonl"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from triage.service import TriageService


_CONTRACT_FIELDS = {"alert_id", "triage_score", "explanation"}  # what the assignment PDF requires per line


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triage")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="triage every alert; write verdicts.jsonl (contract) and verdicts_audit.jsonl (full)")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-zip", type=Path)
    src.add_argument("--data-dir", type=Path)
    r.add_argument("--config", type=Path, default=Path("config/scoring.yaml"))
    r.add_argument("--out", type=Path, default=Path("verdicts.jsonl"), help="PDF contract: alert_id, triage_score, explanation only")
    r.add_argument("--audit-out", type=Path, default=Path("verdicts_audit.jsonl"), help="full verdicts (evidence, gaps, status) for review")
    r.add_argument("--max-concurrency", type=int, default=None)
    return p


def _concurrency(arg: int | None) -> int | None:
    if arg is not None:
        return arg
    env = os.environ.get("TRIAGE_MAX_CONCURRENCY")
    return int(env) if env else None


def _write_jsonl(path: Path, lines: Iterable[str]) -> None:
    """Write one line per item to a temp file beside `path`, then rename it into place, so a crash never leaves a truncated output."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(f"{line}\n" for line in lines)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


async def _run(args: argparse.Namespace) -> int:
    if args.out.resolve() == args.audit_out.resolve():
        raise SystemExit("--out and --audit-out must be different files (the audit file would overwrite the contract file)")
    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    workers = _concurrency(args.max_concurrency)
    svc = (
        TriageService.from_zip(args.data_zip, args.config, max_concurrency=workers)
        if args.data_zip
        else TriageService.from_dir(args.data_dir, args.config, max_concurrency=workers)
    )
    verdicts = await svc.triage_all()
    _write_jsonl(args.out, (v.model_dump_json(include=_CONTRACT_FIELDS) for v in verdicts))
    _write_jsonl(args.audit_out, (v.model_dump_json() for v in verdicts))
    failed = sum(v.status == "failed" for v in verdicts)
    print(f"wrote {len(verdicts)} verdicts to {args.out} and {args.audit_out} ({failed} failed)", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
