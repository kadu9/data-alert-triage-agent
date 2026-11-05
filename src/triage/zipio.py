"""Hardened in-memory zip reader (CWE-22 Zip-Slip, CWE-409 zip bomb). Never extracts to disk."""
from __future__ import annotations

import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

EXPECTED_FILES = frozenset(
    {
        "alerts.jsonl",
        "alert_history.jsonl",
        "monitors.jsonl",
        "table_criticality.jsonl",
        "bi_reports.json",
        "lineage_edges.csv",
        "README.md",
        ".gitignore",  # shipped inside the dataset zip; read but unused
    }
)
MAX_MEMBERS = 32
MAX_TOTAL_BYTES = 250 * 1024 * 1024
MAX_MEMBER_BYTES = 150 * 1024 * 1024
MAX_RATIO = 100
CHUNK = 64 * 1024


class ZipSafetyError(ValueError):
    """Raised when the archive violates a security or size rule.

    Covers Zip-Slip (path traversal), zip bombs (size/ratio caps), encrypted files,
    symlinks, unexpected members, and duplicates. Raised by read_zip; not recoverable.
    """


def _basename(name: str) -> str:
    """Extract the basename for allowlist checking; raise ZipSafetyError on traversal/absolute paths.

    In plain words: security check that prevents Zip-Slip (CWE-22) attacks where a malicious
    archive contains paths like "../../../etc/passwd". Rejects absolute paths, ".." components,
    and Windows-style backslashes.
    """
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ZipSafetyError(f"unsafe member path: {name!r}")
    return p.name


def read_zip(path: str | Path) -> dict[str, bytes]:
    """Read allowlisted members into memory; keyed by basename with security checks.

    In plain words: safely reads a zip file entirely in memory (never extracts to disk).
    Applies multiple security checks to prevent attacks: Zip-Slip (path traversal),
    zip bombs (size/ratio limits), encrypted files, and symlinks. Returns a dict mapping
    filename to contents for the allowed files only.

    Checks: Zip-Slip (traversal), member/total size caps, compression ratio, encryption, symlinks.
    Raises ZipSafetyError on any violation.
    """
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ZipSafetyError(f"not a valid zip: {e}") from e
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename.split("/")]
        if len(infos) > MAX_MEMBERS:
            raise ZipSafetyError(f"too many members: {len(infos)} > {MAX_MEMBERS}")
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise ZipSafetyError("archive uncompressed size exceeds cap")
        names = [_basename(i.filename) for i in infos]
        dups = [n for n, c in Counter(names).items() if c > 1]
        if dups:
            raise ZipSafetyError(f"duplicate members: {dups}")
        out: dict[str, bytes] = {}
        budget = MAX_TOTAL_BYTES
        for info, name in zip(infos, names, strict=True):
            if name not in EXPECTED_FILES:
                raise ZipSafetyError(f"unexpected member: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ZipSafetyError(f"encrypted member: {info.filename!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ZipSafetyError(f"symlink member: {info.filename!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ZipSafetyError(f"member too large: {info.filename!r}")
            if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
                raise ZipSafetyError(f"suspicious compression ratio: {info.filename!r}")
            # Bounded streaming: header sizes can lie, so enforce caps on actual bytes.
            limit = min(MAX_MEMBER_BYTES, budget)
            buf = bytearray()
            with zf.open(info) as fh:
                while chunk := fh.read(CHUNK):
                    buf += chunk
                    if len(buf) > limit:
                        raise ZipSafetyError(f"size cap exceeded while reading: {info.filename!r}")
            budget -= len(buf)
            out[name] = bytes(buf)
        return out
