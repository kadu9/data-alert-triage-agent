import stat
import zipfile

import pytest

from triage import zipio
from triage.zipio import MAX_RATIO, ZipSafetyError, read_zip


def make(tmp_path, members, name="t.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in members:
            z.writestr(n, d)
    return p


def test_zip_reader_accepts_dataset_and_rejects_hostile_archives(tmp_path, monkeypatch, real_zip):
    assert read_zip(make(tmp_path, [("v3/alerts.jsonl", b"x")])) == {"alerts.jsonl": b"x"}  # nested prefix ignored
    assert "alerts.jsonl" in read_zip(real_zip)
    for bad in ["../alerts.jsonl", "/alerts.jsonl", "a/../../alerts.jsonl", "a\\alerts.jsonl"]:  # Zip-Slip (CWE-22)
        with pytest.raises(ZipSafetyError):
            read_zip(make(tmp_path, [(bad, b"x")], "bad.zip"))
    with pytest.raises(ZipSafetyError, match="unexpected"):
        read_zip(make(tmp_path, [("evil.sh", b"x")], "u.zip"))
    with pytest.raises(ZipSafetyError, match="duplicate"):
        read_zip(make(tmp_path, [("a/alerts.jsonl", b"1"), ("b/alerts.jsonl", b"2")], "d.zip"))
    with pytest.raises(ZipSafetyError, match="ratio"):  # zip bomb
        read_zip(make(tmp_path, [("alerts.jsonl", b"0" * (MAX_RATIO * 10_000))], "b.zip"))
    notzip = tmp_path / "n.zip"
    notzip.write_bytes(b"nope")
    with pytest.raises(ZipSafetyError, match="not a valid zip"):
        read_zip(notzip)
    sym = tmp_path / "s.zip"
    with zipfile.ZipFile(sym, "w") as zf:
        info = zipfile.ZipInfo("alerts.jsonl")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ZipSafetyError, match="symlink"):
        read_zip(sym)
    monkeypatch.setattr(zipio, "MAX_MEMBER_BYTES", 5)
    with pytest.raises(ZipSafetyError, match="too large"):
        read_zip(make(tmp_path, [("alerts.jsonl", b"123456")], "m.zip"))
    enc = make(tmp_path, [("alerts.jsonl", b"x")], "e.zip")
    raw = bytearray(enc.read_bytes())
    for sig, off in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):  # set the encryption flag bit in local + central headers
        raw[raw.find(sig) + off] |= 1
    enc.write_bytes(bytes(raw))
    with pytest.raises(ZipSafetyError, match="encrypted"):
        read_zip(enc)
    monkeypatch.undo()
    monkeypatch.setattr(zipio, "MAX_MEMBERS", 1)
    with pytest.raises(ZipSafetyError, match="too many"):
        read_zip(make(tmp_path, [("alerts.jsonl", b"1"), ("monitors.jsonl", b"2")], "many.zip"))
    monkeypatch.undo()
    monkeypatch.setattr(zipio, "MAX_TOTAL_BYTES", 5)
    with pytest.raises(ZipSafetyError, match="uncompressed size"):
        read_zip(make(tmp_path, [("alerts.jsonl", b"123456")], "tot.zip"))
    monkeypatch.undo()
    before = set(tmp_path.iterdir())  # reading never writes to disk
    read_zip(make(tmp_path, [("alerts.jsonl", b"x")], "ok.zip"))
    assert set(tmp_path.iterdir()) - before == {tmp_path / "ok.zip"}
