"""Independent proof: re-derive every cited evidence value straight from the raw files (no triage.* imports)."""
import csv
import json
import re
import sys
import zipfile
from collections import defaultdict, deque

ZIP, VERDICTS = sys.argv[1], sys.argv[2]
NAME = "v3-assignment-data/"
zf = zipfile.ZipFile(ZIP)


def jl(n):
    return [json.loads(x) for x in zf.read(NAME + n).decode().splitlines() if x.strip()]


alerts = {a["alert_id"]: a for a in jl("alerts.jsonl")}
crit = {c["table_id"]: c for c in jl("table_criticality.jsonl")}
hist = jl("alert_history.jsonl")
bi = json.loads(zf.read(NAME + "bi_reports.json"))
adj = defaultdict(set)
for r in csv.DictReader(zf.read(NAME + "lineage_edges.csv").decode().splitlines()):
    adj[r["source"]].add(r["target"])


def downstream(t):
    seen, q = set(), deque([t])
    while q:
        for n in adj.get(q.popleft(), ()):
            if n not in seen and n != t:
                seen.add(n)
                q.append(n)
    return seen


def hist_stats(mid, cfg_acted=("fixed", "acknowledged"), cfg_not=("no action needed", "false positive", "expected")):
    rows = [h for h in hist if h["monitor_id"] == mid and h["alert_type"] != "rule_run_execution_error"]
    lab = [h for h in rows if h["status"] in cfg_acted + cfg_not]
    return len(lab), sum(h["status"] in cfg_acted for h in lab)


monitors = {m["monitor_id"]: m for m in jl("monitors.jsonl")}


def notes(mid):
    cfg = json.loads(monitors[mid]["config"]) if isinstance(monitors[mid]["config"], str) else monitors[mid]["config"]
    inner = cfg.get(monitors[mid]["type"]) or [{}]
    return inner[0].get("notes")


def squash(t):
    return " ".join(t.split())


checked = bad = policy = 0
for line in open(VERDICTS):
    v = json.loads(line)
    a = alerts[v["alert_id"]]
    for e in v["evidence"]:
        f, val, src = e["field"], e["value"], e["source"]
        if f.startswith(("axis.", "combined.")) or src == "config/scoring.yaml":
            policy += 1  # scoring outputs / config rules: not raw record values, so not re-derivable here
            continue
        if (e.get("note") or "").startswith("LLM label:"):
            raw = a.get(f) if src == "alerts.jsonl" else notes(a["monitor_id"])
            ok = squash(val) in squash(raw or "")
            checked += 1
            if not ok:
                bad += 1
                print("MISMATCH quote", v["alert_id"], src, f)
            continue
        if src == "alerts.jsonl":
            if f == "alert_type":
                exp = a["alert_type"]
            elif f == "events.length":
                exp = len(a["events"])
            else:
                m = re.match(r"events\[(\d+)\]\.(\w+)", f)
                assert m, f"unexpected evidence field {f}"
                i, k = m.groups()
                exp = a["events"][int(i)][k]
                if k != "event_type":
                    exp = float(exp)
            ok = (float(val) == exp) if isinstance(exp, float) else str(exp) == val
        elif src == "table_criticality.jsonl":
            t, k = f.rsplit(".", 1)
            ok = str(crit[t][k]) == val
        elif f == "downstream_tables.count":
            ok = len(set().union(*(downstream(t) for t in a["tables"] if t in adj or any(t in s for s in adj.values())))) == int(val)
        elif f == "bi_reports.count":
            reach = set(a["tables"]).union(*(downstream(t) for t in a["tables"]))
            ok = sum(bool(reach & set(r["upstream_tables"])) for r in bi) == int(val)
        elif f == "history.n_total":
            ok = sum(h["monitor_id"] == a["monitor_id"] and h["alert_type"] != "rule_run_execution_error" for h in hist) == int(val)
        elif f == "history.n_labeled":
            ok = hist_stats(a["monitor_id"])[0] == int(val)
        elif f == "history.n_acted":
            ok = hist_stats(a["monitor_id"])[1] == int(val)
        elif src == "monitors.jsonl":
            ok = f == "status" and monitors[a["monitor_id"]]["status"] == val
        else:
            ok = False
        checked += 1
        if not ok:
            bad += 1
            print("MISMATCH", v["alert_id"], src, f, val)
print(f"raw-data evidence items re-derived={checked} mismatches={bad}; skipped scoring-output/config citations={policy}")
sys.exit(1 if bad else 0)
