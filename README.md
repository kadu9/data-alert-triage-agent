# Data Alert Triage Agent

## Overview

Data monitors fire many alerts and most are noise. For each alert this agent answers one question: **how soon should a person look at it?** The Data Alert Triage Agent reads each alert, weighs it against the customer's own data, and rates it HIGH, MEDIUM or LOW with a short reason behind its judgment.

1. **Join** the alert to its monitor, the importance of the affected tables, what depends on them (lineage, BI reports) and how similar past alerts were handled.
2. **Score** several questions ("axes") with deterministic math. Weights and thresholds are in `config/scoring.yaml`.
3. **Extract features from free text** with an LLM (call 1): three fields (stated noise, recurrence, impact), each backed by a quote that must exist in the source text. It never sets the tier.
4. **Combine** the axis values into one score; fixed thresholds and config rules (for example paused monitor: down one tier) give the tier.
5. **Explain** with a second LLM call that writes a short explanation from the verified facts, keeping a deterministic explanation if its output breaks a rule twice.

It does not guess: a missing value is listed as a data gap and lowers confidence, and a LOW verdict that rests on at most one axis with real data says "I can't tell from this data"; a missing value is never treated as zero. It never invents `severity`, `priority` or `owner`, treats alert text as data rather than instructions, and uses past outcomes only as a noisy hint (83 of 1393 history rows are labeled).

- **Input:** a zip (`--data-zip`) or folder (`--data-dir`) with `alerts.jsonl`, `monitors.jsonl`, `alert_history.jsonl`, `table_criticality.jsonl`, `lineage_edges.csv`, `bi_reports.json`.
- **Output:** `verdicts.jsonl` (per alert: `alert_id`, `triage_score`, `explanation`) and `verdicts_audit.jsonl` (full evidence, gaps, status).
- **Pipeline:** `zipio -> loader -> context/history -> axes -> features (LLM call 1) -> combine -> explain -> summary (LLM call 2) -> service -> cli`
- **Status:** runs end to end. Only a partial check of the scoring exists (AUC 0.827 on 5 of 7 axes, 83 labeled rows); no precision/recall of final tiers. Terms are in the [Glossary](#glossary).

---

## Quick start

**Prerequisites:** macOS/Linux, `git`, [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Python 3.11 is fetched automatically.

```bash
# 1. Clone and enter
git clone <repo-url> data-alert-triage-agent && cd data-alert-triage-agent

# 2. Install locked dependencies
uv sync --locked

# 3. Configure LLM (required)
cp .env.example .env
# Edit .env: set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (e.g. claude-sonnet-4-5)

# 4. Place dataset (gitignored)
mkdir -p data && cp /path/to/AI-eng-homeassignment-data.zip data/

# 5. Run tests (offline, LLM faked)
uv run --locked pytest -q

# 6. Produce verdicts
uv run --locked python -m triage.cli run \
  --data-zip data/AI-eng-homeassignment-data.zip \
  --config config/scoring.yaml \
  --out verdicts.jsonl \
  --audit-out verdicts_audit.jsonl

# 7. Verify every citation traces to raw data
uv run --locked python eval/verify_claims.py data/AI-eng-homeassignment-data.zip verdicts_audit.jsonl
```

**Environment variables** (`.env` or shell):
| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_BASE_URL` | yes | OpenAI-compatible endpoint |
| `LLM_API_KEY` | yes | API key (never printed/committed) |
| `LLM_MODEL` | yes | Feature-extraction model (e.g. `claude-sonnet-4-5`) |
| `LLM_SUMMARY_MODEL` | no | Explanation-writing model (default `gpt-4.1`) |
| `TRIAGE_MAX_CONCURRENCY` | no | Override `config/scoring.yaml` concurrency |
| `LOG_LEVEL` | no | Logging level (default `INFO`) |

**CLI flags:** `--data-zip` or `--data-dir` (mutually exclusive), `--config`, `--out`, `--audit-out`, `--max-concurrency`.

**Dependency audit:**
```bash
uv export --no-hashes --no-emit-project > /tmp/req.txt && uvx pip-audit -r /tmp/req.txt --disable-pip --no-deps
```

---

## Output format

### verdicts.jsonl (PDF contract)

Exactly 3 fields per line:
- `alert_id` — UUID
- `triage_score` — `HIGH`, `MEDIUM`, or `LOW`
- `explanation` — short reasoning from the summary LLM (or the deterministic explanation if its output was rejected twice)

### verdicts_audit.jsonl (full record)

Same 3 fields plus:
- `confidence_support` — `high`/`medium`/`low`
- `score` — 0-1 weighted combination
- `axis_summary` — value of each axis (`null` = could not be computed)
- `evidence` — citations (source file, field, value, note)
- `data_gaps` — fields that could not be determined
- `deterministic_tier` — tier before the `context_signal` axis was added
- `tier_override_reason` — rules that changed the tier
- `status` — `ok`, `llm_fallback`, or `failed`
- `error_type`, `error_message` — set on non-ok status, or when the summary was rejected twice (status stays `ok`)

**Status values:**
| Status | Meaning |
|--------|---------|
| `ok` | Scored; both LLM calls returned usable output, or only the summary was rejected (see `error_type`) |
| `llm_fallback` | Feature extraction failed; its output is not used, deterministic tier kept |
| `failed` | Unexpected exception; no-signal tier used |

**Exit code:** 1 if any alert has `status="failed"`, else 0.

**Atomic writes:** Output files are written to a temp file beside the target and renamed into place (crash-safe). File mode is 0600.

**Invariants:** `severity`, `priority`, `owner` are null on every input alert and are never invented.

---

## How scoring works

### From input files to axis inputs

Every step is deterministic code. Nothing below uses the LLM except the last row.

| Step | What happens | Where |
|------|--------------|-------|
| 1. Read | The zip is read in memory (safety checks in [Guardrails](#guardrails-and-security)); only the six required files are used. | `zipio.read_zip`, `loader.load_zip` |
| 2. Parse | JSON lines and CSV become typed objects. Timestamps must carry a timezone and are converted to UTC. Numbers stored as strings (`actual_value`, `lower_threshold`, `upper_threshold`) are parsed; a bad one becomes `None` plus a data gap. JSON stored as text (monitor `config`, event `schema_changes`) is parsed; malformed becomes `None` plus a gap. A null list becomes `[]` plus a gap. | `loader.load_files`, `loader.parse_float`, `loader.parse_json_field` |
| 3. Keep what is scored | History rows keep only `alert_id`, `monitor_id`, `table_id`, `created_at`, `alert_type`, `status`. Criticality rows keep only the modelled fields (`is_key_asset` is dropped). See [Known gaps](#known-gaps). | `loader._history`, `loader._criticality` |
| 4. Index | Monitors by `monitor_id`; criticality by `table_id`; lineage as source -> targets; BI reports by every upstream table they list; history grouped by `monitor_id` and by `alert_type`. | `schema.Dataset`, `context.ContextIndex`, `history.HistoryIndex` |
| 5. Join per alert | The alert's `tables[]` is the table list (monitor `tables[]` is empty for table monitors). For each table: its criticality row, its downstream closure (all tables reachable through lineage, cycle-safe). BI reports are those listing the alert's tables or any of their downstream tables, counted once by `report_id`. A missing monitor, criticality row or lineage entry is recorded as a data gap. | `context.ContextIndex.resolve` |
| 6. History | Rows of the alert's monitor, and a pool of rows for the alert's type (through `history.type_map`). Rows of excluded types are dropped. `fixed`/`acknowledged` count as acted, `no action needed`/`false positive`/`expected` as not acted, null as unlabeled (not counted), anything else is ignored with a gap. Rate = `(acted + 1) / (labeled + 2)`. | `history.history_stats`, `history.HistoryIndex.monitor_stats`, `history.HistoryIndex.pool_stats` |
| 7. Peers | Times of the other alerts of the same monitor, for the recency axis. | `service.TriageService._score` |
| 8. Free text (LLM) | Title, description and monitor notes are turned into features by the LLM and mapped to numbers for `context_signal`. | `features.FeatureExtractor.extract`, `axes.context_signal` |

### Seven axes

Each axis turns the joined data into a number in 0-1, or "unknown" (not counted, recorded as a data gap). Weights, caps and value maps are in `config/scoring.yaml`; code reads them and hard-codes none. `log_scale(x, cap)` is `min(log1p(x) / log1p(cap), 1)`.

| Axis (weight) | Normalised from | Value | Unknown or low support when | Code |
|---------------|-----------------|-------|------------------------------|------|
| `business_criticality` (0.28) | `table_criticality.jsonl`: `importance_score`, `criticality`, `reads_30d`, `writes_30d`, `active_users_30d`, `is_golden` | Per table: weighted mean of importance clipped to 0-1 (0.4), level high/medium/low = 1.0/0.6/0.2 (0.3) and usage (0.3, mean of `log_scale` of reads/writes/users with caps 10000/2000/20); `+0.1` if `is_golden`. Max over the alert's tables | No table has usable data: neutral 0.5, low support | `axes.business_criticality` |
| `blast_radius` (0.22) | `lineage_edges.csv`, `bi_reports.json` | `0.5 * log_scale(downstream tables, 2000) + 0.5 * log_scale(BI reports, 50)`; downstream tables are the union over the alert's tables | No alert table in lineage: BI part only, low support. Some tables missing: lower bound, low support | `axes.blast_radius` |
| `historical_action_rate` (0.20) | `alert_history.jsonl`: `monitor_id`, `alert_type`, `status` | The monitor's smoothed rate. Fewer than 5 labeled rows: weight halved. No monitor labels: pool rate at half weight | Neither monitor nor pool has labels | `axes.historical_action_rate` |
| `anomaly_severity` (0.17) | `alerts.jsonl` events: `event_type`, `actual_value`, `lower_threshold`, `upper_threshold`, `schema_changes` | Rule per `event_type` from config. **band**: distance outside the band divided by band width, capped at 2 widths. **breach**: `log_scale(actual - upper_threshold, 1000)`, using `lower_threshold` if no upper. **schema**: weighted change count (added 0.3, removed 1.0, type changed 1.0) over cap 4. **none**: constant 0.5. Band and breach map to `0.3 + 0.7 * magnitude`. Result is the highest event plus 0.02 per extra event (max +0.2) | No events or no usable numbers. A "none" event is low support | `axes.anomaly_severity` |
| `monitor_signal_quality` (0.08) | `monitors.jsonl`: `sensitivity` (or inner `config[type][0].sensitivity`) and inner `priority` | Mean of the sensitivity map (low 0.7, medium 0.5, high 0.3) and the priority map (p1 1.0, p2 0.75, p3 0.5, p4 0.25, p5 0.0) | Neither is set | `axes.monitor_signal_quality` |
| `recency_pressure` (0.05) | `alerts.jsonl`: `created_at` of this and sibling alerts | 1.0 if the same monitor fired another alert within 24 h, else 0.5 | Never | `axes.recency_pressure` |
| `context_signal` (0.10) | LLM features from title, description and monitor notes | Mean of the stated features: noise (expected_or_known 0.1, retired_or_draft 0.0, historical_backlog 0.2), recurrence (first 0.4, repeated 0.8), impact (low 0.2, medium 0.5, high 0.9) | No feature with a valid quote (skipped, not zero), or LLM failed | `axes.context_signal` |

**Override inputs:** `table_criticality.is_pii` drives `pii_boost`; `monitors.status` drives `paused_monitor`.

### Combine logic (`combine.combine`)

Steps run in this order:

1. **Base confidence** from the number of unknown or low-support axes: 0 -> high, 1 -> medium, 2 or more -> low (`confidence.medium_at`, `confidence.low_at`). Later rules can only lower it.
2. **Score** = `sum(value * weight * weight_factor) / sum(weight * weight_factor)` over axes with a value (weights renormalised). No axis has a value: tier `overrides.no_signal_tier` (MEDIUM), confidence low, stop.
3. **PII boost:** any alert table with `is_pii=true` adds `pii_boost` (0.08), capped at 1.0.
4. **Tier:** HIGH if score >= 0.65, MEDIUM if >= 0.35, else LOW.
5. **Thin evidence:** if the share of configured weight that has a value is below `min_axes_weight` (0.5), confidence becomes low and HIGH is capped at MEDIUM.
6. **Paused monitor:** `monitor.status == "paused"` downgrades one tier (HIGH -> MEDIUM, MEDIUM -> LOW) and caps confidence at medium.
7. **Missing criticality:** if every alert table lacks a criticality level, confidence is capped at medium.
8. **Cannot tell:** `cannot_tell` is true when the final tier is LOW and at most `confidence.cannot_tell_max_axes` (1) axes carry real data (a value, not low-support, weight above zero). Only then does the explanation say "I can't tell from this data" (see Assumptions).

**Worked example (from the golden dataset):**

| Axis | Value | Weight |
|------|-------|--------|
| business_criticality | 0.93 | 0.28 |
| blast_radius | 0.64 | 0.22 |
| historical_action_rate | 0.92 | 0.20 |
| monitor_signal_quality | 0.50 | 0.08 |
| recency_pressure | 1.00 | 0.05 |

Score = (0.28×0.93 + 0.22×0.64 + 0.20×0.92 + 0.08×0.50 + 0.05×1.00) / 0.83 = **0.81 -> HIGH**

(Weights sum to 0.83 because `anomaly_severity` and `context_signal` are absent: history rows have no events or free text.)


### How the customer's context changes a verdict

The same alert type gets different tiers because the tier depends on the account's own tables, lineage, history and monitor settings. Two `custom_sql` alerts from the 2026-11-04 run (ids from `verdicts_audit.jsonl`):

| | `64ac1e2b-dd37-59c4-a26d-83cc6ca4bace` | `43a4e388-bcb0-512d-a7de-3eb3aed0a9d1` |
|---|---|---|
| Tables | `raw.dim_customer`, `raw.fct_orders_opportunity` | `mart.fct_revenue_consumption` |
| `business_criticality` | 0.92 | 0.72 |
| `blast_radius` | 0.79 (345 downstream tables, 23 BI reports) | 0.70 (342 downstream tables, 11 BI reports) |
| `historical_action_rate` | 0.75 | 0.70 (alert-type pool; the monitor has no labeled history) |
| Rule applied | `pii_boost` (+0.08, `dim_customer` has `is_pii=true`) | monitor `status=paused`: one tier down |
| Score -> tier | 0.88 -> `HIGH` | 0.65 -> `MEDIUM` -> `LOW` |

LLM features move scores slightly between runs, so exact values can differ on a new run; the rules shown (`pii_boost`, `paused_monitor`) are deterministic.

### Why these axes

Each axis answers one question about "should a person look at this first?", using only data that exists and varies in the dataset. Weights and thresholds are configured defaults chosen by judgment, not fitted. The order of weights is: how much the table matters, who is downstream, what the team did before, how big this firing is, how the monitor is set up, recency. The only evidence on whether an axis helps is the golden check (AUC per axis, 0.5 = chance; see [Golden dataset](#golden-dataset)).

| Axis | Question | Why it is a usable signal | Evidence and caveats |
|------|----------|---------------------------|----------------------|
| `business_criticality` (0.28) | How much does the affected table matter? | `table_criticality.jsonl` is the only table-importance data: 48 tables, level HIGH 30 / MEDIUM 12 / LOW 5 / empty 1, plus importance and 30-day usage. `is_key_asset` is unused because it is false on every row. | Golden AUC 0.786. 2 of the 50 tables in alerts have no row; the axis then uses a neutral 0.5 and lowers confidence instead of guessing. |
| `blast_radius` (0.22) | Who is hurt if the data is wrong? | Lineage (5881 edges) and BI reports (606) are the only data on downstream consumers. Counts are log-scaled so the first downstream tables matter more than the thousandth. | Golden AUC 0.637 (modest). Tables missing from lineage give a lower bound, flagged as a data gap. |
| `historical_action_rate` (0.20) | Does the team usually act on this monitor? | Past outcomes are the only direct record of what people cared about. Only labeled rows vote (`fixed`/`acknowledged` vs `no action needed`/`false positive`/`expected`); unlabeled rows are not counted as "ignored". | Golden AUC 0.848, but it is computed from earlier labeled rows of the same monitor, the same kind of label the check scores against, so it is the least independent evidence. Only 18 of 40 monitors have any label; others fall back to the alert-type pool at half weight. |
| `anomaly_severity` (0.17) | How big is this firing? | The alert's own events carry numbers (actual value, thresholds, schema changes): the only per-firing size signal. All 46 alerts have events (86 in total). | Not measurable on golden (history has no events). 15 of 86 events (freshness, unchanged size, agent types) carry no magnitude and get a constant 0.5 with low confidence. Which threshold is "the limit" for breach events is an unverified assumption. |
| `monitor_signal_quality` (0.08) | Is this monitor's setup a credible one? | Sensitivity and inner-config priority are the only monitor settings that vary and compare. Kept small because the direction is an assumption: lower sensitivity is treated as more credible. | Golden AUC 0.330 (below chance on 68 rows), i.e. the assumption points the wrong way on this sample. The value map is unchanged; review it before relying on this axis. Top-level `sensitivity` is set on only 12 of 40 monitors. |
| `recency_pressure` (0.05) | Is this part of an ongoing burst? | All 46 alerts have `status = not_acknowledged`, so status cannot separate them. The only differentiator is the same monitor firing again within 24 h (4 of the 46 alerts). | Golden AUC 0.506 (chance) using an earlier-row proxy. Weight is deliberately small. |
| `context_signal` (0.10) | Does the free text state something the structured data lacks? | Titles, descriptions and monitor notes can say "expected", "retired", "backlog", "recurring" or the impact. The LLM only extracts the features, each with a verbatim quote that is checked; code maps them to numbers. | Effect on tier quality is not measured. Skipped, not zeroed, when nothing is stated. |

Fields that were considered and left out are listed in [Known gaps](#known-gaps).

### Config keys

All in `config/scoring.yaml`:
- `weights.*` — per-axis weights
- `thresholds.HIGH`, `thresholds.MEDIUM` — score cutoffs
- `min_axes_weight` — thin-evidence threshold
- `overrides.pii_boost`, `overrides.paused_monitor`, `overrides.missing_criticality`, `overrides.no_signal_tier`
- `confidence.medium_at`, `confidence.low_at`, `confidence.cannot_tell_max_axes`
- `axes.*` — per-axis parameters (caps, mappings, values)
- `history.*` — label sets, `type_map`, smoothing
- `llm.*`, `summary.*`, `runtime.max_concurrency`, `evidence.max_events_cited`

---

## LLM usage

Two calls per alert. The LLM **never sets the tier**.

### Call 1: Feature extraction (`features.FeatureExtractor.extract`)

**Input:** Only free text: `title`, `description` (from `alerts.jsonl`), `notes` (from `monitors.jsonl` `config[type][0].notes`). Sanitized (secrets/emails redacted via `guardrails.sanitize_untrusted`), capped at `llm.max_text_chars` (1000), wrapped in `<untrusted_data>` tags.

**Output:** `AlertFeatures`: three features, each with a verbatim quote:
- `stated_noise`: `none`, `expected_or_known`, `retired_or_draft`, `historical_backlog`
- `recurrence`: `unknown`, `first`, `repeated`
- `impact`: `unknown`, `low`, `medium`, `high`

**Guardrails (`guardrails.validate_features`):**
- Quote must be >= 8 characters (after whitespace normalization)
- Quote must not be `[redacted]` or `[email]`
- Quote must appear verbatim in the sanitized text sent

**Retry:** One re-ask with fixed feedback on guardrail rejection (`llm.max_feedback_retries=1`); transport errors retried via tenacity (`llm.max_attempts=3`, exponential backoff).

**Failure:** Status = `llm_fallback`; features not used; deterministic tier kept.

### Call 2: Explanation writing (`summary.Summarizer.summarize`)

**Input:** Only the verdict's own `evidence` and `data_gaps` lines, redacted and wrapped. Model: `LLM_SUMMARY_MODEL` env var, default `gpt-4.1`.

**Output:** a short plain-language explanation. The prompt asks for 3 to 4 sentences; the length is not enforced.

**Guardrails (`summary.validate_summary`):**
- Every number in the text must appear in the facts (string match)
- Every free-text source behind an LLM feature (`axis.context_signal`) must be quoted exactly
- Must include "I can't tell from this data:" when `cannot_tell` is true (see Assumptions), and must not use that sentence otherwise (other data gaps are still listed, as "Data gaps:", in the deterministic explanation)

**Failure:** rejected output is re-asked once (`llm.max_feedback_retries`); if still rejected, the deterministic explanation is kept, `status` stays `ok`, and `error_type`/`error_message` in the audit file record the reason.

### Prompt templates

Four Jinja2 templates in `src/triage/prompts/version/v0.0.1/`:
- `features_system.j2`, `features_human.j2` — feature extraction
- `summary_system.j2`, `summary_human.j2` — explanation writing

No prompt text lives in Python. To change a prompt: edit the files in `version/v0.0.1/` (git history keeps the old text).

---

## Input data

Six files, read in-memory via `zipio.read_zip` (Zip-Slip, zip-bomb, symlink, encryption checks; size/count caps). The zip's `README.md` and `.gitignore` are allowed but unused.

| File | Rows | Format | Used by |
|------|------|--------|---------|
| `alerts.jsonl` | 46 | JSON lines: `alert_id`, `monitor_id`, `created_at`, `alert_type`, `title`, `description`, `tables[]`, `events[]`. `severity`/`owner`/`priority` null on all rows. | Input feed; events -> `anomaly_severity`; tables -> criticality/lineage scope; title/description -> LLM |
| `monitors.jsonl` | 40 | JSON lines: `monitor_id`, `type`, `status`, `sensitivity`, `config` (JSON). `tables[]` empty for all 11 table-type monitors. | `status=paused` override; sensitivity/priority -> `monitor_signal_quality`; config.notes -> LLM |
| `alert_history.jsonl` | 1393 | JSON lines: `alert_id`, `monitor_id`, `table_id`, `created_at`, `alert_type`, `status`. 83 labeled (status not null). | `historical_action_rate` (per-monitor, per-type pool) |
| `table_criticality.jsonl` | 48 | JSON lines: `table_id`, `criticality`, `importance_score`, `is_golden`, `is_pii`, `reads_30d`, `writes_30d`, `active_users_30d`. `is_key_asset` false on all rows. | `business_criticality`; `is_pii` -> `pii_boost` |
| `bi_reports.json` | 606 | JSON array: `{report_id, name, upstream_tables[]}`. | `blast_radius` (BI report count) |
| `lineage_edges.csv` | 5881 | CSV: `source`, `target`. 26 self-loops (ignored by BFS). | `blast_radius` (downstream closure) |

**Null policy:** Null stays `None`; bad string-numerics become `None` + `DataGap` (never 0/False). Naive timestamps are errors. Unknown JSON keys are ignored.

**Alert-type taxonomy mismatch:** `alerts.jsonl` and `alert_history.jsonl` use different type names. Bridged via `history.type_map` in config.

---

## Known gaps

### Field coverage: what scoring reads and what it does not

Counts are non-null values in the raw zip, re-checked against it. "Loaded, not scored" fields are kept on the typed object (`schema.py`) but no axis or override reads them. "Not loaded" fields are not modelled in `schema.py` and are ignored at parse (`loader.py`).

| File | Scored | Loaded, not scored | Not loaded |
|------|--------|--------------------|------------|
| `alerts.jsonl` (46) | `alert_id`, `monitor_id`, `created_at`, `alert_type`, `tables`, `events`; `title` and `description` (40 of 46 set) go to the LLM only | `status` (`not_acknowledged` on all 46), `severity`, `owner`, `priority` (null on all 46; never invented) | none |
| `events[]` (86) | `event_type`, `actual_value` (68), `lower_threshold` (68), `upper_threshold` (48), `schema_changes` (5) | `event_id`, `created_at`, `field` (9) | `metric` (4), `segmented_by` (1), `segment` (1): too sparse |
| `monitors.jsonl` (40) | `monitor_id`, `type`, `status` (3 paused), `sensitivity` (12), `config` keys `sensitivity` (7), `priority` (11), `notes` (29, to the LLM) | `name`, `created_at`, `tables` (empty for the 11 `table` monitors, so `alert.tables` is used instead), other `config` keys | `owner` (set on all 40; identifiers of people and service accounts, no axis uses it), `priority` (null on all 40), `tags` (2, both `["pii"]`), `schedule` (7), `table_count`, `updated_at` |
| `alert_history.jsonl` (1393) | `monitor_id`, `alert_type`, `status` (83 labeled), `created_at` and `alert_id` (only for the leakage rule in the golden check) | `table_id` (1185; live scoring groups by monitor and type, the golden check joins it for table fields) | `severity` (1), `owner` (1), `ticket_created` (false on all), `minutes_to_first_response` (83) and `minutes_to_resolved` (76): set only on labeled rows, i.e. they follow the outcome, so using them would leak the label |
| `table_criticality.jsonl` (48) | `criticality` (47), `importance_score`, `reads_30d`, `writes_30d`, `active_users_30d`, `is_golden` (7 true), `is_pii` (2 true) | none | `is_key_asset` (false on all 48), `use_cases` (47, unstructured text) |
| `bi_reports.json` (606) | `upstream_tables`; `report_id` only to count each report once | `name` | `report_type` (set on all 606: dashboard 259, view 182, explore 119, bi_report 42, look 4; blast radius counts reports of any type), `importance` (9 of 606) |
| `lineage_edges.csv` (5881) | `source`, `target` | none | none |

The assignment names monitor `owner` and `tags` as context for judging an alert. Neither is used; whether they would help was not tested.

### Golden dataset gap

The golden check scores 5 of 7 axes because history rows have no events or free text:
- `anomaly_severity` (weight 0.17) needs the alert's events (actual_value, thresholds, schema_changes)
- `context_signal` (weight 0.10) needs the alert's text (title, description, monitor notes)

The 46 live alerts have events and text but no outcome (`status = not_acknowledged` on all 46) and share no `alert_id` with history. Neither missing axis is benchmarked against labeled outcomes.

### Other limitations

- LLM features vary between runs; the guard proves a quote exists, not that the feature value is correct. Whether `context_signal` improves tiers is **not measured**.
- Label sparsity: 83 of 1393 history rows are labeled. Labels are noisy priors, not ground truth.
- Tier split on 46 alerts (example run, 2026-11-05): HIGH 21, MEDIUM 22, LOW 3; all `ok`, no summary rejected after the re-ask; confidence low 23, medium 19, high 4. Counts shift between runs because LLM output varies. Not tuned.
- In that run 23 of 46 verdicts had `confidence_support: low`, 19 medium, 4 high.
- `monitor_signal_quality` unknown for ~15 alerts.
- Weights and thresholds are configured defaults, not tuned on data.
- The `--data-dir` CLI path has no automated test. A manual run on the unzipped dataset (2026-11-04) gave the same tier for all 46 alerts as the zip run, and `verify_claims.py` reported 0 mismatches.
- Live runs do not apply the history leakage controls (`as_of`, `exclude_alert_id`): live alerts share no `alert_id` with history. Only the golden check applies the equivalent rule.

---

## Guardrails and security

Guardrails run automatically; there is nothing to switch on. "Where" names the function that does the work.

| Stage | Guardrail | Where | Triggered |
|-------|-----------|-------|-----------|
| Config | Typos in keys fail (`extra="forbid"`); weights must match the axis names, be >= 0 and sum > 0; `HIGH > MEDIUM`; YAML via `safe_load` | `config.load_config`, `config.ScoringConfig` | At start-up |
| Input zip | Read in memory; rejects `..`, absolute paths, `\` (Zip-Slip), encrypted members and symlinks; caps 32 files, 250 MB total, 150 MB per file, 100x compression ratio | `zipio.read_zip`, called by `loader.load_zip` | When `--data-zip` is read |
| Input data | Bad JSON/CSV/timestamp raises `LoaderError` naming file and row; null stays null; bad numeric becomes `None` + data gap | `loader.load_files`, `loader.parse_float`, `loader.parse_json_field` | When data is loaded |
| Before LLM | Only title, description and monitor notes are sent; secrets and emails redacted, fake delimiter tags removed, capped at `llm.max_text_chars` (1000), wrapped in `<untrusted_data>` so it is treated as data | `guardrails.sanitize_untrusted`, `guardrails.wrap_untrusted`, called from `features.FeatureExtractor.extract` and `summary.Summarizer.summarize` | Before every LLM call |
| LLM features | Must match the `AlertFeatures` schema; a non-default value needs a quote of at least 8 characters that appears in the text sent; rejected output is re-asked once (`llm.max_feedback_retries`), then falls back | `guardrails.validate_features`, `features.FeatureExtractor.extract` | After LLM call 1 |
| LLM text | every number appears in the facts; LLM-feature source quotes are exact; says "I can't tell from this data" when `cannot_tell` is true and only then; rejected output is re-asked once, then the deterministic explanation is used | `summary.validate_summary`, `summary.Summarizer.summarize` | After LLM call 2 |
| Evidence quotes | Only quotes that still match the raw text are added to `evidence` | `guardrails.find_quote`, `service._quote_evidence` | While building the verdict |
| LLM power | The model cannot set the tier; feature values are mapped to numbers by code | `axes.context_signal`, `combine.combine` | Always |
| Transport errors | Retried with exponential backoff (`llm.max_attempts`) | `features.with_retry` | On LLM call errors |
| Scoring rules | HIGH capped at MEDIUM on thin evidence; paused monitor downgraded one tier; missing criticality caps confidence | `combine.combine` | Every alert |
| One bad alert | Exception becomes a `failed` verdict; the run continues; exit code 1 | `service.TriageService.triage_one`, `cli._run` | On any unexpected exception |
| Output | Verdict validated (tier in HIGH/MEDIUM/LOW, status in the allowed set); files written to a temp file then renamed, owner-only mode | `verdict.Verdict`, `cli._write_jsonl` | When writing output |
| Runtime | Concurrency bounded by a semaphore | `service.TriageService.triage_all` | Always |
| History leakage | `HistoryIndex.monitor_stats` and `pool_stats` can ignore rows dated at or after `as_of` and the row with `exclude_alert_id` (unit-tested), but live runs do not pass them: live alerts share no `alert_id` with history. The golden check applies the same rule itself: only earlier rows of the same monitor, never the row's own outcome | `history.HistoryIndex.monitor_stats`, `history.HistoryIndex.pool_stats`, `eval/build_golden.py::golden_row` | Golden build only |
| Secrets in logs | Fixed-field logs; the API key is never printed | `features.log_failure`, `cli` logging setup | Always |
| Repo | `data/`, `verdicts_audit.jsonl`, `.env`, `eval/reports/` gitignored (`verdicts.jsonl` is the deliverable and is tracked); deps pinned in `uv.lock` | `.gitignore`, `uv.lock` | n/a |

---

## Evaluation

Three checks, none of which needs an LLM key except the live run. Run them by hand; none runs automatically.

| Check | Question it answers | Command | When |
|-------|--------------------|---------|------|
| Tests | Does the code behave as designed? | `uv run --locked pytest -q` | After every code change |
| Claim verification | Is every number and quote in the output really in the raw data? | `eval/verify_claims.py` | After every real run |
| Golden check | Do the axes rank past alerts the way people acted on them? | `eval/golden.py` | After every change to `config/scoring.yaml` or `axes.py` |

### Tests

`uv run --locked pytest -q` runs 15 offline tests (LLM faked):

| File | Tests | Covers |
|------|-------|--------|
| `test_scoring.py` | 5 | all 7 axes, combine/overrides, config validation |
| `test_gold.py` | 2 | synthetic gold worlds, null policy, LLM injection/citation/fallback |
| `test_service.py` | 3 | async batch, concurrency, failure isolation, real-zip end-to-end |
| `test_guardrails.py` | 1 | quote validation, redaction, delimiting |
| `test_history.py` | 1 | labeled-only rates, leakage control |
| `test_loader.py` | 1 | defensive parsing, null policy, strict inputs |
| `test_context.py` | 1 | lineage, scope, gaps, BI blast radius |
| `test_zipio.py` | 1 | Zip-Slip, zip bomb, symlink, size caps |

### Claim verification

**Purpose:** show that the evidence behind each verdict is real. Each `evidence` item in `verdicts_audit.jsonl` names a source file, field and value. `eval/verify_claims.py <zip> <audit file>` recomputes the value from the raw zip with its own code (it imports nothing from `triage`), so a bug in `loader`, `context` or `history` cannot confirm itself. Exit code 1 on any mismatch.

| Checked | How |
|---------|-----|
| `alerts.jsonl` fields | `alert_type`, number of events, `events[i].*` values compared to the raw row |
| `table_criticality.jsonl` fields | value compared to the raw row |
| Downstream table count, BI report count | recomputed from `lineage_edges.csv` and `bi_reports.json` |
| `history.n_total`, `n_labeled`, `n_acted` | recounted per monitor from `alert_history.jsonl` |
| Monitor `status` | compared to `monitors.jsonl` |
| LLM quotes (evidence note `LLM label:`) | must be a substring of the raw title, description or monitor notes (whitespace ignored) |

**Not checked:** scoring outputs (`axis.*`, `combined.*`) and config citations, which are counted as "skipped"; the explanation sentences (their numbers are checked while writing, by `summary.validate_summary`); and whether an LLM feature value is right, only that its quote exists. The acted / not-acted status lists are hard-coded in the script, so they must be kept in step with `history.*` in `config/scoring.yaml`.

Example run: `raw-data evidence items re-derived=757 mismatches=0; skipped scoring-output/config citations=442`.

### Golden dataset

```bash
# Build once (needs zip):
uv run --locked python eval/build_golden.py data/AI-eng-homeassignment-data.zip

# Check any time (no zip):
uv run --locked python eval/golden.py
```

The "golden dataset" here is the 83 labeled history rows used as a check set. It is unrelated to `table_criticality.is_golden`, a flag on tables.

**How it works:** `build_golden.py` joins each labeled row to its table data and to earlier rows of its monitor only (`created_at` before the row, own `alert_id` excluded, so a row never sees its own outcome), and saves the inputs. `golden.py` needs no zip: it re-scores the saved rows with the production axes (`golden.rescore`), and prints, in order: when the baseline was built and whether the config file hash changed; each config value that differs; the headline metrics saved baseline -> now (score AUC, per-axis AUC, final tier counts, with `<- CHANGED` marks); then, for every changed row, its monitor, table, alert type, time and label with old -> new tier, score and axis values and the reasons now. It exits 1 on any row change. An intended change is accepted by re-running `build_golden.py`, which saves a new baseline.

**Not covered:** the saved rows hold the joined inputs, so a change in `loader`, `ContextIndex` or `HistoryIndex` (the joins) is not detected; only changes in the axis functions, `history_stats`, `combine` and the config are.

**Scope:** 83 labeled history rows scored on 5 axes (`business_criticality`, `blast_radius`, `historical_action_rate`, `monitor_signal_quality`, `recency_pressure`), reported as AUC (how well the score orders acted above not-acted rows) and as acted counts per tier.

**Results:**
- AUC = 0.827, 95% CI over 18 monitors = (0.499, 0.916)
- Per-axis AUC: business_criticality 0.786, blast_radius 0.637, historical_action_rate 0.848, recency_pressure 0.506, monitor_signal_quality 0.330
- Threshold sensitivity (acted/total): config 0.65/0.35 gives HIGH 16/31, MEDIUM 6/27, LOW 0/25

**Files in `eval/reports/` (gitignored):**
- `golden_labeled.jsonl` — 83 rows with inputs, axis values, score, tier
- `golden_run.json` — build time, config and zip hashes, the config values used; the `golden_labeled.jsonl` hash is enforced

**Limits:** 18 monitors, 83 labels, labels are noisy priors, partial score (5/7 axes), not a benchmark of final tiers.

### What is NOT measured

- Precision/recall/confusion matrix of final tiers
- Whether `context_signal` improves triage quality
- Live-LLM test
- `--data-dir` path (manual run only, see Known gaps)
- Drift monitoring

---

## Operations

### 50,000 alerts/day considerations

(Reasoning, not measurements.)

- **Cost/latency:** One LLM call per alert is expensive. Mitigations: cache by redacted-text hash, skip when text is null, batch per monitor, use smaller model, use provider batch API. Token usage/latency are **not tracked today**.
- **Throughput:** Service is async with bounded semaphore (`runtime.max_concurrency`). Core is importable (`triage.service`); FastAPI wrapper or queue consumer can reuse it.
- **Data access:** At scale, `ContextIndex`/`HistoryIndex` become queries against a store (precomputed rates, refreshed on schedule).
- **Failures:** Per-key rate limiting, dead-letter queue. Deterministic tier always computed first.
- **Alert storms:** Group alerts per monitor/table within a window; triage once.

### Production monitoring (proposed, not implemented)

- **Per run:** Share of non-ok status, low confidence, tier mix, data gaps by field, p95 latency
- **Daily:** Drift in tier mix, new alert types, evidence sample check
- **Weekly:** Compare verdicts to actual team actions, recompute metrics, review config
- **Page when:** LLM failure rate high, HIGH share jumps/collapses, any `failed` status, evidence mismatch

---

## Future improvements

**Scoring quality**
- Expose scoring through an API (a thin wrapper over `triage.service`) so users can set weights, thresholds and overrides per request or per tenant, validated by the same `ScoringConfig` model, instead of editing `config/scoring.yaml`.
- Review `monitor_signal_quality`: its golden AUC is 0.330 (below chance), so the "lower sensitivity is more credible" assumption points the wrong way on this sample. Re-test the direction or lower the weight.
- Benchmark all seven axes and the final tiers (precision/recall) on labeled history, then tune weights and thresholds. Needs more than the 83 labels from 18 monitors.


**Regression safety**
- Add a baseline of the deterministic axes and tiers of the 46 live alerts (`TriageService.triage_deterministic`, no LLM) and diff it after every change. The golden check stores its inputs, so it does not notice a change in `loader`, `ContextIndex` or `HistoryIndex`.
- One function that assembles the deterministic axes. `TriageService._score` and `eval/golden.py::rescore` call the same axis functions, `history_stats` and `combine`, but each lists the axes itself, so an axis added to one can be forgotten in the other.
- More adversarial prompt-injection cases in `tests/gold/cases.json`.

**LLM calls**
- Track token usage and latency per LLM call (input/output/cached tokens).
- Check prompt caching (static system message first; no `cache_control` marker is set).
- Circuit breaker after N consecutive LLM failures, and token-bucket coordination for 429s.

**Engineering**
- JSON/OpenTelemetry-compatible logging (trace ids, token and latency metrics).
- For large files that wont fit in memory , use chunking for e.g
- Expose this as an api, containerization, stateless (if possible) scalable service
- Data Persistence of triaged alerts
- Rate limiting, authentication, Authorization for service
  

---

## What I'd do with more data

The assignment leaves out the warehouse tables, the query log and the pipeline run history. Proposals, none implemented:

- **Warehouse tables:** replace the crude breach magnitude with real distribution checks (distance from a learned band, null-rate trend); check whether a schema change touches columns that are actually used.
- **Query log:** use real consumers instead of the `reads_30d` / `active_users_30d` proxies; see which downstream queries would break.
- **Pipeline run history:** tell a late upstream job from a real freshness problem; group alerts that share a failed run into one incident.
- **Going deeper on the alerts that matter:** run these extra checks only for `HIGH` alerts, so cost stays bounded.

---

## Assumptions

- History joins to alerts only by `monitor_id` (alert_id spaces are disjoint)
- Unlabeled history rows (status=None) are not votes
- `severity`, `priority`, `owner` are null on every input alert and never invented
- Breach rule direction: `upper_threshold` if present, else `lower_threshold` (config assumption, unvalidated)
- Sensitivity direction: lower = more credible (config assumption, unverifiable from data)
- Weights and thresholds are configured defaults, not tuned on data
- **"I can't tell from this data."** Said when `Verdict.cannot_tell` is true, never otherwise. The PDF defines no trigger, so this is an assumption (user decision): `cannot_tell` = final tier is LOW **and** at most `confidence.cannot_tell_max_axes` (1) axes carry real data. An axis counts only if it has a value, is not `low_support` and has weight above zero, so neutral defaults, lower bounds, thin evidence and zero-weight axes do not count (`combine._cannot_tell`). Other gaps, and every HIGH/MEDIUM verdict, are listed as `Data gaps: ...` without the sentence, even at low confidence. `TriageService._score` always names a reason (a gap per thin axis, or one generic `confidence` gap), `summary.validate_summary` rejects a summary that lacks the sentence when `cannot_tell` is true or has it otherwise, and a failed alert sets `cannot_tell` too. `recency_pressure` always has a value and is never thin, so the rule needs every other axis unknown or thin; this is rare. In the 46 example alerts no verdict has `cannot_tell` (the 3 LOW verdicts have 5 to 7 axes with a value; 23 verdicts have low confidence but are not LOW or have several axes). The rule is unit-tested (`tests/test_scoring.py`, `tests/test_service.py`, `tests/test_gold.py`)

---

## Glossary

| Term | Meaning |
|------|---------|
| Feature (LLM) | One of the three fields the LLM extracts from free text (`stated_noise`, `recurrence`, `impact`); code in `features.py`, schema `AlertFeatures`. (The code constant `LABEL_NOTE` and evidence notes still say "LLM label".) |
| Label / labeled | Only for history outcomes: a history row is labeled when its `status` is set (for example `fixed`). Not the same as an LLM feature. |
| Explanation vs summary | `summary.py` is the LLM call that writes the verdict's `explanation`; if its output is rejected twice, the deterministic explanation is kept and `error_type` is set. |
| Axis | One scored question about an alert, giving a value in 0-1 (or unknown). |
| Tier | `HIGH`, `MEDIUM` or `LOW`: how soon a person should look. |
| Confidence (`confidence_support`) | How much data backs the tier: `high`, `medium` or `low`. Separate from the tier. |
| Data gap | A recorded fact that a value is missing or unusable (`field` + `reason`). |
| Low support | An axis value that was computed from partial data (for example a neutral default or a lower bound). |
| Laplace smoothing | `(acted + 1) / (labeled + 2)`: keeps a rate away from 0 and 1 when there are few labels. |
| Log-scaled | `min(log1p(x) / log1p(cap), 1)`: the first few units count more than the thousandth. |
| Downstream closure | All tables reachable by following lineage edges from a table, repeatedly. |
| Leakage | Using information that would not have been known when the alert fired (for example the alert's own outcome). |
| AUC | Chance that a randomly picked acted-on row scores above a not-acted row. 0.5 is chance, 1.0 is perfect. |
| Golden dataset | The evaluation set of 83 labeled history rows (see [Golden dataset](#golden-dataset)). Not related to `is_golden`. |
| Gold cases | The 12 hand-built synthetic worlds in `tests/gold/cases.json`, used by `test_gold.py`; unrelated to the golden dataset. |
| `is_golden` | A flag on `table_criticality.jsonl` rows marking a table as golden; it adds `golden_bonus` (0.1) to `business_criticality`. In `eval/` it appears as `table_is_golden`. |

---

## AI assistance and time spent

- **At run time:** two LLM calls per alert (feature extraction, explanation); neither sets the tier (see [LLM usage](#llm-usage)).
- **During development:** an AI coding assistant (OpenCode with Claude and GPT models) wrote much of the code, tests and this README under written project rules, including a review loop with a second model. I must be able to explain and defend all of it.
- **Time spent:** Spend around 3hr 45 minutes across 3 sitting

---

