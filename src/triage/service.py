"""Importable core: typed in/out, no CLI or web dependency. A FastAPI wrapper would call triage_one/triage_all."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from triage import axes as ax
from triage.combine import combine
from triage.config import ScoringConfig, load_config
from triage.context import AlertContext, ContextIndex
from triage.explain import GAP_PHRASE, build_evidence, build_explanation
from triage.history import HistoryIndex
from triage.features import Extraction, FeatureExtractor, build_chain, log_failure
from triage.guardrails import find_quote
from triage.loader import load_dir, load_zip
from triage.schema import Alert, DataGap, Dataset
from triage.summary import LABEL_NOTE, Summarizer, build_summary_chain
from triage.verdict import EvidenceCitation, Verdict

log = logging.getLogger(__name__)


class TriageService:
    """API-ready core service for triaging alerts with typed in/out.

    Initialized with Dataset, ScoringConfig, FeatureExtractor, optional Summarizer,
    and max_concurrency. Provides triage_one (async, per-alert), triage_all (async batch
    with semaphore), and triage_deterministic (sync, no LLM). Used by cli.py; can be
    wrapped by FastAPI without changing core logic.
    """

    def __init__(self, ds: Dataset, cfg: ScoringConfig, extractor: FeatureExtractor, *, summarizer: Summarizer | None = None, max_concurrency: int | None = None):
        self._ds, self._cfg, self._extractor, self._summarizer = ds, cfg, extractor, summarizer
        self._ctx = ContextIndex(ds)
        self._hist = HistoryIndex(ds.history, cfg.history)
        self._times: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
        for a in ds.alerts:
            self._times[a.monitor_id].append((a.alert_id, a.created_at))
        self._max_concurrency = max_concurrency or cfg.runtime.max_concurrency
        if self._max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

    @classmethod
    def from_zip(cls, zip_path: str | Path, config_path: str | Path, *, extractor: FeatureExtractor | None = None, summarizer: Summarizer | None = None, max_concurrency: int | None = None) -> TriageService:
        """Create a TriageService from a dataset zip and config file."""
        cfg = load_config(config_path)
        return cls(load_zip(zip_path), cfg, extractor or _default_extractor(cfg), summarizer=summarizer or _default_summarizer(cfg), max_concurrency=max_concurrency)

    @classmethod
    def from_dir(cls, data_dir: str | Path, config_path: str | Path, *, extractor: FeatureExtractor | None = None, summarizer: Summarizer | None = None, max_concurrency: int | None = None) -> TriageService:
        """Create a TriageService from an unzipped data directory and config file."""
        cfg = load_config(config_path)
        return cls(load_dir(data_dir), cfg, extractor or _default_extractor(cfg), summarizer=summarizer or _default_summarizer(cfg), max_concurrency=max_concurrency)

    @property
    def alerts(self) -> list[Alert]:
        return self._ds.alerts

    def triage_deterministic(self, alert: Alert) -> Verdict:
        """Sync path without the LLM: context joins -> axes -> combine -> evidence-backed explanation."""
        return self._score(alert, None)

    def _score(self, alert: Alert, extraction: Extraction | None) -> Verdict:
        """Core scoring: context joins -> axes -> combine -> evidence-backed explanation.

        Logic:
          1. Resolve alert context (monitor, criticality, lineage, BI).
          2. Compute 6 deterministic axes; combine into base tier.
          3. If extraction has LLM features, add context_signal axis and re-combine.
          4. Build evidence citations and explanation from the same data.
        """
        cfg = self._cfg
        ctx: AlertContext = self._ctx.resolve(alert)
        mon_stats = self._hist.monitor_stats(alert.monitor_id)
        pool = self._hist.pool_stats(alert.alert_type)
        peers = [t for aid, t in self._times[alert.monitor_id] if aid != alert.alert_id]
        axes = {
            "business_criticality": ax.business_criticality(ctx, cfg.axes.business_criticality),
            "blast_radius": ax.blast_radius(ctx, cfg.axes.blast_radius),
            "historical_action_rate": ax.historical_action_rate(mon_stats, pool, cfg.axes.historical_action_rate),
            "anomaly_severity": ax.anomaly_severity(alert.events, cfg.axes.anomaly_severity),
            "monitor_signal_quality": ax.monitor_signal_quality(ctx.monitor, cfg.axes.monitor_signal_quality),
            "recency_pressure": ax.recency_pressure(alert.created_at, peers, cfg.axes.recency_pressure),
        }
        base = combine(axes, ctx, cfg)
        comb, evidence_extra = base, []
        gaps = [*ctx.data_gaps, *(g for e in alert.events for g in e.data_gaps), *(g for r in axes.values() for g in r.gaps)]
        if extraction is not None:
            gaps += extraction.gaps
            signal = ax.context_signal(extraction.features, cfg.axes.context_signal)
            if signal.value is not None:  # nothing stated -> axis skipped, so it neither helps nor hurts confidence
                axes["context_signal"] = signal
                comb = combine(axes, ctx, cfg)
                evidence_extra = _quote_evidence(extraction)
        if comb.cannot_tell or comb.confidence == "low":
            gaps = [*_low_confidence_gaps(axes, has_gaps=bool(gaps)), *gaps]  # first: the explanation shows only the first 5 gaps
        gaps = _dedupe(gaps)
        verdict = Verdict(
            alert_id=alert.alert_id,
            triage_score=comb.tier,
            deterministic_tier=base.tier,
            confidence_support=comb.confidence,
            cannot_tell=comb.cannot_tell,
            explanation=build_explanation(ctx, axes, comb, gaps),
            evidence=[*build_evidence(ctx, mon_stats, axes, comb, max_events=cfg.evidence.max_events_cited), *evidence_extra],
            axis_summary={k: r.value for k, r in axes.items()},
            score=comb.score,
            tier_override_reason="; ".join(comb.reasons) or None,
            data_gaps=gaps,
        )
        if extraction is not None and extraction.error_type:
            verdict.status, verdict.error_type, verdict.error_message = "llm_fallback", extraction.error_type, extraction.error_message
        return verdict

    def _failed(self, alert: Alert, exc: Exception) -> Verdict:
        """Return a failed verdict with the configured no_signal_tier when processing crashes."""
        tier = self._cfg.overrides.no_signal_tier
        return Verdict(
            alert_id=alert.alert_id,
            triage_score=tier,
            deterministic_tier=tier,
            confidence_support="low",
            cannot_tell=tier == "LOW",  # no axis was scored, so the same rule as combine._cannot_tell applies
            explanation=f"{f'{GAP_PHRASE}: processing' if tier == 'LOW' else 'Processing'} failed ({type(exc).__name__}). Tier is the configured no-signal default.",
            status="failed",
            error_type=type(exc).__name__,
            error_message="unexpected error while scoring this alert; see logs for the traceback",
        )

    async def triage_one(self, alert: Alert) -> Verdict:
        """Score one alert: extract LLM features, compute verdict, write summary; never raises."""
        try:
            extraction = await self._extractor.extract(alert, self._ds.monitors.get(alert.monitor_id))
            verdict = self._score(alert, extraction)
            if self._summarizer is not None:
                result = await self._summarizer.summarize(verdict)
                if result.text is not None:
                    verdict.explanation = result.text
                elif verdict.error_type is None:  # keep the deterministic explanation; record why the summary was not used
                    verdict.error_type, verdict.error_message = result.error_type, result.error_message
            return verdict
        except Exception as exc:  # noqa: BLE001 - converted to a flagged verdict, never swallowed or silently dropped
            log_failure(log, f"triage failed alert_id={alert.alert_id}", exc)  # frames + type only, never exception text
            return self._failed(alert, exc)

    async def triage_all(self, alerts: list[Alert] | None = None) -> list[Verdict]:
        """Triage all alerts in parallel with bounded concurrency (max_concurrency semaphore)."""
        sem = asyncio.Semaphore(self._max_concurrency)

        async def run(a: Alert) -> Verdict:
            async with sem:
                return await self.triage_one(a)

        return list(await asyncio.gather(*(run(a) for a in (self.alerts if alerts is None else alerts))))


def _default_extractor(cfg: ScoringConfig) -> FeatureExtractor:
    return FeatureExtractor(build_chain(cfg.llm), cfg.llm)


def _default_summarizer(cfg: ScoringConfig) -> Summarizer:
    return Summarizer(build_summary_chain(cfg), cfg)


def _low_confidence_gaps(axes: dict[str, ax.AxisResult], *, has_gaps: bool) -> list[DataGap]:
    """Name why confidence is low or the verdict cannot be told, so the explanation always lists a reason.

    Low confidence normally comes with gaps already (an unknown axis records one). Axes that only
    rest on thin evidence (low_support) record none, so they are named here; if the verdict still has
    no gap at all, one generic gap is added so a low-confidence verdict can never read as confident.
    """
    gaps = [DataGap(field=f"axis.{k}", reason=f"weak evidence: {r.note or 'thin data'}") for k, r in axes.items() if r.low_support and not r.gaps]
    if not gaps and not has_gaps:
        gaps = [DataGap(field="confidence", reason="low confidence: too little evidence behind the score")]
    return gaps


def _dedupe(gaps: list[DataGap]) -> list[DataGap]:
    """Remove duplicate DataGap entries by (field, reason)."""
    seen: dict[tuple[str, str], DataGap] = {}
    for g in gaps:
        seen.setdefault((g.field, g.reason), g)
    return list(seen.values())


_QUOTE_SOURCE = {"title": "alerts.jsonl", "description": "alerts.jsonl", "notes": "monitors.jsonl"}
_QUOTE_FIELD = {"title": "title", "description": "description", "notes": "config.notes"}


def _quote_evidence(x: Extraction) -> list[EvidenceCitation]:
    """Build evidence citations for LLM labels with their verbatim quotes."""
    f = x.features
    if f is None:
        return []
    out = []
    for label, quote in ((f.stated_noise, f.noise_quote), (f.recurrence, f.recurrence_quote), (f.impact, f.impact_quote)):
        if label in ("none", "unknown"):
            continue
        name = find_quote(quote, x.sources)
        if name is not None:
            out.append(EvidenceCitation(source=_QUOTE_SOURCE[name], field=_QUOTE_FIELD[name], value=quote, note=f"{LABEL_NOTE} {label}"))
    return out
