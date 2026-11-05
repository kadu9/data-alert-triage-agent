"""Axes -> tier. Pure and config-driven; no I/O. Overrides are general rules from config, never per-alert."""
from __future__ import annotations

from pydantic import BaseModel

from triage.axes import AxisResult
from triage.config import Level, ScoringConfig, Tier
from triage.context import AlertContext

_TIERS: tuple[Tier, ...] = ("LOW", "MEDIUM", "HIGH")
_LEVELS: tuple[Level, ...] = ("low", "medium", "high")


class Combined(BaseModel):
    """Result of combining axis scores into a final tier.

    Holds score (weighted mean or None), tier (HIGH/MEDIUM/LOW), confidence level,
    cannot_tell flag, and reasons list recording every override applied. Produced by combine.combine;
    used by explain and service to build the Verdict.
    """

    score: float | None
    tier: Tier
    confidence: Level
    reasons: list[str]  # every rule that changed tier or confidence, in order
    cannot_tell: bool  # final tier LOW and at most confidence.cannot_tell_max_axes axes carry real data


def _tier_for(score: float, cfg: ScoringConfig) -> Tier:
    """Map a [0,1] score to HIGH/MEDIUM/LOW using thresholds from config."""
    if score >= cfg.thresholds.HIGH:
        return "HIGH"
    return "MEDIUM" if score >= cfg.thresholds.MEDIUM else "LOW"


def _cap_level(current: Level, cap: Level) -> Level:
    """Lower confidence to at most cap (high > medium > low)."""
    return _LEVELS[min(_LEVELS.index(current), _LEVELS.index(cap))]


def _cannot_tell(tier: Tier, supported: int, cfg: ScoringConfig) -> bool:
    """True when the verdict is LOW and rests on at most confidence.cannot_tell_max_axes axes with real data."""
    return tier == "LOW" and supported <= cfg.confidence.cannot_tell_max_axes


def combine(axes: dict[str, AxisResult], ctx: AlertContext, cfg: ScoringConfig) -> Combined:
    """Combine axis scores into a final tier with confidence and override rules.

    In plain words: takes all the axis scores, computes a weighted average (skipping axes
    with no value), applies boosts and overrides (PII, paused monitor, thin evidence), then
    maps the final score to HIGH/MEDIUM/LOW using thresholds.

    Logic:
      1. Base confidence (high/medium/low) from the count of unknown/low_support axes vs
         confidence.medium_at/low_at; later rules can only lower it.
      2. If no axis has a value: score None, tier = overrides.no_signal_tier, confidence low; stop.
      3. Weighted mean: sum(value * weight * weight_factor) / sum(weight * weight_factor)
         over axes that have a value; weights from config.weights, renormalised over present axes.
      4. Add pii_boost (+overrides.pii_boost, score capped at 1.0) if any alert table has is_pii=true.
      5. Map score to tier via thresholds.HIGH / thresholds.MEDIUM.
      6. If the weight share of axes with a value (after weight_factor) < min_axes_weight:
         confidence = low and a HIGH tier is capped at MEDIUM.
      7. If monitor.status == paused: apply overrides.paused_monitor (downgrade_one_tier, confidence cap).
      8. If no alert table has a criticality level: cap confidence via overrides.missing_criticality.
      9. cannot_tell = final tier is LOW and at most confidence.cannot_tell_max_axes axes carry real
         data (a value with weight that is not low_support, so neutral defaults, thin evidence and zero-weight axes do not count).

    Example (config: weights 0.28/0.22/0.20/0.08/0.05, thresholds HIGH=0.65, MEDIUM=0.35):
        axes: business_criticality=0.93, blast_radius=0.64, historical_action_rate=0.92,
              monitor_signal_quality=0.50, recency_pressure=1.00
        numer = 0.93*0.28 + 0.64*0.22 + 0.92*0.20 + 0.50*0.08 + 1.00*0.05 = 0.6752
        denom = 0.28 + 0.22 + 0.20 + 0.08 + 0.05 = 0.83
        score = 0.6752 / 0.83 = 0.8135 -> tier HIGH (>= 0.65)
    """
    reasons: list[str] = []
    supported = sum(1 for r in axes.values() if r.value is not None and not r.low_support and r.weight_factor > 0)
    present = {k: r for k, r in axes.items() if r.value is not None}
    weighted = {k: cfg.weights[k] * r.weight_factor for k, r in present.items()}
    total_w = sum(cfg.weights[k] for k in axes)  # optional axes left out of `axes` do not count
    weak = sum(1 for k in axes if k not in present or axes[k].low_support)

    if cfg.confidence.low_at <= weak:
        confidence: Level = "low"
    elif cfg.confidence.medium_at <= weak:
        confidence = "medium"
    else:
        confidence = "high"

    denom = sum(weighted.values())
    if not present or denom <= 0:
        reasons.append(f"no axis has a value; tier set to overrides.no_signal_tier={cfg.overrides.no_signal_tier}")
        tier = cfg.overrides.no_signal_tier
        return Combined(score=None, tier=tier, confidence="low", reasons=reasons, cannot_tell=_cannot_tell(tier, supported, cfg))

    score = sum((r.value or 0.0) * weighted[k] for k, r in present.items()) / denom
    if any(c is not None and c.is_pii for c in ctx.criticality.values()):
        score = min(score + cfg.overrides.pii_boost, 1.0)
        reasons.append(f"pii_boost +{cfg.overrides.pii_boost:g}: an alert table is_pii=true")
    tier = _tier_for(score, cfg)

    present_share = denom / total_w  # effective evidence share (after weight_factor), not nominal
    if present_share < cfg.min_axes_weight:
        confidence = "low"
        reasons.append(f"axes with a value carry {present_share:.2f} of weight (< min_axes_weight {cfg.min_axes_weight:g})")
        if _TIERS.index(tier) > _TIERS.index("MEDIUM"):
            tier = "MEDIUM"
            reasons.append("tier capped at MEDIUM: too little evidence for HIGH")

    if ctx.monitor is not None and ctx.monitor.status == "paused":
        confidence = _cap_level(confidence, cfg.overrides.paused_monitor.confidence_penalty)
        if cfg.overrides.paused_monitor.action == "downgrade_one_tier" and tier != "LOW":
            tier = _TIERS[_TIERS.index(tier) - 1]
            reasons.append(f"monitor.status=paused: tier lowered one step to {tier}")
        else:
            reasons.append(f"monitor.status=paused: confidence capped at {confidence}")

    if ctx.criticality and all(c is None or c.criticality is None for c in ctx.criticality.values()):
        confidence = _cap_level(confidence, cfg.overrides.missing_criticality.confidence_penalty)
        reasons.append("no alert table has a criticality level: confidence capped")

    return Combined(score=score, tier=tier, confidence=confidence, reasons=reasons, cannot_tell=_cannot_tell(tier, supported, cfg))
