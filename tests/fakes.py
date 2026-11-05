import re

from triage.config import LlmConfig, ScoringConfig
from triage.guardrails import AlertFeatures
from triage.summary import Summary


class FakeChain:
    """Stands in for the LangChain chain: records the inputs, returns `fn(inputs)` (or raises what fn raises)."""

    def __init__(self, fn=None):
        self.calls = 0
        self.inputs: list[dict[str, str]] = []
        self._fn = fn or (lambda inputs: AlertFeatures())

    async def ainvoke(self, inputs):
        self.calls += 1
        self.inputs.append(inputs)
        return self._fn(inputs)


def fast(cfg: ScoringConfig) -> LlmConfig:
    """LLM config without backoff sleeps so retry tests stay instant."""
    return cfg.llm.model_copy(update={"backoff_initial_seconds": 0.0, "backoff_max_seconds": 0.0, "backoff_jitter_seconds": 0.0})


def good_summary(inputs: dict[str, str]) -> Summary:
    """A summary that follows the rules, built only from the facts it was given (stands in for the summary model)."""
    tier = re.search(r"combined\.tier = (\w+)", inputs["facts"])
    assert tier is not None
    gap = " I can't tell from this data: see the listed data gaps." if "combined.cannot_tell = True" in inputs["facts"] else ""
    return Summary(sentences=["The score comes from the listed axis values.", "Every value is taken from the facts.", f"Nothing else is used.{gap}"])
