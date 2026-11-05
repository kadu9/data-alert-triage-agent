"""Versioned Jinja2 prompt templates: `version/<PROMPT_VERSION>/<name>_system.j2` and `<name>_human.j2`, never inside Python code.

To change a prompt, edit the files in the current version folder in place (git history keeps the old text).
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPT_VERSION = "v0.0.1"
_DIR = Path(__file__).parent / "version"


def chat_prompt(name: str, version: str = PROMPT_VERSION) -> ChatPromptTemplate:
    """Build the system+human chat prompt from its two .j2 files; alert text only ever arrives as a template variable."""
    system, human = ((_DIR / version / f"{name}_{role}.j2").read_text(encoding="utf-8").rstrip("\n") for role in ("system", "human"))
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)], template_format="jinja2")
