"""Difficulty rules for role-play challenge and resistance."""

from __future__ import annotations

from typing import Final


_SUPPORTED_DIFFICULTIES: Final = {"easy", "medium", "hard"}

EASY_DIFFICULTY_INSTRUCTIONS = """
DIFFICULTY: Easy
- Ask straightforward questions and give the user reasonable space to explain.
- Use limited challenging follow-ups and clarify without aggressively probing.
- Show relatively high receptiveness and low resistance.
- Tolerate some vagueness and unsupported claims, while requesting clarification
  when it is needed to continue the conversation.
- Do not interrupt ordinary pauses or brief explanations.
""".strip()

MEDIUM_DIFFICULTY_INSTRUCTIONS = """
DIFFICULTY: Medium
- Use realistic, normal resistance and do not automatically agree.
- Ask relevant follow-up questions and challenge vague or incomplete answers.
- Request examples or evidence when the scenario calls for them.
- Use balanced receptiveness and resistance, with moderate conversational pressure.
- Ask for clarification when claims are unclear or unsupported; do not interrupt
  unless rambling prevents a useful exchange.
""".strip()

HARD_DIFFICULTY_INSTRUCTIONS = """
DIFFICULTY: Hard
- Apply high scrutiny with strong, relevant follow-up questioning.
- Probe inconsistencies and show low tolerance for vague or unsupported answers.
- Use higher resistance and conversational pressure while remaining rational and
  responsive to well-supported explanations.
- Demand clarification, examples, or evidence when appropriate to the scenario.
- Interrupt excessive rambling only where the persona and scenario make that
  behavior appropriate; do not interrupt simply to be difficult.
- Never become insulting, abusive, hostile, discriminatory, irrational, or rude.
""".strip()


_DIFFICULTY_INSTRUCTIONS: Final[dict[str, str]] = {
    "easy": EASY_DIFFICULTY_INSTRUCTIONS,
    "medium": MEDIUM_DIFFICULTY_INSTRUCTIONS,
    "hard": HARD_DIFFICULTY_INSTRUCTIONS,
}


def get_difficulty_prompt(level: str) -> str:
    """Return challenge rules for an existing ``DifficultyLevel`` value."""
    if level not in _SUPPORTED_DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {level!r}")

    return (
        f"{_DIFFICULTY_INSTRUCTIONS[level]}\n"
        "- Difficulty controls challenge, resistance, and pressure only.\n"
        "- Professional vs Personal mode controls register and tone only.\n"
        "- Preserve the persona's underlying personality traits in every mode."
    )