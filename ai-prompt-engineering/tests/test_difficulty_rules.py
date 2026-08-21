from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_context.difficulty_rules import get_difficulty_prompt
from session_context.tone_rules import get_mode_tone_prompt


def test_easy_rules_are_receptive_and_low_pressure() -> None:
    prompt = get_difficulty_prompt("easy")

    assert "straightforward questions" in prompt
    assert "limited challenging follow-ups" in prompt
    assert "high receptiveness and low resistance" in prompt
    assert "reasonable space" in prompt
    assert "aggressively probing" in prompt


def test_medium_rules_add_realistic_challenge() -> None:
    prompt = get_difficulty_prompt("medium")

    assert "realistic, normal resistance" in prompt
    assert "do not automatically agree" in prompt
    assert "challenge vague or incomplete answers" in prompt
    assert "examples or evidence" in prompt


def test_hard_rules_increase_scrutiny_without_abuse() -> None:
    prompt = get_difficulty_prompt("hard")

    assert "high scrutiny" in prompt
    assert "strong, relevant follow-up questioning" in prompt
    assert "low tolerance for vague or unsupported answers" in prompt
    assert "Probe inconsistencies" in prompt
    assert "Never become insulting, abusive, hostile, discriminatory" in prompt


def test_invalid_difficulty_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported difficulty"):
        get_difficulty_prompt("extreme")


def test_professional_hard_keeps_professional_register() -> None:
    prompt = f"{get_mode_tone_prompt('professional')}\n{get_difficulty_prompt('hard')}"

    assert "measured, polite, and face-saving" in prompt
    assert "high scrutiny" in prompt
    assert "Professional vs Personal mode controls register and tone only" in prompt
    assert "abusive" in prompt


def test_personal_hard_keeps_casual_register_without_abuse() -> None:
    prompt = f"{get_mode_tone_prompt('personal')}\n{get_difficulty_prompt('hard')}"

    assert "direct, casual, and conversational" in prompt
    assert "high scrutiny" in prompt
    assert "Professional vs Personal mode controls register and tone only" in prompt
    assert "Never become insulting, abusive, hostile, discriminatory" in prompt