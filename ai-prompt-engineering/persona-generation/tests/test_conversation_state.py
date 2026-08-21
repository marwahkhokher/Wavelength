import pytest

from conversation_state import (
    DynamicState,
    TurnClassification,
    TurnPhase,
    apply_turn_classification,
    build_classification_messages,
    parse_classification,
    seed_dynamic_state,
)
from schema import BaselineDynamics, DifficultyLevel


def test_seed_easy_difficulty_applies_no_penalty():
    baseline = BaselineDynamics(patience=0.4, receptiveness=0.6, trust=0.5)
    state = seed_dynamic_state(baseline, DifficultyLevel.EASY)
    assert state.patience == 0.4
    assert state.receptiveness == 0.6
    assert state.trust == 0.5
    assert state.defensiveness == 0.0
    assert state.turn_phase == TurnPhase.OPENING


def test_seed_hard_difficulty_lowers_starting_dials():
    baseline = BaselineDynamics(patience=0.4, receptiveness=0.6, trust=0.5)
    state = seed_dynamic_state(baseline, DifficultyLevel.HARD)
    assert state.patience == pytest.approx(0.2)
    assert state.receptiveness == pytest.approx(0.4)
    assert state.trust == pytest.approx(0.3)


def test_seed_clamps_at_zero_for_already_low_baseline():
    baseline = BaselineDynamics(patience=0.1, receptiveness=0.5, trust=0.5)
    state = seed_dynamic_state(baseline, DifficultyLevel.HARD)  # penalty 0.2
    assert state.patience == 0.0  # would be -0.1 unclamped


def test_strong_argument_raises_receptiveness_and_patience():
    state = DynamicState(receptiveness=0.4, patience=0.4)
    updated, log = apply_turn_classification(state, TurnClassification.STRONG_ARGUMENT, turn=1)
    assert updated.receptiveness == pytest.approx(0.5)
    assert updated.patience == pytest.approx(0.45)
    assert log.changes["receptiveness"] == (0.4, pytest.approx(0.5))
    assert log.turn == 1
    assert log.classification == TurnClassification.STRONG_ARGUMENT


def test_disrespectful_raises_defensiveness_and_lowers_trust_and_patience():
    state = DynamicState(trust=0.5, patience=0.5, defensiveness=0.0)
    updated, log = apply_turn_classification(state, TurnClassification.DISRESPECTFUL, turn=3)
    assert updated.defensiveness == pytest.approx(0.20)
    assert updated.trust == pytest.approx(0.35)
    assert updated.patience == pytest.approx(0.40)


def test_neutral_classification_changes_nothing():
    state = DynamicState(receptiveness=0.4, patience=0.4, trust=0.4, defensiveness=0.1)
    updated, log = apply_turn_classification(state, TurnClassification.NEUTRAL, turn=2)
    assert updated == state
    assert log.changes == {}


def test_dial_deltas_clamp_at_upper_bound():
    state = DynamicState(receptiveness=0.95, patience=0.98)
    updated, _ = apply_turn_classification(state, TurnClassification.STRONG_ARGUMENT, turn=1)
    assert updated.receptiveness == 1.0
    assert updated.patience == 1.0


def test_apply_turn_classification_does_not_mutate_input_state():
    state = DynamicState(receptiveness=0.4)
    apply_turn_classification(state, TurnClassification.STRONG_ARGUMENT, turn=1)
    assert state.receptiveness == 0.4  # original untouched


def test_classification_messages_include_conversation_and_latest_message():
    messages = build_classification_messages(
        conversation_so_far="CEO: What's this about?", latest_user_message="I deserve a raise."
    )
    assert "What's this about?" in messages[1]["content"]
    assert "I deserve a raise." in messages[1]["content"]


def test_parse_classification_is_case_and_whitespace_insensitive():
    assert parse_classification("  Strong_Argument \n") == TurnClassification.STRONG_ARGUMENT
