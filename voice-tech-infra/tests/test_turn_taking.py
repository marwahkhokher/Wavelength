"""Tests for the turn-taking state machine, including barge-in."""

from __future__ import annotations

import pytest

from wavelength_voice.voice_pipeline.turn_taking import (
    InvalidTurnTransition,
    TurnState,
    TurnTakingController,
)


def test_starts_waiting_for_user() -> None:
    controller = TurnTakingController()
    assert controller.state == TurnState.WAITING_FOR_USER


def test_full_turn_without_interruption_cycles_back_to_waiting() -> None:
    controller = TurnTakingController()

    barged_in = controller.user_speech_started()
    assert barged_in is False
    assert controller.state == TurnState.USER_SPEAKING

    controller.user_speech_ended("what's the plan for launch")
    assert controller.state == TurnState.PROCESSING

    controller.agent_response_ready()
    assert controller.state == TurnState.AGENT_SPEAKING

    controller.agent_speech_ended()
    assert controller.state == TurnState.WAITING_FOR_USER
    assert controller.barge_in_count == 0


def test_repeated_speech_started_while_already_speaking_is_a_noop() -> None:
    controller = TurnTakingController()
    controller.user_speech_started()

    barged_in = controller.user_speech_started()

    assert barged_in is False
    assert controller.state == TurnState.USER_SPEAKING


@pytest.mark.parametrize(
    "method,args",
    [
        ("user_speech_ended", ("hi",)),
        ("agent_response_ready", ()),
        ("agent_speech_ended", ()),
    ],
)
def test_events_invalid_from_waiting_for_user_raise(method: str, args: tuple) -> None:
    controller = TurnTakingController()

    with pytest.raises(InvalidTurnTransition):
        getattr(controller, method)(*args)


def test_barge_in_during_agent_speaking_interrupts_and_notifies() -> None:
    events: list[tuple[TurnState, TurnState]] = []
    barge_ins: list[TurnState] = []
    controller = TurnTakingController(
        on_state_change=lambda old, new: events.append((old, new)),
        on_barge_in=lambda interrupted, at: barge_ins.append(interrupted),
    )
    controller.user_speech_started()
    controller.user_speech_ended("tell me about the roadmap")
    controller.agent_response_ready()
    assert controller.state == TurnState.AGENT_SPEAKING

    barged_in = controller.user_speech_started()

    assert barged_in is True
    assert controller.state == TurnState.USER_SPEAKING
    assert controller.barge_in_count == 1
    assert barge_ins == [TurnState.AGENT_SPEAKING]
    assert events[-1] == (TurnState.AGENT_SPEAKING, TurnState.USER_SPEAKING)


def test_barge_in_while_agent_is_still_thinking() -> None:
    """The user can interrupt before TTS audio even starts (during PROCESSING)."""
    barge_ins: list[TurnState] = []
    controller = TurnTakingController(on_barge_in=lambda interrupted, at: barge_ins.append(interrupted))
    controller.user_speech_started()
    controller.user_speech_ended("actually, wait")
    assert controller.state == TurnState.PROCESSING

    barged_in = controller.user_speech_started()

    assert barged_in is True
    assert controller.state == TurnState.USER_SPEAKING
    assert barge_ins == [TurnState.PROCESSING]


def test_after_barge_in_the_turn_can_proceed_normally() -> None:
    controller = TurnTakingController()
    controller.user_speech_started()
    controller.user_speech_ended("first thing")
    controller.agent_response_ready()
    controller.user_speech_started()  # barge-in

    controller.user_speech_ended("actually here's what I meant")
    controller.agent_response_ready()
    controller.agent_speech_ended()

    assert controller.state == TurnState.WAITING_FOR_USER
    assert controller.barge_in_count == 1


def test_user_speech_ended_without_active_speech_raises() -> None:
    controller = TurnTakingController()
    controller.user_speech_started()
    controller.user_speech_ended("hello")

    with pytest.raises(InvalidTurnTransition):
        controller.user_speech_ended("hello again")


def test_agent_response_ready_out_of_order_raises() -> None:
    controller = TurnTakingController()

    with pytest.raises(InvalidTurnTransition):
        controller.agent_response_ready()


def test_reset_forces_back_to_waiting_for_user() -> None:
    controller = TurnTakingController()
    controller.user_speech_started()
    controller.user_speech_ended("hi")

    controller.reset()

    assert controller.state == TurnState.WAITING_FOR_USER
