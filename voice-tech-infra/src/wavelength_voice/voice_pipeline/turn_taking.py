"""Turn-taking state machine, including barge-in detection.

Deliberately has no knowledge of websockets, Deepgram, or ElevenLabs - it
only tracks *whose turn it is* and raises on illegal sequencing, which is
what makes it cheap to unit test exhaustively. The orchestrator wires VAD/STT
events and AI-service/TTS lifecycle events into this controller and reacts to
its barge-in callback (e.g. by cancelling an in-flight TTS stream).

State machine::

    WAITING_FOR_USER --user_speech_started--> USER_SPEAKING
    USER_SPEAKING --user_speech_ended--------> PROCESSING
    PROCESSING --agent_response_ready--------> AGENT_SPEAKING
    AGENT_SPEAKING --agent_speech_ended------> WAITING_FOR_USER

    # Barge-in: the user starts talking while the agent has (or is about to
    # have) the floor. Valid from either PROCESSING or AGENT_SPEAKING.
    PROCESSING --user_speech_started (barge-in)-----> USER_SPEAKING
    AGENT_SPEAKING --user_speech_started (barge-in)-> USER_SPEAKING
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum

from wavelength_voice.session_state.models import utcnow


class TurnState(str, Enum):
    WAITING_FOR_USER = "waiting_for_user"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    AGENT_SPEAKING = "agent_speaking"


class InvalidTurnTransition(RuntimeError):
    """Raised when an event doesn't make sense in the controller's current state."""


#: States from which a user_speech_started event counts as a barge-in on the
#: agent's turn, rather than the normal start of the user's turn.
_BARGE_IN_STATES = frozenset({TurnState.PROCESSING, TurnState.AGENT_SPEAKING})

OnStateChange = Callable[[TurnState, TurnState], None]
OnBargeIn = Callable[[TurnState, datetime], None]


class TurnTakingController:
    """Tracks conversational turn state for a single session connection."""

    def __init__(
        self,
        on_state_change: OnStateChange | None = None,
        on_barge_in: OnBargeIn | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._state = TurnState.WAITING_FOR_USER
        self._on_state_change = on_state_change
        self._on_barge_in = on_barge_in
        self._now = clock
        self.barge_in_count = 0

    @property
    def state(self) -> TurnState:
        return self._state

    def _transition(self, new_state: TurnState) -> None:
        old_state = self._state
        self._state = new_state
        if self._on_state_change is not None:
            self._on_state_change(old_state, new_state)

    def user_speech_started(self) -> bool:
        """Signal that STT/VAD detected the user starting to talk.

        Returns True if this interrupted the agent's turn (a barge-in).
        Safe to call repeatedly while the user keeps talking - a call while
        already USER_SPEAKING is a no-op.
        """
        if self._state == TurnState.USER_SPEAKING:
            return False

        if self._state in _BARGE_IN_STATES:
            interrupted_state = self._state
            self._transition(TurnState.USER_SPEAKING)
            self.barge_in_count += 1
            if self._on_barge_in is not None:
                self._on_barge_in(interrupted_state, self._now())
            return True

        # WAITING_FOR_USER -> USER_SPEAKING: the normal start of a user turn.
        self._transition(TurnState.USER_SPEAKING)
        return False

    def user_speech_ended(self, transcript: str) -> None:
        """Signal that STT finalized the user's utterance; hand off to the AI service."""
        if self._state != TurnState.USER_SPEAKING:
            raise InvalidTurnTransition(
                f"user_speech_ended() is invalid from state {self._state}; "
                "expected USER_SPEAKING"
            )
        self._transition(TurnState.PROCESSING)

    def agent_response_ready(self) -> None:
        """Signal that the AI service replied and TTS playback is starting."""
        if self._state != TurnState.PROCESSING:
            raise InvalidTurnTransition(
                f"agent_response_ready() is invalid from state {self._state}; "
                "expected PROCESSING"
            )
        self._transition(TurnState.AGENT_SPEAKING)

    def agent_speech_ended(self) -> None:
        """Signal that TTS playback finished (naturally, not via barge-in)."""
        if self._state != TurnState.AGENT_SPEAKING:
            raise InvalidTurnTransition(
                f"agent_speech_ended() is invalid from state {self._state}; "
                "expected AGENT_SPEAKING"
            )
        self._transition(TurnState.WAITING_FOR_USER)

    def reset(self) -> None:
        """Force back to the initial state, e.g. after a reconnect mid-turn."""
        self._transition(TurnState.WAITING_FOR_USER)
