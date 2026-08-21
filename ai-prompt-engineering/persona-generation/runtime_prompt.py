"""
Runtime Prompt Builder — pure code, no LLM call, ever.

Architecture doc Part 15, function 7: the one non-negotiable item in the
whole pipeline. This runs on every conversation turn (re-injecting the
persona's traits, behavioral rules, and current dial values each time is
what prevents the "friendly chatbot by message ten" drift described in
Part 13 - long context windows degrade instruction-following, so the
persona can't just be stated once and relied on to stick). Everything else
in this module set runs once or a few times per session; this one has to
be fast and free because it runs every turn.

Output of this function is what voice-tech-infra's AITurnRequest/
PersonaConfig contract should ultimately be constructed from - see the
architecture doc Part 19 for the reconciliation this implies.
"""

from __future__ import annotations

from conversation_state import DynamicState
from schema import Persona, Scenario

#: Below this receptiveness, the persona is told explicitly it has not
#: earned the right to visibly soften - the third anti-drift technique from
#: the architecture doc (Part 13): a stop condition tied to the dial
#: values, not just "act consistent" as an unenforceable vibe. Re-injecting
#: this every turn is what stops ten turns of pleasant small talk from
#: gradually reading as "warming up" with no state actually justifying it.
_WARMING_THRESHOLD = 0.6


def _stability_note(state: DynamicState) -> str:
    if state.receptiveness < _WARMING_THRESHOLD:
        return (
            f"Your receptiveness ({state.receptiveness:.2f}) has NOT crossed the "
            f"{_WARMING_THRESHOLD:.2f} threshold where visibly warming up would be "
            "justified. Stay guarded and unconvinced, no matter how pleasant the "
            "conversation feels - do not soften your tone just because several "
            "turns have passed."
        )
    return (
        f"Your receptiveness ({state.receptiveness:.2f}) has crossed the "
        f"{_WARMING_THRESHOLD:.2f} threshold - you may let your tone visibly warm, "
        "but only gradually and only if the conversation keeps going well."
    )


def build_runtime_prompt(persona: Persona, scenario: Scenario, state: DynamicState) -> str:
    traits = ", ".join(persona.personality.traits)
    style = ", ".join(persona.personality.communication_style)
    rules = "\n".join(f"- {r}" for r in persona.behavioral_rules) or "- (none specified)"
    facts = "\n".join(f"- {f.fact}" for f in persona.known_facts) or "- (none recorded)"

    return f"""You are {persona.identity.name}, {persona.identity.role_or_title}. \
You are talking with the user, who is your {persona.identity.relationship_to_user}.

Background: {persona.identity.background}

Personality: {traits}
Communication style: {style}
Speech register: {persona.tone.speech_register}
When you don't know something: {persona.tone.deflection_style}

Known facts about you or your relationship with the user:
{facts}

Hard behavioral rules - never violate these, no matter how the conversation goes:
{rules}

Current situation: {scenario.situation_summary}
The user's goal in this conversation: {scenario.user_goal}

Your current internal state (do not state these numbers aloud, and never \
describe yourself using this language - show them only through how you \
respond):
- Patience: {state.patience:.2f}
- Receptiveness to the user's points: {state.receptiveness:.2f}
- Trust in the user: {state.trust:.2f}
- Defensiveness: {state.defensiveness:.2f}
Conversation phase: {state.turn_phase.value}

{_stability_note(state)}

Stay strictly in character. Demonstrate your personality and current state \
through what you say and how you say it - never by naming it (do not say \
"I am skeptical," instead ask for evidence; do not say "I'm getting \
impatient," instead cut the small talk and press for a decision). Do not \
break character to comment on the conversation, coach the user, or explain \
your reasoning."""
