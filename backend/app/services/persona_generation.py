import json

from app.models.enums import Mode
from app.services.llm import get_llm
from app.services.tone_rules import build_tone_system_prompt

# Default persona profile shape. Section 9 open item #3 asks exactly which
# fields the profile should have - this is a reasonable starting point that
# covers what the tone/live-session logic needs. Extend, don't replace, once
# design/product confirm the final field list, so existing sessions don't break.
PERSONA_PROFILE_FIELDS = [
    "name",
    "role_or_relationship",   # e.g. "hiring manager" or "older brother"
    "personality_traits",     # list[str]
    "speech_style",           # short description
    "likely_reactions",       # how they tend to respond under pressure
    "background",             # 1-2 sentences of context
]


def generate_persona_profile(scenario_text: str, persona_description: str, mode: Mode) -> dict:
    """FR-PERS-4: turns the user's free-text description into a full profile."""
    llm = get_llm(temperature=0.6)

    system_prompt = (
        "You build realistic conversation-partner personas for a communication "
        "practice app. Given a scenario and a description of who the user needs "
        "to talk to, produce a JSON persona profile.\n"
        f"{build_tone_system_prompt(mode)}\n"
        f"Return ONLY valid JSON with these keys: {PERSONA_PROFILE_FIELDS}."
    )
    user_prompt = f"Scenario: {scenario_text}\nPerson description: {persona_description}"

    response = llm.invoke([("system", system_prompt), ("user", user_prompt)])

    try:
        return json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        # Fail safe rather than fail loud - the user can still edit/retry (FR-PERS-5)
        return {field: "" for field in PERSONA_PROFILE_FIELDS}
