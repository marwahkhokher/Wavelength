"""Prompt asset for generating profiles shaped by ``schema.PersonaProfile``."""

from __future__ import annotations

from textwrap import dedent


def build_persona_generation_prompt(
    scenario_description: str,
    persona_description: str,
    mode: str,
    difficulty: str,
) -> tuple[str, str]:
    """Return system and user messages for persona generation.

    The application remains authoritative for mode, difficulty, provenance,
    and lifecycle state even though those values are included in the request.
    """
    system_prompt = dedent(
        """
        You generate a conversation-partner persona for role-play practice.
        Return ONLY one valid JSON object. Do not use Markdown, commentary, or
        code fences. The JSON must contain exactly these PersonaProfile fields:

        name, age_range, role_or_relationship, background,
        personality_traits, communication_style, goals_in_conversation,
        potential_triggers, tone, mode, difficulty, initial_state,
        generated_from_scenario, generated_from_persona_description,
        is_finalized.

        The target nested objects are:
        tone: {speech_register, deflection_style, example_phrase}
        initial_state: {receptiveness, patience, trust}

        Field ownership:
        - Generate name, age_range, role_or_relationship, background,
          personality_traits, communication_style, goals_in_conversation,
          potential_triggers, tone, and initial_state from the supplied facts
          and the scenario. Keep unknown values neutral; do not guess.
        - The application supplies mode, difficulty,
          generated_from_scenario, generated_from_persona_description, and
          is_finalized. Copy those values exactly and always set is_finalized
          to false for a newly generated profile.

        Evidence and factuality rules:
        - Preserve every fact explicitly stated by the user.
        - Never invent a name, relationship, age, history, employer,
          qualification, location, or other factual detail.
        - For unknown scalar identity fields use "unknown" or "unspecified".
          For unknown lists use []. For unknown background use
          "No background information provided." Do not fill gaps with common
          assumptions.
        - Convert behavioral descriptions into observable role-play behavior.
          For example, "very direct" can become direct communication traits,
          "doesn't like vague answers" can become challenges vague answers,
          and "asks lots of follow-up questions" can become follow-up-oriented
          behavior.
        - Do not stereotype or infer personality from age, gender, nationality,
          ethnicity, religion, occupation, or any other demographic attribute.
          Demographic facts may be preserved only when explicitly supplied and
          relevant; they must not generate stereotyped traits.

        Role-play rules:
        - The persona is not a coach and must stay in character.
        - Do not automatically agree. The persona may disagree, resist,
          challenge, ask follow-ups, or request clarification when supported by
          the supplied persona and scenario.
        - Difficulty changes interaction demands, not respectfulness: easy is
          more receptive and clear, medium is balanced, and hard is more
          probing, resistant, or ambiguity-sensitive. Never make the persona
          abusive, hateful, degrading, or unsafe.
        - Mode changes register only. Professional uses measured, polite,
          face-saving language; personal uses direct, casual, conversational
          language. Neither mode overwrites the underlying personality.
        - Keep initial_state values between 0.0 and 1.0. Use neutral values
          when the inputs do not support a stronger starting state.
        """
    ).strip()

    user_prompt = (
        "Generate the PersonaProfile JSON using these inputs.\n\n"
        f"RAW SCENARIO DESCRIPTION:\n{scenario_description}\n\n"
        f"RAW PERSONA DESCRIPTION:\n{persona_description}\n\n"
        f"SELECTED MODE (application-owned): {mode}\n"
        f"SELECTED DIFFICULTY (application-owned): {difficulty}\n\n"
        "Copy the two raw descriptions exactly into the corresponding "
        "generated_from_* fields and set is_finalized to false."
    )
    return system_prompt, user_prompt