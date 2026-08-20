"""
FR-TONE-2: tone rules maintained as a defined, reviewable asset.

This is intentionally a plain, editable data structure (not hardcoded into
a prompt string somewhere) so ai-prompt-engineering can iterate on tone
without touching backend code. Import TONE_RULES wherever a mode-specific
system prompt is built (persona generation, live session, evaluation).
"""

from app.models.enums import Mode

TONE_RULES: dict[Mode, dict] = {
    Mode.PROFESSIONAL: {
        "register": "measured, polite, face-saving",
        "deflection_style": "diplomatic - e.g. 'Great question, I'll get back to you.'",
        "example_lines": [
            "Great question, I'll get back to you.",
            "Let me loop in the right person on that.",
        ],
        "avoid": ["slang", "blunt admissions of ignorance", "casual profanity"],
    },
    Mode.PERSONAL: {
        "register": "direct, casual - the way people talk to friends and family",
        "deflection_style": "blunt and unpolished - e.g. 'Yaar, I don't know.'",
        "example_lines": [
            "Yaar, I don't know.",
            "Honestly no idea, don't ask me that.",
        ],
        "avoid": ["corporate phrasing", "excessive formality"],
    },
}


def build_tone_system_prompt(mode: Mode) -> str:
    rules = TONE_RULES[mode]
    return (
        f"Speak in a {rules['register']} register. "
        f"When you don't know something, respond in this style: {rules['deflection_style']}. "
        f"Avoid: {', '.join(rules['avoid'])}."
    )
