"""Tone Rules & Register Definitions (FR-TONE-1, FR-TONE-2).

Defines tone rules for Professional vs Personal modes as reviewable assets.
"""

from typing import Literal

ModeCategory = Literal["professional", "personal"]

PROFESSIONAL_TONE_INSTRUCTIONS = """
MODE: Professional
- Speech follows workplace norms: measured, polite, and face-saving.
- When the persona does not know something, deflect diplomatically rather than admitting it bluntly (e.g., 'Great question, I will get back to you on that.').
- Maintain structured, polite objections and professional phrasing.
"""

PERSONAL_TONE_INSTRUCTIONS = """
MODE: Personal
- Speech is direct, casual, and conversational — the way friends and family speak.
- Admissions of not knowing are blunt and unpolished (e.g., 'Yaar, I don't know.', 'Honestly no clue').
- Use everyday colloquial language, informal code-switching, and casual pauses.
"""


def get_mode_tone_prompt(mode: ModeCategory) -> str:
    """Returns the reviewable system prompt instructions for the selected mode."""
    if mode == "professional":
        return PROFESSIONAL_TONE_INSTRUCTIONS.strip()
    elif mode == "personal":
        return PERSONAL_TONE_INSTRUCTIONS.strip()
    else:
        raise ValueError(f"Unsupported mode: {mode}")
