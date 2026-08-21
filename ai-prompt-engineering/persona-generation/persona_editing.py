"""
Persona Editing — architecture doc Part 14.

Routes a natural-language edit request to either a PERMANENT persona
mutation ("my CEO is actually very friendly") or a SESSION-LEVEL change
("make today's conversation harder"), and applies it. An ambiguous request
defaults to session-level - the lower-blast-radius, reversible
interpretation, per the architecture doc's own reasoning.

No persistence here - apply_permanent_edit() returns the new Persona plus
a PersonaVersionRecord ready to hand to a storage layer; nothing in this
module writes anywhere, and the old Persona object is never mutated in
place. Any conversation that already pinned the previous
persona_version_id (see the DB design in the architecture doc, Part 9) is
therefore unaffected by a later edit - it keeps reading the persona as it
was when that conversation happened.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field

from schema import (
    DifficultyLevel,
    Persona,
    PersonaIdentity,
    PersonaPersonality,
    PersonaVersionRecord,
    utcnow,
)
from validation import ValidationIssue, ValidationResult, check_contradictions


class EditScope(str, Enum):
    PERMANENT = "permanent"
    SESSION = "session"


class DifficultyShift(str, Enum):
    HARDER = "harder"
    EASIER = "easier"
    UNCHANGED = "unchanged"


# ---------------------------------------------------------------------------
# Scope classification - "make him friendlier" vs "make today harder"
# ---------------------------------------------------------------------------

_SCOPE_SYSTEM_TEMPLATE = """Classify the user's edit request as EXACTLY ONE of:

- permanent: changes the OTHER PERSON's stable characteristics - who they \
are, still true next session. Examples: "make him friendlier", "she should \
be more direct", "he's actually pretty casual, not formal".
- session: changes only THIS conversation's parameters - difficulty, \
length, how hard the going gets today. Examples: "make this harder", \
"go easier on me today", "make this conversation shorter", "be tougher \
on me right now".

If genuinely ambiguous, prefer "session" - it's the reversible, \
lower-impact interpretation.

Output ONLY the single label, nothing else.
"""


def build_scope_classification_messages(edit_text: str) -> list[dict[str, str]]:
    if not edit_text.strip():
        raise ValueError("edit_text must not be empty")
    return [
        {"role": "system", "content": _SCOPE_SYSTEM_TEMPLATE},
        {"role": "user", "content": edit_text.strip()},
    ]


def parse_edit_scope(raw_response: str) -> EditScope:
    label = raw_response.strip().lower()
    if label not in (EditScope.PERMANENT.value, EditScope.SESSION.value):
        # Unrecognized output defaults to the reversible interpretation too.
        return EditScope.SESSION
    return EditScope(label)


# ---------------------------------------------------------------------------
# Permanent edits - regenerate the editable subset of the persona
# ---------------------------------------------------------------------------


class EditedPersonaFields(BaseModel):
    """The editable subset of Persona (matches Persona.EDITABLE_FIELDS,
    minus display_name - that's a user-chosen label, not AI-revised
    content, and is left untouched by permanent edits)."""

    identity: PersonaIdentity
    personality: PersonaPersonality
    behavioral_rules: list[str] = Field(default_factory=list, max_length=6)


_EDIT_SYSTEM_TEMPLATE = """You are revising an existing persona for a communication-practice app, \
based on the user's edit request. Keep everything about the person the \
same EXCEPT what the edit request asks you to change - this is a targeted \
revision, not a regeneration from scratch.

Current persona:
{current_json}

Edit request: {edit_text}

## Output

Return ONLY a single JSON object matching this shape (the full revised \
identity/personality/behavioral_rules, not a diff), no prose before or \
after it:

{output_schema}
"""


def build_edit_messages(persona: Persona, edit_text: str) -> list[dict[str, str]]:
    if not edit_text.strip():
        raise ValueError("edit_text must not be empty")
    current = EditedPersonaFields(
        identity=persona.identity,
        personality=persona.personality,
        behavioral_rules=persona.behavioral_rules,
    )
    system = _EDIT_SYSTEM_TEMPLATE.format(
        current_json=current.model_dump_json(indent=2),
        edit_text=edit_text.strip(),
        output_schema=json.dumps(EditedPersonaFields.model_json_schema(), indent=2),
    )
    return [{"role": "system", "content": system}]


def parse_edited_fields(raw_response: str) -> EditedPersonaFields:
    return EditedPersonaFields.model_validate_json(raw_response)


def validate_edit(edited: EditedPersonaFields) -> ValidationResult:
    issues = check_contradictions(edited.personality)
    if not edited.personality.traits:
        issues.append(
            ValidationIssue(
                layer="deterministic",
                severity="error",
                message="No personality traits after edit",
            )
        )
    if not edited.identity.background.strip():
        issues.append(
            ValidationIssue(
                layer="deterministic", severity="error", message="Empty background after edit"
            )
        )
    passed = not any(i.severity == "error" for i in issues)
    return ValidationResult(passed=passed, issues=issues)


class PersonaEditFailed(RuntimeError):
    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        messages = "; ".join(i.message for i in result.issues if i.severity == "error")
        super().__init__(f"Persona edit failed validation: {messages}")


def apply_permanent_edit(
    persona: Persona, edited: EditedPersonaFields, edit_text: str
) -> tuple[Persona, PersonaVersionRecord]:
    """Validates, then bumps the persona's version. Does not mutate
    `persona` in place - callers must swap in the returned object."""
    check = validate_edit(edited)
    if not check.passed:
        raise PersonaEditFailed(check)

    new_persona = persona.model_copy(
        update={
            "identity": edited.identity,
            "personality": edited.personality,
            "behavioral_rules": edited.behavioral_rules,
            "version": persona.version + 1,
            "updated_at": utcnow(),
        }
    )
    version_record = PersonaVersionRecord(
        persona_id=persona.persona_id,
        version=new_persona.version,
        snapshot=new_persona,
        changed_by="user_edit",
        change_summary=edit_text.strip(),
    )
    return new_persona, version_record


# ---------------------------------------------------------------------------
# Session-level edits - never touch the persona at all
# ---------------------------------------------------------------------------

_DIFFICULTY_ORDER = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

_SHIFT_SYSTEM_TEMPLATE = """Classify the user's request as EXACTLY ONE of: harder, easier, unchanged. \
This is about THIS conversation's difficulty only. Output ONLY the single word.
"""


def build_shift_classification_messages(edit_text: str) -> list[dict[str, str]]:
    if not edit_text.strip():
        raise ValueError("edit_text must not be empty")
    return [
        {"role": "system", "content": _SHIFT_SYSTEM_TEMPLATE},
        {"role": "user", "content": edit_text.strip()},
    ]


def parse_difficulty_shift(raw_response: str) -> DifficultyShift:
    label = raw_response.strip().lower()
    if label not in (s.value for s in DifficultyShift):
        return DifficultyShift.UNCHANGED
    return DifficultyShift(label)


def apply_difficulty_shift(current: DifficultyLevel, shift: DifficultyShift) -> DifficultyLevel:
    """Session-level only - never touches Persona.baseline_dynamics. Clamps
    at the easy/hard boundaries rather than erroring on 'harder' from HARD."""
    index = _DIFFICULTY_ORDER.index(current)
    if shift == DifficultyShift.HARDER:
        index = min(index + 1, len(_DIFFICULTY_ORDER) - 1)
    elif shift == DifficultyShift.EASIER:
        index = max(index - 1, 0)
    return _DIFFICULTY_ORDER[index]


# ---------------------------------------------------------------------------
# End-to-end orchestration for one edit request
# ---------------------------------------------------------------------------


class EditOutcome(BaseModel):
    scope: EditScope
    persona: Persona | None = None
    version_record: PersonaVersionRecord | None = None
    new_difficulty: DifficultyLevel | None = None


async def route_and_apply_edit(
    llm,  # LLMClient - not type-hinted directly to avoid a circular import with llm_client
    *,
    persona: Persona,
    current_difficulty: DifficultyLevel,
    edit_text: str,
) -> EditOutcome:
    """Classifies one edit request and applies it end to end - the single
    entry point a caller (pipeline.py, or eventually a backend endpoint)
    needs for Part 14's "user edits the persona" flow."""
    scope_raw = await llm.complete(build_scope_classification_messages(edit_text))
    scope = parse_edit_scope(scope_raw)

    if scope is EditScope.PERMANENT:
        edited_raw = await llm.complete(build_edit_messages(persona, edit_text))
        edited = parse_edited_fields(edited_raw)
        new_persona, version_record = apply_permanent_edit(persona, edited, edit_text)
        return EditOutcome(scope=scope, persona=new_persona, version_record=version_record)

    shift_raw = await llm.complete(build_shift_classification_messages(edit_text))
    shift = parse_difficulty_shift(shift_raw)
    new_difficulty = apply_difficulty_shift(current_difficulty, shift)
    return EditOutcome(scope=scope, new_difficulty=new_difficulty)
