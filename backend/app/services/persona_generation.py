import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from app.models.enums import Mode
from app.services.llm import get_llm

_AI_PROMPT_ENGINEERING_ROOT = Path(__file__).resolve().parents[3] / "ai-prompt-engineering"


def _load_prompt_module() -> ModuleType:
    prompt_path = _AI_PROMPT_ENGINEERING_ROOT / "persona-generation" / "prompt.py"
    spec = importlib.util.spec_from_file_location("wavelength_persona_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load persona prompt from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_schema_module() -> ModuleType:
    schema_path = _AI_PROMPT_ENGINEERING_ROOT / "persona-generation" / "schema.py"
    spec = importlib.util.spec_from_file_location("wavelength_persona_schema", schema_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load persona schema from {schema_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_persona_profile(
    scenario_text: str,
    persona_description: str,
    mode: Mode,
    difficulty: str = "medium",
) -> dict:
    """FR-PERS-4: turns the user's free-text description into a full profile."""
    llm = get_llm(temperature=0.6)
    prompt_module = _load_prompt_module()
    schema_module = _load_schema_module()
    system_prompt, user_prompt = prompt_module.build_persona_generation_prompt(
        scenario_description=scenario_text,
        persona_description=persona_description,
        mode=mode.value,
        difficulty=difficulty,
    )

    response = llm.invoke([("system", system_prompt), ("user", user_prompt)])

    try:
        profile_data = json.loads(response.content)
        if not isinstance(profile_data, dict):
            raise TypeError("Persona response must be a JSON object")
        profile_data.update(
            mode=mode.value,
            difficulty=difficulty,
            generated_from_scenario=scenario_text,
            generated_from_persona_description=persona_description,
            is_finalized=False,
        )
        profile = schema_module.PersonaProfile.model_validate(profile_data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Persona generation returned an invalid PersonaProfile") from exc

    return profile.model_dump(exclude={"EDITABLE_FIELDS"})
