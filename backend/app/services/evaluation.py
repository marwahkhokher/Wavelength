import json
import re

from app.models.enums import CONFIRMED_METRIC_KEYS

FILLER_WORDS = ["um", "uh", "like", "you know", "sort of", "kind of", "basically"]


def count_filler_words(transcript: list[dict]) -> int:
    """
    Deterministic count rather than LLM-estimated, since FR-SESS-8 specifically
    calls out filler words as something that can only be measured from captured
    speech - worth keeping exact.
    """
    user_text = " ".join(
        turn["text"].lower() for turn in transcript if turn.get("speaker") == "user"
    )
    count = 0
    for filler in FILLER_WORDS:
        count += len(re.findall(rf"\b{re.escape(filler)}\b", user_text))
    return count


def score_transcript(transcript: list[dict]) -> dict:
    """
    FR-EVAL-1/2: score the session against the standard metric set.
    clarity/empathy are LLM-judged (they require semantic understanding);
    filler_words is computed deterministically and merged in.
    Remaining metrics (Section 9 item #1) plug in here once finalized.
    """
    from app.services.llm import get_llm  # local import keeps module import-light for tests

    llm = get_llm(temperature=0.2)
    llm_metric_keys = [k for k in CONFIRMED_METRIC_KEYS if k != "filler_words"]

    transcript_text = "\n".join(
        f"{turn.get('speaker')}: {turn.get('text')}" for turn in transcript
    )
    system_prompt = (
        "You are scoring a user's side of a practice conversation on a 0-100 "
        f"scale for each of these metrics: {llm_metric_keys}. "
        "Return ONLY a JSON object mapping metric name to a number 0-100."
    )
    response = llm.invoke([("system", system_prompt), ("user", transcript_text)])

    try:
        scores = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        scores = {k: 0 for k in llm_metric_keys}

    scores["filler_words"] = count_filler_words(transcript)
    return scores


def generate_coach_recommendations(metrics: dict) -> list[dict]:
    """FR-COACH-2/3: recommendations based on this session's results."""
    from app.services.llm import get_llm

    llm = get_llm(temperature=0.5)
    system_prompt = (
        "Given a user's communication metric scores from a practice session, "
        "return ONLY a JSON list of objects, one per weak metric (below 70, or "
        "the single weakest if all are strong), each with keys: "
        "'metric', 'observation', 'recommendation'. Keep each field to 1-2 sentences."
    )
    response = llm.invoke([("system", system_prompt), ("user", json.dumps(metrics))])

    try:
        return json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return []
