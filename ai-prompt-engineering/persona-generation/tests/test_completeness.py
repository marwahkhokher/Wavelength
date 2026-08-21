from completeness import (
    MAX_USEFUL_QUESTIONS_TOTAL,
    USEFUL_QUESTIONS_PER_ROUND,
    check_completeness,
)
from situation_extraction import InformationGap, SituationDraft


def gap(field, tier, importance=1, question=None):
    return InformationGap(
        field=field, tier=tier, importance=importance, question=question or f"What about {field}?"
    )


def test_required_gap_forces_need_info_regardless_of_budget():
    draft = SituationDraft(gaps=[gap("other_person_role", "required")])
    result = check_completeness(draft, useful_questions_asked_so_far=MAX_USEFUL_QUESTIONS_TOTAL)
    assert result.status == "need_info"
    assert "other_person_role" not in result.questions  # question text, not field name
    assert result.questions == ["What about other_person_role?"]


def test_no_gaps_is_ready():
    draft = SituationDraft(gaps=[])
    result = check_completeness(draft)
    assert result.status == "ready"
    assert result.questions == []


def test_useful_gaps_sorted_by_importance_and_capped_per_round():
    draft = SituationDraft(
        gaps=[
            gap("a", "useful", importance=2),
            gap("b", "useful", importance=5),
            gap("c", "useful", importance=4),
        ]
    )
    result = check_completeness(draft)
    assert result.status == "need_info"
    assert len(result.questions) == USEFUL_QUESTIONS_PER_ROUND
    assert result.questions == ["What about b?", "What about c?"]  # highest importance first


def test_total_useful_budget_caps_across_rounds():
    draft = SituationDraft(gaps=[gap("a", "useful", importance=5), gap("b", "useful", importance=4)])
    # Only 1 slot left in the total budget, even though per-round cap is 2.
    result = check_completeness(
        draft, useful_questions_asked_so_far=MAX_USEFUL_QUESTIONS_TOTAL - 1
    )
    assert result.status == "need_info"
    assert len(result.questions) == 1
    assert result.questions == ["What about a?"]


def test_budget_exhausted_proceeds_as_ready_rather_than_asking_forever():
    draft = SituationDraft(gaps=[gap("a", "useful", importance=5)])
    result = check_completeness(draft, useful_questions_asked_so_far=MAX_USEFUL_QUESTIONS_TOTAL)
    assert result.status == "ready"


def test_reuse_flow_typically_has_no_required_gaps():
    # Persona already known - situation_extraction wouldn't emit
    # other_person_role/relationship gaps at all, only situational ones.
    draft = SituationDraft(
        interaction_type=None,
        other_person_role=None,  # not re-derived, per situation_extraction's known_persona path
        gaps=[],
    )
    result = check_completeness(draft)
    assert result.status == "ready"
