import enum


class Mode(str, enum.Enum):
    """FR-MODE-1: exactly two categories."""
    PROFESSIONAL = "professional"
    PERSONAL = "personal"


class InputMethod(str, enum.Enum):
    """FR-SCEN-2/3, FR-PERS-2/3: typed or speech-to-text."""
    TYPED = "typed"
    SPEECH = "speech"


class Gender(str, enum.Enum):
    """FR-ONB-5."""
    MALE = "male"
    FEMALE = "female"


class SessionStatus(str, enum.Enum):
    DRAFT = "draft"                # scenario/persona being built, not started
    IN_PROGRESS = "in_progress"    # FR-SESS-* live session running
    ENDED = "ended"                # FR-SESS-7
    EVALUATED = "evaluated"        # FR-EVAL-1 complete


class MetricSource(str, enum.Enum):
    BASELINE = "baseline"   # FR-ONB-6 / FR-MET-2
    SESSION = "session"     # FR-EVAL-2 / FR-MET-3


# FR-MET / Section 9 item #1: the full metric list is NOT finalized in the PRD.
# clarity, empathy, and filler_words are confirmed. This list is deliberately
# a single source of truth so the rest can be added here once product-research
# confirms them, without touching the DB schema (metrics are stored as JSON,
# validated against this list at the API layer).
CONFIRMED_METRIC_KEYS = [
    "clarity",
    "empathy",
    "filler_words",
    # TODO(product-research): add remaining metrics from PRD Section 9, item 1
]
