import re
from typing import Tuple

CITATION_PATTERN = re.compile(
    r"\[source:\s*[^\]]+#[^\]]+\]"
)

FORBIDDEN_ACTION_CLAIMS = [
    "has been cancelled",
    "have cancelled",
    "successfully cancelled",
    "has been refunded",
    "have refunded",
    "successfully refunded",
    "address has been changed",
    "updated your address",
    "has been updated",
]

ABSTENTION_STRING = (
    "I don't have enough information to answer that reliably. "
    "Please contact human support for assistance."
)

CONFLICT_STRING = (
    "I found conflicting information in the current policy sources, "
    "so I can't give you a reliable answer. "
    "Please contact human support for clarification."
)


def check_citations(
    response_text: str,
    used_policy_context: bool
) -> bool:
    """If policy context was used, the response MUST contain a valid citation,
    unless it's an escalation.
    """
    if not used_policy_context:
        return True

    # If the model correctly escalates, it does not need to cite a source.
    if ABSTENTION_STRING in response_text or CONFLICT_STRING in response_text:
        return True

    return bool(CITATION_PATTERN.search(response_text))


def check_action_claims(response_text: str) -> bool:
    """Ensures the LLM doesn't hallucinate that it executed a system action."""
    lower_text = response_text.lower()

    for phrase in FORBIDDEN_ACTION_CLAIMS:
        if phrase in lower_text:
            return False

    return True


def apply_guardrails(
    response_text: str,
    used_policy_context: bool
) -> Tuple[bool, str, str]:

    if not check_action_claims(response_text):
        return (
            False,
            "I cannot confirm that action was executed. "
            "Please use the supported process or contact human support.",
            "Failed action-claim guardrail.",
        )

    if not check_citations(response_text, used_policy_context):
        return (
            False,
            ABSTENTION_STRING,
            "Failed citation guardrail.",
        )

    return True, response_text, ""