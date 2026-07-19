from __future__ import annotations

from enum import Enum

from .errors import StylometryError
from .verification_models import VerificationDecision


class VerificationExpectedClass(str, Enum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"


class VerificationEvaluationOutcome(str, Enum):
    CORRECT_ACCEPT = "correct_accept"
    FALSE_REJECT = "false_reject"
    GENUINE_INCONCLUSIVE = "genuine_inconclusive"
    CORRECT_REJECT = "correct_reject"
    FALSE_ACCEPT = "false_accept"
    IMPOSTOR_INCONCLUSIVE = "impostor_inconclusive"


def classify_verification_evaluation_outcome(
    expected: VerificationExpectedClass,
    decision: VerificationDecision,
) -> VerificationEvaluationOutcome:
    try:
        return {
            (VerificationExpectedClass.GENUINE, VerificationDecision.ACCEPT): VerificationEvaluationOutcome.CORRECT_ACCEPT,
            (VerificationExpectedClass.GENUINE, VerificationDecision.REJECT): VerificationEvaluationOutcome.FALSE_REJECT,
            (VerificationExpectedClass.GENUINE, VerificationDecision.INCONCLUSIVE): VerificationEvaluationOutcome.GENUINE_INCONCLUSIVE,
            (VerificationExpectedClass.IMPOSTOR, VerificationDecision.REJECT): VerificationEvaluationOutcome.CORRECT_REJECT,
            (VerificationExpectedClass.IMPOSTOR, VerificationDecision.ACCEPT): VerificationEvaluationOutcome.FALSE_ACCEPT,
            (VerificationExpectedClass.IMPOSTOR, VerificationDecision.INCONCLUSIVE): VerificationEvaluationOutcome.IMPOSTOR_INCONCLUSIVE,
        }[(expected, decision)]
    except (KeyError, TypeError) as exc:
        raise StylometryError("invalid verification evaluation classification") from exc
