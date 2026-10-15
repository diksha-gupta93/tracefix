from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.agent.policy import PatchPolicyResult
from app.agent.schemas import BaselineOutcome, BaselineResult
from app.agent.static_checks import StaticValidationOutcome, StaticValidationResult
from app.agent.verification import CandidateVerificationResult, VerificationOutcome
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationResult,
    CandidateFunctionalStatus,
)


class TerminalOutcome(StrEnum):
    SUCCESS = "success"
    CANDIDATE_FAILURE = "candidate_failure"
    POLICY_VIOLATION = "policy_violation"
    STATIC_CHECK_FAILURE = "static_check_failure"
    BASELINE_NOT_REPRODUCIBLE = "baseline_not_reproducible"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"


class TerminalDiagnosticCode(StrEnum):
    CANDIDATE_FAILURE = "candidate_failure"
    POLICY_VIOLATION = "policy_violation"
    STATIC_CHECK_FAILURE = "static_check_failure"
    BASELINE_NOT_REPRODUCIBLE = "baseline_not_reproducible"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"


EXIT_CODES = MappingProxyType(
    {
        TerminalOutcome.SUCCESS: 0,
        TerminalOutcome.CANDIDATE_FAILURE: 1,
        TerminalOutcome.POLICY_VIOLATION: 3,
        TerminalOutcome.STATIC_CHECK_FAILURE: 4,
        TerminalOutcome.BASELINE_NOT_REPRODUCIBLE: 5,
        TerminalOutcome.INFRASTRUCTURE_FAILURE: 6,
        TerminalOutcome.MALFORMED_MODEL_OUTPUT: 7,
    }
)

_SUMMARIES = MappingProxyType(
    {
        TerminalOutcome.SUCCESS: "repair candidate passed all required checks",
        TerminalOutcome.CANDIDATE_FAILURE: "repair candidate did not pass required verification",
        TerminalOutcome.POLICY_VIOLATION: "patch proposal violated the repair policy",
        TerminalOutcome.STATIC_CHECK_FAILURE: "repair candidate failed static validation",
        TerminalOutcome.BASELINE_NOT_REPRODUCIBLE: "declared baseline failure was not reproducible",
        TerminalOutcome.INFRASTRUCTURE_FAILURE: "repair run encountered an infrastructure failure",
        TerminalOutcome.MALFORMED_MODEL_OUTPUT: "model output was not a valid repair response",
    }
)


def terminal_summary(outcome: TerminalOutcome) -> str:
    return _SUMMARIES[outcome]


class TerminalDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: TerminalDiagnosticCode
    message: str

    _OUTCOME_CODES: ClassVar[dict[TerminalOutcome, TerminalDiagnosticCode]] = {
        TerminalOutcome.CANDIDATE_FAILURE: TerminalDiagnosticCode.CANDIDATE_FAILURE,
        TerminalOutcome.POLICY_VIOLATION: TerminalDiagnosticCode.POLICY_VIOLATION,
        TerminalOutcome.STATIC_CHECK_FAILURE: TerminalDiagnosticCode.STATIC_CHECK_FAILURE,
        TerminalOutcome.BASELINE_NOT_REPRODUCIBLE: (
            TerminalDiagnosticCode.BASELINE_NOT_REPRODUCIBLE
        ),
        TerminalOutcome.INFRASTRUCTURE_FAILURE: TerminalDiagnosticCode.INFRASTRUCTURE_FAILURE,
        TerminalOutcome.MALFORMED_MODEL_OUTPUT: TerminalDiagnosticCode.MALFORMED_MODEL_OUTPUT,
    }

    @classmethod
    def for_outcome(cls, outcome: TerminalOutcome) -> TerminalDiagnostic:
        if outcome is TerminalOutcome.SUCCESS:
            raise ValueError("success does not have a terminal diagnostic")
        return cls(code=cls._OUTCOME_CODES[outcome], message=terminal_summary(outcome))

    @model_validator(mode="after")
    def validate_constant(self) -> Self:
        outcome = TerminalOutcome(self.code.value)
        if self.message != terminal_summary(outcome):
            raise ValueError("terminal diagnostic message must be the safe constant")
        return self


class TerminalRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: TerminalOutcome
    exit_code: int
    diagnostic: TerminalDiagnostic | None

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        if type(self.exit_code) is not int or self.exit_code != EXIT_CODES[self.outcome]:
            raise ValueError("terminal exit code does not match the outcome")
        expected = (
            None
            if self.outcome is TerminalOutcome.SUCCESS
            else TerminalDiagnostic.for_outcome(self.outcome)
        )
        if self.diagnostic != expected:
            raise ValueError("terminal diagnostic does not match the outcome")
        return self

    @classmethod
    def for_outcome(cls, outcome: TerminalOutcome) -> TerminalRoute:
        return cls(
            outcome=outcome,
            exit_code=EXIT_CODES[outcome],
            diagnostic=(
                None
                if outcome is TerminalOutcome.SUCCESS
                else TerminalDiagnostic.for_outcome(outcome)
            ),
        )


class RoutingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    baseline: BaselineResult | None = None
    policy: PatchPolicyResult | None = None
    static: StaticValidationResult | None = None
    verification: CandidateVerificationResult | None = None
    evaluation: CandidateEvaluationResult | None = None
    failure_override: TerminalOutcome | None = None

    @model_validator(mode="after")
    def validate_override(self) -> Self:
        if self.failure_override in {
            TerminalOutcome.SUCCESS,
            TerminalOutcome.POLICY_VIOLATION,
            TerminalOutcome.STATIC_CHECK_FAILURE,
            TerminalOutcome.BASELINE_NOT_REPRODUCIBLE,
        }:
            raise ValueError("terminal override is not permitted for evidence-derived outcomes")
        later = (self.policy, self.static, self.verification, self.evaluation)
        if self.baseline is None and any(value is not None for value in later):
            raise ValueError("later routing evidence requires a baseline")
        if (
            self.baseline is not None
            and self.baseline.outcome is not BaselineOutcome.REPRODUCED_FAILURE
            and any(value is not None for value in later)
        ):
            raise ValueError("terminal baseline evidence cannot have later evidence")
        if self.policy is None and any(
            value is not None for value in (self.static, self.verification, self.evaluation)
        ):
            raise ValueError("candidate evidence requires policy evidence")
        if (
            self.policy is not None
            and not self.policy.approved
            and any(
                value is not None for value in (self.static, self.verification, self.evaluation)
            )
        ):
            raise ValueError("rejected policy cannot have candidate evidence")
        if self.static is None and any(
            value is not None for value in (self.verification, self.evaluation)
        ):
            raise ValueError("verification evidence requires static evidence")
        if (
            self.static is not None
            and self.static.outcome is not StaticValidationOutcome.PASSED
            and (self.verification is not None or self.evaluation is not None)
        ):
            raise ValueError("non-passing static evidence cannot have verification evidence")
        if self.verification is None and self.evaluation is not None:
            raise ValueError("evaluation requires verification evidence")
        if self.failure_override in {
            TerminalOutcome.CANDIDATE_FAILURE,
            TerminalOutcome.MALFORMED_MODEL_OUTPUT,
        } and any(value is not None for value in later):
            raise ValueError("terminal override contradicts later evidence")
        return self


def _strict_evidence(evidence: RoutingEvidence) -> RoutingEvidence | None:
    if type(evidence) is not RoutingEvidence:
        return None
    try:
        raw = evidence.model_dump(warnings="none")
        checked = RoutingEvidence.model_validate(raw, strict=True)
        if checked.model_dump(warnings="none") != raw:
            return None
        return checked
    except (AttributeError, TypeError, ValueError, ValidationError):
        return None


def route_terminal(evidence: RoutingEvidence) -> TerminalRoute:
    checked = _strict_evidence(evidence)
    if checked is None:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if checked.failure_override is not None:
        return TerminalRoute.for_outcome(checked.failure_override)

    baseline = checked.baseline
    if baseline is None:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if baseline.outcome is BaselineOutcome.NOT_REPRODUCIBLE:
        return TerminalRoute.for_outcome(TerminalOutcome.BASELINE_NOT_REPRODUCIBLE)
    if baseline.outcome is BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if baseline.outcome is not BaselineOutcome.REPRODUCED_FAILURE:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)

    policy = checked.policy
    if policy is None:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if not policy.approved:
        return TerminalRoute.for_outcome(TerminalOutcome.POLICY_VIOLATION)

    static = checked.static
    if static is None:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if static.outcome is StaticValidationOutcome.CANDIDATE_FAILED:
        return TerminalRoute.for_outcome(TerminalOutcome.STATIC_CHECK_FAILURE)
    if static.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if static.outcome is not StaticValidationOutcome.PASSED:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)

    verification = checked.verification
    evaluation = checked.evaluation
    if verification is None or evaluation is None:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    if verification.outcome is VerificationOutcome.PASSED:
        outcome = (
            TerminalOutcome.SUCCESS
            if evaluation.functional_status is CandidateFunctionalStatus.PASSED
            else TerminalOutcome.INFRASTRUCTURE_FAILURE
        )
        return TerminalRoute.for_outcome(outcome)
    if verification.outcome is VerificationOutcome.CANDIDATE_FAILED:
        outcome = (
            TerminalOutcome.CANDIDATE_FAILURE
            if evaluation.functional_status is CandidateFunctionalStatus.FAILED
            else TerminalOutcome.INFRASTRUCTURE_FAILURE
        )
        return TerminalRoute.for_outcome(outcome)
    if verification.outcome is VerificationOutcome.INFRASTRUCTURE_DIAGNOSTIC:
        return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
    return TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE)
