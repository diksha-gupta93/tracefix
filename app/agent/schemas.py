from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictInteger = Annotated[int, Field(strict=True)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]


def _require_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("value must be a float")
    return value


Confidence = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]


class TestStatus(StrEnum):
    passed = "passed"
    failed = "failed"
    timed_out = "timed_out"
    error = "error"


class RepairStatus(StrEnum):
    pending = "pending"
    baseline_complete = "baseline_complete"
    analysis_complete = "analysis_complete"
    plan_complete = "plan_complete"
    patch_proposed = "patch_proposed"
    verified = "verified"
    failed = "failed"


class FailureCategory(StrEnum):
    syntax_failure = "syntax_failure"
    import_or_dependency_failure = "import_or_dependency_failure"
    assertion_failure = "assertion_failure"
    incorrect_exception_behaviour = "incorrect_exception_behaviour"
    type_related_failure = "type_related_failure"
    timeout = "timeout"
    test_environment_failure = "test_environment_failure"
    unsupported_or_ambiguous = "unsupported_or_ambiguous"


class EvaluationStatus(StrEnum):
    passed = "passed"
    failed = "failed"


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    status: TestStatus
    exit_code: StrictInteger
    stdout: str
    stderr: str
    duration_seconds: NonNegativeFiniteFloat


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    category: FailureCategory
    summary: NonBlankString


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    suspected_root_cause: NonBlankString
    files_expected_to_change: list[NonBlankString]
    intended_behavioural_correction: NonBlankString
    risks: list[NonBlankString]
    validation_strategy: NonBlankString
    autonomous_repair_suitable: Annotated[bool, Field(strict=True)]


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    summary: NonBlankString
    root_cause: NonBlankString
    files_changed: list[NonBlankString]
    unified_diff: NonBlankString
    expected_effect: NonBlankString
    risks: list[NonBlankString]
    confidence: Confidence


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    status: EvaluationStatus
    summary: NonBlankString


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LocalRepairCaseState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    case_id: NonBlankString
    status: RepairStatus = RepairStatus.pending
    attempt_number: Annotated[int, Field(strict=True, ge=0, le=1)] = 0
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    baseline_result: TestResult | None = None
    failure_analysis: FailureAnalysis | None = None
    repair_plan: RepairPlan | None = None
    candidate_patch: PatchProposal | None = None
    verification_results: list[TestResult] = Field(default_factory=list)
    evaluation_results: list[EvaluationResult] = Field(default_factory=list)
    model_provider: NonBlankString | None = None
    model_name: NonBlankString | None = None
    prompt_version: NonBlankString | None = None
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)
