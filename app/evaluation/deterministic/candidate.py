from __future__ import annotations

import math
import re
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from app.agent.patches import (
    DiffLineKind,
    UnifiedDiffError,
    normalize_repository_paths,
    parse_unified_diff,
)
from app.agent.schemas import PatchProposal
from app.agent.verification import CandidateVerificationResult, VerificationOutcome
from app.sandbox.results import NonNegativeFiniteFloat

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_TOKEN_COUNT = 2**63 - 1
_MAX_LINE_COUNT = 2**63 - 1


def _require_decimal(value: object, info: ValidationInfo) -> object:
    if info.mode == "json" and type(value) is str:
        try:
            value = Decimal(value)
        except Exception:
            raise ValueError("estimated cost must be a Decimal") from None
    if type(value) is not Decimal:
        raise ValueError("estimated cost must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError("estimated cost must be finite and nonnegative")
    return value


NonNegativeUsdDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_decimal),
    Field(ge=Decimal(0), allow_inf_nan=False),
]
StrictTokenCount = Annotated[int, Field(strict=True, ge=0, le=_MAX_TOKEN_COUNT)]
StrictLineCount = Annotated[int, Field(strict=True, ge=0, le=_MAX_LINE_COUNT)]


class CandidateFunctionalStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class CandidateModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model_latency_seconds: NonNegativeFiniteFloat | None = None
    input_tokens: StrictTokenCount | None = None
    output_tokens: StrictTokenCount | None = None
    estimated_cost_usd: NonNegativeUsdDecimal | None = None
    estimated_cost_currency: Literal["USD"] = "USD"


class CandidateEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=128)]
    proposal: PatchProposal
    verification: CandidateVerificationResult
    model_usage: CandidateModelUsage | None = None


class CandidateEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=128)]
    functional_status: CandidateFunctionalStatus
    files_changed: tuple[str, ...]
    lines_added: StrictLineCount
    lines_removed: StrictLineCount
    sandbox_duration_seconds: NonNegativeFiniteFloat
    model_latency_seconds: NonNegativeFiniteFloat | None = None
    input_tokens: StrictTokenCount | None = None
    output_tokens: StrictTokenCount | None = None
    estimated_cost_usd: NonNegativeUsdDecimal | None = None
    estimated_cost_currency: Literal["USD"] = "USD"

    @field_validator("files_changed", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("files_changed")
    @classmethod
    def validate_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        try:
            normalized = normalize_repository_paths(value)
        except UnifiedDiffError:
            raise ValueError("evaluation paths are invalid") from None
        expected = tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))
        if value != expected:
            raise ValueError("evaluation paths must be deterministically ordered")
        return value


class EvaluationInputError(ValueError):
    """Candidate evaluation input was malformed or internally inconsistent."""


def _strict_request(request: CandidateEvaluationRequest) -> CandidateEvaluationRequest:
    if type(request) is not CandidateEvaluationRequest:
        raise EvaluationInputError("candidate evaluation input is invalid")
    if (
        type(request.case_id) is not str
        or _CASE_ID.fullmatch(request.case_id) is None
        or type(request.proposal) is not PatchProposal
        or type(request.verification) is not CandidateVerificationResult
        or (
            request.model_usage is not None and type(request.model_usage) is not CandidateModelUsage
        )
    ):
        raise EvaluationInputError("candidate evaluation input is invalid")
    try:
        proposal_raw = request.proposal.model_dump(warnings="none")
        proposal = PatchProposal.model_validate(proposal_raw, strict=True)
        verification_raw = request.verification.model_dump(warnings="none")
        verification = CandidateVerificationResult.model_validate(verification_raw, strict=True)
        if (
            proposal.model_dump(warnings="none") != proposal_raw
            or verification.model_dump(warnings="none") != verification_raw
        ):
            raise ValueError
        usage: CandidateModelUsage | None = None
        if request.model_usage is not None:
            usage_raw = request.model_usage.model_dump(warnings="none")
            usage = CandidateModelUsage.model_validate(usage_raw, strict=True)
            if usage.model_dump(warnings="none") != usage_raw:
                raise ValueError
        rebuilt = CandidateEvaluationRequest(
            case_id=request.case_id,
            proposal=proposal,
            verification=verification,
            model_usage=usage,
        )
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise EvaluationInputError("candidate evaluation input is invalid") from None
    return rebuilt


def evaluate_candidate(request: CandidateEvaluationRequest) -> CandidateEvaluationResult:
    validated = _strict_request(request)
    try:
        proposal_paths = normalize_repository_paths(tuple(validated.proposal.files_changed))
        parsed = parse_unified_diff(validated.proposal.unified_diff)
        if parsed.has_ignored_content:
            raise UnifiedDiffError("unified diff contains unsupported content")
        parsed_paths = normalize_repository_paths(parsed.changed_paths)
        if len(proposal_paths) != len(parsed_paths) or set(proposal_paths) != set(parsed_paths):
            raise UnifiedDiffError("proposal and parsed diff paths disagree")
        ordered_paths = tuple(sorted(parsed_paths, key=lambda item: (item.casefold(), item)))
        lines_added = sum(
            1 for file in parsed.files for line in file.lines if line.kind is DiffLineKind.ADDED
        )
        lines_removed = sum(
            1 for file in parsed.files for line in file.lines if line.kind is DiffLineKind.REMOVED
        )
        if lines_added > _MAX_LINE_COUNT or lines_removed > _MAX_LINE_COUNT:
            raise UnifiedDiffError("diff line count is too large")
    except (TypeError, ValueError, UnifiedDiffError):
        raise EvaluationInputError("candidate proposal is invalid") from None

    status = {
        VerificationOutcome.PASSED: CandidateFunctionalStatus.PASSED,
        VerificationOutcome.CANDIDATE_FAILED: CandidateFunctionalStatus.FAILED,
        VerificationOutcome.INFRASTRUCTURE_DIAGNOSTIC: (CandidateFunctionalStatus.NOT_EVALUATED),
    }[validated.verification.outcome]
    try:
        duration = math.fsum(
            stage.duration_seconds
            for stage in validated.verification.stages
            if stage.duration_seconds is not None
        )
    except (OverflowError, ValueError):
        raise EvaluationInputError("verification duration is invalid") from None
    if not math.isfinite(duration) or duration < 0.0:
        raise EvaluationInputError("verification duration is invalid")
    usage = validated.model_usage
    return CandidateEvaluationResult(
        case_id=validated.case_id,
        functional_status=status,
        files_changed=ordered_paths,
        lines_added=lines_added,
        lines_removed=lines_removed,
        sandbox_duration_seconds=duration,
        model_latency_seconds=None if usage is None else usage.model_latency_seconds,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        estimated_cost_usd=None if usage is None else usage.estimated_cost_usd,
        estimated_cost_currency="USD",
    )
