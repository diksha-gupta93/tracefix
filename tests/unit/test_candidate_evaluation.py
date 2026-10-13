from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.agent.schemas import PatchProposal
from app.agent.verification import (
    CandidateVerificationResult,
    VerificationDiagnostic,
    VerificationDiagnosticCode,
    VerificationStage,
    VerificationStageResult,
    VerificationStageStatus,
)
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationRequest,
    CandidateFunctionalStatus,
    CandidateModelUsage,
    EvaluationInputError,
    evaluate_candidate,
)
from app.sandbox.results import SandboxCompletion


def _stage(
    stage: VerificationStage, exit_code: int = 0, duration: float = 0.25
) -> VerificationStageResult:
    return VerificationStageResult(
        stage=stage,
        status=(
            VerificationStageStatus.PASSED if exit_code == 0 else VerificationStageStatus.FAILED
        ),
        completion=SandboxCompletion.NORMAL,
        exit_code=exit_code,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=duration,
    )


def _verification() -> CandidateVerificationResult:
    return CandidateVerificationResult(stages=tuple(_stage(stage) for stage in VerificationStage))


def _proposal() -> PatchProposal:
    return PatchProposal(
        summary="repair",
        root_cause="defect",
        files_changed=["src/a.py", "src/b.py"],
        unified_diff=(
            "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1,2 @@\n-old\n+new\n+extra\n"
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n keep\n-remove\n+add\n"
            "\\ No newline at end of file\n"
        ),
        expected_effect="fixed",
        risks=[],
        confidence=1.0,
    )


def test_evaluator_counts_diff_orders_paths_and_sums_only_sandbox_duration() -> None:
    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one",
            proposal=_proposal(),
            verification=_verification(),
        )
    )

    assert result.functional_status is CandidateFunctionalStatus.PASSED
    assert result.files_changed == ("src/a.py", "src/b.py")
    assert result.lines_added == 3
    assert result.lines_removed == 2
    assert result.sandbox_duration_seconds == 0.75
    assert result.model_latency_seconds is None
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.estimated_cost_usd is None


def test_model_usage_preserves_zero_and_decimal_exactly() -> None:
    usage = CandidateModelUsage(
        model_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=Decimal("0.0000"),
    )
    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one",
            proposal=_proposal(),
            verification=_verification(),
            model_usage=usage,
        )
    )

    assert result.model_latency_seconds == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.estimated_cost_usd == Decimal("0.0000")
    assert '"estimated_cost_usd":"0.0000"' in result.model_dump_json()
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_latency_seconds", 1),
        ("model_latency_seconds", float("inf")),
        ("input_tokens", True),
        ("input_tokens", -1),
        ("output_tokens", 2**63),
        ("estimated_cost_usd", Decimal("NaN")),
        ("estimated_cost_usd", Decimal("-0.01")),
    ],
)
def test_usage_rejects_coercion_nonfinite_negative_and_oversized(field: str, value: object) -> None:
    values: dict[str, object] = {field: value}
    with pytest.raises(ValidationError):
        CandidateModelUsage.model_validate(values)


def test_malformed_or_path_mismatched_diff_is_domain_error() -> None:
    proposal = _proposal()
    proposal.files_changed = ["src/a.py"]
    with pytest.raises(EvaluationInputError):
        evaluate_candidate(
            CandidateEvaluationRequest(
                case_id="case_one",
                proposal=proposal,
                verification=_verification(),
            )
        )


def test_infrastructure_verification_maps_to_not_evaluated() -> None:
    diagnostic = VerificationDiagnostic(
        code=VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED,
        message="sandbox execution failed",
    )
    verification = CandidateVerificationResult(
        stages=(
            VerificationStageResult(
                stage=VerificationStage.TARGETED,
                status=VerificationStageStatus.INFRASTRUCTURE_ERROR,
                diagnostic=diagnostic,
            ),
            *(
                VerificationStageResult(
                    stage=stage,
                    status=VerificationStageStatus.NOT_RUN,
                    diagnostic=VerificationDiagnostic(
                        code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
                        message="a required earlier verification stage did not pass",
                    ),
                )
                for stage in (VerificationStage.VISIBLE_SUITE, VerificationStage.HIDDEN_SUITE)
            ),
        )
    )
    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one", proposal=_proposal(), verification=verification
        )
    )
    assert result.functional_status is CandidateFunctionalStatus.NOT_EVALUATED
    assert result.sandbox_duration_seconds == 0.0


def test_candidate_failure_maps_to_failed_and_sums_only_executed_stage() -> None:
    prerequisite = VerificationDiagnostic(
        code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
        message="a required earlier verification stage did not pass",
    )
    verification = CandidateVerificationResult(
        stages=(
            _stage(VerificationStage.TARGETED, exit_code=3, duration=1.25),
            VerificationStageResult(
                stage=VerificationStage.VISIBLE_SUITE,
                status=VerificationStageStatus.NOT_RUN,
                diagnostic=prerequisite,
            ),
            VerificationStageResult(
                stage=VerificationStage.HIDDEN_SUITE,
                status=VerificationStageStatus.NOT_RUN,
                diagnostic=prerequisite,
            ),
        )
    )

    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one", proposal=_proposal(), verification=verification
        )
    )

    assert result.functional_status is CandidateFunctionalStatus.FAILED
    assert result.sandbox_duration_seconds == 1.25


def test_partial_model_usage_preserves_missing_values_as_null() -> None:
    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one",
            proposal=_proposal(),
            verification=_verification(),
            model_usage=CandidateModelUsage(input_tokens=17),
        )
    )

    assert result.input_tokens == 17
    assert result.output_tokens is None
    assert result.model_latency_seconds is None
    assert result.estimated_cost_usd is None
    payload = result.model_dump_json()
    assert '"output_tokens":null' in payload
    assert '"estimated_cost_currency":"USD"' in payload


@pytest.mark.parametrize(
    "diff",
    [
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n",
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\ntrailing metadata\n",
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n--- a/src/A.py\n+++ b/src/A.py\n@@ -1 +1 @@\n-old\n+new\n",
    ],
)
def test_evaluator_rejects_ignored_content_and_case_aliased_paths(diff: str) -> None:
    proposal = _proposal()
    proposal.files_changed = ["src/a.py"]
    proposal.unified_diff = diff

    with pytest.raises(EvaluationInputError):
        evaluate_candidate(
            CandidateEvaluationRequest(
                case_id="case_one", proposal=proposal, verification=_verification()
            )
        )


def test_evaluation_request_subclass_and_mutated_verification_fail_closed() -> None:
    class RequestSubclass(CandidateEvaluationRequest):
        pass

    request = CandidateEvaluationRequest(
        case_id="case_one", proposal=_proposal(), verification=_verification()
    )
    subclass = RequestSubclass.model_validate(request.model_dump())
    with pytest.raises(EvaluationInputError):
        evaluate_candidate(subclass)

    object.__setattr__(request.verification, "outcome", "candidate_failed")
    with pytest.raises(EvaluationInputError):
        evaluate_candidate(request)


def test_evaluator_rejects_normalized_mutated_stage_collection() -> None:
    request = CandidateEvaluationRequest(
        case_id="case_one", proposal=_proposal(), verification=_verification()
    )
    object.__setattr__(request.verification, "stages", list(request.verification.stages))

    with pytest.raises(EvaluationInputError):
        evaluate_candidate(request)


def test_evaluator_maps_finite_duration_sum_overflow_to_domain_error() -> None:
    verification = CandidateVerificationResult(
        stages=tuple(_stage(stage, duration=1e308) for stage in VerificationStage)
    )

    with pytest.raises(EvaluationInputError, match="verification duration is invalid"):
        evaluate_candidate(
            CandidateEvaluationRequest(
                case_id="case_one", proposal=_proposal(), verification=verification
            )
        )


def test_evaluation_result_is_strict_frozen_and_rejects_extra_fields() -> None:
    result = evaluate_candidate(
        CandidateEvaluationRequest(
            case_id="case_one", proposal=_proposal(), verification=_verification()
        )
    )
    with pytest.raises(ValidationError):
        result.lines_added = 99  # type: ignore[misc]
    payload = result.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)
