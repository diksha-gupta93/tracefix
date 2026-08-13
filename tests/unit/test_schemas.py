from __future__ import annotations

from datetime import UTC, datetime
from math import inf, nan

import pytest
from pydantic import BaseModel, ValidationError

from app.agent.schemas import (
    EvaluationResult,
    EvaluationStatus,
    FailureAnalysis,
    FailureCategory,
    LocalRepairCaseState,
    PatchProposal,
    RepairPlan,
    RepairStatus,
)
from app.agent.schemas import (
    TestResult as SchemaTestResult,
)
from app.agent.schemas import (
    TestStatus as SchemaTestStatus,
)


def make_test_result() -> SchemaTestResult:
    return SchemaTestResult(
        status=SchemaTestStatus.failed,
        exit_code=1,
        stdout="",
        stderr="assertion failed",
        duration_seconds=0.25,
    )


def make_repair_plan() -> RepairPlan:
    return RepairPlan(
        suspected_root_cause="Incorrect boundary check",
        files_expected_to_change=["app/example.py"],
        intended_behavioural_correction="Include the upper boundary",
        risks=["May affect adjacent inputs"],
        validation_strategy="Run the focused and full test suites",
        autonomous_repair_suitable=True,
    )


def make_patch_proposal() -> PatchProposal:
    return PatchProposal(
        summary="Correct the boundary check",
        root_cause="The upper boundary was excluded",
        files_changed=["app/example.py"],
        unified_diff="--- a/app/example.py\n+++ b/app/example.py\n",
        expected_effect="The upper boundary is accepted",
        risks=[],
        confidence=0.9,
    )


def test_constructs_each_schema_and_minimal_state() -> None:
    result = make_test_result()
    analysis = FailureAnalysis(
        category=FailureCategory.assertion_failure,
        summary="A boundary assertion failed",
    )
    plan = make_repair_plan()
    patch = make_patch_proposal()
    evaluation = EvaluationResult(
        status=EvaluationStatus.passed,
        summary="All deterministic checks passed",
    )
    state = LocalRepairCaseState(case_id="case-001")

    assert result.exit_code == 1
    assert analysis.category is FailureCategory.assertion_failure
    assert plan.autonomous_repair_suitable is True
    assert patch.confidence == 0.9
    assert evaluation.status is EvaluationStatus.passed
    assert state.status is RepairStatus.pending
    assert state.attempt_number == 0
    assert state.max_attempts == 1
    assert state.baseline_result is None
    assert state.failure_analysis is None
    assert state.repair_plan is None
    assert state.candidate_patch is None
    assert state.model_provider is None
    assert state.model_name is None
    assert state.prompt_version is None


def test_constructs_fully_populated_state() -> None:
    result = make_test_result()
    state = LocalRepairCaseState(
        case_id="case-002",
        status=RepairStatus.verified,
        attempt_number=1,
        max_attempts=1,
        baseline_result=result,
        failure_analysis=FailureAnalysis(
            category=FailureCategory.assertion_failure,
            summary="An assertion failed",
        ),
        repair_plan=make_repair_plan(),
        candidate_patch=make_patch_proposal(),
        verification_results=[result],
        evaluation_results=[EvaluationResult(status=EvaluationStatus.passed, summary="Passed")],
        model_provider="provider",
        model_name="model",
        prompt_version="v1",
    )

    assert state.status is RepairStatus.verified
    assert state.verification_results == [result]
    assert state.evaluation_results[0].status is EvaluationStatus.passed


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (SchemaTestResult, {}),
        (FailureAnalysis, {}),
        (RepairPlan, {}),
        (PatchProposal, {}),
        (EvaluationResult, {}),
        (LocalRepairCaseState, {}),
        (LocalRepairCaseState, {"case_id": "case", "unknown": "value"}),
    ],
)
def test_rejects_missing_required_and_unknown_fields(
    model: type[SchemaTestResult]
    | type[FailureAnalysis]
    | type[RepairPlan]
    | type[PatchProposal]
    | type[EvaluationResult]
    | type[LocalRepairCaseState],
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)


@pytest.mark.parametrize("exit_code", [True, False, 1.0, "1"])
def test_exit_code_is_a_strict_integer(exit_code: object) -> None:
    with pytest.raises(ValidationError):
        SchemaTestResult(
            status=SchemaTestStatus.error,
            exit_code=exit_code,
            stdout="",
            stderr="error",
            duration_seconds=0.0,
        )


@pytest.mark.parametrize("attempt_number", [True, False, -1, 2, 1.0, "1"])
def test_attempt_number_is_a_bounded_strict_integer(attempt_number: object) -> None:
    with pytest.raises(ValidationError):
        LocalRepairCaseState(case_id="case", attempt_number=attempt_number)


@pytest.mark.parametrize("max_attempts", [True, False, 0, 2, 1.0, "1"])
def test_max_attempts_accepts_only_integer_one(max_attempts: object) -> None:
    with pytest.raises(ValidationError):
        LocalRepairCaseState(case_id="case", max_attempts=max_attempts)


def test_enums_accept_only_declared_values() -> None:
    assert make_test_result().status is SchemaTestStatus.failed
    assert (
        LocalRepairCaseState(case_id="case", status=RepairStatus.failed).status
        is RepairStatus.failed
    )
    assert (
        FailureAnalysis(category=FailureCategory.timeout, summary="Timed out").category
        is FailureCategory.timeout
    )
    assert (
        EvaluationResult(status=EvaluationStatus.failed, summary="Failed").status
        is EvaluationStatus.failed
    )

    with pytest.raises(ValidationError):
        SchemaTestResult(
            status="unknown",
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        FailureAnalysis(category="boundary_condition", summary="Not a runtime category")


@pytest.mark.parametrize("duration", [-0.01, inf, -inf, nan])
def test_duration_must_be_finite_and_non_negative(duration: float) -> None:
    with pytest.raises(ValidationError):
        SchemaTestResult(
            status=SchemaTestStatus.error,
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=duration,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, inf, -inf, nan])
def test_confidence_must_be_finite_and_bounded(confidence: float) -> None:
    values = make_patch_proposal().model_dump()
    values["confidence"] = confidence
    with pytest.raises(ValidationError):
        PatchProposal.model_validate(values)


@pytest.mark.parametrize("confidence", [True, False, 1, "0.5"])
def test_confidence_is_a_strict_float(confidence: object) -> None:
    values = make_patch_proposal().model_dump()
    values["confidence"] = confidence
    with pytest.raises(ValidationError):
        PatchProposal.model_validate(values)


@pytest.mark.parametrize("suitable", [1, 0, "true"])
def test_autonomous_repair_suitable_is_a_strict_boolean(suitable: object) -> None:
    values = make_repair_plan().model_dump()
    values["autonomous_repair_suitable"] = suitable
    with pytest.raises(ValidationError):
        RepairPlan.model_validate(values)


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (FailureAnalysis, "summary"),
        (RepairPlan, "suspected_root_cause"),
        (RepairPlan, "intended_behavioural_correction"),
        (RepairPlan, "validation_strategy"),
        (PatchProposal, "summary"),
        (PatchProposal, "root_cause"),
        (PatchProposal, "unified_diff"),
        (PatchProposal, "expected_effect"),
        (EvaluationResult, "summary"),
        (LocalRepairCaseState, "case_id"),
        (LocalRepairCaseState, "model_provider"),
        (LocalRepairCaseState, "model_name"),
        (LocalRepairCaseState, "prompt_version"),
    ],
)
def test_semantic_strings_reject_blank_values(model: type[BaseModel], field: str) -> None:
    valid_values: dict[type[BaseModel], dict[str, object]] = {
        FailureAnalysis: {"category": FailureCategory.assertion_failure, "summary": "valid"},
        RepairPlan: make_repair_plan().model_dump(),
        PatchProposal: make_patch_proposal().model_dump(),
        EvaluationResult: {"status": EvaluationStatus.passed, "summary": "valid"},
        LocalRepairCaseState: {"case_id": "valid"},
    }
    values = valid_values[model]
    values[field] = " \t "

    with pytest.raises(ValidationError):
        model.model_validate(values)


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (RepairPlan, "files_expected_to_change"),
        (RepairPlan, "risks"),
        (PatchProposal, "files_changed"),
        (PatchProposal, "risks"),
    ],
)
def test_list_items_reject_blank_values(model: type[BaseModel], field: str) -> None:
    values = (
        make_repair_plan().model_dump()
        if model is RepairPlan
        else make_patch_proposal().model_dump()
    )
    values[field] = [" \t "]

    with pytest.raises(ValidationError):
        model.model_validate(values)


def test_empty_output_and_empty_canonical_lists_are_valid() -> None:
    result = SchemaTestResult(
        status=SchemaTestStatus.passed,
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )
    plan_values = make_repair_plan().model_dump()
    plan_values["files_expected_to_change"] = []
    plan_values["risks"] = []

    assert result.stdout == result.stderr == ""
    assert RepairPlan.model_validate(plan_values).files_expected_to_change == []
    assert make_patch_proposal().risks == []


def test_fully_populated_state_round_trips_through_json() -> None:
    result = make_test_result()
    state = LocalRepairCaseState(
        case_id="café-例",
        status=RepairStatus.verified,
        attempt_number=1,
        baseline_result=result,
        failure_analysis=FailureAnalysis(
            category=FailureCategory.incorrect_exception_behaviour,
            summary="Unexpected exception: échec",
        ),
        repair_plan=make_repair_plan(),
        candidate_patch=make_patch_proposal(),
        verification_results=[result],
        evaluation_results=[EvaluationResult(status=EvaluationStatus.passed, summary="Vérifié")],
        model_provider="fournisseur",
        model_name="modèle",
        prompt_version="版本-1",
    )

    restored = LocalRepairCaseState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.created_at == state.created_at
    assert restored.created_at.utcoffset() == state.created_at.utcoffset()


def test_timestamp_defaults_are_per_instance_timezone_aware_and_utc() -> None:
    before = datetime.now(UTC)
    first = LocalRepairCaseState(case_id="first")
    second = LocalRepairCaseState(case_id="second")
    after = datetime.now(UTC)

    for timestamp in (
        first.created_at,
        first.updated_at,
        second.created_at,
        second.updated_at,
    ):
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset() == UTC.utcoffset(timestamp)
        assert before <= timestamp <= after

    assert first.created_at is not second.created_at
    assert first.updated_at is not second.updated_at


def test_default_lists_are_independent() -> None:
    first = LocalRepairCaseState(case_id="first")
    second = LocalRepairCaseState(case_id="second")

    first.verification_results.append(make_test_result())
    first.evaluation_results.append(
        EvaluationResult(status=EvaluationStatus.passed, summary="Passed")
    )

    assert second.verification_results == []
    assert second.evaluation_results == []


def test_assignment_validation_rejects_invalid_updates() -> None:
    state = LocalRepairCaseState(case_id="case")
    result = make_test_result()

    with pytest.raises(ValidationError):
        state.attempt_number = 2
    with pytest.raises(ValidationError):
        result.duration_seconds = -1.0
