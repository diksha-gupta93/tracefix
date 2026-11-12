from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import PurePosixPath

import pytest
from pydantic import ValidationError

from app.agent.routing import TerminalOutcome
from app.evaluation.benchmark import (
    DEVELOPMENT_CASE_IDS,
    BenchmarkCaseExecution,
    BenchmarkClassification,
    BenchmarkExecutionStatus,
    BenchmarkReport,
    build_benchmark_report,
    run_benchmark_cases,
)
from app.evaluation.deterministic.candidate import CandidateModelUsage
from app.evaluation.live_model import LiveBenchmarkConfiguration
from benchmarks.loader import load_development_cases

START = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)


def test_observability_does_not_change_benchmark_report_schema() -> None:
    assert not {
        "trace_id",
        "spans",
        "langsmith_project",
        "terminal_diagnostic_code",
    }.intersection(BenchmarkReport.model_fields)


def live_configuration() -> LiveBenchmarkConfiguration:
    return LiveBenchmarkConfiguration(
        request_timeout_seconds=120,
        input_price_usd_per_million_tokens=Decimal("0.15"),
        output_price_usd_per_million_tokens=Decimal("0.60"),
    )


def execution(case_id: str, outcome: TerminalOutcome) -> BenchmarkCaseExecution:
    return BenchmarkCaseExecution(
        case_id=case_id,
        terminal_outcome=outcome,
        model_usage=CandidateModelUsage(
            model_latency_seconds=0.5,
            input_tokens=10,
            output_tokens=5,
            estimated_cost_usd=Decimal("0.0000045"),
        ),
        artifact_path=PurePosixPath(".tracefix", "artifacts", f"{case_id}-{'a' * 32}.json"),
    )


def test_runner_keeps_repair_failures_and_preserves_order() -> None:
    outcomes = iter(
        (
            TerminalOutcome.SUCCESS,
            TerminalOutcome.CANDIDATE_FAILURE,
            TerminalOutcome.POLICY_VIOLATION,
            TerminalOutcome.STATIC_CHECK_FAILURE,
            TerminalOutcome.MALFORMED_MODEL_OUTPUT,
        )
    )
    cases = load_development_cases()
    results = run_benchmark_cases(
        cases, lambda case: execution(case.manifest.case_id, next(outcomes))
    )
    assert tuple(item.case_id for item in results) == DEVELOPMENT_CASE_IDS
    assert results[0].classification is BenchmarkClassification.REPAIR_SUCCESS
    assert all(
        item.classification is BenchmarkClassification.REPAIR_FAILURE for item in results[1:]
    )

    report = build_benchmark_report(
        benchmark_run_id="b" * 32,
        model="openai/gpt-oss-120b",
        prompt_version="repair-v1.0",
        started_at=START,
        completed_at=START + timedelta(seconds=1),
        live_configuration=live_configuration(),
        cases=results,
    )
    assert report.execution_status is BenchmarkExecutionStatus.COMPLETED
    assert report.counts.repair_success == 1
    assert report.counts.repair_failure == 4
    assert report.success_rate == Decimal("0.2")
    assert report.aggregate_usage.input_tokens == 50
    assert report.aggregate_usage.output_tokens == 25
    assert report.aggregate_usage.estimated_cost_usd == Decimal("0.0000225")


def test_first_execution_failure_stops_later_cases() -> None:
    calls: list[str] = []

    def runner(case: object) -> BenchmarkCaseExecution:
        case_id = case.manifest.case_id  # type: ignore[attr-defined]
        calls.append(case_id)
        outcome = (
            TerminalOutcome.INFRASTRUCTURE_FAILURE if len(calls) == 2 else TerminalOutcome.SUCCESS
        )
        return execution(case_id, outcome)

    results = run_benchmark_cases(load_development_cases(), runner)
    assert calls == list(DEVELOPMENT_CASE_IDS[:2])
    assert results[1].classification is BenchmarkClassification.EXECUTION_FAILURE
    assert all(item.classification is BenchmarkClassification.NOT_RUN for item in results[2:])

    report = build_benchmark_report(
        benchmark_run_id="c" * 32,
        model="openai/gpt-oss-120b",
        prompt_version="repair-v1.0",
        started_at=START,
        completed_at=START + timedelta(seconds=1),
        live_configuration=live_configuration(),
        cases=results,
    )
    assert report.execution_status is BenchmarkExecutionStatus.EXECUTION_FAILURE
    assert report.counts.attempted == 2
    assert report.counts.not_run == 3
    assert report.counts.execution_failure == 1
    assert report.success_rate == Decimal(1)


def test_runner_maps_case_runner_exception_to_safe_execution_failure() -> None:
    def runner(case: object) -> BenchmarkCaseExecution:
        del case
        raise RuntimeError("secret provider body")

    results = run_benchmark_cases(load_development_cases(), runner)
    assert results[0].classification is BenchmarkClassification.EXECUTION_FAILURE
    assert results[0].terminal_outcome is None
    assert all(item.classification is BenchmarkClassification.NOT_RUN for item in results[1:])
    assert "secret provider body" not in repr(results)


def test_report_is_strict_frozen_and_rejects_inconsistent_counts() -> None:
    results = run_benchmark_cases(
        load_development_cases(),
        lambda case: execution(case.manifest.case_id, TerminalOutcome.SUCCESS),
    )
    report = build_benchmark_report(
        benchmark_run_id="d" * 32,
        model="openai/gpt-oss-120b",
        prompt_version="repair-v1.0",
        started_at=START,
        completed_at=START + timedelta(seconds=1),
        live_configuration=live_configuration(),
        cases=results,
    )
    with pytest.raises(ValidationError):
        report.counts = report.counts.model_copy(update={"attempted": 0})
    with pytest.raises(ValidationError):
        BenchmarkReport.model_validate(
            {
                **report.model_dump(warnings="none"),
                "counts": {**report.counts.model_dump(), "attempted": 4},
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        BenchmarkReport.model_validate(
            {**report.model_dump(warnings="none"), "extra": True}, strict=True
        )
    tampered_cases = list(report.model_dump(warnings="none")["cases"])
    tampered_cases[0] = {**tampered_cases[0], "estimated_cost_usd": Decimal("99")}
    with pytest.raises(ValidationError, match="case cost"):
        BenchmarkReport.model_validate(
            {**report.model_dump(warnings="none"), "cases": tuple(tampered_cases)}, strict=True
        )


def test_case_execution_rejects_unsafe_artifact_path() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCaseExecution(
            case_id="boundary_condition",
            terminal_outcome=TerminalOutcome.SUCCESS,
            artifact_path=PurePosixPath("..", "secret.json"),
        )


def test_repair_result_requires_complete_usage_and_matching_artifact_identity() -> None:
    cases = load_development_cases()

    def missing_usage(case: object) -> BenchmarkCaseExecution:
        case_id = case.manifest.case_id  # type: ignore[attr-defined]
        return BenchmarkCaseExecution(
            case_id=case_id,
            terminal_outcome=TerminalOutcome.SUCCESS,
            artifact_path=PurePosixPath(".tracefix", "artifacts", f"{case_id}-{'a' * 32}.json"),
        )

    results = run_benchmark_cases(cases, missing_usage)
    assert results[0].classification is BenchmarkClassification.EXECUTION_FAILURE
    assert all(item.classification is BenchmarkClassification.NOT_RUN for item in results[1:])

    with pytest.raises(ValidationError, match="artifact path"):
        BenchmarkCaseExecution(
            case_id="boundary_condition",
            terminal_outcome=TerminalOutcome.SUCCESS,
            model_usage=execution("boundary_condition", TerminalOutcome.SUCCESS).model_usage,
            artifact_path=PurePosixPath(
                ".tracefix", "artifacts", f"exception_handling-{'a' * 32}.json"
            ),
        )
