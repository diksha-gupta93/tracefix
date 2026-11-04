from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.agent.routing import TerminalOutcome
from app.evaluation.deterministic.candidate import (
    CandidateModelUsage,
    NonNegativeUsdDecimal,
    StrictTokenCount,
)
from app.evaluation.live_model import (
    LiveBenchmarkConfiguration,
    calculate_estimated_cost_usd,
)
from benchmarks.loader import TrustedCase

BENCHMARK_SCHEMA_VERSION: Literal["tracefix-live-benchmark-v1"] = "tracefix-live-benchmark-v1"
DEVELOPMENT_SPLIT: Literal["development"] = "development"
OPENAI_REPORT_PROVIDER: Literal["openai"] = "openai"
DEVELOPMENT_CASE_IDS = (
    "boundary_condition",
    "exception_handling",
    "fixture_or_mocking",
    "incorrect_conditional",
    "incorrect_return_value",
)
_REPAIR_FAILURES = frozenset(
    {
        TerminalOutcome.CANDIDATE_FAILURE,
        TerminalOutcome.POLICY_VIOLATION,
        TerminalOutcome.STATIC_CHECK_FAILURE,
        TerminalOutcome.MALFORMED_MODEL_OUTPUT,
    }
)


class BenchmarkClassification(StrEnum):
    REPAIR_SUCCESS = "repair_success"
    REPAIR_FAILURE = "repair_failure"
    EXECUTION_FAILURE = "execution_failure"
    NOT_RUN = "not_run"


class BenchmarkExecutionStatus(StrEnum):
    COMPLETED = "completed"
    EXECUTION_FAILURE = "execution_failure"


def _artifact_path(value: PurePosixPath, case_id: str) -> None:
    rendered = value.as_posix()
    windows = PureWindowsPath(rendered)
    if (
        value.is_absolute()
        or windows.drive
        or windows.root
        or value.parts[:2] != (".tracefix", "artifacts")
        or len(value.parts) != 3
        or any(part in {"", ".", ".."} for part in value.parts)
        or "\\" in rendered
        or "\x00" in rendered
        or re.fullmatch(rf"{re.escape(case_id)}-[0-9a-f]{{32}}\.json", value.name) is None
    ):
        raise ValueError("artifact path is unsafe")


class BenchmarkCaseExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    terminal_outcome: TerminalOutcome
    model_usage: CandidateModelUsage | None = None
    artifact_path: PurePosixPath

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _artifact_path(self.artifact_path, self.case_id)
        return self


class BenchmarkCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    classification: BenchmarkClassification
    terminal_outcome: TerminalOutcome | None = None
    model_latency_seconds: (
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None
    ) = None
    input_tokens: StrictTokenCount | None = None
    output_tokens: StrictTokenCount | None = None
    estimated_cost_usd: NonNegativeUsdDecimal | None = None
    artifact_path: PurePosixPath | None = None

    @model_validator(mode="after")
    def validate_classification(self) -> Self:
        measurements = (
            self.model_latency_seconds,
            self.input_tokens,
            self.output_tokens,
            self.estimated_cost_usd,
        )
        if any(item is None for item in measurements) and any(
            item is not None for item in measurements
        ):
            raise ValueError("case measurements must be all present or all absent")
        if self.artifact_path is not None:
            _artifact_path(self.artifact_path, self.case_id)
        if self.classification is BenchmarkClassification.NOT_RUN:
            if self.terminal_outcome is not None or any(
                item is not None for item in (*measurements, self.artifact_path)
            ):
                raise ValueError("not-run case cannot contain execution evidence")
        elif self.classification is BenchmarkClassification.REPAIR_SUCCESS:
            if (
                self.terminal_outcome is not TerminalOutcome.SUCCESS
                or self.artifact_path is None
                or any(item is None for item in measurements)
            ):
                raise ValueError("repair success evidence is inconsistent")
        elif self.classification is BenchmarkClassification.REPAIR_FAILURE:
            if (
                self.terminal_outcome not in _REPAIR_FAILURES
                or self.artifact_path is None
                or any(item is None for item in measurements)
            ):
                raise ValueError("repair failure evidence is inconsistent")
        elif self.terminal_outcome is not None and self.terminal_outcome not in {
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            TerminalOutcome.BASELINE_NOT_REPRODUCIBLE,
        }:
            raise ValueError("execution failure outcome is inconsistent")
        return self


class BenchmarkCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    attempted: int = Field(strict=True, ge=0, le=5)
    not_run: int = Field(strict=True, ge=0, le=5)
    repair_success: int = Field(strict=True, ge=0, le=5)
    repair_failure: int = Field(strict=True, ge=0, le=5)
    execution_failure: int = Field(strict=True, ge=0, le=1)


class BenchmarkAggregateUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model_latency_seconds: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    input_tokens: StrictTokenCount
    output_tokens: StrictTokenCount
    estimated_cost_usd: NonNegativeUsdDecimal


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["tracefix-live-benchmark-v1"] = BENCHMARK_SCHEMA_VERSION
    benchmark_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    split: Literal["development"] = DEVELOPMENT_SPLIT
    provider: Literal["openai"] = OPENAI_REPORT_PROVIDER
    model: str = Field(min_length=1)
    prompt_version: Literal["repair-v1.0"] = "repair-v1.0"
    max_attempts: Literal[1] = 1
    started_at: AwareDatetime
    completed_at: AwareDatetime
    execution_status: BenchmarkExecutionStatus
    request_timeout_seconds: int = Field(strict=True, ge=1, le=300)
    input_price_usd_per_million_tokens: NonNegativeUsdDecimal
    output_price_usd_per_million_tokens: NonNegativeUsdDecimal
    cases: tuple[BenchmarkCaseResult, ...]
    counts: BenchmarkCounts
    success_rate: NonNegativeUsdDecimal | None
    aggregate_usage: BenchmarkAggregateUsage

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if (
            self.started_at.utcoffset() != UTC.utcoffset(self.started_at)
            or self.completed_at.utcoffset() != UTC.utcoffset(self.completed_at)
            or self.completed_at < self.started_at
        ):
            raise ValueError("benchmark timestamps are invalid")
        if tuple(item.case_id for item in self.cases) != DEVELOPMENT_CASE_IDS:
            raise ValueError("benchmark cases are not the development split")
        actual = {classification: 0 for classification in BenchmarkClassification}
        latency = 0.0
        input_tokens = 0
        output_tokens = 0
        cost = Decimal(0)
        for item in self.cases:
            actual[item.classification] += 1
            if item.model_latency_seconds is not None:
                latency += item.model_latency_seconds
                input_tokens += item.input_tokens or 0
                output_tokens += item.output_tokens or 0
                cost += item.estimated_cost_usd or Decimal(0)
                expected_cost = calculate_estimated_cost_usd(
                    item.input_tokens or 0,
                    item.output_tokens or 0,
                    LiveBenchmarkConfiguration(
                        request_timeout_seconds=self.request_timeout_seconds,
                        input_price_usd_per_million_tokens=(
                            self.input_price_usd_per_million_tokens
                        ),
                        output_price_usd_per_million_tokens=(
                            self.output_price_usd_per_million_tokens
                        ),
                    ),
                )
                if item.estimated_cost_usd != expected_cost:
                    raise ValueError("benchmark case cost is inconsistent")
        expected_counts = BenchmarkCounts(
            attempted=len(self.cases) - actual[BenchmarkClassification.NOT_RUN],
            not_run=actual[BenchmarkClassification.NOT_RUN],
            repair_success=actual[BenchmarkClassification.REPAIR_SUCCESS],
            repair_failure=actual[BenchmarkClassification.REPAIR_FAILURE],
            execution_failure=actual[BenchmarkClassification.EXECUTION_FAILURE],
        )
        if self.counts != expected_counts:
            raise ValueError("benchmark counts are inconsistent")
        completed_repairs = self.counts.repair_success + self.counts.repair_failure
        expected_rate = (
            None
            if completed_repairs == 0
            else Decimal(self.counts.repair_success) / Decimal(completed_repairs)
        )
        if self.success_rate != expected_rate:
            raise ValueError("benchmark success rate is inconsistent")
        if not math.isfinite(latency) or self.aggregate_usage != BenchmarkAggregateUsage(
            model_latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        ):
            raise ValueError("benchmark aggregate usage is inconsistent")
        expected_status = (
            BenchmarkExecutionStatus.EXECUTION_FAILURE
            if self.counts.execution_failure
            else BenchmarkExecutionStatus.COMPLETED
        )
        if self.execution_status is not expected_status:
            raise ValueError("benchmark execution status is inconsistent")
        failure_positions = [
            index
            for index, item in enumerate(self.cases)
            if item.classification is BenchmarkClassification.EXECUTION_FAILURE
        ]
        if failure_positions and any(
            item.classification is not BenchmarkClassification.NOT_RUN
            for item in self.cases[failure_positions[0] + 1 :]
        ):
            raise ValueError("benchmark did not fail fast")
        return self


CaseRunner = Callable[[TrustedCase], BenchmarkCaseExecution]


def _case_result(execution: BenchmarkCaseExecution) -> BenchmarkCaseResult:
    if execution.terminal_outcome is TerminalOutcome.SUCCESS:
        classification = BenchmarkClassification.REPAIR_SUCCESS
    elif execution.terminal_outcome in _REPAIR_FAILURES:
        classification = BenchmarkClassification.REPAIR_FAILURE
    else:
        classification = BenchmarkClassification.EXECUTION_FAILURE
    usage = execution.model_usage
    return BenchmarkCaseResult(
        case_id=execution.case_id,
        classification=classification,
        terminal_outcome=execution.terminal_outcome,
        model_latency_seconds=None if usage is None else usage.model_latency_seconds,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        estimated_cost_usd=None if usage is None else usage.estimated_cost_usd,
        artifact_path=execution.artifact_path,
    )


def run_benchmark_cases(
    cases: Sequence[TrustedCase], case_runner: CaseRunner
) -> tuple[BenchmarkCaseResult, ...]:
    identities = tuple(case.manifest.case_id for case in cases)
    if identities != DEVELOPMENT_CASE_IDS:
        raise ValueError("development cases are invalid or unordered")
    results: list[BenchmarkCaseResult] = []
    failed = False
    for case in cases:
        if failed:
            results.append(
                BenchmarkCaseResult(
                    case_id=case.manifest.case_id,
                    classification=BenchmarkClassification.NOT_RUN,
                )
            )
            continue
        try:
            execution = case_runner(case)
            checked = BenchmarkCaseExecution.model_validate(
                execution.model_dump(warnings="none"), strict=True
            )
            if checked.case_id != case.manifest.case_id:
                raise ValueError
            result = _case_result(checked)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            result = BenchmarkCaseResult(
                case_id=case.manifest.case_id,
                classification=BenchmarkClassification.EXECUTION_FAILURE,
            )
        results.append(result)
        failed = result.classification is BenchmarkClassification.EXECUTION_FAILURE
    return tuple(results)


def build_benchmark_report(
    *,
    benchmark_run_id: str,
    model: str,
    prompt_version: str,
    started_at: datetime,
    completed_at: datetime,
    live_configuration: LiveBenchmarkConfiguration,
    cases: tuple[BenchmarkCaseResult, ...],
) -> BenchmarkReport:
    if prompt_version != "repair-v1.0":
        raise ValueError("benchmark prompt version is unsupported")
    checked_prompt: Literal["repair-v1.0"] = "repair-v1.0"
    success = sum(item.classification is BenchmarkClassification.REPAIR_SUCCESS for item in cases)
    repair_failure = sum(
        item.classification is BenchmarkClassification.REPAIR_FAILURE for item in cases
    )
    execution_failure = sum(
        item.classification is BenchmarkClassification.EXECUTION_FAILURE for item in cases
    )
    not_run = sum(item.classification is BenchmarkClassification.NOT_RUN for item in cases)
    latency = sum(item.model_latency_seconds or 0.0 for item in cases)
    input_tokens = sum(item.input_tokens or 0 for item in cases)
    output_tokens = sum(item.output_tokens or 0 for item in cases)
    cost = sum((item.estimated_cost_usd or Decimal(0) for item in cases), start=Decimal(0))
    completed_repairs = success + repair_failure
    return BenchmarkReport(
        benchmark_run_id=benchmark_run_id,
        model=model,
        prompt_version=checked_prompt,
        started_at=started_at,
        completed_at=completed_at,
        execution_status=(
            BenchmarkExecutionStatus.EXECUTION_FAILURE
            if execution_failure
            else BenchmarkExecutionStatus.COMPLETED
        ),
        request_timeout_seconds=live_configuration.request_timeout_seconds,
        input_price_usd_per_million_tokens=(live_configuration.input_price_usd_per_million_tokens),
        output_price_usd_per_million_tokens=(
            live_configuration.output_price_usd_per_million_tokens
        ),
        cases=cases,
        counts=BenchmarkCounts(
            attempted=len(cases) - not_run,
            not_run=not_run,
            repair_success=success,
            repair_failure=repair_failure,
            execution_failure=execution_failure,
        ),
        success_rate=(
            None if completed_repairs == 0 else Decimal(success) / Decimal(completed_repairs)
        ),
        aggregate_usage=BenchmarkAggregateUsage(
            model_latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        ),
    )
