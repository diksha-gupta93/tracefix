from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from app.sandbox.results import TRUNCATION_MARKER, SandboxCompletion, SandboxLimits, SandboxResult


def result_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "exit_code": 0,
        "completion": SandboxCompletion.NORMAL,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "duration_seconds": 0.0,
    }
    values.update(changes)
    return values


def test_accepts_completed_passing_and_failing_results() -> None:
    passing = SandboxResult.model_validate(result_values(stdout="passed"))
    failing = SandboxResult.model_validate(
        result_values(exit_code=1, stderr="failed", duration_seconds=0.25)
    )

    assert passing.exit_code == 0
    assert failing.exit_code == 1


@pytest.mark.parametrize(
    "values",
    [
        {},
        result_values(extra=1),
    ],
)
def test_rejects_missing_and_unknown_fields(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(values)


@pytest.mark.parametrize("exit_code", [True, False, 1.0, "1"])
def test_exit_code_is_a_strict_integer(exit_code: object) -> None:
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(result_values(exit_code=exit_code))


@pytest.mark.parametrize("duration", [-0.01, inf, -inf, nan, 0, True, "0.1"])
def test_duration_is_a_strict_non_negative_finite_float(duration: object) -> None:
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(result_values(duration_seconds=duration))


@pytest.mark.parametrize("field", ["stdout", "stderr"])
def test_output_fields_are_strict_strings(field: str) -> None:
    values = result_values()
    values[field] = b"output"
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(values)


def test_limits_have_exact_defaults_and_are_serializable_and_immutable() -> None:
    limits = SandboxLimits()
    assert limits.model_dump() == {
        "cpu_count": 1.0,
        "memory_bytes": 536_870_912,
        "pids_limit": 128,
        "timeout_seconds": 120.0,
        "stdout_bytes": 1_048_576,
        "stderr_bytes": 1_048_576,
    }
    with pytest.raises(ValidationError):
        limits.cpu_count = 2.0


@pytest.mark.parametrize("field", ["cpu_count", "timeout_seconds"])
@pytest.mark.parametrize("value", [True, 1, "1", 0.0, -1.0, inf, -inf, nan])
def test_float_limits_are_strict_positive_and_finite(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SandboxLimits.model_validate({field: value})


@pytest.mark.parametrize("field", ["memory_bytes", "pids_limit", "stdout_bytes", "stderr_bytes"])
@pytest.mark.parametrize("value", [True, 1.0, "128", 0, -1])
def test_integer_limits_are_strict_and_positive(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SandboxLimits.model_validate({field: value})


def test_limits_reject_unknown_fields_and_output_smaller_than_marker() -> None:
    with pytest.raises(ValidationError):
        SandboxLimits.model_validate({"unknown": 1})
    for field in ("stdout_bytes", "stderr_bytes"):
        with pytest.raises(ValidationError):
            SandboxLimits.model_validate({field: len(TRUNCATION_MARKER) - 1})


def test_result_completion_combinations_are_enforced() -> None:
    SandboxResult.model_validate(
        result_values(exit_code=None, completion=SandboxCompletion.TIMED_OUT)
    )
    SandboxResult.model_validate(
        result_values(exit_code=None, completion=SandboxCompletion.MEMORY_LIMIT)
    )
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(result_values(exit_code=None))
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(result_values(completion=SandboxCompletion.TIMED_OUT))
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(result_values(stdout_truncated=True))
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(
            result_values(stdout=TRUNCATION_MARKER.decode(), stdout_truncated=False)
        )
