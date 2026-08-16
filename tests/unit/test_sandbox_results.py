from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from app.sandbox.results import SandboxResult


def test_accepts_completed_passing_and_failing_results() -> None:
    passing = SandboxResult(exit_code=0, stdout="passed", stderr="", duration_seconds=0.0)
    failing = SandboxResult(exit_code=1, stdout="", stderr="failed", duration_seconds=0.25)

    assert passing.exit_code == 0
    assert failing.exit_code == 1


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"exit_code": 0, "stdout": "", "stderr": "", "duration_seconds": 0.0, "extra": 1},
    ],
)
def test_rejects_missing_and_unknown_fields(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(values)


@pytest.mark.parametrize("exit_code", [True, False, 1.0, "1"])
def test_exit_code_is_a_strict_integer(exit_code: object) -> None:
    with pytest.raises(ValidationError):
        SandboxResult(exit_code=exit_code, stdout="", stderr="", duration_seconds=0.0)


@pytest.mark.parametrize("duration", [-0.01, inf, -inf, nan, 0, True, "0.1"])
def test_duration_is_a_strict_non_negative_finite_float(duration: object) -> None:
    with pytest.raises(ValidationError):
        SandboxResult(exit_code=0, stdout="", stderr="", duration_seconds=duration)


@pytest.mark.parametrize("field", ["stdout", "stderr"])
def test_output_fields_are_strict_strings(field: str) -> None:
    values: dict[str, object] = {
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
    }
    values[field] = b"output"
    with pytest.raises(ValidationError):
        SandboxResult.model_validate(values)
