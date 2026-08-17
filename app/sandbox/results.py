from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

TRUNCATION_MARKER = b"\n[tracefix output truncated]\n"

StrictInteger = Annotated[int, Field(strict=True)]
PositiveStrictInteger = Annotated[int, Field(strict=True, gt=0)]


def _require_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("value must be a float")
    return value


NonNegativeFiniteFloat = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(ge=0.0, allow_inf_nan=False),
]
PositiveFiniteFloat = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(gt=0.0, allow_inf_nan=False),
]


class SandboxLimits(BaseModel):
    """Immutable resource and returned-output limits for one execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    cpu_count: PositiveFiniteFloat = 1.0
    memory_bytes: PositiveStrictInteger = 536_870_912
    pids_limit: PositiveStrictInteger = 128
    timeout_seconds: PositiveFiniteFloat = 120.0
    stdout_bytes: PositiveStrictInteger = 1_048_576
    stderr_bytes: PositiveStrictInteger = 1_048_576

    @model_validator(mode="after")
    def marker_must_fit(self) -> Self:
        minimum = len(TRUNCATION_MARKER)
        if self.stdout_bytes < minimum or self.stderr_bytes < minimum:
            raise ValueError("output limits must contain the complete truncation marker")
        return self


class SandboxCompletion(StrEnum):
    NORMAL = "normal"
    TIMED_OUT = "timed_out"
    MEMORY_LIMIT = "memory_limit"


class SandboxResult(BaseModel):
    """The complete observable result of one pytest container execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    exit_code: StrictInteger | None
    completion: SandboxCompletion
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def valid_completion_state(self) -> Self:
        if self.completion is SandboxCompletion.NORMAL and self.exit_code is None:
            raise ValueError("normal completion requires an exit code")
        if self.completion is SandboxCompletion.TIMED_OUT and self.exit_code is not None:
            raise ValueError("timed-out completion must not claim an exit code")
        marker = TRUNCATION_MARKER.decode("ascii")
        if self.stdout.endswith(marker) is not self.stdout_truncated:
            raise ValueError("stdout truncation metadata is inconsistent")
        if self.stderr.endswith(marker) is not self.stderr_truncated:
            raise ValueError("stderr truncation metadata is inconsistent")
        if not isfinite(self.duration_seconds):
            raise ValueError("duration must be finite")
        return self
