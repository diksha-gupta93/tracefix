from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

StrictInteger = Annotated[int, Field(strict=True)]


def _require_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("value must be a float")
    return value


NonNegativeFiniteFloat = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(ge=0.0, allow_inf_nan=False),
]


class SandboxResult(BaseModel):
    """The complete observable result of one pytest container execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    exit_code: StrictInteger
    stdout: str
    stderr: str
    duration_seconds: NonNegativeFiniteFloat
