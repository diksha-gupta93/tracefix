from __future__ import annotations

import os
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import NoReturn, Protocol

from pydantic import ValidationError

from app.agent.schemas import (
    ModelConfiguration,
    ModelDiagnostic,
    ModelDiagnosticCode,
    ModelDiagnosticStage,
    ModelProviderRequest,
    ModelProviderResponse,
)

SUPPORTED_PROMPT_VERSION = "repair-v1.0"
_REQUIRED_ENVIRONMENT_VARIABLES = (
    "MODEL_PROVIDER",
    "MODEL_NAME",
    "MODEL_TEMPERATURE",
    "MODEL_MAX_TOKENS",
    "PROMPT_VERSION",
)
_DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_INTEGER = re.compile(r"^[0-9]+$")
_BOOLEAN_LIKE = frozenset({"true", "false", "yes", "no", "on", "off"})


class ModelProvider(Protocol):
    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse: ...


class ModelDiagnosticError(Exception):
    def __init__(self, diagnostic: ModelDiagnostic) -> None:
        super().__init__(f"{diagnostic.stage.value}:{diagnostic.code.value}")
        self.diagnostic = diagnostic


def _raise_configuration_error(code: ModelDiagnosticCode, message: str) -> NoReturn:
    raise ModelDiagnosticError(
        ModelDiagnostic(
            stage=ModelDiagnosticStage.CONFIGURATION,
            code=code,
            message=message,
        )
    )


def _parse_temperature(value: object) -> float:
    if (
        type(value) is not str
        or value.casefold() in _BOOLEAN_LIKE
        or _DECIMAL.fullmatch(value) is None
    ):
        raise ValueError("temperature is not a strict decimal")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("temperature is not a strict decimal") from error
    if not parsed.is_finite() or not Decimal("0.0") <= parsed <= Decimal("2.0"):
        raise ValueError("temperature is outside the supported range")
    return float(parsed)


def _parse_max_output_tokens(value: object) -> int:
    if (
        type(value) is not str
        or value.casefold() in _BOOLEAN_LIKE
        or _INTEGER.fullmatch(value) is None
    ):
        raise ValueError("maximum output tokens is not a strict integer")
    parsed = int(value, 10)
    if not 1 <= parsed <= 32_768:
        raise ValueError("maximum output tokens is outside the supported range")
    return parsed


def load_model_configuration(
    environment: Mapping[str, str] | None = None,
) -> ModelConfiguration:
    source = os.environ if environment is None else environment
    if any(name not in source for name in _REQUIRED_ENVIRONMENT_VARIABLES):
        _raise_configuration_error(
            ModelDiagnosticCode.MISSING_CONFIGURATION,
            "required model configuration is missing",
        )
    configuration: ModelConfiguration | None = None
    try:
        provider = source["MODEL_PROVIDER"]
        model_name = source["MODEL_NAME"]
        prompt_version = source["PROMPT_VERSION"]
        configuration = ModelConfiguration(
            provider=provider,
            model_name=model_name,
            temperature=_parse_temperature(source["MODEL_TEMPERATURE"]),
            max_output_tokens=_parse_max_output_tokens(source["MODEL_MAX_TOKENS"]),
            prompt_version=prompt_version,
        )
    except (InvalidOperation, KeyError, TypeError, ValueError, ValidationError):
        pass
    if configuration is None:
        _raise_configuration_error(
            ModelDiagnosticCode.INVALID_CONFIGURATION,
            "model configuration is invalid",
        )
    if configuration.prompt_version != SUPPORTED_PROMPT_VERSION:
        _raise_configuration_error(
            ModelDiagnosticCode.UNSUPPORTED_PROMPT_VERSION,
            "model prompt version is unsupported",
        )
    return configuration
