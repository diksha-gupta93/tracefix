from __future__ import annotations

import traceback
from math import inf

import pytest
from pydantic import ValidationError

from app.agent.model import (
    ModelDiagnosticError,
    ModelProvider,
    load_model_configuration,
)
from app.agent.schemas import (
    ModelConfiguration,
    ModelDiagnostic,
    ModelDiagnosticCode,
    ModelDiagnosticStage,
    ModelOperation,
    ModelProviderRequest,
    ModelProviderResponse,
)


def valid_environment() -> dict[str, str]:
    return {
        "MODEL_PROVIDER": "test-provider",
        "MODEL_NAME": "test-model",
        "MODEL_TEMPERATURE": "0.25",
        "MODEL_MAX_TOKENS": "4096",
        "PROMPT_VERSION": "repair-v1.0",
    }


def test_loads_complete_configuration_without_defaults() -> None:
    configuration = load_model_configuration(valid_environment())

    assert configuration == ModelConfiguration(
        provider="test-provider",
        model_name="test-model",
        temperature=0.25,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
    )


@pytest.mark.parametrize("missing", tuple(valid_environment()))
def test_missing_configuration_has_stable_safe_diagnostic(missing: str) -> None:
    environment = valid_environment()
    del environment[missing]

    with pytest.raises(ModelDiagnosticError) as raised:
        load_model_configuration(environment)

    assert raised.value.diagnostic == ModelDiagnostic(
        stage=ModelDiagnosticStage.CONFIGURATION,
        code=ModelDiagnosticCode.MISSING_CONFIGURATION,
        message="required model configuration is missing",
    )
    assert missing not in str(raised.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MODEL_PROVIDER", ""),
        ("MODEL_PROVIDER", " \t "),
        ("MODEL_NAME", ""),
        ("MODEL_TEMPERATURE", ""),
        ("MODEL_TEMPERATURE", "true"),
        ("MODEL_TEMPERATURE", "False"),
        ("MODEL_TEMPERATURE", "NaN"),
        ("MODEL_TEMPERATURE", "Infinity"),
        ("MODEL_TEMPERATURE", "-0.1"),
        ("MODEL_TEMPERATURE", "+1.0"),
        ("MODEL_TEMPERATURE", "1e0"),
        ("MODEL_TEMPERATURE", "2.01"),
        ("MODEL_MAX_TOKENS", ""),
        ("MODEL_MAX_TOKENS", "true"),
        ("MODEL_MAX_TOKENS", "False"),
        ("MODEL_MAX_TOKENS", "+1"),
        ("MODEL_MAX_TOKENS", "-1"),
        ("MODEL_MAX_TOKENS", "1.0"),
        ("MODEL_MAX_TOKENS", "1e3"),
        ("MODEL_MAX_TOKENS", "0"),
        ("MODEL_MAX_TOKENS", "32769"),
        ("PROMPT_VERSION", ""),
    ],
)
def test_invalid_configuration_has_stable_safe_diagnostic(name: str, value: str) -> None:
    environment = valid_environment()
    environment[name] = value

    with pytest.raises(ModelDiagnosticError) as raised:
        load_model_configuration(environment)

    assert raised.value.diagnostic.code is ModelDiagnosticCode.INVALID_CONFIGURATION
    assert raised.value.diagnostic.stage is ModelDiagnosticStage.CONFIGURATION
    if value:
        assert value not in str(raised.value)


def test_invalid_configuration_does_not_retain_the_source_exception() -> None:
    sentinel = "secret-invalid-temperature"
    environment = valid_environment()
    environment["MODEL_TEMPERATURE"] = sentinel

    with pytest.raises(ModelDiagnosticError) as raised:
        load_model_configuration(environment)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(raised.value))


def test_unsupported_prompt_version_is_distinct_and_not_disclosed() -> None:
    environment = valid_environment()
    environment["PROMPT_VERSION"] = "secret-experimental-version"

    with pytest.raises(ModelDiagnosticError) as raised:
        load_model_configuration(environment)

    assert raised.value.diagnostic.code is ModelDiagnosticCode.UNSUPPORTED_PROMPT_VERSION
    assert "secret-experimental-version" not in str(raised.value)


def test_loader_ignores_unrelated_environment_values() -> None:
    environment = valid_environment()
    environment["API_KEY"] = "must-not-be-read-or-retained"
    environment["UNRELATED"] = "value"

    serialized = load_model_configuration(environment).model_dump_json()

    assert "must-not-be-read-or-retained" not in serialized
    assert "UNRELATED" not in serialized


@pytest.mark.parametrize(
    "environment",
    [
        {**valid_environment(), "MODEL_TEMPERATURE": "0"},
        {**valid_environment(), "MODEL_TEMPERATURE": "2.0"},
        {**valid_environment(), "MODEL_MAX_TOKENS": "1"},
        {**valid_environment(), "MODEL_MAX_TOKENS": "32768"},
    ],
)
def test_configuration_accepts_inclusive_numeric_boundaries(
    environment: dict[str, str],
) -> None:
    load_model_configuration(environment)


def test_typed_provider_protocol_and_request_round_trip() -> None:
    class Provider:
        def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
            assert request.operation is ModelOperation.REPAIR_PLANNING
            return ModelProviderResponse(raw_json='{"result":"ok"}')

    provider: ModelProvider = Provider()
    request = ModelProviderRequest(
        operation=ModelOperation.REPAIR_PLANNING,
        model_name="test-model",
        temperature=0.0,
        max_output_tokens=128,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="prompt",
    )

    restored = ModelProviderRequest.model_validate_json(request.model_dump_json(), strict=True)

    assert restored == request
    assert provider.complete(restored).raw_json == '{"result":"ok"}'


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (
            ModelConfiguration,
            {
                "provider": "provider",
                "model_name": "model",
                "temperature": 0.0,
                "max_output_tokens": 1,
                "prompt_version": "repair-v1.0",
                "extra": "rejected",
            },
        ),
        (
            ModelProviderRequest,
            {
                "operation": ModelOperation.REPAIR_PLANNING,
                "model_name": "model",
                "temperature": 0.0,
                "max_output_tokens": 1,
                "prompt_version": "repair-v1.0",
                "attempt_number": 1,
                "rendered_prompt": "prompt",
                "extra": "rejected",
            },
        ),
        (ModelProviderResponse, {"raw_json": "{}", "extra": "rejected"}),
        (
            ModelDiagnostic,
            {
                "stage": ModelDiagnosticStage.PLANNING,
                "code": ModelDiagnosticCode.PROVIDER_EXCEPTION,
                "message": "safe",
                "extra": "rejected",
            },
        ),
    ],
)
def test_new_boundary_models_forbid_extra_fields(
    model: type[ModelConfiguration]
    | type[ModelProviderRequest]
    | type[ModelProviderResponse]
    | type[ModelDiagnostic],
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0),
        ("temperature", True),
        ("temperature", "0.0"),
        ("temperature", inf),
        ("max_output_tokens", True),
        ("max_output_tokens", 1.0),
        ("max_output_tokens", "1"),
        ("attempt_number", 0),
        ("attempt_number", 2),
        ("attempt_number", True),
    ],
)
def test_request_numeric_fields_are_strict(field: str, value: object) -> None:
    values: dict[str, object] = {
        "operation": ModelOperation.REPAIR_PLANNING,
        "model_name": "model",
        "temperature": 0.0,
        "max_output_tokens": 1,
        "prompt_version": "repair-v1.0",
        "attempt_number": 1,
        "rendered_prompt": "prompt",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        ModelProviderRequest.model_validate(values)


def test_new_boundary_models_are_frozen() -> None:
    configuration = load_model_configuration(valid_environment())
    request = ModelProviderRequest(
        operation=ModelOperation.REPAIR_PLANNING,
        model_name="model",
        temperature=0.0,
        max_output_tokens=1,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="prompt",
    )
    response = ModelProviderResponse(raw_json="{}")
    diagnostic = ModelDiagnostic(
        stage=ModelDiagnosticStage.PLANNING,
        code=ModelDiagnosticCode.MALFORMED_STRUCTURED_OUTPUT,
        message="safe",
    )

    with pytest.raises(ValidationError):
        configuration.model_name = "changed"
    with pytest.raises(ValidationError):
        request.rendered_prompt = "changed"
    with pytest.raises(ValidationError):
        response.raw_json = "changed"
    with pytest.raises(ValidationError):
        diagnostic.message = "changed"


def test_boundary_models_strict_json_round_trip() -> None:
    configuration = load_model_configuration(valid_environment())
    request = ModelProviderRequest(
        operation=ModelOperation.PATCH_GENERATION,
        model_name="model-ü",
        temperature=0.25,
        max_output_tokens=128,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="prompt-ü",
    )
    response = ModelProviderResponse(raw_json='{"summary":"ü"}')
    diagnostic = ModelDiagnostic(
        stage=ModelDiagnosticStage.PATCH_GENERATION,
        code=ModelDiagnosticCode.INVALID_PATCH_ENVELOPE,
        message="safe diagnostic",
    )

    assert (
        ModelConfiguration.model_validate_json(configuration.model_dump_json(), strict=True)
        == configuration
    )
    assert (
        ModelProviderRequest.model_validate_json(request.model_dump_json(), strict=True) == request
    )
    assert (
        ModelProviderResponse.model_validate_json(response.model_dump_json(), strict=True)
        == response
    )
    restored_diagnostic = ModelDiagnostic.model_validate_json(
        diagnostic.model_dump_json(), strict=True
    )
    assert restored_diagnostic == diagnostic
    assert restored_diagnostic.stage is ModelDiagnosticStage.PATCH_GENERATION
    assert restored_diagnostic.code is ModelDiagnosticCode.INVALID_PATCH_ENVELOPE


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (ModelProviderResponse, {"raw_json": 1}),
        (
            ModelDiagnostic,
            {
                "stage": "unknown",
                "code": ModelDiagnosticCode.INVALID_INPUT_STATE,
                "message": "safe",
            },
        ),
        (
            ModelDiagnostic,
            {
                "stage": ModelDiagnosticStage.PLANNING,
                "code": "unknown",
                "message": "safe",
            },
        ),
    ],
)
def test_boundary_models_reject_wrong_types_and_unknown_enums(
    model: type[ModelProviderResponse] | type[ModelDiagnostic],
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)
