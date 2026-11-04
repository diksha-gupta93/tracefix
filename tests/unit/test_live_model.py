from __future__ import annotations

from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.schemas import ModelConfiguration, ModelOperation, ModelProviderRequest
from app.evaluation.live_model import (
    LiveBenchmarkConfiguration,
    LiveModelConfigurationError,
    LiveModelProviderError,
    OpenAICompletionStatus,
    OpenAIModelProvider,
    OpenAITransportRequest,
    OpenAITransportResponse,
    SdkOpenAITransport,
    apply_live_model_configuration,
    calculate_estimated_cost_usd,
    create_openai_model_provider,
    load_live_benchmark_configuration,
)


def environment() -> dict[str, str]:
    return {
        "MODEL_REQUEST_TIMEOUT_SECONDS": "120",
        "MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS": "0.15",
        "MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS": "0.60",
        "OPENAI_API_KEY": "credential-must-not-escape",
    }


def model_configuration() -> ModelConfiguration:
    return ModelConfiguration(
        provider="openai",
        model_name="reviewed-model-name",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
    )


def live_configuration() -> LiveBenchmarkConfiguration:
    return load_live_benchmark_configuration(environment())


class RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[OpenAITransportRequest] = []
        self.responses = [
            OpenAITransportResponse(
                status=OpenAICompletionStatus.COMPLETED,
                raw_json='{"autonomous_repair_suitable":false,"files_expected_to_change":[],"intended_behavioural_correction":"none","risks":[],"suspected_root_cause":"unknown","validation_strategy":"none"}',
                input_tokens=10,
                output_tokens=5,
            ),
            OpenAITransportResponse(
                status=OpenAICompletionStatus.CONTENT_FAILURE,
                raw_json="{}",
                input_tokens=20,
                output_tokens=8,
            ),
        ]

    def complete(self, request: OpenAITransportRequest) -> OpenAITransportResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def request(operation: ModelOperation) -> ModelProviderRequest:
    return ModelProviderRequest(
        case_id="boundary_condition",
        operation=operation,
        model_name="reviewed-model-name",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="safe rendered prompt",
    )


def transport_request() -> OpenAITransportRequest:
    return OpenAITransportRequest(
        operation=ModelOperation.REPAIR_PLANNING,
        case_id="boundary_condition",
        model_name="reviewed-model-name",
        temperature=0.0,
        max_output_tokens=4096,
        rendered_prompt="prompt",
        prompt_version="repair-v1.0",
        attempt_number=1,
        response_schema_json='{"type":"object","properties":{},"additionalProperties":false}',
        request_timeout_seconds=45,
    )


def test_live_configuration_is_strict_and_exact() -> None:
    configuration = live_configuration()
    assert configuration.request_timeout_seconds == 120
    assert configuration.input_price_usd_per_million_tokens == Decimal("0.15")
    assert configuration.output_price_usd_per_million_tokens == Decimal("0.60")

    for name, value in (
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "0"),
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "301"),
        ("MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS", "nan"),
        ("MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS", "-1"),
    ):
        invalid = environment()
        invalid[name] = value
        with pytest.raises(LiveModelConfigurationError):
            load_live_benchmark_configuration(invalid)
    for timeout in ("1", "300"):
        valid = environment()
        valid["MODEL_REQUEST_TIMEOUT_SECONDS"] = timeout
        valid["UNRELATED_SECRET"] = "ignored"
        assert load_live_benchmark_configuration(valid).request_timeout_seconds == int(timeout)


def test_live_model_configuration_requires_exact_profile() -> None:
    configured = apply_live_model_configuration(model_configuration(), "openai", environment())
    assert configured == model_configuration()

    with pytest.raises(LiveModelConfigurationError):
        apply_live_model_configuration(model_configuration(), "OpenAI", environment())
    configured_alias = apply_live_model_configuration(
        model_configuration().model_copy(update={"model_name": "reviewed-other-model"}),
        "openai",
        environment(),
    )
    assert configured_alias.model_name == "reviewed-other-model"


def test_adapter_maps_request_measures_usage_and_calculates_exact_cost() -> None:
    transport = RecordingTransport()
    clock_values = iter((10.0, 10.25, 20.0, 20.75))
    provider = OpenAIModelProvider(
        transport,
        live_configuration(),
        monotonic_clock=lambda: next(clock_values),
    )

    first = provider.complete(request(ModelOperation.REPAIR_PLANNING))
    second = provider.complete(request(ModelOperation.PATCH_GENERATION))

    assert first.usage is not None and first.usage.latency_seconds == 0.25
    assert second.usage is not None and second.usage.latency_seconds == 0.75
    assert [item.operation for item in transport.requests] == [
        ModelOperation.REPAIR_PLANNING,
        ModelOperation.PATCH_GENERATION,
    ]
    assert all(
        '"additionalProperties":false' in item.response_schema_json for item in transport.requests
    )
    usage = provider.snapshot_usage()
    assert usage.model_latency_seconds == 1.0
    assert usage.input_tokens == 30
    assert usage.output_tokens == 13
    assert usage.estimated_cost_usd == Decimal("0.0000123")
    rendered = first.model_dump_json() + second.model_dump_json() + usage.model_dump_json()
    assert environment()["OPENAI_API_KEY"] not in rendered


def test_adapter_preserves_interrupt_and_rejects_invalid_measurement() -> None:
    class InterruptingTransport:
        def complete(self, request: OpenAITransportRequest) -> OpenAITransportResponse:
            del request
            raise KeyboardInterrupt

    provider = OpenAIModelProvider(
        InterruptingTransport(), live_configuration(), monotonic_clock=lambda: 1.0
    )
    with pytest.raises(KeyboardInterrupt):
        provider.complete(request(ModelOperation.REPAIR_PLANNING))

    transport = RecordingTransport()
    backwards = iter((2.0, 1.0))
    provider = OpenAIModelProvider(
        transport, live_configuration(), monotonic_clock=lambda: next(backwards)
    )
    with pytest.raises(LiveModelProviderError):
        provider.complete(request(ModelOperation.REPAIR_PLANNING))


def test_provider_factory_reads_credential_without_retaining_it() -> None:
    captured: list[tuple[str, int]] = []
    transport = RecordingTransport()

    def factory(credential: str, timeout: int) -> RecordingTransport:
        captured.append((credential, timeout))
        return transport

    configuration = apply_live_model_configuration(model_configuration(), "openai", environment())
    provider = create_openai_model_provider(
        configuration,
        live_configuration(),
        environment(),
        transport_factory=factory,
    )
    assert captured == [(environment()["OPENAI_API_KEY"], 120)]
    assert environment()["OPENAI_API_KEY"] not in repr(provider.__dict__)

    missing = environment()
    del missing["OPENAI_API_KEY"]
    with pytest.raises(LiveModelConfigurationError):
        create_openai_model_provider(
            configuration,
            live_configuration(),
            missing,
            transport_factory=factory,
        )


def test_transport_contracts_reject_extra_and_wrong_types() -> None:
    with pytest.raises(ValidationError):
        OpenAITransportResponse.model_validate(
            {
                "status": "completed",
                "raw_json": "{}",
                "input_tokens": True,
                "output_tokens": 0,
            },
            strict=True,
        )


def test_exact_cost_does_not_use_process_decimal_precision() -> None:
    configuration = LiveBenchmarkConfiguration(
        request_timeout_seconds=1,
        input_price_usd_per_million_tokens=Decimal("12345678901234567890.123456789"),
        output_price_usd_per_million_tokens=Decimal("0.000000000123456789"),
    )
    with localcontext() as context:
        context.prec = 100
        expected = Decimal(9_223_372_036_854_775_000) * Decimal(
            "12345678901234567890.123456789"
        ) / Decimal(1_000_000) + Decimal(17) * Decimal("0.000000000123456789") / Decimal(1_000_000)
    assert calculate_estimated_cost_usd(9_223_372_036_854_775_000, 17, configuration) == expected


def test_sdk_transport_disables_capabilities_streaming_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_arguments: dict[str, object] = {}
    request_arguments: dict[str, object] = {}

    class Responses:
        def create(self, **kwargs: object) -> object:
            request_arguments.update(kwargs)
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text='{"ok":true}')],
                    )
                ],
                usage=SimpleNamespace(input_tokens=7, output_tokens=2, total_tokens=9),
            )

    class Client:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.update(kwargs)
            self.responses = Responses()

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: SimpleNamespace(__version__="2.1.0", OpenAI=Client)
        if name == "openai"
        else None,
    )
    transport = SdkOpenAITransport("credential", 45)
    response = transport.complete(transport_request())
    assert constructor_arguments == {
        "api_key": "credential",
        "timeout": 45.0,
        "max_retries": 0,
    }
    assert request_arguments["tools"] == []
    assert request_arguments["tool_choice"] == "none"
    assert request_arguments["stream"] is False
    assert request_arguments["store"] is False
    assert request_arguments["timeout"] == 45.0
    assert "reasoning_effort" not in request_arguments
    assert request_arguments["metadata"] == {
        "case_id": "boundary_condition",
        "operation": "repair_planning",
        "attempt_number": "1",
        "prompt_version": "repair-v1.0",
    }
    text = request_arguments["text"]
    assert isinstance(text, dict)
    assert text["format"]["strict"] is True  # type: ignore[index]
    assert response.input_tokens == 7
    assert response.output_tokens == 2


@pytest.mark.parametrize(
    ("status", "output", "usage", "expected_content_failure"),
    [
        ("incomplete", [], (7, 2, 9), True),
        ("completed", [], (7, 2, 9), True),
        (
            "completed",
            [
                SimpleNamespace(type="message", content=[]),
                SimpleNamespace(type="message", content=[]),
            ],
            (7, 2, 9),
            True,
        ),
        ("completed", [], (7, 2, 10), False),
        ("failed", [], (7, 2, 9), False),
    ],
)
def test_sdk_transport_fails_closed_on_response_integrity(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    output: list[object],
    usage: tuple[int, int, int],
    expected_content_failure: bool,
) -> None:
    class Responses:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(
                status=status,
                output=output,
                usage=SimpleNamespace(
                    input_tokens=usage[0], output_tokens=usage[1], total_tokens=usage[2]
                ),
            )

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.responses = Responses()

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: SimpleNamespace(__version__="2.9.0", OpenAI=Client)
        if name == "openai"
        else None,
    )
    transport = SdkOpenAITransport("credential", 45)
    if expected_content_failure:
        response = transport.complete(transport_request())
        assert response.status is OpenAICompletionStatus.CONTENT_FAILURE
        assert response.raw_json == "{}"
    else:
        with pytest.raises(LiveModelProviderError):
            transport.complete(transport_request())


def test_sdk_transport_rejects_unsupported_major_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: SimpleNamespace(__version__="3.0.0", OpenAI=object)
        if name == "openai"
        else None,
    )
    with pytest.raises(LiveModelProviderError, match="SDK version"):
        SdkOpenAITransport("credential", 45)


def test_sdk_transport_maps_missing_dependency_and_provider_exception_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> object:
        del name
        raise ModuleNotFoundError("secret import path")

    monkeypatch.setattr("app.evaluation.live_model.importlib.import_module", missing_dependency)
    with pytest.raises(LiveModelProviderError, match="provider is unavailable") as missing:
        SdkOpenAITransport("credential", 45)
    assert "secret import path" not in str(missing.value)

    class Responses:
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("secret provider body")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.responses = Responses()

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: SimpleNamespace(__version__="2.1.0", OpenAI=Client)
        if name == "openai"
        else None,
    )
    transport = SdkOpenAITransport("credential", 45)
    with pytest.raises(LiveModelProviderError, match="provider request failed") as failed:
        transport.complete(transport_request())
    assert "secret provider body" not in str(failed.value)
