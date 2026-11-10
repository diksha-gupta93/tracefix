from __future__ import annotations

from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.schemas import ModelConfiguration, ModelOperation, ModelProviderRequest
from app.evaluation.live_model import (
    GroqCompletionStatus,
    GroqModelProvider,
    GroqTransportRequest,
    GroqTransportResponse,
    LiveBenchmarkConfiguration,
    LiveModelConfigurationError,
    LiveModelProviderError,
    SdkGroqTransport,
    apply_live_model_configuration,
    calculate_estimated_cost_usd,
    create_groq_model_provider,
    load_live_benchmark_configuration,
)


def environment() -> dict[str, str]:
    return {
        "MODEL_REQUEST_TIMEOUT_SECONDS": "120",
        "MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS": "0.15",
        "MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS": "0.60",
        "MODEL_REASONING_EFFORT": "low",
        "GROQ_API_KEY": "credential-must-not-escape",
    }


def model_configuration() -> ModelConfiguration:
    return ModelConfiguration(
        provider="groq",
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
    )


def live_configuration() -> LiveBenchmarkConfiguration:
    return load_live_benchmark_configuration(environment())


class RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[GroqTransportRequest] = []
        self.responses = [
            GroqTransportResponse(
                status=GroqCompletionStatus.COMPLETED,
                raw_json='{"autonomous_repair_suitable":false,"files_expected_to_change":[],"intended_behavioural_correction":"none","risks":[],"suspected_root_cause":"unknown","validation_strategy":"none"}',
                input_tokens=10,
                output_tokens=5,
            ),
            GroqTransportResponse(
                status=GroqCompletionStatus.CONTENT_FAILURE,
                raw_json="{}",
                input_tokens=20,
                output_tokens=8,
            ),
        ]

    def complete(self, request: GroqTransportRequest) -> GroqTransportResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def request(operation: ModelOperation) -> ModelProviderRequest:
    return ModelProviderRequest(
        case_id="boundary_condition",
        operation=operation,
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="safe rendered prompt",
    )


def transport_request() -> GroqTransportRequest:
    return GroqTransportRequest(
        operation=ModelOperation.REPAIR_PLANNING,
        case_id="boundary_condition",
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
        max_output_tokens=4096,
        rendered_prompt="prompt",
        prompt_version="repair-v1.0",
        attempt_number=1,
        response_schema_json='{"type":"object","properties":{},"additionalProperties":false}',
        request_timeout_seconds=45,
        reasoning_effort="low",
    )


def test_live_configuration_is_strict_and_exact() -> None:
    configuration = live_configuration()
    assert configuration.reasoning_effort == "low"
    assert configuration.request_timeout_seconds == 120
    assert configuration.input_price_usd_per_million_tokens == Decimal("0.15")
    assert configuration.output_price_usd_per_million_tokens == Decimal("0.60")

    for name, value in (
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "0"),
        ("MODEL_REQUEST_TIMEOUT_SECONDS", "301"),
        ("MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS", "nan"),
        ("MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS", "-1"),
        ("MODEL_REASONING_EFFORT", "medium"),
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
    missing_reasoning = environment()
    del missing_reasoning["MODEL_REASONING_EFFORT"]
    with pytest.raises(LiveModelConfigurationError):
        load_live_benchmark_configuration(missing_reasoning)


def test_live_model_configuration_requires_exact_profile() -> None:
    configured = apply_live_model_configuration(model_configuration(), "groq", environment())
    assert configured == model_configuration()

    with pytest.raises(LiveModelConfigurationError):
        apply_live_model_configuration(model_configuration(), "Groq", environment())
    with pytest.raises(LiveModelConfigurationError):
        apply_live_model_configuration(
            model_configuration().model_copy(update={"model_name": "reviewed-other-model"}),
            "groq",
            environment(),
        )
    for field, value in (("provider", "openai"), ("max_output_tokens", 4095)):
        with pytest.raises(LiveModelConfigurationError):
            apply_live_model_configuration(
                model_configuration().model_copy(update={field: value}),
                "groq",
                environment(),
            )


def test_adapter_maps_request_measures_usage_and_calculates_exact_cost() -> None:
    transport = RecordingTransport()
    clock_values = iter((10.0, 10.25, 20.0, 20.75))
    provider = GroqModelProvider(
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
    assert environment()["GROQ_API_KEY"] not in rendered


def test_adapter_preserves_interrupt_and_rejects_invalid_measurement() -> None:
    class InterruptingTransport:
        def complete(self, request: GroqTransportRequest) -> GroqTransportResponse:
            del request
            raise KeyboardInterrupt

    provider = GroqModelProvider(
        InterruptingTransport(), live_configuration(), monotonic_clock=lambda: 1.0
    )
    with pytest.raises(KeyboardInterrupt):
        provider.complete(request(ModelOperation.REPAIR_PLANNING))

    transport = RecordingTransport()
    backwards = iter((2.0, 1.0))
    provider = GroqModelProvider(
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

    configuration = apply_live_model_configuration(model_configuration(), "groq", environment())
    provider = create_groq_model_provider(
        configuration,
        live_configuration(),
        environment(),
        transport_factory=factory,
    )
    assert captured == [(environment()["GROQ_API_KEY"], 120)]
    assert environment()["GROQ_API_KEY"] not in repr(provider.__dict__)

    missing = environment()
    del missing["GROQ_API_KEY"]
    with pytest.raises(LiveModelConfigurationError):
        create_groq_model_provider(
            configuration,
            live_configuration(),
            missing,
            transport_factory=factory,
        )
    for invalid_credential in ("", "   "):
        invalid = environment()
        invalid["GROQ_API_KEY"] = invalid_credential
        with pytest.raises(LiveModelConfigurationError):
            create_groq_model_provider(
                configuration,
                live_configuration(),
                invalid,
                transport_factory=factory,
            )


def test_transport_contracts_reject_extra_and_wrong_types() -> None:
    with pytest.raises(ValidationError):
        GroqTransportResponse.model_validate(
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
        reasoning_effort="low",
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


class FakeChatCompletion:
    pass


def _sdk_module(client: type[object], version: str = "1.7.0") -> object:
    return SimpleNamespace(
        __version__=version,
        Groq=client,
        types=SimpleNamespace(chat=SimpleNamespace(ChatCompletion=FakeChatCompletion)),
    )


def _chat_completion(
    *, choices: object | None = None, usage: object | None = None
) -> FakeChatCompletion:
    response = FakeChatCompletion()
    response.object = "chat.completion"
    response.choices = (
        [
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content='{"ok":true}',
                    tool_calls=None,
                    function_call=None,
                    executed_tools=None,
                ),
            )
        ]
        if choices is None
        else choices
    )
    response.usage = (
        SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9)
        if usage is None
        else usage
    )
    return response


def test_sdk_transport_uses_exact_chat_completions_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_arguments: dict[str, object] = {}
    request_arguments: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> object:
            request_arguments.update(kwargs)
            return _chat_completion()

    class Client:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.update(kwargs)
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("GROQ_BASE_URL", "https://credential-stealing.invalid")
    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(Client) if name == "groq" else None,
    )
    transport = SdkGroqTransport("credential", 45)
    response = transport.complete(transport_request())
    assert constructor_arguments == {
        "api_key": "credential",
        "base_url": "https://api.groq.com",
        "timeout": 45.0,
        "max_retries": 0,
    }
    assert request_arguments == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "prompt"}],
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "repair_planning",
                "strict": True,
                "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        "tools": [],
        "tool_choice": "none",
        "stream": False,
        "n": 1,
        "timeout": 45.0,
    }
    assert all(
        field not in request_arguments
        for field in ("input", "text", "store", "metadata", "max_output_tokens", "temperature")
    )
    assert response.status is GroqCompletionStatus.COMPLETED
    assert response.raw_json == '{"ok":true}'
    assert response.input_tokens == 7
    assert response.output_tokens == 2


@pytest.mark.parametrize(
    "choice",
    [
        SimpleNamespace(
            index=0,
            finish_reason="length",
            message=SimpleNamespace(
                role="assistant",
                content='{"partial":true}',
                tool_calls=None,
                function_call=None,
                executed_tools=None,
            ),
        ),
        SimpleNamespace(
            index=0,
            finish_reason="stop",
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=None,
                function_call=None,
                executed_tools=None,
            ),
        ),
        SimpleNamespace(
            index=0,
            finish_reason="tool_calls",
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[object()],
                function_call=None,
                executed_tools=None,
            ),
        ),
        SimpleNamespace(
            index=0,
            finish_reason="function_call",
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=None,
                function_call=object(),
                executed_tools=None,
            ),
        ),
        SimpleNamespace(
            index=0,
            finish_reason="stop",
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=None,
                function_call=None,
                executed_tools=None,
                refusal="refused",
            ),
        ),
    ],
)
def test_sdk_transport_maps_trustworthy_non_content_results_to_content_failure(
    monkeypatch: pytest.MonkeyPatch, choice: object
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return _chat_completion(choices=[choice])

    _install_fake_sdk(monkeypatch, Completions())
    response = SdkGroqTransport("credential", 45).complete(transport_request())
    assert response.status is GroqCompletionStatus.CONTENT_FAILURE
    assert response.raw_json == "{}"
    assert response.input_tokens == 7
    assert response.output_tokens == 2


@pytest.mark.parametrize(
    "choices",
    [
        None,
        [],
        [object(), object()],
        [SimpleNamespace(index=1, finish_reason="stop", message=object())],
        [
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(role="tool"),
            )
        ],
        [
            SimpleNamespace(
                index=0,
                finish_reason="unexpected",
                message=SimpleNamespace(role="assistant"),
            )
        ],
    ],
)
def test_sdk_transport_rejects_malformed_response_structures(
    monkeypatch: pytest.MonkeyPatch, choices: object
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            response = _chat_completion()
            response.choices = choices
            return response

    _install_fake_sdk(monkeypatch, Completions())
    with pytest.raises(LiveModelProviderError, match="provider response"):
        SdkGroqTransport("credential", 45).complete(transport_request())


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, completions: object) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(Client) if name == "groq" else None,
    )


@pytest.mark.parametrize("version", ["1.6.9", "2.0.0", "invalid"])
def test_sdk_transport_rejects_unsupported_sdk_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(object, version) if name == "groq" else None,
    )
    with pytest.raises(LiveModelProviderError, match="SDK version"):
        SdkGroqTransport("credential", 45)


@pytest.mark.parametrize("version", ["1.7.0", "1.99.0"])
def test_sdk_transport_accepts_supported_sdk_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(Client, version) if name == "groq" else None,
    )
    SdkGroqTransport("credential", 45)


def test_sdk_transport_maps_missing_dependency_and_provider_exception_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> object:
        del name
        raise ModuleNotFoundError("secret import path")

    monkeypatch.setattr("app.evaluation.live_model.importlib.import_module", missing_dependency)
    with pytest.raises(LiveModelProviderError, match="provider is unavailable") as missing:
        SdkGroqTransport("credential", 45)
    assert "secret import path" not in str(missing.value)

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("secret provider body")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(Client) if name == "groq" else None,
    )
    transport = SdkGroqTransport("credential", 45)
    with pytest.raises(LiveModelProviderError, match="provider request failed") as failed:
        transport.complete(transport_request())
    assert "secret provider body" not in str(failed.value)


def test_sdk_transport_maps_client_construction_failure_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise RuntimeError("credential and raw client failure")

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        lambda name: _sdk_module(Client) if name == "groq" else None,
    )
    with pytest.raises(LiveModelProviderError, match="provider is unavailable") as failed:
        SdkGroqTransport("credential", 45)
    assert "credential" not in str(failed.value)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(prompt_tokens=True, completion_tokens=2, total_tokens=3),
        SimpleNamespace(prompt_tokens="7", completion_tokens=2, total_tokens=9),
        SimpleNamespace(prompt_tokens=-1, completion_tokens=2, total_tokens=1),
        SimpleNamespace(prompt_tokens=2**63, completion_tokens=0, total_tokens=2**63),
        SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=10),
        SimpleNamespace(prompt_tokens=7, completion_tokens=2),
    ],
)
def test_sdk_transport_rejects_malformed_usage(
    monkeypatch: pytest.MonkeyPatch, usage: object
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            response = _chat_completion()
            response.usage = usage
            return response

    _install_fake_sdk(monkeypatch, Completions())
    with pytest.raises(LiveModelProviderError, match="provider usage"):
        SdkGroqTransport("credential", 45).complete(transport_request())


def test_sdk_transport_rejects_wrong_official_response_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace()

    _install_fake_sdk(monkeypatch, Completions())
    with pytest.raises(LiveModelProviderError, match="provider response"):
        SdkGroqTransport("credential", 45).complete(transport_request())


@pytest.mark.parametrize(
    "provider_error",
    [
        RuntimeError("authentication secret body"),
        TimeoutError("timeout secret body"),
        ConnectionError("rate-limit secret body"),
        OSError("transport secret body"),
    ],
    ids=("authentication", "timeout", "rate_limit", "transport"),
)
def test_sdk_transport_maps_provider_error_categories_safely(
    monkeypatch: pytest.MonkeyPatch, provider_error: Exception
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise provider_error

    _install_fake_sdk(monkeypatch, Completions())
    with pytest.raises(LiveModelProviderError, match="provider request failed") as failed:
        SdkGroqTransport("credential", 45).complete(transport_request())
    assert "secret body" not in str(failed.value)
