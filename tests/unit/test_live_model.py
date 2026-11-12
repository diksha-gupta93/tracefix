from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, localcontext
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.schemas import ModelConfiguration, ModelOperation, ModelProviderRequest
from app.evaluation.live_model import (
    GROQ_LIVE_BENCHMARK_REQUEST_INTERVAL_SECONDS,
    GroqCompletionStatus,
    GroqModelProvider,
    GroqRequestPacer,
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
from app.observability import DiagnosticSpan, SafeMetadata, safe_metadata


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

    def complete(
        self,
        request: GroqTransportRequest,
        request_pacer: GroqRequestPacer,
    ) -> tuple[GroqTransportResponse, float]:
        def invoke() -> GroqTransportResponse:
            self.requests.append(request)
            return self.responses.pop(0)

        return request_pacer.pace_request(invoke)


class MetadataRecorder:
    def __init__(self) -> None:
        self.values: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def span(self, name: str, metadata: SafeMetadata | None = None) -> Iterator[DiagnosticSpan]:
        del name
        self.add_metadata(safe_metadata(metadata))
        yield self  # type: ignore[misc]

    def add_metadata(self, metadata: SafeMetadata) -> None:
        self.values.append(safe_metadata(metadata))


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


def response() -> GroqTransportResponse:
    return GroqTransportResponse(
        status=GroqCompletionStatus.COMPLETED,
        raw_json="{}",
        input_tokens=1,
        output_tokens=1,
    )


def pacer(clock: ManualClock | None = None) -> GroqRequestPacer:
    selected = ManualClock() if clock is None else clock
    return GroqRequestPacer(
        monotonic_clock=selected,
        sleeper=selected.sleep,
    )


def sdk_complete(transport: SdkGroqTransport) -> GroqTransportResponse:
    response_value, _ = transport.complete(transport_request(), pacer())
    return response_value


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


def test_request_pacer_first_request_is_immediate_and_second_waits_remaining_time() -> None:
    assert GROQ_LIVE_BENCHMARK_REQUEST_INTERVAL_SECONDS == 61.0
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)
    starts: list[float] = []

    def record_start() -> GroqTransportResponse:
        starts.append(clock())
        return response()

    request_pacer.pace_request(record_start)
    assert clock.sleeps == []

    clock.advance(20.0)
    request_pacer.pace_request(record_start)

    assert clock.sleeps == [41.0]
    assert starts == [100.0, 161.0]


def test_request_pacer_does_not_sleep_after_full_interval_elapsed() -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)

    request_pacer.pace_request(response)
    clock.advance(61.0)
    request_pacer.pace_request(response)

    assert clock.sleeps == []


def test_request_pacer_measures_start_to_start_not_completion_to_start() -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)

    def three_second_request() -> GroqTransportResponse:
        clock.advance(3.0)
        return response()

    request_pacer.pace_request(three_second_request)
    clock.advance(17.0)
    request_pacer.pace_request(response)

    assert clock.sleeps == [41.0]
    assert clock() == 161.0


@pytest.mark.parametrize("failure_source", ["clock", "sleeper"])
def test_request_pacer_failures_fail_closed_with_safe_diagnostics(
    failure_source: str,
) -> None:
    sensitive = "sensitive pacing implementation detail"
    calls = 0
    clock = ManualClock()
    clock_failures = 1
    sleeper_failures = 1

    def failing_clock() -> float:
        nonlocal clock_failures
        if clock_failures:
            clock_failures -= 1
            raise RuntimeError(sensitive)
        return clock()

    def failing_sleeper(seconds: float) -> None:
        nonlocal sleeper_failures
        if sleeper_failures:
            sleeper_failures -= 1
            raise RuntimeError(sensitive)
        clock.sleep(seconds)

    request_pacer = GroqRequestPacer(
        monotonic_clock=failing_clock if failure_source == "clock" else clock,
        sleeper=failing_sleeper if failure_source == "sleeper" else clock.sleep,
    )

    def provider_request() -> GroqTransportResponse:
        nonlocal calls
        calls += 1
        return response()

    if failure_source == "sleeper":
        request_pacer.pace_request(provider_request)
    with pytest.raises(LiveModelProviderError) as raised:
        request_pacer.pace_request(provider_request)
    request_pacer.pace_request(provider_request)

    assert calls == (2 if failure_source == "sleeper" else 1)
    assert sensitive not in str(raised.value)


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
    clock = ManualClock(10.0)

    class AdvancingRecordingTransport(RecordingTransport):
        def __init__(self) -> None:
            super().__init__()
            self._durations = iter((0.25, 0.75))

        def complete(
            self,
            request: GroqTransportRequest,
            request_pacer: GroqRequestPacer,
        ) -> tuple[GroqTransportResponse, float]:
            def invoke() -> GroqTransportResponse:
                self.requests.append(request)
                result = self.responses.pop(0)
                clock.advance(next(self._durations))
                return result

            return request_pacer.pace_request(invoke)

    transport = AdvancingRecordingTransport()
    provider = GroqModelProvider(
        transport,
        live_configuration(),
        pacer(clock),
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


def test_adapter_emits_safe_provider_finish_category_without_content() -> None:
    recorder = MetadataRecorder()
    provider = GroqModelProvider(
        RecordingTransport(),
        live_configuration(),
        pacer(),
        recorder,
    )

    provider.complete(request(ModelOperation.REPAIR_PLANNING))
    provider.complete(request(ModelOperation.PATCH_GENERATION))

    assert recorder.values[0] == {
        "finish_category": "completed",
        "terminal_diagnostic_code": "provider_response_completed",
    }
    assert recorder.values[1] == {
        "finish_category": "content_failure",
        "terminal_diagnostic_code": "provider_content_failure",
    }
    assert all("raw_json" not in values for values in recorder.values)


def test_pacing_delay_is_excluded_from_provider_latency() -> None:
    class AdvancingTransport:
        def __init__(self, clock: ManualClock) -> None:
            self._clock = clock

        def complete(
            self,
            request: GroqTransportRequest,
            request_pacer: GroqRequestPacer,
        ) -> tuple[GroqTransportResponse, float]:
            del request

            def invoke() -> GroqTransportResponse:
                self._clock.advance(2.0)
                return response()

            return request_pacer.pace_request(invoke)

    clock = ManualClock(100.0)
    provider = GroqModelProvider(
        AdvancingTransport(clock),
        live_configuration(),
        pacer(clock),
    )

    first = provider.complete(request(ModelOperation.REPAIR_PLANNING))
    second = provider.complete(request(ModelOperation.PATCH_GENERATION))

    assert clock.sleeps == [59.0]
    assert first.usage is not None and first.usage.latency_seconds == 2.0
    assert second.usage is not None and second.usage.latency_seconds == 2.0
    assert provider.snapshot_usage().model_latency_seconds == 4.0


def test_shared_pacer_coordinates_separate_sdk_provider_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(100.0)
    shared_pacer = pacer(clock)
    sdk_starts: list[float] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            sdk_starts.append(clock())
            return _chat_completion()

    _install_fake_sdk(monkeypatch, Completions())
    first_provider = GroqModelProvider(
        SdkGroqTransport("credential", 45),
        live_configuration(),
        shared_pacer,
    )
    second_provider = GroqModelProvider(
        SdkGroqTransport("credential", 45),
        live_configuration(),
        shared_pacer,
    )

    first_provider.complete(request(ModelOperation.REPAIR_PLANNING))
    clock.advance(20.0)
    second_provider.complete(
        request(ModelOperation.PATCH_GENERATION).model_copy(
            update={"case_id": "exception_handling"}
        )
    )

    assert clock.sleeps == [41.0]
    assert sdk_starts == [100.0, 161.0]


def test_adapter_preserves_interrupt_and_rejects_invalid_measurement() -> None:
    class InterruptingTransport:
        def complete(
            self,
            request: GroqTransportRequest,
            request_pacer: GroqRequestPacer,
        ) -> tuple[GroqTransportResponse, float]:
            del request

            def invoke() -> GroqTransportResponse:
                raise KeyboardInterrupt

            return request_pacer.pace_request(invoke)

    provider = GroqModelProvider(InterruptingTransport(), live_configuration(), pacer())
    with pytest.raises(KeyboardInterrupt):
        provider.complete(request(ModelOperation.REPAIR_PLANNING))

    transport = RecordingTransport()
    backwards = iter((2.0, 1.0))
    provider = GroqModelProvider(
        transport,
        live_configuration(),
        GroqRequestPacer(monotonic_clock=lambda: next(backwards), sleeper=lambda delay: None),
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
        request_pacer=pacer(),
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
            request_pacer=pacer(),
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
                request_pacer=pacer(),
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
    )


def _sdk_importer(client: type[object], version: str = "1.7.0"):
    modules = {
        "groq": _sdk_module(client, version),
        "groq.types.chat": SimpleNamespace(ChatCompletion=FakeChatCompletion),
    }
    return modules.__getitem__


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
        _sdk_importer(Client),
    )
    transport = SdkGroqTransport("credential", 45)
    response, latency_seconds = transport.complete(transport_request(), pacer())
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
    assert latency_seconds == 0.0


def test_sdk_transport_paces_actual_create_after_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)
    preparation_delays = iter((2.0, 0.0))
    request_durations = iter((3.0, 4.0))
    transport_entries: list[float] = []
    sdk_starts: list[float] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            sdk_starts.append(clock())
            clock.advance(next(request_durations))
            return _chat_completion()

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self._completions = Completions()

        @property
        def chat(self) -> object:
            clock.advance(next(preparation_delays))
            return SimpleNamespace(completions=self._completions)

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        _sdk_importer(Client),
    )
    transport = SdkGroqTransport("credential", 45)

    transport_entries.append(clock())
    _, first_latency = transport.complete(transport_request(), request_pacer)
    clock.advance(56.0)
    transport_entries.append(clock())
    _, second_latency = transport.complete(transport_request(), request_pacer)

    assert transport_entries == [100.0, 161.0]
    assert sdk_starts == [102.0, 163.0]
    assert clock.sleeps == [2.0]
    assert first_latency == 3.0
    assert second_latency == 4.0


def test_sdk_transport_does_not_sleep_after_interval_since_actual_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)
    sdk_starts: list[float] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            sdk_starts.append(clock())
            return _chat_completion()

    _install_fake_sdk(monkeypatch, Completions())
    transport = SdkGroqTransport("credential", 45)

    transport.complete(transport_request(), request_pacer)
    clock.advance(61.0)
    transport.complete(transport_request(), request_pacer)

    assert sdk_starts == [100.0, 161.0]
    assert clock.sleeps == []


def test_sdk_exception_releases_pacing_lock_and_preserves_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)
    sdk_starts: list[float] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            sdk_starts.append(clock())
            if len(sdk_starts) == 1:
                raise RuntimeError("sensitive provider failure")
            return _chat_completion()

    _install_fake_sdk(monkeypatch, Completions())
    transport = SdkGroqTransport("credential", 45)

    with pytest.raises(LiveModelProviderError) as raised:
        transport.complete(transport_request(), request_pacer)
    response_value, _ = transport.complete(transport_request(), request_pacer)

    assert "sensitive provider failure" not in str(raised.value)
    assert response_value.status is GroqCompletionStatus.COMPLETED
    assert sdk_starts == [100.0, 161.0]
    assert clock.sleeps == [61.0]


def test_concurrent_sdk_create_calls_cannot_bypass_shared_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(100.0)
    request_pacer = pacer(clock)
    first_create_started = Event()
    both_prepared = Event()
    release_first_create = Event()
    preparation_lock = Lock()
    prepared = 0
    sdk_starts: list[float] = []
    errors: list[BaseException] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            sdk_starts.append(clock())
            if len(sdk_starts) == 1:
                first_create_started.set()
                if not release_first_create.wait(timeout=5.0):
                    raise RuntimeError("test synchronization failed")
            return _chat_completion()

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self._completions = Completions()

        @property
        def chat(self) -> object:
            nonlocal prepared
            with preparation_lock:
                prepared += 1
                if prepared == 2:
                    both_prepared.set()
            return SimpleNamespace(completions=self._completions)

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        _sdk_importer(Client),
    )
    transport = SdkGroqTransport("credential", 45)

    def invoke() -> None:
        try:
            transport.complete(transport_request(), request_pacer)
        except BaseException as error:
            errors.append(error)

    first = Thread(target=invoke)
    second = Thread(target=invoke)
    first.start()
    assert first_create_started.wait(timeout=5.0)
    second.start()
    assert both_prepared.wait(timeout=5.0)
    release_first_create.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sdk_starts == [100.0, 161.0]
    assert clock.sleeps == [61.0]


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
    response = sdk_complete(SdkGroqTransport("credential", 45))
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
        sdk_complete(SdkGroqTransport("credential", 45))


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, completions: object) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        _sdk_importer(Client),
    )


@pytest.mark.parametrize("version", ["1.6.9", "2.0.0", "invalid"])
def test_sdk_transport_rejects_unsupported_sdk_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(
        "app.evaluation.live_model.importlib.import_module",
        _sdk_importer(object, version),
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
        _sdk_importer(Client, version),
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
        _sdk_importer(Client),
    )
    transport = SdkGroqTransport("credential", 45)
    with pytest.raises(LiveModelProviderError, match="provider request failed") as failed:
        sdk_complete(transport)
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
        _sdk_importer(Client),
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
        sdk_complete(SdkGroqTransport("credential", 45))


def test_sdk_transport_rejects_wrong_official_response_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace()

    _install_fake_sdk(monkeypatch, Completions())
    with pytest.raises(LiveModelProviderError, match="provider response"):
        sdk_complete(SdkGroqTransport("credential", 45))


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
        sdk_complete(SdkGroqTransport("credential", 45))
    assert "secret body" not in str(failed.value)
