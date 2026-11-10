from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from time import monotonic
from typing import Annotated, Literal, Protocol, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from app.agent.model import SUPPORTED_PROMPT_VERSION, ModelProvider
from app.agent.planning import MAX_PROVIDER_RESPONSE_UTF8_BYTES
from app.agent.schemas import (
    ModelCallUsage,
    ModelConfiguration,
    ModelOperation,
    ModelProviderRequest,
    ModelProviderResponse,
    PatchProposal,
    RepairPlan,
)
from app.evaluation.deterministic.candidate import CandidateModelUsage

GROQ_PROVIDER_NAME = "groq"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_CREDENTIAL_NAME = "GROQ_API_KEY"
GROQ_REASONING_EFFORT: Literal["low"] = "low"
GROQ_API_BASE_URL = "https://api.groq.com"
GROQ_MAX_COMPLETION_TOKENS = 4096
_REASONING_EFFORT_NAME = "MODEL_REASONING_EFFORT"
_TIMEOUT_NAME = "MODEL_REQUEST_TIMEOUT_SECONDS"
_INPUT_PRICE_NAME = "MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS"
_OUTPUT_PRICE_NAME = "MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS"
_INTEGER = re.compile(r"^[0-9]+$")
_DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_SDK_VERSION = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.|$)")
_MILLION = Decimal(1_000_000)
_MAX_TOKEN_COUNT = 2**63 - 1


class LiveModelConfigurationError(ValueError):
    """Live benchmark configuration failed without exposing its input values."""


class LiveModelProviderError(RuntimeError):
    """The provider boundary failed without exposing provider content or exceptions."""


def _require_decimal(value: object) -> object:
    if type(value) is not Decimal or not value.is_finite() or value < 0:
        raise ValueError("value must be a finite nonnegative Decimal")
    return value


NonNegativeDecimal = Annotated[
    Decimal,
    BeforeValidator(_require_decimal),
    Field(ge=Decimal(0), allow_inf_nan=False),
]


class LiveBenchmarkConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reasoning_effort: Literal["low"] = GROQ_REASONING_EFFORT
    request_timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=300)]
    input_price_usd_per_million_tokens: NonNegativeDecimal
    output_price_usd_per_million_tokens: NonNegativeDecimal


def _parse_timeout(value: object) -> int:
    if type(value) is not str or _INTEGER.fullmatch(value) is None:
        raise ValueError
    parsed = int(value, 10)
    if not 1 <= parsed <= 300:
        raise ValueError
    return parsed


def _parse_price(value: object) -> Decimal:
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise ValueError
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise ValueError from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError
    return parsed


def _parse_reasoning_effort(value: object) -> Literal["low"]:
    if value != GROQ_REASONING_EFFORT or type(value) is not str:
        raise ValueError
    return GROQ_REASONING_EFFORT


def load_live_benchmark_configuration(
    environment: Mapping[str, str],
) -> LiveBenchmarkConfiguration:
    try:
        return LiveBenchmarkConfiguration(
            reasoning_effort=_parse_reasoning_effort(environment[_REASONING_EFFORT_NAME]),
            request_timeout_seconds=_parse_timeout(environment[_TIMEOUT_NAME]),
            input_price_usd_per_million_tokens=_parse_price(environment[_INPUT_PRICE_NAME]),
            output_price_usd_per_million_tokens=_parse_price(environment[_OUTPUT_PRICE_NAME]),
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        raise LiveModelConfigurationError("live benchmark configuration is invalid") from None


def apply_live_model_configuration(
    configuration: ModelConfiguration,
    provider_argument: str,
    environment: Mapping[str, str],
) -> ModelConfiguration:
    del environment
    try:
        checked = ModelConfiguration.model_validate(
            configuration.model_dump(warnings="none"), strict=True
        )
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError):
        raise LiveModelConfigurationError("live model configuration is invalid") from None
    if (
        type(provider_argument) is not str
        or provider_argument != GROQ_PROVIDER_NAME
        or checked.provider != provider_argument
        or checked.model_name != GROQ_MODEL_NAME
        or checked.max_output_tokens != GROQ_MAX_COMPLETION_TOKENS
        or checked.prompt_version != SUPPORTED_PROMPT_VERSION
    ):
        raise LiveModelConfigurationError("live model configuration is unsupported")
    return checked


class GroqCompletionStatus(StrEnum):
    COMPLETED = "completed"
    CONTENT_FAILURE = "content_failure"


class GroqTransportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    operation: ModelOperation
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    model_name: str = Field(min_length=1)
    temperature: float = Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)
    max_output_tokens: int = Field(strict=True, ge=1, le=32_768)
    rendered_prompt: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    attempt_number: int = Field(strict=True, ge=1, le=1)
    response_schema_json: str = Field(min_length=1)
    request_timeout_seconds: int = Field(strict=True, ge=1, le=300)
    reasoning_effort: Literal["low"]


class GroqTransportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: GroqCompletionStatus
    raw_json: str
    input_tokens: int = Field(strict=True, ge=0, le=2**63 - 1)
    output_tokens: int = Field(strict=True, ge=0, le=2**63 - 1)


class GroqTransport(Protocol):
    def complete(self, request: GroqTransportRequest) -> GroqTransportResponse: ...


def _attribute(value: object, name: str) -> object:
    try:
        return cast(object, getattr(value, name))
    except (AttributeError, TypeError):
        raise LiveModelProviderError("provider response is invalid") from None


def _integer_attribute(value: object, name: str) -> int:
    try:
        candidate = cast(object, getattr(value, name))
    except (AttributeError, TypeError):
        raise LiveModelProviderError("provider usage is invalid") from None
    if type(candidate) is not int or not 0 <= candidate <= _MAX_TOKEN_COUNT:
        raise LiveModelProviderError("provider usage is invalid")
    return candidate


def _sdk_components(module: object) -> tuple[Callable[..., object], type[object]]:
    version = _attribute(module, "__version__")
    match = _SDK_VERSION.match(version) if type(version) is str else None
    if match is None or int(match.group("major")) != 1 or int(match.group("minor")) < 7:
        raise LiveModelProviderError("live provider SDK version is unsupported")
    constructor = cast(Callable[..., object], _attribute(module, "Groq"))
    types = _attribute(module, "types")
    chat_types = _attribute(types, "chat")
    response_type = _attribute(chat_types, "ChatCompletion")
    if not isinstance(response_type, type):
        raise LiveModelProviderError("live provider SDK version is unsupported")
    return constructor, response_type


def _optional_attribute(value: object, name: str) -> object | None:
    try:
        return cast(object, getattr(value, name))
    except AttributeError:
        return None
    except TypeError:
        raise LiveModelProviderError("provider response is invalid") from None


def _completed_output(response: object) -> tuple[GroqCompletionStatus, str]:
    if _attribute(response, "object") != "chat.completion":
        raise LiveModelProviderError("provider response is invalid")
    choices = _attribute(response, "choices")
    if type(choices) is not list or len(choices) != 1:
        raise LiveModelProviderError("provider response is invalid")
    choice = choices[0]
    choice_index = _attribute(choice, "index")
    if type(choice_index) is not int or choice_index != 0:
        raise LiveModelProviderError("provider response is invalid")
    message = _attribute(choice, "message")
    if _attribute(message, "role") != "assistant":
        raise LiveModelProviderError("provider response is invalid")
    finish_reason = _attribute(choice, "finish_reason")
    if type(finish_reason) is not str:
        raise LiveModelProviderError("provider response is invalid")
    if finish_reason not in {
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
        "cancelled",
        "incomplete",
    }:
        raise LiveModelProviderError("provider response is invalid")
    tool_calls = _attribute(message, "tool_calls")
    function_call = _attribute(message, "function_call")
    executed_tools = _attribute(message, "executed_tools")
    refusal = _optional_attribute(message, "refusal")
    if (
        finish_reason != "stop"
        or tool_calls not in (None, [])
        or function_call is not None
        or executed_tools not in (None, [])
        or refusal is not None
    ):
        return GroqCompletionStatus.CONTENT_FAILURE, "{}"
    content = _attribute(message, "content")
    if type(content) is not str or not content:
        return GroqCompletionStatus.CONTENT_FAILURE, "{}"
    try:
        response_size = len(content.encode("utf-8"))
    except UnicodeError:
        raise LiveModelProviderError("provider response is invalid") from None
    if response_size > MAX_PROVIDER_RESPONSE_UTF8_BYTES:
        return GroqCompletionStatus.CONTENT_FAILURE, "{}"
    return GroqCompletionStatus.COMPLETED, content


class SdkGroqTransport:
    """The only boundary that imports and observes objects from the optional Groq SDK."""

    def __init__(self, api_key: str, request_timeout_seconds: int) -> None:
        try:
            module = importlib.import_module("groq")
            constructor, self._response_type = _sdk_components(module)
            self._client = constructor(
                api_key=api_key,
                base_url=GROQ_API_BASE_URL,
                timeout=float(request_timeout_seconds),
                max_retries=0,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except LiveModelProviderError:
            raise
        except Exception:
            raise LiveModelProviderError("live provider is unavailable") from None

    def complete(self, request: GroqTransportRequest) -> GroqTransportResponse:
        checked = GroqTransportRequest.model_validate(
            request.model_dump(warnings="none"), strict=True
        )
        try:
            schema_value: object = json.loads(checked.response_schema_json)
            if type(schema_value) is not dict:
                raise ValueError
            chat = _attribute(self._client, "chat")
            completions = _attribute(chat, "completions")
            create = cast(Callable[..., object], _attribute(completions, "create"))
            response = create(
                model=checked.model_name,
                messages=[{"role": "user", "content": checked.rendered_prompt}],
                reasoning_effort=checked.reasoning_effort,
                include_reasoning=False,
                max_completion_tokens=checked.max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": checked.operation.value,
                        "strict": True,
                        "schema": schema_value,
                    },
                },
                tools=[],
                tool_choice="none",
                stream=False,
                n=1,
                timeout=float(checked.request_timeout_seconds),
            )
            if type(response) is not self._response_type:
                raise LiveModelProviderError("provider response is invalid")
            status, raw_json = _completed_output(response)
            usage = _optional_attribute(response, "usage")
            if usage is None:
                raise LiveModelProviderError("provider usage is invalid")
            input_tokens = _integer_attribute(usage, "prompt_tokens")
            output_tokens = _integer_attribute(usage, "completion_tokens")
            total_tokens = _integer_attribute(usage, "total_tokens")
            if input_tokens + output_tokens != total_tokens:
                raise LiveModelProviderError("provider usage is invalid")
            return GroqTransportResponse(
                status=status,
                raw_json=raw_json,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except LiveModelProviderError:
            raise
        except Exception:
            raise LiveModelProviderError("provider request failed") from None


MonotonicClock = Callable[[], float]
TransportFactory = Callable[[str, int], GroqTransport]


def _schema_json(operation: ModelOperation) -> str:
    model: type[RepairPlan] | type[PatchProposal]
    if operation is ModelOperation.REPAIR_PLANNING:
        model = RepairPlan
    elif operation is ModelOperation.PATCH_GENERATION:
        model = PatchProposal
    else:
        raise LiveModelProviderError("provider operation is unsupported")
    return json.dumps(
        model.model_json_schema(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def calculate_estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    configuration: LiveBenchmarkConfiguration,
) -> Decimal:
    if (
        type(input_tokens) is not int
        or type(output_tokens) is not int
        or not 0 <= input_tokens <= _MAX_TOKEN_COUNT
        or not 0 <= output_tokens <= _MAX_TOKEN_COUNT
    ):
        raise LiveModelProviderError("provider usage is invalid")
    checked = LiveBenchmarkConfiguration.model_validate(
        configuration.model_dump(warnings="none"), strict=True
    )
    rates = (
        checked.input_price_usd_per_million_tokens,
        checked.output_price_usd_per_million_tokens,
    )
    token_values = (input_tokens, output_tokens)
    adjusted = [
        rate.adjusted() + len(str(tokens or 1))
        for rate, tokens in zip(rates, token_values, strict=True)
    ]
    raw_exponents = [rate.as_tuple().exponent for rate in rates]
    if any(type(exponent) is not int for exponent in raw_exponents):
        raise LiveModelProviderError("provider usage is invalid")
    exponents = [cast(int, exponent) - 6 for exponent in raw_exponents]
    precision = max(28, max(adjusted) - min(exponents) + 3)
    with localcontext() as context:
        context.prec = precision
        context.Emax = max(context.Emax, max(adjusted) + 3)
        context.Emin = min(context.Emin, min(exponents) - 3)
        return (
            Decimal(input_tokens) * rates[0] / _MILLION
            + Decimal(output_tokens) * rates[1] / _MILLION
        )


class GroqModelProvider(ModelProvider):
    def __init__(
        self,
        transport: GroqTransport,
        configuration: LiveBenchmarkConfiguration,
        monotonic_clock: MonotonicClock = monotonic,
    ) -> None:
        self._transport = transport
        self._configuration = LiveBenchmarkConfiguration.model_validate(
            configuration.model_dump(warnings="none"), strict=True
        )
        self._clock = monotonic_clock
        self._latency = 0.0
        self._input_tokens = 0
        self._output_tokens = 0

    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        try:
            checked = ModelProviderRequest.model_validate(
                request.model_dump(warnings="none"), strict=True
            )
            transport_request = GroqTransportRequest(
                operation=checked.operation,
                case_id=checked.case_id,
                model_name=checked.model_name,
                temperature=checked.temperature,
                max_output_tokens=checked.max_output_tokens,
                rendered_prompt=checked.rendered_prompt,
                prompt_version=checked.prompt_version,
                attempt_number=checked.attempt_number,
                response_schema_json=_schema_json(checked.operation),
                request_timeout_seconds=self._configuration.request_timeout_seconds,
                reasoning_effort=self._configuration.reasoning_effort,
            )
            started = self._clock()
            response = self._transport.complete(transport_request)
            completed = self._clock()
            if (
                type(started) is not float
                or type(completed) is not float
                or not math.isfinite(started)
                or not math.isfinite(completed)
                or completed < started
            ):
                raise ValueError
            checked_response = GroqTransportResponse.model_validate(
                response.model_dump(warnings="none"), strict=True
            )
            usage = ModelCallUsage(
                latency_seconds=completed - started,
                input_tokens=checked_response.input_tokens,
                output_tokens=checked_response.output_tokens,
            )
            next_latency = self._latency + usage.latency_seconds
            next_input = self._input_tokens + usage.input_tokens
            next_output = self._output_tokens + usage.output_tokens
            if (
                not math.isfinite(next_latency)
                or next_input > _MAX_TOKEN_COUNT
                or next_output > _MAX_TOKEN_COUNT
            ):
                raise ValueError
            self._latency = next_latency
            self._input_tokens = next_input
            self._output_tokens = next_output
            self.snapshot_usage()
            return ModelProviderResponse(raw_json=checked_response.raw_json, usage=usage)
        except (KeyboardInterrupt, SystemExit):
            raise
        except LiveModelProviderError:
            raise
        except Exception:
            raise LiveModelProviderError("provider response is invalid") from None

    def snapshot_usage(self) -> CandidateModelUsage:
        try:
            return CandidateModelUsage(
                model_latency_seconds=self._latency,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                estimated_cost_usd=calculate_estimated_cost_usd(
                    self._input_tokens, self._output_tokens, self._configuration
                ),
            )
        except (ValidationError, LiveModelProviderError):
            raise LiveModelProviderError("provider usage is invalid") from None


def create_groq_model_provider(
    configuration: ModelConfiguration,
    live_configuration: LiveBenchmarkConfiguration,
    environment: Mapping[str, str],
    *,
    transport_factory: TransportFactory = SdkGroqTransport,
    monotonic_clock: MonotonicClock = monotonic,
) -> GroqModelProvider:
    try:
        checked = ModelConfiguration.model_validate(
            configuration.model_dump(warnings="none"), strict=True
        )
        live_checked = LiveBenchmarkConfiguration.model_validate(
            live_configuration.model_dump(warnings="none"), strict=True
        )
        credential = environment[GROQ_CREDENTIAL_NAME]
        if (
            type(credential) is not str
            or not credential.strip()
            or checked.provider != GROQ_PROVIDER_NAME
            or checked.model_name != GROQ_MODEL_NAME
            or checked.max_output_tokens != GROQ_MAX_COMPLETION_TOKENS
            or checked.prompt_version != SUPPORTED_PROMPT_VERSION
            or live_checked.reasoning_effort != GROQ_REASONING_EFFORT
        ):
            raise ValueError
        transport = transport_factory(credential, live_checked.request_timeout_seconds)
        return GroqModelProvider(transport, live_checked, monotonic_clock)
    except (KeyboardInterrupt, SystemExit):
        raise
    except LiveModelProviderError:
        raise
    except Exception:
        raise LiveModelConfigurationError("live provider configuration is invalid") from None
