from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from time import monotonic
from typing import Annotated, Protocol, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from app.agent.model import SUPPORTED_PROMPT_VERSION, ModelProvider
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

OPENAI_PROVIDER_NAME = "openai"
OPENAI_CREDENTIAL_NAME = "OPENAI_API_KEY"
_TIMEOUT_NAME = "MODEL_REQUEST_TIMEOUT_SECONDS"
_INPUT_PRICE_NAME = "MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS"
_OUTPUT_PRICE_NAME = "MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS"
_INTEGER = re.compile(r"^[0-9]+$")
_DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_SDK_VERSION = re.compile(r"^(?P<major>[0-9]+)(?:\.|$)")
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


def load_live_benchmark_configuration(
    environment: Mapping[str, str],
) -> LiveBenchmarkConfiguration:
    try:
        return LiveBenchmarkConfiguration(
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
        or provider_argument != OPENAI_PROVIDER_NAME
        or checked.provider != provider_argument
        or checked.prompt_version != SUPPORTED_PROMPT_VERSION
    ):
        raise LiveModelConfigurationError("live model configuration is unsupported")
    return checked


class OpenAICompletionStatus(StrEnum):
    COMPLETED = "completed"
    CONTENT_FAILURE = "content_failure"


class OpenAITransportRequest(BaseModel):
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


class OpenAITransportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: OpenAICompletionStatus
    raw_json: str
    input_tokens: int = Field(strict=True, ge=0, le=2**63 - 1)
    output_tokens: int = Field(strict=True, ge=0, le=2**63 - 1)


class OpenAITransport(Protocol):
    def complete(self, request: OpenAITransportRequest) -> OpenAITransportResponse: ...


def _attribute(value: object, name: str) -> object:
    try:
        return cast(object, getattr(value, name))
    except (AttributeError, TypeError):
        raise LiveModelProviderError("provider response is invalid") from None


def _integer_attribute(value: object, name: str) -> int:
    candidate = _attribute(value, name)
    if type(candidate) is not int or not 0 <= candidate <= _MAX_TOKEN_COUNT:
        raise LiveModelProviderError("provider usage is invalid")
    return candidate


def _sdk_constructor(module: object) -> Callable[..., object]:
    version = _attribute(module, "__version__")
    match = _SDK_VERSION.match(version) if type(version) is str else None
    if match is None or int(match.group("major")) != 2:
        raise LiveModelProviderError("live provider SDK version is unsupported")
    return cast(Callable[..., object], _attribute(module, "OpenAI"))


def _completed_output(response: object) -> tuple[OpenAICompletionStatus, str]:
    status = _attribute(response, "status")
    if status == "incomplete":
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    if status != "completed":
        raise LiveModelProviderError("provider response did not complete")
    output = _attribute(response, "output")
    if type(output) is not list or len(output) != 1:
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    message = output[0]
    if _attribute(message, "type") != "message":
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    content = _attribute(message, "content")
    if type(content) is not list or len(content) != 1:
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    output_item = content[0]
    if _attribute(output_item, "type") != "output_text":
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    text = _attribute(output_item, "text")
    if type(text) is not str or not text:
        return OpenAICompletionStatus.CONTENT_FAILURE, "{}"
    return OpenAICompletionStatus.COMPLETED, text


class SdkOpenAITransport:
    """The only boundary that imports and observes objects from the optional OpenAI SDK."""

    def __init__(self, api_key: str, request_timeout_seconds: int) -> None:
        try:
            module = importlib.import_module("openai")
            self._client = _sdk_constructor(module)(
                api_key=api_key,
                timeout=float(request_timeout_seconds),
                max_retries=0,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except LiveModelProviderError:
            raise
        except Exception:
            raise LiveModelProviderError("live provider is unavailable") from None

    def complete(self, request: OpenAITransportRequest) -> OpenAITransportResponse:
        checked = OpenAITransportRequest.model_validate(
            request.model_dump(warnings="none"), strict=True
        )
        try:
            schema_value: object = json.loads(checked.response_schema_json)
            if type(schema_value) is not dict:
                raise ValueError
            responses = _attribute(self._client, "responses")
            create = cast(Callable[..., object], _attribute(responses, "create"))
            response = create(
                model=checked.model_name,
                input=checked.rendered_prompt,
                temperature=checked.temperature,
                max_output_tokens=checked.max_output_tokens,
                metadata={
                    "case_id": checked.case_id,
                    "operation": checked.operation.value,
                    "attempt_number": str(checked.attempt_number),
                    "prompt_version": checked.prompt_version,
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": checked.operation.value,
                        "strict": True,
                        "schema": schema_value,
                    }
                },
                tools=[],
                tool_choice="none",
                stream=False,
                store=False,
                timeout=float(checked.request_timeout_seconds),
            )
            usage = _attribute(response, "usage")
            input_tokens = _integer_attribute(usage, "input_tokens")
            output_tokens = _integer_attribute(usage, "output_tokens")
            total_tokens = _integer_attribute(usage, "total_tokens")
            if input_tokens + output_tokens != total_tokens:
                raise LiveModelProviderError("provider usage is invalid")
            status, raw_json = _completed_output(response)
            return OpenAITransportResponse(
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
TransportFactory = Callable[[str, int], OpenAITransport]


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


class OpenAIModelProvider(ModelProvider):
    def __init__(
        self,
        transport: OpenAITransport,
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
            transport_request = OpenAITransportRequest(
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
            checked_response = OpenAITransportResponse.model_validate(
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


def create_openai_model_provider(
    configuration: ModelConfiguration,
    live_configuration: LiveBenchmarkConfiguration,
    environment: Mapping[str, str],
    *,
    transport_factory: TransportFactory = SdkOpenAITransport,
    monotonic_clock: MonotonicClock = monotonic,
) -> OpenAIModelProvider:
    try:
        checked = ModelConfiguration.model_validate(
            configuration.model_dump(warnings="none"), strict=True
        )
        live_checked = LiveBenchmarkConfiguration.model_validate(
            live_configuration.model_dump(warnings="none"), strict=True
        )
        credential = environment[OPENAI_CREDENTIAL_NAME]
        if (
            type(credential) is not str
            or not credential.strip()
            or checked.provider != OPENAI_PROVIDER_NAME
            or checked.prompt_version != SUPPORTED_PROMPT_VERSION
        ):
            raise ValueError
        transport = transport_factory(credential, live_checked.request_timeout_seconds)
        return OpenAIModelProvider(transport, live_checked, monotonic_clock)
    except (KeyboardInterrupt, SystemExit):
        raise
    except LiveModelProviderError:
        raise
    except Exception:
        raise LiveModelConfigurationError("live provider configuration is invalid") from None
