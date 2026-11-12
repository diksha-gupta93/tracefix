from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, cast

from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree
from pydantic import ValidationError

LANGSMITH_DIAGNOSTIC_PROJECT = "tracefix-live-diagnostics"
LANGSMITH_API_ENDPOINT = "https://api.smith.langchain.com"
TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE = "TRACEFIX_LANGSMITH_DIAGNOSTICS"

_AUTOMATIC_TRACING_ENVIRONMENT_VARIABLES = frozenset(
    {
        "LANGCHAIN_HANDLER",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
    }
)
_ENDPOINT_OVERRIDE_ENVIRONMENT_VARIABLES = frozenset(
    {
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_RUNS_ENDPOINTS",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_RUNS_ENDPOINTS",
    }
)
_TRANSPORT_OVERRIDE_ENVIRONMENT_VARIABLES = frozenset(
    {
        "LANGCHAIN_OTEL_ENABLED",
        "LANGCHAIN_OTEL_ONLY",
        "LANGCHAIN_TRACING_MODE",
        "LANGSMITH_OTEL_ENABLED",
        "LANGSMITH_OTEL_ONLY",
        "LANGSMITH_TRACING_MODE",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    }
)

_TRACE_NAMES = frozenset(
    {
        "tracefix.live_benchmark",
        "prepare_repository",
        "run_baseline",
        "classify_failure",
        "build_context",
        "plan_repair",
        "provider_request",
        "parse_repair_plan",
        "validate_repair_plan",
        "generate_patch",
        "parse_patch_proposal",
        "validate_patch_proposal",
        "validate_patch_envelope",
        "patch_policy",
        "static_checks",
        "verification",
        "evaluation",
    }
)
_CASE_TRACE_NAME = re.compile(r"^case\.[a-z0-9][a-z0-9_-]{0,127}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")

SAFE_METADATA_ALLOW_LIST = frozenset(
    {
        "benchmark_run_id",
        "case_id",
        "operation",
        "node_name",
        "stage_name",
        "provider_name",
        "model_name",
        "prompt_version",
        "attempt_number",
        "response_byte_count",
        "input_token_count",
        "output_token_count",
        "provider_latency_seconds",
        "finish_category",
        "json_parse_result",
        "schema_validation_result",
        "validation_field_path",
        "validation_error_type",
        "patch_envelope_result",
        "changed_file_count",
        "path_agreement_result",
        "unified_diff_result",
        "policy_result",
        "static_check_result",
        "verification_result",
        "evaluation_result",
        "terminal_diagnostic_code",
        "terminal_repair_status",
        "benchmark_classification",
        "execution_status",
    }
)

SafeMetadataValue = str | int | float | bool
SafeMetadata = Mapping[str, SafeMetadataValue]


class TracingConfigurationError(ValueError):
    """Enabled diagnostic tracing was not configured safely."""


class DiagnosticSpan(Protocol):
    def add_metadata(self, metadata: SafeMetadata) -> None: ...


class DiagnosticTracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def span(
        self, name: str, metadata: SafeMetadata | None = None
    ) -> AbstractContextManager[DiagnosticSpan]: ...

    def add_metadata(self, metadata: SafeMetadata) -> None: ...


class DiagnosticRun(Protocol):
    def add_metadata(self, metadata: SafeMetadata) -> None: ...

    def close(self) -> None: ...


class DiagnosticRunExporter(Protocol):
    def start_run(
        self,
        name: str,
        metadata: SafeMetadata,
        parent: DiagnosticRun | None,
    ) -> DiagnosticRun: ...


def _validated_trace_name(name: str) -> str:
    if type(name) is not str or (
        name not in _TRACE_NAMES and _CASE_TRACE_NAME.fullmatch(name) is None
    ):
        raise ValueError("diagnostic trace name is not allow-listed")
    return name


def safe_metadata(
    metadata: SafeMetadata | None = None, /, **values: SafeMetadataValue
) -> dict[str, SafeMetadataValue]:
    candidate = {} if metadata is None else dict(metadata)
    candidate.update(values)
    if not set(candidate).issubset(SAFE_METADATA_ALLOW_LIST):
        raise ValueError("diagnostic metadata key is not allow-listed")
    for key, value in candidate.items():
        if type(value) not in (str, int, float, bool):
            raise ValueError("diagnostic metadata value has an unsafe type")
        if type(value) is str:
            if not value or len(value) > 256 or any(ord(character) < 32 for character in value):
                raise ValueError("diagnostic metadata string is unsafe")
        elif type(value) is int:
            if not -(2**63) <= value <= 2**63 - 1:
                raise ValueError("diagnostic metadata integer is unsafe")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("diagnostic metadata float is unsafe")
        if key in {
            "finish_category",
            "json_parse_result",
            "schema_validation_result",
            "validation_error_type",
            "patch_envelope_result",
            "path_agreement_result",
            "unified_diff_result",
            "policy_result",
            "static_check_result",
            "verification_result",
            "evaluation_result",
            "terminal_diagnostic_code",
            "terminal_repair_status",
            "benchmark_classification",
            "execution_status",
        } and (type(value) is not str or _SAFE_CODE.fullmatch(value) is None):
            raise ValueError("diagnostic metadata code is unsafe")
    return candidate


def validation_error_metadata(
    error: ValidationError,
    *,
    allowed_fields: frozenset[str],
) -> dict[str, SafeMetadataValue]:
    """Return only the first safe field location and normalized Pydantic error type."""
    try:
        item = error.errors(include_url=False, include_context=False, include_input=False)[0]
        location = item.get("loc", ())
        error_type = item.get("type", "validation_error")
    except Exception:
        return {
            "validation_field_path": "$.unknown",
            "validation_error_type": "validation_error",
        }
    path = "$"
    if not isinstance(location, tuple):
        location = ()
    for part in location:
        if type(part) is int and 0 <= part <= 1_000_000:
            path += f"[{part}]"
        elif type(part) is str and part in allowed_fields:
            path += f".{part}"
        else:
            path = "$.unknown"
            break
    normalized_type = (
        error_type
        if type(error_type) is str and _SAFE_CODE.fullmatch(error_type) is not None
        else "validation_error"
    )
    return {
        "validation_field_path": path,
        "validation_error_type": normalized_type,
    }


class _NoOpSpan:
    def add_metadata(self, metadata: SafeMetadata) -> None:
        safe_metadata(metadata)


class NoOpDiagnosticTracer:
    @property
    def enabled(self) -> bool:
        return False

    @contextmanager
    def span(self, name: str, metadata: SafeMetadata | None = None) -> Iterator[DiagnosticSpan]:
        _validated_trace_name(name)
        safe_metadata(metadata)
        yield _NoOpSpan()

    def add_metadata(self, metadata: SafeMetadata) -> None:
        safe_metadata(metadata)


NOOP_TRACER: DiagnosticTracer = NoOpDiagnosticTracer()


class _LangSmithSpan:
    def __init__(self, run: DiagnosticRun) -> None:
        self._run = run

    def add_metadata(self, metadata: SafeMetadata) -> None:
        checked = safe_metadata(metadata)
        with suppress(Exception):
            self._run.add_metadata(checked)


class _LangSmithRun:
    def __init__(self, run: RunTree) -> None:
        self.run = run

    def add_metadata(self, metadata: SafeMetadata) -> None:
        self.run.add_metadata(dict(metadata))

    def close(self) -> None:
        self.run.end(outputs={})
        self.run.patch(exclude_inputs=True)


class LangSmithRunExporter:
    """Create metadata-only SDK runs with TraceFix-owned explicit parents."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def start_run(
        self,
        name: str,
        metadata: SafeMetadata,
        parent: DiagnosticRun | None,
    ) -> DiagnosticRun:
        if parent is not None and not isinstance(parent, _LangSmithRun):
            raise TypeError("diagnostic run parent is invalid")
        run = RunTree(
            name=name,
            run_type="chain",
            inputs={},
            outputs={},
            serialized={},
            events=[],
            tags=[],
            attachments={},
            extra={"metadata": dict(metadata)},
            parent_run=None if parent is None else parent.run,
            project_name=LANGSMITH_DIAGNOSTIC_PROJECT,
            ls_client=self._client,
            dangerously_allow_filesystem=False,
            replicas=(),
        )
        run.post()
        return _LangSmithRun(run)


_FAILED_RUN = object()


class LangSmithDiagnosticTracer:
    """Manual metadata-only LangSmith tracing with best-effort export."""

    def __init__(self, exporter: DiagnosticRunExporter) -> None:
        self._exporter = exporter
        self._current: ContextVar[DiagnosticRun | object | None] = ContextVar(
            "tracefix_diagnostic_run", default=None
        )

    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def span(self, name: str, metadata: SafeMetadata | None = None) -> Iterator[DiagnosticSpan]:
        checked_name = _validated_trace_name(name)
        checked_metadata = safe_metadata(metadata)
        parent = self._current.get()
        if parent is _FAILED_RUN:
            yield _NoOpSpan()
            return
        try:
            run = self._exporter.start_run(
                checked_name,
                checked_metadata,
                cast(DiagnosticRun | None, parent),
            )
        except Exception:
            failed_token = self._current.set(_FAILED_RUN)
            try:
                yield _NoOpSpan()
            finally:
                self._current.reset(failed_token)
            return
        span = _LangSmithSpan(run)
        token = self._current.set(run)
        try:
            yield span
        finally:
            self._current.reset(token)
            with suppress(Exception):
                run.close()

    def add_metadata(self, metadata: SafeMetadata) -> None:
        checked = safe_metadata(metadata)
        current = self._current.get()
        if current is not None and current is not _FAILED_RUN:
            with suppress(Exception):
                cast(DiagnosticRun, current).add_metadata(checked)


@contextmanager
def automatic_langsmith_tracing_disabled() -> Iterator[None]:
    """Prevent LangChain and LangGraph from installing automatic SDK tracers."""
    with tracing_context(enabled=False, parent=False):
        yield


@dataclass(frozen=True, slots=True)
class DiagnosticTracingConfiguration:
    enabled: bool
    project_name: str = LANGSMITH_DIAGNOSTIC_PROJECT


def load_diagnostic_tracing_configuration(
    environment: Mapping[str, str],
) -> DiagnosticTracingConfiguration:
    value = environment.get(TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE, "false")
    if value == "false":
        return DiagnosticTracingConfiguration(enabled=False)
    if value != "true":
        raise TracingConfigurationError("diagnostic tracing configuration is invalid")
    if any(
        environment.get(name, "").strip() not in {"", "false"}
        for name in _AUTOMATIC_TRACING_ENVIRONMENT_VARIABLES
    ):
        raise TracingConfigurationError("diagnostic tracing configuration is invalid")
    if any(environment.get(name, "").strip() for name in _ENDPOINT_OVERRIDE_ENVIRONMENT_VARIABLES):
        raise TracingConfigurationError("diagnostic tracing configuration is invalid")
    if any(environment.get(name, "").strip() for name in _TRANSPORT_OVERRIDE_ENVIRONMENT_VARIABLES):
        raise TracingConfigurationError("diagnostic tracing configuration is invalid")
    credential = environment.get("LANGSMITH_API_KEY")
    project = environment.get("LANGSMITH_PROJECT", LANGSMITH_DIAGNOSTIC_PROJECT)
    if (
        type(credential) is not str
        or not credential.strip()
        or environment.get("LANGSMITH_HIDE_INPUTS") != "true"
        or environment.get("LANGSMITH_HIDE_OUTPUTS") != "true"
        or project != LANGSMITH_DIAGNOSTIC_PROJECT
    ):
        raise TracingConfigurationError("diagnostic tracing configuration is invalid")
    return DiagnosticTracingConfiguration(enabled=True)


def create_diagnostic_tracer(environment: Mapping[str, str]) -> DiagnosticTracer:
    configuration = load_diagnostic_tracing_configuration(environment)
    if not configuration.enabled:
        return NOOP_TRACER
    credential = environment["LANGSMITH_API_KEY"]
    try:
        client = Client(
            api_url=LANGSMITH_API_ENDPOINT,
            api_key=credential,
            api_urls={},
            auto_batch_tracing=True,
            hide_inputs=True,
            hide_outputs=True,
            omit_traced_runtime_info=True,
            tracing_mode="langsmith",
            tracing_error_callback=lambda _error: None,
        )
        if (
            client.tracing_mode != "langsmith"
            or client.api_url != LANGSMITH_API_ENDPOINT
            or client.otel_exporter is not None
        ):
            raise TracingConfigurationError("diagnostic tracing initialization failed")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise TracingConfigurationError("diagnostic tracing initialization failed") from None
    return LangSmithDiagnosticTracer(LangSmithRunExporter(client))
