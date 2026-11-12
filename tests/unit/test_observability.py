from __future__ import annotations

import json
from pathlib import PurePosixPath

import pytest
from pydantic import ValidationError

import app.observability as observability
from app.agent.schemas import PatchProposal
from app.observability import (
    LANGSMITH_API_ENDPOINT,
    LANGSMITH_DIAGNOSTIC_PROJECT,
    NOOP_TRACER,
    SAFE_METADATA_ALLOW_LIST,
    TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE,
    DiagnosticRun,
    LangSmithDiagnosticTracer,
    SafeMetadata,
    TracingConfigurationError,
    create_diagnostic_tracer,
    load_diagnostic_tracing_configuration,
    safe_metadata,
    validation_error_metadata,
)


def enabled_environment(**updates: str) -> dict[str, str]:
    environment = {
        TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE: "true",
        "LANGSMITH_API_KEY": "sentinel-api-key",
        "LANGSMITH_PROJECT": LANGSMITH_DIAGNOSTIC_PROJECT,
        "LANGSMITH_HIDE_INPUTS": "true",
        "LANGSMITH_HIDE_OUTPUTS": "true",
        "LANGSMITH_TRACING": "false",
    }
    environment.update(updates)
    return environment


class RecordingRun:
    def __init__(
        self,
        name: str,
        metadata: SafeMetadata,
        parent: DiagnosticRun | None,
        *,
        close_failure: bool = False,
        metadata_failure: bool = False,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata)
        self.parent = parent
        self.close_failure = close_failure
        self.metadata_failure = metadata_failure
        self.closed = False

    def add_metadata(self, metadata: SafeMetadata) -> None:
        if self.metadata_failure:
            raise RuntimeError("SENTINEL_METADATA_EXPORT_FAILURE")
        self.metadata.update(metadata)

    def close(self) -> None:
        self.closed = True
        if self.close_failure:
            raise RuntimeError("SENTINEL_EXPORT_CLOSE_FAILURE")


class RecordingExporter:
    def __init__(self) -> None:
        self.runs: list[RecordingRun] = []
        self.fail_next_start = False
        self.close_failure_names: set[str] = set()
        self.metadata_failure_names: set[str] = set()

    def start_run(
        self, name: str, metadata: SafeMetadata, parent: DiagnosticRun | None
    ) -> DiagnosticRun:
        if self.fail_next_start:
            self.fail_next_start = False
            raise RuntimeError("SENTINEL_EXPORT_START_FAILURE")
        run = RecordingRun(
            name,
            metadata,
            parent,
            close_failure=name in self.close_failure_names,
            metadata_failure=name in self.metadata_failure_names,
        )
        self.runs.append(run)
        return run


def test_tracing_is_disabled_by_default_and_needs_no_credential() -> None:
    configuration = load_diagnostic_tracing_configuration({})

    assert configuration.enabled is False
    assert create_diagnostic_tracer({}) is NOOP_TRACER


@pytest.mark.parametrize(
    "environment",
    [
        {TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE: "TRUE"},
        {TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE: "true"},
        {
            TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE: "true",
            "LANGSMITH_API_KEY": "key",
            "LANGSMITH_HIDE_INPUTS": "true",
            "LANGSMITH_HIDE_OUTPUTS": "false",
        },
        {
            TRACEFIX_DIAGNOSTICS_ENVIRONMENT_VARIABLE: "true",
            "LANGSMITH_API_KEY": "key",
            "LANGSMITH_HIDE_INPUTS": "true",
            "LANGSMITH_HIDE_OUTPUTS": "true",
            "LANGSMITH_PROJECT": "unsafe-project",
        },
        enabled_environment(LANGSMITH_TRACING="true"),
        enabled_environment(LANGCHAIN_TRACING_V2="true"),
    ],
)
def test_enabled_tracing_requires_complete_fixed_safe_configuration(
    environment: dict[str, str],
) -> None:
    with pytest.raises(TracingConfigurationError):
        load_diagnostic_tracing_configuration(environment)


def test_langsmith_client_boundary_always_hides_inputs_outputs_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **values: object) -> None:
            captured.update(values)
            self.api_url = values["api_url"]
            self.tracing_mode = values["tracing_mode"]
            self.otel_exporter = None

    monkeypatch.setattr(observability, "Client", FakeClient)
    tracer = create_diagnostic_tracer(enabled_environment())

    assert isinstance(tracer, LangSmithDiagnosticTracer)
    assert captured["api_url"] == LANGSMITH_API_ENDPOINT
    assert captured["api_urls"] == {}
    assert captured["hide_inputs"] is True
    assert captured["hide_outputs"] is True
    assert captured["omit_traced_runtime_info"] is True
    assert captured["tracing_mode"] == "langsmith"
    assert "sentinel-api-key" not in json.dumps(
        {key: value for key, value in captured.items() if key != "api_key"}, default=str
    )


@pytest.mark.parametrize(
    "name",
    [
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_RUNS_ENDPOINTS",
        "LANGCHAIN_RUNS_ENDPOINTS",
    ],
)
def test_ambient_endpoint_overrides_are_rejected_without_exposing_values_or_creating_client(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "https://SENTINEL-CREDENTIAL-EXFILTRATION.invalid"
    constructed = False

    class UnexpectedClient:
        def __init__(self, **values: object) -> None:
            del values
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(observability, "Client", UnexpectedClient)
    environment = enabled_environment(**{name: sentinel})

    with pytest.raises(TracingConfigurationError) as raised:
        create_diagnostic_tracer(environment)

    assert constructed is False
    assert sentinel not in str(raised.value)


@pytest.mark.parametrize(
    "name",
    [
        "LANGSMITH_TRACING_MODE",
        "LANGCHAIN_TRACING_MODE",
        "LANGSMITH_OTEL_ENABLED",
        "LANGCHAIN_OTEL_ENABLED",
        "LANGSMITH_OTEL_ONLY",
        "LANGCHAIN_OTEL_ONLY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ],
)
def test_ambient_transport_overrides_are_rejected_before_client_construction(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SENTINEL-UNSAFE-TRANSPORT-VALUE"
    constructed = False

    class UnexpectedClient:
        def __init__(self, **values: object) -> None:
            del values
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(observability, "Client", UnexpectedClient)

    with pytest.raises(TracingConfigurationError) as raised:
        create_diagnostic_tracer(enabled_environment(**{name: sentinel}))

    assert constructed is False
    assert sentinel not in str(raised.value)


@pytest.mark.parametrize(
    ("effective_mode", "effective_url", "otel_exporter"),
    [
        ("otel", LANGSMITH_API_ENDPOINT, object()),
        ("hybrid", LANGSMITH_API_ENDPOINT, object()),
        ("langsmith", "https://SENTINEL-UNSAFE-ENDPOINT.invalid", None),
        ("langsmith", LANGSMITH_API_ENDPOINT, object()),
    ],
)
def test_client_effective_transport_invariants_fail_closed_without_network(
    effective_mode: str,
    effective_url: str,
    otel_exporter: object | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsafeClient:
        def __init__(self, **values: object) -> None:
            assert values["tracing_mode"] == "langsmith"
            self.tracing_mode = effective_mode
            self.api_url = effective_url
            self.otel_exporter = otel_exporter

    monkeypatch.setattr(observability, "Client", UnsafeClient)

    with pytest.raises(TracingConfigurationError) as raised:
        create_diagnostic_tracer(enabled_environment())

    assert "SENTINEL" not in str(raised.value)


def test_metadata_boundary_rejects_unknown_keys_and_unsafe_value_types() -> None:
    with pytest.raises(ValueError):
        safe_metadata({"prompt": "must-not-leave"})
    with pytest.raises(ValueError):
        safe_metadata({"case_id": {"source": "must-not-leave"}})  # type: ignore[dict-item]

    checked = safe_metadata(case_id="boundary_condition", changed_file_count=1)

    assert set(checked).issubset(SAFE_METADATA_ALLOW_LIST)


def test_pydantic_validation_metadata_excludes_input_context_and_unknown_field_name() -> None:
    sentinel = "SENTINEL_SECRET_SOURCE_PROMPT_DIFF"
    values = {
        "summary": "summary",
        "root_cause": "cause",
        "files_changed": ["src/example.py"],
        "unified_diff": "not relevant",
        "expected_effect": "effect",
        "risks": ["risk"],
        "confidence": sentinel,
        sentinel: sentinel,
    }
    with pytest.raises(ValidationError) as raised:
        PatchProposal.model_validate(values, strict=True)

    metadata = validation_error_metadata(
        raised.value,
        allowed_fields=frozenset(PatchProposal.model_fields),
    )
    rendered = json.dumps(metadata, sort_keys=True)

    assert metadata["validation_field_path"] == "$.confidence"
    assert metadata["validation_error_type"] == "value_error"
    assert sentinel not in rendered
    assert "input" not in rendered
    assert "ctx" not in rendered


def test_noop_tracer_validates_but_emits_nothing() -> None:
    with NOOP_TRACER.span(
        "tracefix.live_benchmark",
        safe_metadata(benchmark_run_id="a" * 32),
    ) as span:
        span.add_metadata(safe_metadata(execution_status="completed"))

    assert NOOP_TRACER.enabled is False


def test_langsmith_span_creation_and_export_failures_are_best_effort() -> None:
    exporter = RecordingExporter()
    exporter.fail_next_start = True
    exporter.metadata_failure_names.add("generate_patch")
    tracer = LangSmithDiagnosticTracer(exporter)
    reached = False

    with tracer.span("plan_repair", safe_metadata(case_id="boundary_condition")):
        reached = True
    with tracer.span("generate_patch") as span:
        span.add_metadata(safe_metadata(schema_validation_result="passed"))
        tracer.add_metadata(safe_metadata(terminal_repair_status="repair_failed"))

    assert reached is True


def test_explicit_parent_tracking_survives_start_close_and_parent_completion_failures() -> None:
    exporter = RecordingExporter()
    exporter.close_failure_names.add("case.boundary_condition")
    tracer = LangSmithDiagnosticTracer(exporter)

    with tracer.span("tracefix.live_benchmark"):
        with tracer.span("case.boundary_condition"), tracer.span("plan_repair"):
            pass
        exporter.fail_next_start = True
        with tracer.span("case.pagination"), tracer.span("plan_repair"):
            pass
        with tracer.span("case.schema_validation"), tracer.span("plan_repair"):
            pass

    root, first_case, first_plan, final_case, final_plan = exporter.runs
    assert first_case.parent is root
    assert first_plan.parent is first_case
    assert final_case.parent is root
    assert final_plan.parent is final_case
    assert final_case.parent is not first_case
    assert all(run.closed for run in exporter.runs)


def test_langsmith_run_exporter_builds_content_free_runs_with_explicit_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeRunTree:
        def __init__(self, **values: object) -> None:
            self.values = values
            captured.append(values)

        def post(self) -> None:
            return None

        def add_metadata(self, metadata: dict[str, object]) -> None:
            extra = self.values["extra"]
            assert isinstance(extra, dict)
            recorded = extra["metadata"]
            assert isinstance(recorded, dict)
            recorded.update(metadata)

        def end(self, *, outputs: dict[str, object]) -> None:
            self.values["closed_outputs"] = outputs

        def patch(self, *, exclude_inputs: bool) -> None:
            self.values["exclude_inputs"] = exclude_inputs

    monkeypatch.setattr(observability, "RunTree", FakeRunTree)
    exporter = observability.LangSmithRunExporter(object())  # type: ignore[arg-type]
    tracer = LangSmithDiagnosticTracer(exporter)

    with (
        tracer.span("tracefix.live_benchmark", safe_metadata(benchmark_run_id="b" * 32)),
        tracer.span("case.boundary_condition", safe_metadata(case_id="boundary_condition")),
    ):
        tracer.add_metadata(safe_metadata(execution_status="completed"))

    assert len(captured) == 2
    root, child = captured
    assert root["parent_run"] is None
    assert child["parent_run"] is not None
    for run in captured:
        assert run["inputs"] == {}
        assert run["outputs"] == {}
        assert run["serialized"] == {}
        assert run["tags"] == []
        assert run["events"] == []
        assert run["attachments"] == {}
        assert run["dangerously_allow_filesystem"] is False
        assert run["replicas"] == ()
        assert run["project_name"] == LANGSMITH_DIAGNOSTIC_PROJECT
        assert run["closed_outputs"] == {}
        assert run["exclude_inputs"] is True


def test_enabled_fake_tracing_does_not_change_benchmark_case_outputs() -> None:
    from app.agent.routing import TerminalOutcome
    from app.evaluation.benchmark import BenchmarkCaseExecution, run_benchmark_cases
    from benchmarks.loader import load_development_cases

    def execute(case: object) -> BenchmarkCaseExecution:
        case_id = case.manifest.case_id  # type: ignore[attr-defined]
        return BenchmarkCaseExecution(
            case_id=case_id,
            terminal_outcome=TerminalOutcome.MALFORMED_MODEL_OUTPUT,
            model_usage=None,
            artifact_path=PurePosixPath(".tracefix", "artifacts", f"{case_id}-{'a' * 32}.json"),
        )

    exporter = RecordingExporter()
    tracer = LangSmithDiagnosticTracer(exporter)
    expected = run_benchmark_cases(load_development_cases(), execute)
    with tracer.span("tracefix.live_benchmark", safe_metadata(benchmark_run_id="b" * 32)):
        actual = run_benchmark_cases(load_development_cases(), execute)

    assert actual == expected
