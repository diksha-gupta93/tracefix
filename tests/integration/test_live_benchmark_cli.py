from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import distribution
from io import StringIO
from pathlib import Path

import pytest

from app.agent.model import load_model_configuration
from app.agent.schemas import PatchProposal, RepairPlan
from app.artifacts import LocalArtifactWriter
from app.cli import (
    ApplicationDependencies,
    BenchmarkApplication,
    BenchmarkApplicationDependencies,
    DefaultGraphDependenciesFactory,
    LocalWorkspaceManager,
    ProviderRegistry,
    RepairApplication,
    main,
)
from app.evaluation.benchmark import BenchmarkExecutionStatus, BenchmarkReport
from app.evaluation.live_model import (
    LiveBenchmarkConfiguration,
    OpenAICompletionStatus,
    OpenAITransportRequest,
    OpenAITransportResponse,
    create_openai_model_provider,
    load_live_benchmark_configuration,
)
from app.evaluation.reporting import LocalBenchmarkReportWriter
from app.sandbox.runner import SANDBOX_IMAGE
from app.sandbox.static_runner import STATIC_CHECK_IMAGE
from benchmarks.loader import load_development_cases, load_trusted_case

pytestmark = pytest.mark.integration

_FILES = {
    "boundary_condition": "src/pagination/pagination.py",
    "exception_handling": "src/parser/parser.py",
    "fixture_or_mocking": "src/greeter/greeter.py",
    "incorrect_conditional": "src/eligibility/eligibility.py",
    "incorrect_return_value": "src/normalizer/normalizer.py",
}
_DIFFS = {
    "boundary_condition": (
        "--- a/src/pagination/pagination.py\n"
        "+++ b/src/pagination/pagination.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def page_count(items: int, size: int) -> int:\n"
        "-    return items // size + 1\n"
        "+    return (items + size - 1) // size\n"
    ),
    "exception_handling": (
        "--- a/src/parser/parser.py\n"
        "+++ b/src/parser/parser.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def parse_port(value: str) -> int | None:\n"
        "     try:\n"
        "         return int(value)\n"
        "-    except TypeError:\n"
        "+    except (TypeError, ValueError):\n"
        "         return None\n"
    ),
    "fixture_or_mocking": (
        "--- a/src/greeter/greeter.py\n"
        "+++ b/src/greeter/greeter.py\n"
        "@@ -6,4 +6,4 @@ class UserClient(Protocol):\n"
        " \n"
        " \n"
        " def greeting(client: UserClient) -> str:\n"
        '-    return f"Hello, {client.fetch_users()}!"  # type: ignore[attr-defined]\n'
        '+    return f"Hello, {client.fetch_user()}!"\n'
    ),
    "incorrect_conditional": (
        "--- a/src/eligibility/eligibility.py\n"
        "+++ b/src/eligibility/eligibility.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def is_eligible(age: int) -> bool:\n"
        "-    return age > 18\n"
        "+    return age >= 18\n"
    ),
    "incorrect_return_value": (
        "--- a/src/normalizer/normalizer.py\n"
        "+++ b/src/normalizer/normalizer.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def normalize(name: str) -> str:\n"
        "-    return name.upper()\n"
        "+    return name.strip().lower()\n"
    ),
}


def _docker(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("docker", *arguments),
        check=False,
        capture_output=True,
        shell=False,
        timeout=120.0,
    )


def _unavailable(message: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module", autouse=True)
def require_docker_and_images() -> None:
    try:
        version = _docker("version", "--format", "{{.Server.Os}}")
    except (OSError, subprocess.TimeoutExpired):
        _unavailable("Docker CLI is unavailable")
        return
    if version.returncode != 0 or version.stdout.strip() != b"linux":
        _unavailable("a usable Linux-container Docker daemon is unavailable")
        return
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.returncode != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
    for image in (SANDBOX_IMAGE, STATIC_CHECK_IMAGE):
        if _docker("image", "inspect", image).returncode != 0:
            pytest.fail(f"required integration image is absent: {image}")


def _environment(runtime_temp: Path) -> dict[str, str]:
    return {
        "MODEL_PROVIDER": "openai",
        "MODEL_NAME": "reviewed-model-name",
        "MODEL_TEMPERATURE": "0",
        "MODEL_MAX_TOKENS": "4096",
        "PROMPT_VERSION": "repair-v1.0",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "120",
        "MODEL_INPUT_PRICE_USD_PER_MILLION_TOKENS": "0.15",
        "MODEL_OUTPUT_PRICE_USD_PER_MILLION_TOKENS": "0.60",
        "OPENAI_API_KEY": "integration-credential-must-not-leak",
        "TEMP": str(runtime_temp),
        "TMP": str(runtime_temp),
        "TMPDIR": str(runtime_temp),
    }


class TickClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        current = self._value
        self._value += 0.1
        return current


class ScriptedOpenAITransport:
    def __init__(self, requests: list[OpenAITransportRequest], *, fail: bool = False) -> None:
        self._requests = requests
        self._fail = fail

    def complete(self, request: OpenAITransportRequest) -> OpenAITransportResponse:
        self._requests.append(request)
        if self._fail:
            raise RuntimeError("untrusted provider failure body")
        path = _FILES[request.case_id]
        if request.operation.value == "repair_planning":
            raw_json = RepairPlan(
                suspected_root_cause="scripted integration diagnosis",
                files_expected_to_change=[path],
                intended_behavioural_correction="restore declared behavior",
                risks=["preserve all other behavior"],
                validation_strategy="run all configured checks",
                autonomous_repair_suitable=True,
            ).model_dump_json()
        else:
            raw_json = PatchProposal(
                summary="scripted typed transport proposal",
                root_cause="the fixture contains its declared defect",
                files_changed=[path],
                unified_diff=_DIFFS[request.case_id],
                expected_effect="the declared behavior is restored",
                risks=["preserve all other behavior"],
                confidence=1.0,
            ).model_dump_json()
        return OpenAITransportResponse(
            status=OpenAICompletionStatus.COMPLETED,
            raw_json=raw_json,
            input_tokens=100,
            output_tokens=50,
        )


def _application(
    invocation: Path,
    runtime_temp: Path,
    requests: list[OpenAITransportRequest],
    *,
    fail_provider: bool = False,
    stdout: StringIO | None = None,
    remove_environment: str | None = None,
) -> BenchmarkApplication:
    environment = _environment(runtime_temp)
    if remove_environment is not None:
        environment.pop(remove_environment)
    output = StringIO() if stdout is None else stdout
    stderr = StringIO()
    repair_ids = iter(f"{index:032x}" for index in range(1, 10))
    repair = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=load_model_configuration,
            provider_registry=ProviderRegistry(),
            workspace_manager=LocalWorkspaceManager(runtime_temp),
            artifact_writer=LocalArtifactWriter(),
            graph_dependencies_factory=DefaultGraphDependenciesFactory(),
            clock=lambda: datetime.now(UTC),
            run_id_factory=lambda: next(repair_ids),
            invocation_directory=lambda: invocation,
            environment=environment,
            stdout=output,
            stderr=stderr,
        )
    )

    def provider_factory(
        configuration: object,
        live_configuration: LiveBenchmarkConfiguration,
        source: object,
    ):
        assert isinstance(source, dict)
        return create_openai_model_provider(
            configuration,  # type: ignore[arg-type]
            live_configuration,
            source,
            transport_factory=lambda credential, timeout: ScriptedOpenAITransport(
                requests, fail=fail_provider
            ),
            monotonic_clock=TickClock(),
        )

    return BenchmarkApplication(
        BenchmarkApplicationDependencies(
            repair_application=repair,
            development_case_loader=load_development_cases,
            configuration_loader=load_model_configuration,
            live_configuration_loader=load_live_benchmark_configuration,
            live_provider_factory=provider_factory,
            report_writer=LocalBenchmarkReportWriter(),
            clock=lambda: datetime.now(UTC),
            run_id_factory=lambda: "b" * 32 if not fail_provider else "c" * 32,
            invocation_directory=lambda: invocation,
            environment=environment,
            stdout=output,
            stderr=stderr,
        )
    )


def _unused_repair_application() -> RepairApplication:
    raise AssertionError("repair application must not be composed for benchmark command")


def test_injected_live_transport_runs_all_cases_through_real_pipeline(tmp_path: Path) -> None:
    invocation = (tmp_path / "invocation").resolve()
    runtime_temp = (tmp_path / "runtime").resolve()
    invocation.mkdir()
    runtime_temp.mkdir()
    requests: list[OpenAITransportRequest] = []
    application = _application(invocation, runtime_temp, requests)

    entry_point = next(
        item
        for item in distribution("tracefix").entry_points
        if item.group == "console_scripts" and item.name == "tracefix"
    )
    assert entry_point.value == "app.cli:main"
    result = entry_point.load()(
        ["benchmark", "--split", "development", "--provider", "openai"],
        application_factory=_unused_repair_application,
        benchmark_application_factory=lambda: application,
    )

    assert result == 0
    assert len(requests) == 10
    assert [request.case_id for request in requests[::2]] == list(_FILES)
    report_path = invocation / ".tracefix" / "benchmarks" / ("b" * 32) / "report.json"
    report = BenchmarkReport.model_validate_json(
        report_path.read_text(encoding="utf-8"), strict=True
    )
    assert report.execution_status is BenchmarkExecutionStatus.COMPLETED
    assert report.counts.repair_success == 5
    assert report.aggregate_usage.input_tokens == 1000
    assert report.aggregate_usage.output_tokens == 500
    assert report.aggregate_usage.estimated_cost_usd == Decimal("0.00045")
    assert len(tuple((invocation / ".tracefix" / "artifacts").glob("*.json"))) == 5
    published = report_path.read_text(encoding="utf-8") + (
        report_path.with_suffix(".md").read_text(encoding="utf-8")
    )
    assert _environment(runtime_temp)["OPENAI_API_KEY"] not in published
    assert not tuple(runtime_temp.glob("tracefix-run-*"))


def test_provider_failure_publishes_partial_report_and_stops(tmp_path: Path) -> None:
    invocation = (tmp_path / "invocation-failure").resolve()
    runtime_temp = (tmp_path / "runtime-failure").resolve()
    invocation.mkdir()
    runtime_temp.mkdir()
    requests: list[OpenAITransportRequest] = []
    application = _application(invocation, runtime_temp, requests, fail_provider=True)

    result = main(
        ["benchmark", "--split", "development", "--provider", "openai"],
        application_factory=_unused_repair_application,
        benchmark_application_factory=lambda: application,
    )

    assert result == 6
    assert len(requests) == 1
    report_path = invocation / ".tracefix" / "benchmarks" / ("c" * 32) / "report.json"
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert raw["counts"]["execution_failure"] == 1
    assert raw["counts"]["not_run"] == 4
    assert [item["classification"] for item in raw["cases"]] == [
        "execution_failure",
        "not_run",
        "not_run",
        "not_run",
        "not_run",
    ]
    assert "untrusted provider failure body" not in json.dumps(raw)
    assert not tuple(runtime_temp.glob("tracefix-run-*"))


@pytest.mark.parametrize(
    ("failure", "expected_exit"),
    [(OSError, 6), (KeyboardInterrupt, 130)],
)
def test_console_failure_discards_completed_report_pair(
    tmp_path: Path, failure: type[BaseException], expected_exit: int
) -> None:
    class FailingOutput(StringIO):
        def write(self, value: str) -> int:
            del value
            raise failure("injected console failure")

    invocation = (tmp_path / "invocation-output-failure").resolve()
    runtime_temp = (tmp_path / "runtime-output-failure").resolve()
    invocation.mkdir()
    runtime_temp.mkdir()
    application = _application(invocation, runtime_temp, [], stdout=FailingOutput())

    result = main(
        ["benchmark", "--split", "development", "--provider", "openai"],
        application_factory=_unused_repair_application,
        benchmark_application_factory=lambda: application,
    )

    assert result == expected_exit
    assert not (invocation / ".tracefix" / "benchmarks" / ("b" * 32)).exists()


def test_preflight_failure_does_not_fabricate_report(tmp_path: Path) -> None:
    invocation = (tmp_path / "invocation-preflight-failure").resolve()
    runtime_temp = (tmp_path / "runtime-preflight-failure").resolve()
    invocation.mkdir()
    runtime_temp.mkdir()
    requests: list[OpenAITransportRequest] = []
    application = _application(
        invocation,
        runtime_temp,
        requests,
        remove_environment="MODEL_REQUEST_TIMEOUT_SECONDS",
    )

    result = main(
        ["benchmark", "--split", "development", "--provider", "openai"],
        application_factory=_unused_repair_application,
        benchmark_application_factory=lambda: application,
    )

    assert result == 6
    assert requests == []
    assert not (invocation / ".tracefix" / "benchmarks").exists()
