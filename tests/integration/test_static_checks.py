from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pytest

from app.agent.policy import PatchPolicy, validate_patch_policy
from app.agent.preparation import RepositoryPreparer
from app.agent.schemas import DEFAULT_PROTECTED_PATHS, PatchProposal, ProtectedPathPolicy
from app.agent.static_checks import (
    LocalStaticFilesystem,
    Sha256RepositoryFingerprinter,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationService,
)
from app.sandbox.results import SandboxLimits
from app.sandbox.runner import (
    DockerCommandError,
    DockerCommandResult,
    SubprocessDockerCommandAdapter,
)
from app.sandbox.static_runner import (
    STATIC_CHECK_IMAGE,
    STATIC_CONTAINER_NAME_PREFIX,
    DockerStaticRunner,
    StaticContainerResult,
    StaticRunnerError,
    StaticRunnerErrorCode,
    StaticToolStatus,
)
from benchmarks.loader import load_trusted_case

pytestmark = pytest.mark.integration


def _docker(*arguments: str) -> DockerCommandResult:
    result = SubprocessDockerCommandAdapter().run(
        arguments,
        timeout_seconds=120.0,
        stdout_bytes=65_536,
        stderr_bytes=65_536,
        check=False,
    )
    if result.stdout_truncated or result.stderr_truncated:
        raise DockerCommandError("bounded Docker integration output was truncated")
    return result


@pytest.fixture(scope="module", autouse=True)
def require_docker_and_image() -> None:
    try:
        version = _docker("version", "--format", "{{.Server.Os}}")
    except DockerCommandError:
        pytest.skip("Docker CLI is unavailable")
    if version.exit_code != 0 or version.stdout.strip() != b"linux":
        pytest.skip("a usable Linux-container Docker daemon is unavailable")
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.exit_code != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
    if _docker("image", "inspect", STATIC_CHECK_IMAGE).exit_code != 0:
        pytest.fail(
            "static-check image is absent; run: docker build --tag "
            f"{STATIC_CHECK_IMAGE} --file docker/static-checks/Dockerfile ."
        )


def _candidate(tmp_path: Path, source: str) -> Path:
    candidate = tmp_path / "candidate"
    path = candidate / "src" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    if __import__("os").name != "nt":
        candidate.chmod(0o755)
        path.parent.chmod(0o755)
        path.chmod(0o644)
    return candidate.resolve()


def _containers() -> set[str]:
    response = _docker(
        "ps",
        "--all",
        "--filter",
        f"name=^{STATIC_CONTAINER_NAME_PREFIX}",
        "--format",
        "{{.Names}}",
    )
    assert response.exit_code == 0
    return set(response.stdout.decode("utf-8").splitlines())


class CorruptingDockerAdapter:
    def __init__(self) -> None:
        self._delegate = SubprocessDockerCommandAdapter()

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        result = self._delegate.run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )
        if tuple(arguments)[:1] == ("start",):
            return DockerCommandResult(result.exit_code, b"not-json", result.stderr)
        return result


def test_real_static_tools_pass_clean_candidate(tmp_path: Path) -> None:
    before = _containers()
    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(
        _candidate(tmp_path, "def double(value: int) -> int:\n    return value * 2\n"),
        ("src/example.py",),
    )

    for item in result.results:
        assert item.status is StaticToolStatus.PASSED, (
            item.check,
            item.status,
            item.error_code,
        )
    assert _containers() == before


def test_candidate_mypy_module_cannot_shadow_fixed_tool_executable(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "mypy.py").write_text("raise SystemExit(7)\n", encoding="utf-8")

    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(
        candidate.resolve(), ("mypy.py",)
    )

    assert all(item.status is StaticToolStatus.PASSED for item in result.results), result


def test_real_image_maps_isolated_ast_component_failure(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "value = 1\n")
    harness = tmp_path / "ast_harness.py"
    harness.write_text(
        """
import importlib.util
import json

specification = importlib.util.spec_from_file_location(
    "tracefix_static_image_runner", "/opt/tracefix/static_runner.py"
)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
module.REPOSITORY = module.INPUT_ROOT

def fail_ast(*args, **kwargs):
    raise SyntaxError("synthetic", ("src/example.py", 1, 1, "value = 1"))

module.ast.parse = fail_ast
print(json.dumps([
    module.syntax_result("python_compile", ("src/example.py",), parse_ast=False),
    module.syntax_result("ast_parse", ("src/example.py",), parse_ast=True),
]))
""".lstrip(),
        encoding="utf-8",
    )
    response = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--user",
        "10001:10001",
        "--cpus",
        "1",
        "--memory",
        "536870912",
        "--pids-limit",
        "128",
        "--mount",
        f"type=bind,source={candidate},target=/tracefix/input,readonly",
        "--mount",
        f"type=bind,source={harness.resolve()},target=/tracefix/ast_harness.py,readonly",
        "--mount",
        "type=tmpfs,destination=/tracefix/workspace,tmpfs-mode=1777",
        "--mount",
        "type=tmpfs,destination=/tmp,tmpfs-mode=1777",
        "--entrypoint",
        "python",
        STATIC_CHECK_IMAGE,
        "/tracefix/ast_harness.py",
    )

    assert response.exit_code == 0, response.stderr.decode("utf-8", errors="replace")
    results = json.loads(response.stdout)
    assert results[0]["status"] == "passed"
    assert results[1]["status"] == "failed"
    assert results[1]["findings"][0]["rule_code"] == "AST_PARSE"


def test_real_image_maps_missing_tool_to_infrastructure_evidence(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "value = 1\n")
    blocked_tool = tmp_path / "blocked-ruff"
    blocked_tool.write_bytes(b"")
    response = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--user",
        "10001:10001",
        "--cpus",
        "1",
        "--memory",
        "536870912",
        "--pids-limit",
        "128",
        "--mount",
        f"type=bind,source={candidate},target=/tracefix/input,readonly",
        "--mount",
        f"type=bind,source={blocked_tool.resolve()},target=/usr/local/bin/ruff,readonly",
        "--mount",
        "type=tmpfs,destination=/tracefix/workspace,tmpfs-mode=1777",
        "--mount",
        "type=tmpfs,destination=/tmp,tmpfs-mode=1777",
        STATIC_CHECK_IMAGE,
        "src/example.py",
    )

    assert response.exit_code == 0, response.stderr.decode("utf-8", errors="replace")
    result = StaticContainerResult.model_validate_json(response.stdout, strict=True)
    ruff = next(item for item in result.results if item.check.value == "ruff")
    assert ruff.status.value == "infrastructure_error"
    assert ruff.error_code is not None and ruff.error_code.value == "executable_missing"


def test_real_container_malformed_result_is_blocking_and_removed(tmp_path: Path) -> None:
    before = _containers()
    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(CorruptingDockerAdapter()).execute(
            _candidate(tmp_path, "value = 1\n"), ("src/example.py",)
        )

    assert raised.value.code is StaticRunnerErrorCode.MALFORMED_RESULT
    assert _containers() == before


def test_real_container_host_deadline_is_blocking_and_removed(tmp_path: Path) -> None:
    before = _containers()
    limits = SandboxLimits(timeout_seconds=1e-9, stdout_bytes=65_536, stderr_bytes=4_096)
    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(SubprocessDockerCommandAdapter(), limits=limits).execute(
            _candidate(tmp_path, "value = 1\n"), ("src/example.py",)
        )

    assert raised.value.code is StaticRunnerErrorCode.TIMED_OUT
    assert _containers() == before


@pytest.mark.parametrize(
    ("source", "failed_check"),
    [
        ("import os\n", "ruff"),
        ("def value() -> int:\n    return 'wrong'\n", "mypy"),
        (
            "import subprocess\n\n\ndef run() -> None:\n    subprocess.run(['true'], check=True)\n",
            "bandit",
        ),
    ],
)
def test_real_static_tools_report_isolated_findings(
    tmp_path: Path, source: str, failed_check: str
) -> None:
    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(
        _candidate(tmp_path, source), ("src/example.py",)
    )

    failed = {item.check.value for item in result.results if item.status is StaticToolStatus.FAILED}
    assert failed == {failed_check}, result


def test_real_compile_failure_is_blocking(tmp_path: Path) -> None:
    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(
        _candidate(tmp_path, "return 1\n"), ("src/example.py",)
    )

    failed = {item.check.value for item in result.results if item.status is StaticToolStatus.FAILED}
    assert "python_compile" in failed


def test_real_static_tools_aggregate_independent_findings(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "import subprocess\n\n"
        "def run() -> int:\n"
        "    subprocess.run(['true'])\n"
        "    return 'wrong'\n"
    )

    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(
        _candidate(tmp_path, source), ("src/example.py",)
    )

    failed = tuple(
        item.check.value for item in result.results if item.status is StaticToolStatus.FAILED
    )
    assert failed == ("ruff", "mypy", "bandit"), result
    for item in result.results:
        assert item.findings == tuple(
            sorted(
                item.findings,
                key=lambda finding: (
                    finding.path.casefold(),
                    finding.path,
                    finding.line or 0,
                    finding.column or 0,
                    finding.rule_code,
                ),
            )
        )


@pytest.mark.parametrize("case_id", ["boundary_condition", "exception_handling"])
def test_fixture_shaped_reference_patch_passes_complete_gate(tmp_path: Path, case_id: str) -> None:
    case = load_trusted_case(case_id)
    preparation_root = tmp_path / "preparations"
    candidate_root = tmp_path / "candidates"
    preparation_root.mkdir()
    candidate_root.mkdir()
    preparation = RepositoryPreparer(load_trusted_case).prepare(
        case_id,
        case.manifest.failing_revision,
        preparation_root,
    )
    proposal = PatchProposal(
        summary="trusted fixture repair",
        root_cause="trusted fixture defect",
        files_changed=list(case.manifest.expected_changed_files),
        unified_diff=case.reference_patch.read_text(encoding="utf-8"),
        expected_effect="trusted expected behavior",
        risks=[],
        confidence=1.0,
    )
    policy = PatchPolicy(
        protected_path_policy=ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS),
        forbidden_changed_files=tuple(
            sorted(
                (PurePosixPath(path) for path in case.manifest.forbidden_changed_files),
                key=lambda path: path.as_posix().casefold(),
            )
        ),
    )
    approved = validate_patch_policy(proposal, policy)
    filesystem = LocalStaticFilesystem()
    service = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=Sha256RepositoryFingerprinter(filesystem),
        runner=DockerStaticRunner(SubprocessDockerCommandAdapter()),
    )
    original_fingerprint = preparation.fingerprint

    result = service.validate(
        StaticValidationRequest(
            preparation=preparation,
            proposal=proposal,
            policy_result=approved,
            candidate_workspace_root=candidate_root.resolve(),
        )
    )

    assert result.outcome is StaticValidationOutcome.PASSED, result
    assert result.candidate is not None
    assert (
        Sha256RepositoryFingerprinter(filesystem).fingerprint(preparation.prepared_repository)
        == original_fingerprint
    )


def test_exact_patch_conflict_leaves_preparation_unchanged_and_no_candidate(
    tmp_path: Path,
) -> None:
    case = load_trusted_case("boundary_condition")
    preparation_root = tmp_path / "preparations"
    candidate_root = tmp_path / "candidates"
    preparation_root.mkdir()
    candidate_root.mkdir()
    preparation = RepositoryPreparer(load_trusted_case).prepare(
        case.manifest.case_id,
        case.manifest.failing_revision,
        preparation_root,
    )
    patch_lines = case.reference_patch.read_text(encoding="utf-8").splitlines(keepends=True)
    removal = next(
        index
        for index, line in enumerate(patch_lines)
        if line.startswith("-") and not line.startswith("---")
    )
    patch_lines[removal] = "-tracefix guaranteed context mismatch\n"
    proposal = PatchProposal(
        summary="conflicting fixture repair",
        root_cause="synthetic exact-match conflict",
        files_changed=list(case.manifest.expected_changed_files),
        unified_diff="".join(patch_lines),
        expected_effect="conflict must remain atomic",
        risks=[],
        confidence=1.0,
    )
    policy = PatchPolicy(
        protected_path_policy=ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS),
        forbidden_changed_files=tuple(
            sorted(
                (PurePosixPath(path) for path in case.manifest.forbidden_changed_files),
                key=lambda path: path.as_posix().casefold(),
            )
        ),
    )
    filesystem = LocalStaticFilesystem()
    service = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=Sha256RepositoryFingerprinter(filesystem),
        runner=DockerStaticRunner(SubprocessDockerCommandAdapter()),
    )
    original_fingerprint = preparation.fingerprint

    result = service.validate(
        StaticValidationRequest(
            preparation=preparation,
            proposal=proposal,
            policy_result=validate_patch_policy(proposal, policy),
            candidate_workspace_root=candidate_root.resolve(),
        )
    )

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert list(candidate_root.iterdir()) == []
    assert (
        Sha256RepositoryFingerprinter(filesystem).fingerprint(preparation.prepared_repository)
        == original_fingerprint
    )


def test_no_python_paths_are_not_applicable_in_real_container(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "README.md").write_text("documentation\n", encoding="utf-8")
    if __import__("os").name != "nt":
        candidate.chmod(0o755)
        (candidate / "README.md").chmod(0o644)

    result = DockerStaticRunner(SubprocessDockerCommandAdapter()).execute(candidate.resolve(), ())

    assert {item.status for item in result.results} == {StaticToolStatus.NOT_APPLICABLE}
