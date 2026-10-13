from __future__ import annotations

import itertools
import os
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest

from app.agent.policy import PatchPolicy, PatchPolicyResult, validate_patch_policy
from app.agent.preparation import RepositoryPreparer
from app.agent.schemas import DEFAULT_PROTECTED_PATHS, PatchProposal, ProtectedPathPolicy
from app.agent.static_checks import (
    LocalStaticFilesystem,
    Sha256RepositoryFingerprinter,
    StaticCheck,
    StaticCheckResult,
    StaticCheckStatus,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationResult,
    StaticValidationService,
    ValidatedCandidate,
)
from app.agent.verification import (
    CandidateVerificationRequest,
    LocalCandidateFilesystem,
    LocalEvaluatorBundleFilesystem,
    VerificationService,
    VerificationStageStatus,
)
from app.sandbox.runner import (
    DockerCommandError,
    DockerCommandResult,
    SubprocessDockerCommandAdapter,
)
from app.sandbox.static_runner import STATIC_CHECK_IMAGE, DockerStaticRunner
from app.sandbox.verification_runner import (
    VERIFICATION_CONTAINER_NAME_PREFIX,
    VerificationSandboxRunner,
)
from benchmarks.loader import TrustedCase, load_trusted_case

pytestmark = pytest.mark.integration
_INTEGRATION_IDENTIFIERS = itertools.count(1)

_TARGETED_FAILURE_PATCHES = {
    "boundary_condition": (
        "--- a/src/pagination/pagination.py\n"
        "+++ b/src/pagination/pagination.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def page_count(items: int, size: int) -> int:\n"
        "-    return items // size + 1\n"
        "+    return 0\n"
    ),
    "incorrect_conditional": (
        "--- a/src/eligibility/eligibility.py\n"
        "+++ b/src/eligibility/eligibility.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def is_eligible(age: int) -> bool:\n"
        "-    return age > 18\n"
        "+    return age > 19\n"
    ),
}

_HIDDEN_FAILURE_PATCH = (
    "--- a/src/pagination/pagination.py\n"
    "+++ b/src/pagination/pagination.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def page_count(items: int, size: int) -> int:\n"
    "-    return items // size + 1\n"
    "+    return 1\n"
)


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
def require_docker_and_images() -> None:
    try:
        version = _docker("version", "--format", "{{.Server.Os}}")
    except DockerCommandError:
        pytest.skip("Docker CLI is unavailable")
    if version.exit_code != 0 or version.stdout.strip() != b"linux":
        pytest.skip("a usable Linux-container Docker daemon is unavailable")
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.exit_code != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
    for image in ("tracefix-sandbox:0.1.3a", STATIC_CHECK_IMAGE):
        if _docker("image", "inspect", image).exit_code != 0:
            pytest.fail(f"required integration image is absent: {image}")


def _identifier_factory() -> Callable[[], str]:
    return lambda: f"{next(_INTEGRATION_IDENTIFIERS):032x}"


def _proposal(case: TrustedCase, diff: str | None = None) -> PatchProposal:
    return PatchProposal(
        summary="trusted fixture candidate",
        root_cause="trusted fixture defect",
        files_changed=list(case.manifest.expected_changed_files),
        unified_diff=(case.reference_patch.read_text(encoding="utf-8") if diff is None else diff),
        expected_effect="candidate behavior",
        risks=[],
        confidence=1.0,
    )


def _policy(case: TrustedCase, proposal: PatchProposal) -> PatchPolicyResult:
    policy = PatchPolicy(
        protected_path_policy=ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS),
        forbidden_changed_files=tuple(
            sorted(
                (PurePosixPath(path) for path in case.manifest.forbidden_changed_files),
                key=lambda path: (path.as_posix().casefold(), path.as_posix()),
            )
        ),
    )
    return validate_patch_policy(proposal, policy)


def _static_candidate(
    tmp_path: Path, case_id: str, diff: str | None = None
) -> tuple[TrustedCase, PatchProposal, StaticValidationResult, Path, object]:
    case = load_trusted_case(case_id)
    preparation_root = (tmp_path / "preparations").resolve()
    candidate_root = (tmp_path / "candidates").resolve()
    preparation_root.mkdir()
    candidate_root.mkdir()
    preparation = RepositoryPreparer(load_trusted_case).prepare(
        case_id,
        case.manifest.failing_revision,
        preparation_root,
        destination_name=f"{case_id}-prepared",
    )
    proposal = _proposal(case, diff)
    filesystem = LocalStaticFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    static = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=DockerStaticRunner(
            SubprocessDockerCommandAdapter(), identifier_factory=_identifier_factory()
        ),
        identifier_factory=_identifier_factory(),
    ).validate(
        StaticValidationRequest(
            preparation=preparation,
            proposal=proposal,
            policy_result=_policy(case, proposal),
            candidate_workspace_root=candidate_root,
        )
    )
    assert static.outcome is StaticValidationOutcome.PASSED, static
    assert static.candidate is not None
    return case, proposal, static, preparation.prepared_repository, preparation.fingerprint


def _verification_service() -> VerificationService:
    filesystem = LocalStaticFilesystem()
    return VerificationService(
        loader=load_trusted_case,
        fingerprinter=Sha256RepositoryFingerprinter(filesystem),
        candidate_filesystem=LocalCandidateFilesystem(),
        bundle_filesystem=LocalEvaluatorBundleFilesystem(),
        sandbox=VerificationSandboxRunner(
            SubprocessDockerCommandAdapter(), identifier_factory=_identifier_factory()
        ),
        identifier_factory=_identifier_factory(),
    )


def _verify(tmp_path: Path, case_id: str, static: StaticValidationResult):
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    return _verification_service().verify(
        CandidateVerificationRequest(
            case_id=case_id,
            static_result=static,
            verification_workspace_root=workspace,
        )
    )


def _containers() -> set[str]:
    response = _docker(
        "ps",
        "--all",
        "--filter",
        f"name=^{VERIFICATION_CONTAINER_NAME_PREFIX}",
        "--format",
        "{{.Names}}",
    )
    assert response.exit_code == 0
    return set(response.stdout.decode("utf-8").splitlines())


def _containers_eventually_equal(expected: set[str]) -> bool:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _containers() == expected:
            return True
        time.sleep(0.05)
    return _containers() == expected


@pytest.mark.parametrize("case_id", ["boundary_condition", "exception_handling"])
def test_reference_candidate_passes_all_three_real_stages(tmp_path: Path, case_id: str) -> None:
    case, _, static, preparation_repository, preparation_fingerprint = _static_candidate(
        tmp_path, case_id
    )
    assert static.candidate is not None
    filesystem = LocalStaticFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    candidate_fingerprint = fingerprinter.fingerprint(static.candidate.repository)
    before_containers = _containers()

    result = _verify(tmp_path, case.manifest.case_id, static)

    assert result.outcome.value == "passed", result
    assert tuple(stage.status for stage in result.stages) == (
        VerificationStageStatus.PASSED,
        VerificationStageStatus.PASSED,
        VerificationStageStatus.PASSED,
    )
    assert fingerprinter.fingerprint(static.candidate.repository) == candidate_fingerprint
    assert fingerprinter.fingerprint(preparation_repository) == preparation_fingerprint
    assert _containers_eventually_equal(before_containers)
    assert not any((tmp_path / "verification").iterdir())


@pytest.mark.parametrize("case_id", ["boundary_condition", "incorrect_conditional"])
def test_static_clean_but_functionally_wrong_candidate_fails_targeted_stage(
    tmp_path: Path, case_id: str
) -> None:
    _, _, static, _, _ = _static_candidate(tmp_path, case_id, _TARGETED_FAILURE_PATCHES[case_id])

    result = _verify(tmp_path, case_id, static)

    assert result.outcome.value == "candidate_failed"
    assert tuple(stage.status for stage in result.stages) == (
        VerificationStageStatus.FAILED,
        VerificationStageStatus.NOT_RUN,
        VerificationStageStatus.NOT_RUN,
    )
    assert not any((tmp_path / "verification").iterdir())


def test_visible_regression_short_circuits_before_hidden_real_execution(tmp_path: Path) -> None:
    case = load_trusted_case("boundary_condition")
    candidate = (tmp_path / "candidate").resolve()
    (candidate / "tests").mkdir(parents=True)
    (candidate / "tests" / "test_behavior.py").write_text(
        "def test_targeted():\n    assert True\n", encoding="utf-8"
    )
    (candidate / "tests" / "test_regression.py").write_text(
        "def test_regression():\n    assert False\n", encoding="utf-8"
    )
    _make_readable(candidate)
    static = _trusted_static_result(candidate)

    result = _verify(tmp_path, case.manifest.case_id, static)

    assert tuple(stage.status for stage in result.stages) == (
        VerificationStageStatus.PASSED,
        VerificationStageStatus.FAILED,
        VerificationStageStatus.NOT_RUN,
    )
    assert not any((tmp_path / "verification").iterdir())


def test_visible_pass_hidden_failure_is_candidate_failure(tmp_path: Path) -> None:
    _, _, static, _, _ = _static_candidate(tmp_path, "boundary_condition", _HIDDEN_FAILURE_PATCH)

    result = _verify(tmp_path, "boundary_condition", static)

    assert tuple(stage.status for stage in result.stages) == (
        VerificationStageStatus.PASSED,
        VerificationStageStatus.PASSED,
        VerificationStageStatus.FAILED,
    )
    assert result.outcome.value == "candidate_failed"
    assert not any((tmp_path / "verification").iterdir())


def test_real_hidden_bundle_is_read_only_and_sensitive_host_state_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = (tmp_path / "candidate").resolve()
    (candidate / "tests").mkdir(parents=True)
    (candidate / "src" / "pagination").mkdir(parents=True)
    (candidate / "src" / "pagination" / "__init__.py").write_text("", encoding="utf-8")
    (candidate / "src" / "pagination" / "pagination.py").write_text(
        "from pathlib import Path\n"
        "try:\n"
        "    Path('/tracefix/hidden/test_0000.py').write_text('tampered')\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise RuntimeError('hidden bundle was writable')\n"
        "def page_count(items: int, size: int) -> int:\n"
        "    return (items + size - 1) // size\n",
        encoding="utf-8",
    )
    (candidate / "tests" / "test_behavior.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def test_controls():\n"
        "    assert os.getuid() == 10001\n"
        "    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'\n"
        "    assert os.environ.get('PYTHONUNBUFFERED') == '1'\n"
        "    assert os.environ.get('PYTHONPATH') == '/tracefix/workspace/repository/src'\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n"
        "    assert 'HTTPS_PROXY' not in os.environ\n"
        "    assert not Path('/var/run/docker.sock').exists()\n"
        "    assert not Path('/run/docker.sock').exists()\n",
        encoding="utf-8",
    )
    _make_readable(candidate)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-container")
    monkeypatch.setenv("HTTPS_PROXY", "must-not-enter-container")

    result = _verify(tmp_path, "boundary_condition", _trusted_static_result(candidate))

    assert result.outcome.value == "passed", result
    assert not any((tmp_path / "verification").iterdir())


def _trusted_static_result(candidate: Path) -> StaticValidationResult:
    filesystem = LocalStaticFilesystem()
    fingerprint = Sha256RepositoryFingerprinter(filesystem).fingerprint(candidate)
    return StaticValidationResult(
        outcome=StaticValidationOutcome.PASSED,
        checks=tuple(
            StaticCheckResult(check=check, status=StaticCheckStatus.PASSED) for check in StaticCheck
        ),
        candidate=ValidatedCandidate(repository=candidate, fingerprint=fingerprint),
    )


def _make_readable(root: Path) -> None:
    if os.name == "nt":
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
