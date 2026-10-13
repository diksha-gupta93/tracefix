from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from app.agent.schemas import RepositoryFingerprint
from app.agent.static_checks import (
    StaticCheck,
    StaticCheckResult,
    StaticCheckStatus,
    StaticValidationOutcome,
    StaticValidationResult,
    ValidatedCandidate,
)
from app.agent.verification import (
    BundleCleanupError,
    CandidateInspection,
    CandidateVerificationRequest,
    CandidateVerificationResult,
    EvaluatorInputError,
    LocalCandidateFilesystem,
    LocalEvaluatorBundleFilesystem,
    VerificationDiagnostic,
    VerificationDiagnosticCode,
    VerificationFilesystemError,
    VerificationService,
    VerificationStage,
    VerificationStageResult,
    VerificationStageStatus,
)
from app.sandbox.results import SandboxCompletion
from app.sandbox.verification_runner import (
    HiddenTestBundle,
    VerificationRunnerError,
    VerificationRunnerErrorCode,
    VerificationSandboxEvidence,
)
from benchmarks.loader import TrustedCase, load_trusted_case


def _fingerprint(value: str = "a") -> RepositoryFingerprint:
    return RepositoryFingerprint(digest=value * 64)


def _static_result(repository: Path) -> StaticValidationResult:
    return StaticValidationResult(
        outcome=StaticValidationOutcome.PASSED,
        checks=tuple(
            StaticCheckResult(check=check, status=StaticCheckStatus.PASSED) for check in StaticCheck
        ),
        candidate=ValidatedCandidate(repository=repository, fingerprint=_fingerprint()),
    )


def _evidence(exit_code: int = 0) -> VerificationSandboxEvidence:
    return VerificationSandboxEvidence(
        completion=SandboxCompletion.NORMAL,
        exit_code=exit_code,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=0.25,
    )


class FakeFingerprinter:
    def __init__(self, values: list[RepositoryFingerprint] | None = None) -> None:
        self.values = values or [_fingerprint()]
        self.calls: list[Path] = []

    def fingerprint(self, repository: Path) -> RepositoryFingerprint:
        self.calls.append(repository)
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class FakeCandidateFilesystem:
    def __init__(self) -> None:
        self.inspections: list[Path] = []

    def inspect_candidate(self, repository: Path) -> CandidateInspection:
        self.inspections.append(repository)
        return CandidateInspection(repository=repository, identity="b" * 64)

    def validate_visible_tests(
        self, repository: Path, declared_paths: tuple[str, ...]
    ) -> tuple[str, ...]:
        return declared_paths

    def validate_workspace(
        self, workspace: Path, candidate: Path, trusted_case: TrustedCase
    ) -> Path:
        return workspace


class FakeBundleFilesystem:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.created = 0
        self.cleaned = 0

    def create_bundle(
        self, workspace: Path, trusted_case: TrustedCase, identifier: str
    ) -> HiddenTestBundle:
        self.created += 1
        return HiddenTestBundle(
            workspace_root=workspace,
            root=workspace / f".tracefix-verification-{identifier}",
            test_files=("test_0000.py",),
        )

    def remove_bundle(self, bundle: HiddenTestBundle) -> None:
        self.cleaned += 1


class FakeSandbox:
    def __init__(self, results: list[VerificationSandboxEvidence | Exception]) -> None:
        self.results = results
        self.calls: list[tuple[VerificationStage, tuple[str, ...]]] = []

    def _next(self) -> VerificationSandboxEvidence:
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def execute_targeted(
        self, repository: Path, test_paths: tuple[str, ...]
    ) -> VerificationSandboxEvidence:
        self.calls.append((VerificationStage.TARGETED, test_paths))
        return self._next()

    def execute_visible_suite(self, repository: Path) -> VerificationSandboxEvidence:
        self.calls.append((VerificationStage.VISIBLE_SUITE, ()))
        return self._next()

    def execute_hidden(
        self, repository: Path, bundle: HiddenTestBundle
    ) -> VerificationSandboxEvidence:
        self.calls.append((VerificationStage.HIDDEN_SUITE, bundle.test_files))
        return self._next()


def _service(
    tmp_path: Path,
    sandbox: FakeSandbox,
    *,
    fingerprinter: FakeFingerprinter | None = None,
) -> tuple[VerificationService, FakeBundleFilesystem, FakeCandidateFilesystem]:
    bundles = FakeBundleFilesystem(tmp_path)
    filesystem = FakeCandidateFilesystem()
    service = VerificationService(
        loader=load_trusted_case,
        fingerprinter=fingerprinter or FakeFingerprinter(),
        candidate_filesystem=filesystem,
        bundle_filesystem=bundles,
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )
    return service, bundles, filesystem


def _request(tmp_path: Path, case_id: str = "boundary_condition") -> CandidateVerificationRequest:
    candidate = tmp_path / "candidate"
    candidate.mkdir(exist_ok=True)
    workspace = tmp_path / "verification"
    workspace.mkdir(exist_ok=True)
    if os.name != "nt":
        workspace.chmod(0o700)
    return CandidateVerificationRequest(
        case_id=case_id,
        static_result=_static_result(candidate.resolve()),
        verification_workspace_root=workspace.resolve(),
    )


def test_verification_contracts_are_strict_frozen_and_round_trip(tmp_path: Path) -> None:
    diagnostic = VerificationDiagnostic(
        code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
        message="a required earlier verification stage did not pass",
    )
    not_run = VerificationStageResult(
        stage=VerificationStage.VISIBLE_SUITE,
        status=VerificationStageStatus.NOT_RUN,
        diagnostic=diagnostic,
    )
    with pytest.raises(ValidationError):
        VerificationDiagnostic(
            code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
            message="unsafe detail",
        )
    with pytest.raises(ValidationError):
        VerificationStageResult.model_validate(
            {
                "stage": "targeted",
                "status": "passed",
                "completion": "normal",
                "exit_code": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "duration_seconds": 0,
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        not_run.status = VerificationStageStatus.PASSED  # type: ignore[misc]

    sandbox = FakeSandbox([_evidence(), _evidence(), _evidence()])
    service, _, _ = _service(tmp_path, sandbox)
    result = service.verify(_request(tmp_path))
    assert CandidateVerificationResult.model_validate_json(result.model_dump_json()) == result


def test_verification_runs_exact_order_and_cleans_hidden_bundle(tmp_path: Path) -> None:
    sandbox = FakeSandbox([_evidence(), _evidence(), _evidence()])
    service, bundles, filesystem = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    assert result.outcome.value == "passed"
    assert tuple(stage.status for stage in result.stages) == (
        VerificationStageStatus.PASSED,
        VerificationStageStatus.PASSED,
        VerificationStageStatus.PASSED,
    )
    assert [stage for stage, _ in sandbox.calls] == list(VerificationStage)
    assert sandbox.calls[0][1] == ("tests/test_behavior.py",)
    assert bundles.created == bundles.cleaned == 1
    assert len(filesystem.inspections) >= 4


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_normal_nonzero_is_candidate_failure_and_short_circuits(
    tmp_path: Path, failure_index: int
) -> None:
    results = [_evidence() for _ in range(failure_index)] + [_evidence(1)]
    sandbox = FakeSandbox(results)
    service, bundles, _ = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    statuses = tuple(stage.status for stage in result.stages)
    assert statuses[failure_index] is VerificationStageStatus.FAILED
    assert all(
        status is VerificationStageStatus.NOT_RUN for status in statuses[failure_index + 1 :]
    )
    assert result.outcome.value == "candidate_failed"
    assert len(sandbox.calls) == failure_index + 1
    assert bundles.cleaned == (1 if failure_index == 2 else 0)


def test_timeout_is_infrastructure_not_candidate_failure(tmp_path: Path) -> None:
    timeout = VerificationSandboxEvidence(
        completion=SandboxCompletion.TIMED_OUT,
        exit_code=None,
        stdout_truncated=False,
        stderr_truncated=True,
        duration_seconds=120.0,
    )
    sandbox = FakeSandbox([timeout])
    service, _, _ = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    assert result.outcome.value == "infrastructure_diagnostic"
    assert result.stages[0].status is VerificationStageStatus.INFRASTRUCTURE_ERROR
    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.SANDBOX_TIMED_OUT


def test_nonpassing_static_result_is_rejected_before_dependencies(tmp_path: Path) -> None:
    request = _request(tmp_path)
    object.__setattr__(request.static_result, "outcome", StaticValidationOutcome.CANDIDATE_FAILED)
    sandbox = FakeSandbox([])
    service, _, filesystem = _service(tmp_path, sandbox)

    result = service.verify(request)

    assert result.outcome.value == "infrastructure_diagnostic"
    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.STATIC_PREREQUISITE_FAILED
    assert sandbox.calls == []
    assert filesystem.inspections == []


@pytest.mark.parametrize(
    ("fingerprints", "failure_stage", "sandbox_calls"),
    [
        ([_fingerprint(), _fingerprint("d")], 0, 1),
        ([_fingerprint(), _fingerprint(), _fingerprint("d")], 1, 2),
        (
            [
                _fingerprint(),
                _fingerprint(),
                _fingerprint(),
                _fingerprint(),
                _fingerprint("d"),
            ],
            2,
            3,
        ),
    ],
)
def test_candidate_fingerprint_change_blocks_later_stages(
    tmp_path: Path,
    fingerprints: list[RepositoryFingerprint],
    failure_stage: int,
    sandbox_calls: int,
) -> None:
    fingerprinter = FakeFingerprinter(fingerprints)
    sandbox = FakeSandbox([_evidence(), _evidence(), _evidence()])
    service, _, _ = _service(tmp_path, sandbox, fingerprinter=fingerprinter)

    result = service.verify(_request(tmp_path))

    assert result.outcome.value == "infrastructure_diagnostic"
    assert result.stages[failure_stage].status is VerificationStageStatus.INFRASTRUCTURE_ERROR
    assert result.stages[failure_stage].diagnostic is not None
    assert (
        result.stages[failure_stage].diagnostic.code is VerificationDiagnosticCode.INTEGRITY_FAILED
    )
    assert len(sandbox.calls) == sandbox_calls


def test_stale_candidate_fingerprint_blocks_before_first_sandbox_run(tmp_path: Path) -> None:
    sandbox = FakeSandbox([])
    service, _, _ = _service(
        tmp_path, sandbox, fingerprinter=FakeFingerprinter([_fingerprint("d")])
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.CANDIDATE_INVALID
    assert sandbox.calls == []


def test_subclassed_request_is_revalidated_as_invalid_input(tmp_path: Path) -> None:
    class RequestSubclass(CandidateVerificationRequest):
        pass

    original = _request(tmp_path)
    request = RequestSubclass.model_validate(original.model_dump())
    sandbox = FakeSandbox([])
    service, _, _ = _service(tmp_path, sandbox)

    result = service.verify(cast(CandidateVerificationRequest, request))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.INVALID_INPUT
    assert sandbox.calls == []


def test_complete_result_rejects_wrong_order_impossible_topology_and_outcome() -> None:
    passed = tuple(
        VerificationStageResult(
            stage=stage,
            status=VerificationStageStatus.PASSED,
            completion=SandboxCompletion.NORMAL,
            exit_code=0,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.0,
        )
        for stage in VerificationStage
    )
    with pytest.raises(ValidationError):
        CandidateVerificationResult(stages=tuple(reversed(passed)))
    with pytest.raises(ValidationError):
        CandidateVerificationResult(
            stages=(
                VerificationStageResult(
                    stage=VerificationStage.TARGETED,
                    status=VerificationStageStatus.FAILED,
                    completion=SandboxCompletion.NORMAL,
                    exit_code=1,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_seconds=0.1,
                ),
                *passed[1:],
            )
        )
    with pytest.raises(ValidationError):
        CandidateVerificationResult(stages=passed, outcome="candidate_failed")
    with pytest.raises(ValidationError):
        VerificationStageResult(
            stage=VerificationStage.TARGETED,
            status=VerificationStageStatus.INFRASTRUCTURE_ERROR,
            diagnostic=VerificationDiagnostic(
                code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
                message="a required earlier verification stage did not pass",
            ),
        )


def test_case_mismatch_blocks_before_candidate_or_sandbox(tmp_path: Path) -> None:
    sandbox = FakeSandbox([])
    filesystem = FakeCandidateFilesystem()
    bundles = FakeBundleFilesystem(tmp_path)
    service = VerificationService(
        loader=lambda case_id: load_trusted_case("exception_handling"),
        fingerprinter=FakeFingerprinter(),
        candidate_filesystem=filesystem,
        bundle_filesystem=bundles,
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.CASE_MISMATCH
    assert filesystem.inspections == []
    assert sandbox.calls == []


def test_unexpected_loader_failure_maps_to_safe_case_diagnostic(tmp_path: Path) -> None:
    def failing_loader(case_id: str) -> TrustedCase:
        del case_id
        raise LookupError("secret loader detail")

    sandbox = FakeSandbox([])
    service = VerificationService(
        loader=failing_loader,
        fingerprinter=FakeFingerprinter(),
        candidate_filesystem=FakeCandidateFilesystem(),
        bundle_filesystem=FakeBundleFilesystem(tmp_path),
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.CASE_LOAD_FAILED
    assert "secret" not in result.model_dump_json()
    assert sandbox.calls == []


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("workspace", VerificationDiagnosticCode.WORKSPACE_INVALID),
        ("visible", VerificationDiagnosticCode.VISIBLE_TEST_INVALID),
        ("candidate", VerificationDiagnosticCode.CANDIDATE_INVALID),
    ],
)
def test_candidate_boundary_failures_have_distinct_safe_diagnostics(
    tmp_path: Path, failure: str, expected: VerificationDiagnosticCode
) -> None:
    class FailingCandidateFilesystem(FakeCandidateFilesystem):
        def inspect_candidate(self, repository: Path) -> CandidateInspection:
            if failure == "candidate":
                raise OSError("secret candidate detail")
            return super().inspect_candidate(repository)

        def validate_visible_tests(
            self, repository: Path, declared_paths: tuple[str, ...]
        ) -> tuple[str, ...]:
            if failure == "visible":
                raise OSError("secret visible detail")
            return super().validate_visible_tests(repository, declared_paths)

        def validate_workspace(
            self, workspace: Path, candidate: Path, trusted_case: TrustedCase
        ) -> Path:
            if failure == "workspace":
                raise OSError("secret workspace detail")
            return super().validate_workspace(workspace, candidate, trusted_case)

    sandbox = FakeSandbox([])
    service = VerificationService(
        loader=load_trusted_case,
        fingerprinter=FakeFingerprinter(),
        candidate_filesystem=FailingCandidateFilesystem(),
        bundle_filesystem=FakeBundleFilesystem(tmp_path),
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is expected
    assert "secret" not in result.model_dump_json()
    assert sandbox.calls == []


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            VerificationRunnerError(VerificationRunnerErrorCode.DAEMON_SECURITY),
            VerificationDiagnosticCode.SANDBOX_VALIDATION_FAILED,
        ),
        (
            VerificationRunnerError(VerificationRunnerErrorCode.DOCKER_FAILED),
            VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED,
        ),
        (
            VerificationRunnerError(VerificationRunnerErrorCode.CLEANUP_FAILED),
            VerificationDiagnosticCode.SANDBOX_CLEANUP_FAILED,
        ),
        (
            VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT),
            VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT,
        ),
        (
            VerificationRunnerError(VerificationRunnerErrorCode.TIMED_OUT),
            VerificationDiagnosticCode.SANDBOX_TIMED_OUT,
        ),
        (
            VerificationRunnerError(VerificationRunnerErrorCode.MEMORY_LIMIT),
            VerificationDiagnosticCode.SANDBOX_MEMORY_LIMIT,
        ),
    ],
)
def test_runner_failures_map_to_safe_infrastructure_diagnostics(
    tmp_path: Path,
    error: VerificationRunnerError,
    code: VerificationDiagnosticCode,
) -> None:
    sandbox = FakeSandbox([error])
    service, _, _ = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is code
    serialized = result.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "secret" not in serialized


def test_malformed_runner_error_cannot_leak_or_escape_mapping(tmp_path: Path) -> None:
    error = VerificationRunnerError(VerificationRunnerErrorCode.DOCKER_FAILED)
    error.code = cast(VerificationRunnerErrorCode, "secret-invalid-code")
    sandbox = FakeSandbox([error])
    service, _, _ = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED
    assert "secret" not in result.model_dump_json()


def test_subclassed_sandbox_evidence_is_malformed(tmp_path: Path) -> None:
    class EvidenceSubclass(VerificationSandboxEvidence):
        pass

    sandbox = FakeSandbox([EvidenceSubclass.model_validate(_evidence().model_dump())])
    service, _, _ = _service(tmp_path, sandbox)

    result = service.verify(_request(tmp_path))

    assert result.stages[0].diagnostic is not None
    assert result.stages[0].diagnostic.code is VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT


def test_hidden_bundle_cleanup_failure_overrides_test_pass(tmp_path: Path) -> None:
    class CleanupFailingBundles(FakeBundleFilesystem):
        def remove_bundle(self, bundle: HiddenTestBundle) -> None:
            del bundle
            raise OSError("secret cleanup detail")

    sandbox = FakeSandbox([_evidence(), _evidence(), _evidence()])
    bundles = CleanupFailingBundles(tmp_path)
    service = VerificationService(
        loader=load_trusted_case,
        fingerprinter=FakeFingerprinter(),
        candidate_filesystem=FakeCandidateFilesystem(),
        bundle_filesystem=bundles,
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[2].status is VerificationStageStatus.INFRASTRUCTURE_ERROR
    assert result.stages[2].diagnostic is not None
    assert result.stages[2].diagnostic.code is VerificationDiagnosticCode.CLEANUP_FAILED
    assert "secret cleanup detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EvaluatorInputError(), VerificationDiagnosticCode.EVALUATOR_INVALID),
        (VerificationFilesystemError(), VerificationDiagnosticCode.BUNDLE_FAILED),
    ],
)
def test_evaluator_and_bundle_creation_failures_map_distinctly(
    tmp_path: Path,
    error: VerificationFilesystemError,
    expected: VerificationDiagnosticCode,
) -> None:
    class FailingBundles(FakeBundleFilesystem):
        def create_bundle(
            self, workspace: Path, trusted_case: TrustedCase, identifier: str
        ) -> HiddenTestBundle:
            del workspace, trusted_case, identifier
            raise error

    sandbox = FakeSandbox([_evidence(), _evidence()])
    service = VerificationService(
        loader=load_trusted_case,
        fingerprinter=FakeFingerprinter(),
        candidate_filesystem=FakeCandidateFilesystem(),
        bundle_filesystem=FailingBundles(tmp_path),
        sandbox=sandbox,
        identifier_factory=lambda: "c" * 32,
    )

    result = service.verify(_request(tmp_path))

    assert result.stages[2].diagnostic is not None
    assert result.stages[2].diagnostic.code is expected
    assert len(sandbox.calls) == 2


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(3)])
def test_process_control_propagates_after_one_bundle_cleanup(
    tmp_path: Path, interruption: BaseException
) -> None:
    class InterruptingSandbox(FakeSandbox):
        def execute_hidden(
            self, repository: Path, bundle: HiddenTestBundle
        ) -> VerificationSandboxEvidence:
            del repository, bundle
            raise interruption

    sandbox = InterruptingSandbox([_evidence(), _evidence()])
    service, bundles, _ = _service(tmp_path, sandbox)

    with pytest.raises(type(interruption)):
        service.verify(_request(tmp_path))

    assert bundles.cleaned == 1


def test_local_bundle_copies_only_declared_hidden_bytes_and_removes_it(tmp_path: Path) -> None:
    case = load_trusted_case("boundary_condition")
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    filesystem = LocalEvaluatorBundleFilesystem()

    bundle = filesystem.create_bundle(workspace, case, "d" * 32)

    assert bundle.test_files == ("test_0000.py",)
    assert (bundle.root / bundle.test_files[0]).read_bytes() == case.hidden_tests[0].read_bytes()
    assert tuple(path.name for path in bundle.root.iterdir()) == bundle.test_files
    assert not any("reference" in path.name for path in bundle.root.iterdir())
    if os.name != "nt":
        assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
        assert stat.S_IMODE(bundle.root.stat().st_mode) == 0o755
        assert stat.S_IMODE((bundle.root / bundle.test_files[0]).stat().st_mode) == 0o444
    filesystem.remove_bundle(bundle)
    assert not bundle.root.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX host permission boundary")
def test_world_accessible_verification_workspace_is_rejected(tmp_path: Path) -> None:
    case = load_trusted_case("boundary_condition")
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir(mode=0o755)
    workspace.chmod(0o755)
    filesystem = LocalEvaluatorBundleFilesystem()

    with pytest.raises(VerificationFilesystemError):
        filesystem.create_bundle(workspace, case, "d" * 32)

    assert not any(workspace.iterdir())


def _relocated_case(tmp_path: Path) -> TrustedCase:
    original = load_trusted_case("boundary_condition")
    case_path = (tmp_path / "case").resolve()
    shutil.copytree(original.case_path, case_path)
    return original.model_copy(
        update={
            "case_path": case_path,
            "repository_path": case_path / original.manifest.repository_path,
            "visible_tests": tuple(
                case_path.joinpath(*Path(value).parts) for value in original.manifest.visible_tests
            ),
            "hidden_tests": tuple(
                case_path.joinpath(*Path(value).parts) for value in original.manifest.hidden_tests
            ),
            "reference_patch": case_path / original.manifest.reference_patch,
        }
    )


def test_hidden_source_path_read_swap_cannot_copy_external_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _relocated_case(tmp_path)
    source = case.hidden_tests[0]
    original_bytes = source.read_bytes()
    victim = (tmp_path / "victim.py").resolve()
    victim.write_bytes(b"SECRET_EXTERNAL_BYTES")
    original_read_bytes = Path.read_bytes

    def swapping_read_bytes(path: Path) -> bytes:
        if path == source:
            backup = source.with_suffix(".held")
            source.rename(backup)
            try:
                source.symlink_to(victim)
            except OSError:
                backup.rename(source)
                pytest.skip("symbolic links are unavailable on this filesystem")
            try:
                return original_read_bytes(source)
            finally:
                source.unlink()
                backup.rename(source)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    filesystem = LocalEvaluatorBundleFilesystem()

    bundle = filesystem.create_bundle(workspace, case, "e" * 32)

    assert original_read_bytes(bundle.root / bundle.test_files[0]) == original_bytes
    assert original_read_bytes(bundle.root / bundle.test_files[0]) != victim.read_bytes()
    filesystem.remove_bundle(bundle)


def test_oversized_hidden_source_is_evaluator_invalid_and_partial_bundle_is_removed(
    tmp_path: Path,
) -> None:
    case = _relocated_case(tmp_path)
    with case.hidden_tests[0].open("r+b") as handle:
        handle.truncate(67_108_865)
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    filesystem = LocalEvaluatorBundleFilesystem()

    with pytest.raises(EvaluatorInputError):
        filesystem.create_bundle(workspace, case, "f" * 32)

    assert not any(workspace.iterdir())


def test_hardlinked_hidden_source_is_rejected_and_partial_bundle_is_removed(
    tmp_path: Path,
) -> None:
    case = _relocated_case(tmp_path)
    alias = case.hidden_tests[0].with_name("hardlink-alias.py")
    try:
        os.link(case.hidden_tests[0], alias)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    filesystem = LocalEvaluatorBundleFilesystem()

    with pytest.raises(EvaluatorInputError):
        filesystem.create_bundle(workspace, case, "b" * 32)

    assert not any(workspace.iterdir())


def test_hidden_bundle_name_collision_preserves_existing_directory(tmp_path: Path) -> None:
    case = _relocated_case(tmp_path)
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    collision = workspace / (".tracefix-verification-" + "c" * 32)
    collision.mkdir()
    marker = collision / "existing.txt"
    marker.write_text("unchanged", encoding="utf-8")
    filesystem = LocalEvaluatorBundleFilesystem()

    with pytest.raises(VerificationFilesystemError):
        filesystem.create_bundle(workspace, case, "c" * 32)

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_bundle_cleanup_rejects_symlink_without_mutating_target(tmp_path: Path) -> None:
    case = _relocated_case(tmp_path)
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    filesystem = LocalEvaluatorBundleFilesystem()
    bundle = filesystem.create_bundle(workspace, case, "a" * 32)
    child = bundle.root / bundle.test_files[0]
    original_bytes = child.read_bytes()
    victim = (tmp_path / "victim.txt").resolve()
    victim.write_text("unchanged", encoding="utf-8")
    victim_mode = stat.S_IMODE(victim.stat().st_mode)
    if os.name != "nt":
        child.chmod(0o600)
    child.unlink()
    try:
        child.symlink_to(victim)
    except OSError:
        child.write_bytes(original_bytes)
        filesystem.remove_bundle(bundle)
        pytest.skip("symbolic links are unavailable on this filesystem")

    with pytest.raises(BundleCleanupError):
        filesystem.remove_bundle(bundle)

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(victim.stat().st_mode) == victim_mode
    child.unlink()
    child.write_bytes(original_bytes)
    filesystem.remove_bundle(bundle)
    assert not bundle.root.exists()


def test_local_candidate_filesystem_rejects_hardlinks_and_invalid_visible_paths(
    tmp_path: Path,
) -> None:
    candidate = (tmp_path / "candidate").resolve()
    candidate.mkdir()
    tests = candidate / "tests"
    tests.mkdir()
    visible = tests / "test_behavior.py"
    visible.write_bytes(b"def test_x(): pass\n")
    filesystem = LocalCandidateFilesystem()

    assert filesystem.validate_visible_tests(candidate, ("tests/test_behavior.py",)) == (
        "tests/test_behavior.py",
    )
    with pytest.raises(VerificationFilesystemError):
        filesystem.validate_visible_tests(candidate, ("../test_behavior.py",))

    alias = candidate / "alias.py"
    try:
        os.link(visible, alias)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(VerificationFilesystemError):
        filesystem.inspect_candidate(candidate)


def test_local_candidate_and_workspace_validation_preserve_candidate_bytes(
    tmp_path: Path,
) -> None:
    case = load_trusted_case("boundary_condition")
    candidate = (tmp_path / "candidate").resolve()
    shutil.copytree(case.repository_path, candidate)
    workspace = (tmp_path / "verification").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    before = tuple(
        (path.relative_to(candidate).as_posix(), path.read_bytes())
        for path in sorted(candidate.rglob("*"))
        if path.is_file()
    )
    filesystem = LocalCandidateFilesystem()

    inspection = filesystem.inspect_candidate(candidate)
    validated_workspace = filesystem.validate_workspace(workspace, candidate, case)

    assert inspection.repository == candidate
    assert validated_workspace == workspace
    after = tuple(
        (path.relative_to(candidate).as_posix(), path.read_bytes())
        for path in sorted(candidate.rglob("*"))
        if path.is_file()
    )
    assert after == before

    with pytest.raises(VerificationFilesystemError):
        filesystem.validate_workspace(candidate, candidate, case)
