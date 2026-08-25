from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from app.agent.baseline import BaselineExecutor
from app.agent.schemas import (
    BaselineDiagnosticCode,
    BaselineOutcome,
    RepositoryFingerprint,
    RepositoryPreparation,
    RevisionKind,
)
from app.sandbox import (
    SandboxCleanupError,
    SandboxCompletion,
    SandboxExecutionError,
    SandboxResult,
    SandboxValidationError,
)
from app.sandbox.results import TRUNCATION_MARKER
from benchmarks.loader import TrustedCase, load_trusted_case


def _preparation(repository: Path) -> RepositoryPreparation:
    return RepositoryPreparation(
        case_id="incorrect_conditional",
        requested_revision="incorrect_conditional-failing-v1",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=repository,
        fingerprint=RepositoryFingerprint(digest="0" * 64),
    )


class FakeRunner:
    def __init__(self, result: SandboxResult | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def execute(self, repository: Path, command_tokens: tuple[str, ...]) -> SandboxResult:
        self.calls.append((repository, command_tokens))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class MalformedRunner:
    def execute(self, repository: Path, command_tokens: tuple[str, ...]) -> SandboxResult:
        del repository, command_tokens
        return cast(SandboxResult, object())


def _executor(runner: FakeRunner) -> BaselineExecutor:
    return BaselineExecutor(load_trusted_case, runner)


def test_reproduced_failure_preserves_complete_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    sandbox = SandboxResult(
        exit_code=1,
        completion=SandboxCompletion.NORMAL,
        stdout="failed",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=1.0,
    )
    runner = FakeRunner(sandbox)
    result = _executor(runner).run(_preparation(repository.resolve()))

    assert result.outcome is BaselineOutcome.REPRODUCED_FAILURE
    assert result.evidence is not None
    assert result.evidence.exit_code == 1
    assert runner.calls == [
        (repository.resolve(), ("python", "-m", "pytest", "tests/test_behavior.py"))
    ]


def test_unexpected_pass_is_terminal_and_has_no_model_callback(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    runner = FakeRunner(
        SandboxResult(
            exit_code=0,
            completion=SandboxCompletion.NORMAL,
            stdout="passed",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        )
    )
    result = _executor(runner).run(_preparation(repository.resolve()))
    assert result.outcome is BaselineOutcome.NOT_REPRODUCIBLE
    assert result.diagnostic_code is BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE
    assert result.terminal
    assert not hasattr(_executor(runner), "model")


@pytest.mark.parametrize(
    ("completion", "code"),
    [
        (SandboxCompletion.TIMED_OUT, BaselineDiagnosticCode.SANDBOX_TIMED_OUT),
        (SandboxCompletion.MEMORY_LIMIT, BaselineDiagnosticCode.SANDBOX_MEMORY_LIMIT),
    ],
)
def test_resource_termination_is_infrastructure_diagnostic(
    tmp_path: Path, completion: SandboxCompletion, code: BaselineDiagnosticCode
) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    result = _executor(
        FakeRunner(
            SandboxResult(
                exit_code=None,
                completion=completion,
                stdout="partial",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=2.0,
            )
        )
    ).run(_preparation(repository.resolve()))
    assert result.outcome is BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic_code is code
    assert result.evidence is not None


def test_cleanup_failure_is_distinguished(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    error = SandboxCleanupError("cleanup", cleanup_cause=RuntimeError("rm"))
    result = _executor(FakeRunner(error)).run(_preparation(repository.resolve()))
    assert result.diagnostic_code is BaselineDiagnosticCode.SANDBOX_CLEANUP_FAILED
    assert result.evidence is None


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            SandboxValidationError("invalid"),
            BaselineDiagnosticCode.SANDBOX_VALIDATION_FAILED,
        ),
        (
            SandboxExecutionError("failed"),
            BaselineDiagnosticCode.SANDBOX_EXECUTION_FAILED,
        ),
    ],
)
def test_sandbox_failures_are_distinguished(
    tmp_path: Path, error: BaseException, code: BaselineDiagnosticCode
) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    result = _executor(FakeRunner(error)).run(_preparation(repository.resolve()))
    assert result.outcome is BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic_code is code


def test_truncation_flags_and_marker_are_retained(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    marker = TRUNCATION_MARKER.decode("ascii")
    result = _executor(
        FakeRunner(
            SandboxResult(
                exit_code=1,
                completion=SandboxCompletion.NORMAL,
                stdout=f"partial{marker}",
                stderr=f"partial{marker}",
                stdout_truncated=True,
                stderr_truncated=True,
                duration_seconds=0.5,
            )
        )
    ).run(_preparation(repository.resolve()))
    assert result.evidence is not None
    assert result.evidence.stdout_truncated
    assert result.evidence.stderr_truncated
    assert result.evidence.stdout.endswith(marker)
    assert result.evidence.stderr.endswith(marker)


def test_rejects_visible_test_missing_from_prepared_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    runner = FakeRunner(cast(SandboxResult, RuntimeError("must not run")))
    result = _executor(runner).run(_preparation(repository.resolve()))
    assert result.diagnostic_code is BaselineDiagnosticCode.INVALID_VISIBLE_TEST
    assert runner.calls == []


def test_malformed_sandbox_result_is_diagnostic(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    (repository / "tests" / "test_behavior.py").write_text("", encoding="utf-8")
    result = BaselineExecutor(load_trusted_case, MalformedRunner()).run(
        _preparation(repository.resolve())
    )
    assert result.diagnostic_code is BaselineDiagnosticCode.MALFORMED_SANDBOX_RESULT


def test_all_visible_tests_are_passed_in_manifest_order(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "tests").mkdir(parents=True)
    for name in ("second.py", "first.py"):
        (repository / "tests" / name).write_text("", encoding="utf-8")
    case = load_trusted_case("incorrect_conditional")
    manifest = case.manifest.model_copy(
        update={"visible_tests": ("tests/second.py", "tests/first.py")}
    )

    def loader(_: str) -> TrustedCase:
        return case.model_copy(update={"manifest": manifest})

    runner = FakeRunner(
        SandboxResult(
            exit_code=1,
            completion=SandboxCompletion.NORMAL,
            stdout="failed",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        )
    )
    BaselineExecutor(loader, runner).run(_preparation(repository.resolve()))

    assert runner.calls == [
        (
            repository.resolve(),
            (
                "python",
                "-m",
                "pytest",
                "tests/second.py",
                "tests/first.py",
            ),
        )
    ]
