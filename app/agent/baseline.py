from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import ValidationError

from app.agent.schemas import (
    BaselineDiagnosticCode,
    BaselineEvidence,
    BaselineOutcome,
    BaselineResult,
    RepositoryPreparation,
)
from app.sandbox import (
    SandboxCleanupError,
    SandboxCompletion,
    SandboxError,
    SandboxExecutionError,
    SandboxResult,
    SandboxValidationError,
)
from benchmarks.loader import BenchmarkLoadError, TrustedCase

TrustedCaseLoader = Callable[[str], TrustedCase]


class SandboxExecutionBoundary(Protocol):
    def execute(self, repository: Path, command_tokens: Sequence[str]) -> SandboxResult: ...


class BaselineExecutor:
    def __init__(self, loader: TrustedCaseLoader, runner: SandboxExecutionBoundary) -> None:
        self._loader = loader
        self._runner = runner

    def run(self, preparation: RepositoryPreparation) -> BaselineResult:
        try:
            case = self._loader(preparation.case_id)
            paths = self._visible_test_paths(case, preparation)
        except (BenchmarkLoadError, OSError, RuntimeError, ValueError) as error:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.INVALID_VISIBLE_TEST,
                "declared visible tests are not valid in the prepared repository",
                cause=error,
            )
        try:
            raw = self._runner.execute(
                preparation.prepared_repository,
                ("python", "-m", "pytest", *(path.as_posix() for path in paths)),
            )
            sandbox = SandboxResult.model_validate(raw.model_dump())
            evidence = BaselineEvidence.model_validate(sandbox.model_dump())
        except SandboxCleanupError as error:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.SANDBOX_CLEANUP_FAILED,
                "sandbox cleanup failed",
                cause=error,
            )
        except SandboxValidationError as error:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.SANDBOX_VALIDATION_FAILED,
                "sandbox validation failed",
                cause=error,
            )
        except (SandboxExecutionError, SandboxError, OSError) as error:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.SANDBOX_EXECUTION_FAILED,
                "sandbox execution failed",
                cause=error,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.MALFORMED_SANDBOX_RESULT,
                "sandbox returned malformed execution evidence",
                cause=error,
            )
        if sandbox.completion is SandboxCompletion.TIMED_OUT:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.SANDBOX_TIMED_OUT,
                "baseline execution exceeded its deadline",
                evidence=evidence,
            )
        if sandbox.completion is SandboxCompletion.MEMORY_LIMIT:
            return self._diagnostic(
                preparation,
                BaselineDiagnosticCode.SANDBOX_MEMORY_LIMIT,
                "baseline execution exceeded its memory limit",
                evidence=evidence,
            )
        if sandbox.exit_code == 0:
            return BaselineResult(
                preparation=preparation,
                outcome=BaselineOutcome.NOT_REPRODUCIBLE,
                evidence=evidence,
                diagnostic_code=BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE,
                message="declared visible failure was not reproducible",
            )
        return BaselineResult(
            preparation=preparation,
            outcome=BaselineOutcome.REPRODUCED_FAILURE,
            evidence=evidence,
        )

    @staticmethod
    def _visible_test_paths(
        case: TrustedCase, preparation: RepositoryPreparation
    ) -> tuple[PurePosixPath, ...]:
        if case.manifest.failing_revision != preparation.requested_revision:
            raise ValueError("prepared revision no longer matches the trusted manifest")
        repository = preparation.prepared_repository.resolve(strict=True)
        paths: list[PurePosixPath] = []
        for declared in case.manifest.visible_tests:
            relative = PurePosixPath(declared)
            target = repository.joinpath(*relative.parts).resolve(strict=True)
            target.relative_to(repository)
            if not target.is_file():
                raise ValueError("visible test is not a regular file")
            paths.append(relative)
        return tuple(paths)

    @staticmethod
    def _diagnostic(
        preparation: RepositoryPreparation,
        code: BaselineDiagnosticCode,
        message: str,
        *,
        evidence: BaselineEvidence | None = None,
        cause: BaseException | None = None,
    ) -> BaselineResult:
        del cause
        return BaselineResult(
            preparation=preparation,
            outcome=BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC,
            evidence=evidence,
            diagnostic_code=code,
            message=message,
        )
