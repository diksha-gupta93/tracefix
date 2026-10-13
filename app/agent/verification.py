from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.agent.patches import UnifiedDiffError, normalize_repository_paths
from app.agent.schemas import RepositoryFingerprint
from app.agent.static_checks import (
    StaticCheck,
    StaticValidationOutcome,
    StaticValidationResult,
    ValidatedCandidate,
)
from app.sandbox.results import NonNegativeFiniteFloat, SandboxCompletion
from app.sandbox.verification_runner import (
    HIDDEN_BUNDLE_PREFIX,
    HiddenTestBundle,
    VerificationRunnerError,
    VerificationRunnerErrorCode,
    VerificationRunnerValidationError,
    VerificationSandboxBoundary,
    VerificationSandboxEvidence,
)
from benchmarks.loader import TrustedCase

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_IDENTIFIER = re.compile(r"^[0-9a-f]{32}$")
_IDENTITY = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_REPARSE_POINT = 0x400
_MAX_TEST_FILES = 1_000
_MAX_FILE_BYTES = 67_108_864
_GENERATED_TEST = re.compile(r"^test_[0-9]{4}\.py$")


class VerificationStage(StrEnum):
    TARGETED = "targeted"
    VISIBLE_SUITE = "visible_suite"
    HIDDEN_SUITE = "hidden_suite"


class VerificationStageStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class VerificationOutcome(StrEnum):
    PASSED = "passed"
    CANDIDATE_FAILED = "candidate_failed"
    INFRASTRUCTURE_DIAGNOSTIC = "infrastructure_diagnostic"


class VerificationDiagnosticCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    STATIC_PREREQUISITE_FAILED = "static_prerequisite_failed"
    CASE_LOAD_FAILED = "case_load_failed"
    CASE_MISMATCH = "case_mismatch"
    CANDIDATE_INVALID = "candidate_invalid"
    WORKSPACE_INVALID = "workspace_invalid"
    VISIBLE_TEST_INVALID = "visible_test_invalid"
    EVALUATOR_INVALID = "evaluator_invalid"
    BUNDLE_FAILED = "bundle_failed"
    CLEANUP_FAILED = "cleanup_failed"
    INTEGRITY_FAILED = "integrity_failed"
    SANDBOX_VALIDATION_FAILED = "sandbox_validation_failed"
    SANDBOX_EXECUTION_FAILED = "sandbox_execution_failed"
    SANDBOX_CLEANUP_FAILED = "sandbox_cleanup_failed"
    SANDBOX_TIMED_OUT = "sandbox_timed_out"
    SANDBOX_MEMORY_LIMIT = "sandbox_memory_limit"
    MALFORMED_SANDBOX_RESULT = "malformed_sandbox_result"
    PREREQUISITE_FAILED = "prerequisite_failed"


_DIAGNOSTIC_MESSAGES: dict[VerificationDiagnosticCode, str] = {
    VerificationDiagnosticCode.INVALID_INPUT: "verification input is invalid",
    VerificationDiagnosticCode.STATIC_PREREQUISITE_FAILED: (
        "passing static validation is required"
    ),
    VerificationDiagnosticCode.CASE_LOAD_FAILED: "trusted benchmark case could not be loaded",
    VerificationDiagnosticCode.CASE_MISMATCH: "trusted benchmark case identity does not match",
    VerificationDiagnosticCode.CANDIDATE_INVALID: "candidate repository validation failed",
    VerificationDiagnosticCode.WORKSPACE_INVALID: "verification workspace validation failed",
    VerificationDiagnosticCode.VISIBLE_TEST_INVALID: "declared visible tests are invalid",
    VerificationDiagnosticCode.EVALUATOR_INVALID: "trusted evaluator input is invalid",
    VerificationDiagnosticCode.BUNDLE_FAILED: "hidden test bundle creation failed",
    VerificationDiagnosticCode.CLEANUP_FAILED: "hidden test bundle cleanup failed",
    VerificationDiagnosticCode.INTEGRITY_FAILED: "candidate integrity verification failed",
    VerificationDiagnosticCode.SANDBOX_VALIDATION_FAILED: "sandbox validation failed",
    VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED: "sandbox execution failed",
    VerificationDiagnosticCode.SANDBOX_CLEANUP_FAILED: "sandbox cleanup failed",
    VerificationDiagnosticCode.SANDBOX_TIMED_OUT: "sandbox execution exceeded its deadline",
    VerificationDiagnosticCode.SANDBOX_MEMORY_LIMIT: "sandbox execution exceeded its memory limit",
    VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT: (
        "sandbox returned malformed execution evidence"
    ),
    VerificationDiagnosticCode.PREREQUISITE_FAILED: (
        "a required earlier verification stage did not pass"
    ),
}


class VerificationDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: VerificationDiagnosticCode
    message: Annotated[str, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        if self.message != _DIAGNOSTIC_MESSAGES[self.code]:
            raise ValueError("diagnostic message must be the safe constant for its code")
        return self


class VerificationStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage: VerificationStage
    status: VerificationStageStatus
    completion: SandboxCompletion | None = None
    exit_code: Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)] | None = None
    stdout_truncated: Annotated[bool, Field(strict=True)] | None = None
    stderr_truncated: Annotated[bool, Field(strict=True)] | None = None
    duration_seconds: NonNegativeFiniteFloat | None = None
    diagnostic: VerificationDiagnostic | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        evidence = (
            self.completion,
            self.exit_code,
            self.stdout_truncated,
            self.stderr_truncated,
            self.duration_seconds,
        )
        has_evidence = any(value is not None for value in evidence)
        complete_evidence = (
            self.completion is not None
            and self.stdout_truncated is not None
            and self.stderr_truncated is not None
            and self.duration_seconds is not None
        )
        if has_evidence and not complete_evidence:
            raise ValueError("sandbox evidence must be complete")
        if self.completion is SandboxCompletion.NORMAL and self.exit_code is None:
            raise ValueError("normal evidence requires an exit code")
        if self.completion is SandboxCompletion.TIMED_OUT and self.exit_code is not None:
            raise ValueError("timed-out evidence cannot contain an exit code")
        if self.status is VerificationStageStatus.PASSED:
            if (
                self.completion is not SandboxCompletion.NORMAL
                or self.exit_code != 0
                or self.diagnostic is not None
            ):
                raise ValueError("passing stage requires normal exit zero evidence")
        elif self.status is VerificationStageStatus.FAILED:
            if (
                self.completion is not SandboxCompletion.NORMAL
                or self.exit_code in (None, 0)
                or self.diagnostic is not None
            ):
                raise ValueError("failed stage requires normal nonzero evidence")
        elif self.status is VerificationStageStatus.NOT_RUN:
            if (
                has_evidence
                or self.diagnostic is None
                or (self.diagnostic.code is not VerificationDiagnosticCode.PREREQUISITE_FAILED)
            ):
                raise ValueError("not-run stage requires prerequisite evidence only")
        elif self.diagnostic is None:
            raise ValueError("infrastructure stage requires a diagnostic")
        elif self.diagnostic.code is VerificationDiagnosticCode.PREREQUISITE_FAILED:
            raise ValueError("prerequisite evidence is valid only for a not-run stage")
        elif self.diagnostic.code is VerificationDiagnosticCode.SANDBOX_TIMED_OUT:
            if has_evidence and self.completion is not SandboxCompletion.TIMED_OUT:
                raise ValueError("timeout diagnostic requires timed-out evidence")
        elif self.diagnostic.code is VerificationDiagnosticCode.SANDBOX_MEMORY_LIMIT:
            if has_evidence and self.completion is not SandboxCompletion.MEMORY_LIMIT:
                raise ValueError("memory diagnostic requires memory-limit evidence")
        elif self.completion in {SandboxCompletion.TIMED_OUT, SandboxCompletion.MEMORY_LIMIT}:
            raise ValueError("non-normal completion requires its matching diagnostic")
        return self


class CandidateVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stages: tuple[VerificationStageResult, ...]
    outcome: VerificationOutcome = VerificationOutcome.PASSED

    @field_validator("stages", mode="before")
    @classmethod
    def freeze_stages(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if tuple(item.stage for item in self.stages) != tuple(VerificationStage):
            raise ValueError("verification stages must contain the fixed complete order")
        statuses = tuple(item.status for item in self.stages)
        derived: VerificationOutcome
        if statuses == (VerificationStageStatus.PASSED,) * 3:
            derived = VerificationOutcome.PASSED
        else:
            infrastructure = tuple(
                index
                for index, status in enumerate(statuses)
                if status is VerificationStageStatus.INFRASTRUCTURE_ERROR
            )
            failures = tuple(
                index
                for index, status in enumerate(statuses)
                if status is VerificationStageStatus.FAILED
            )
            blocking = infrastructure or failures
            if len(blocking) != 1:
                raise ValueError("verification result must contain one earliest blocking stage")
            blocked_index = blocking[0]
            if any(
                status is not VerificationStageStatus.PASSED for status in statuses[:blocked_index]
            ) or any(
                status is not VerificationStageStatus.NOT_RUN
                for status in statuses[blocked_index + 1 :]
            ):
                raise ValueError("verification stages do not short-circuit deterministically")
            derived = (
                VerificationOutcome.INFRASTRUCTURE_DIAGNOSTIC
                if infrastructure
                else VerificationOutcome.CANDIDATE_FAILED
            )
        if "outcome" in self.model_fields_set and self.outcome is not derived:
            raise ValueError("verification outcome does not match stage evidence")
        object.__setattr__(self, "outcome", derived)
        return self


class CandidateVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=128)]
    static_result: StaticValidationResult
    verification_workspace_root: Path


class CandidateInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    repository: Path
    identity: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_repository(self) -> Self:
        if not self.repository.is_absolute():
            raise ValueError("candidate path must be absolute")
        return self


class CandidateFingerprinter(Protocol):
    def fingerprint(self, repository: Path) -> RepositoryFingerprint: ...


class CandidateFilesystem(Protocol):
    def inspect_candidate(self, repository: Path) -> CandidateInspection: ...

    def validate_visible_tests(
        self, repository: Path, declared_paths: tuple[str, ...]
    ) -> tuple[str, ...]: ...

    def validate_workspace(
        self, workspace: Path, candidate: Path, trusted_case: TrustedCase
    ) -> Path: ...


class EvaluatorBundleFilesystem(Protocol):
    def create_bundle(
        self, workspace: Path, trusted_case: TrustedCase, identifier: str
    ) -> HiddenTestBundle: ...

    def remove_bundle(self, bundle: HiddenTestBundle) -> None: ...


class VerificationFilesystemError(Exception):
    """A candidate, workspace, or evaluator filesystem boundary failed closed."""


class BundleCleanupError(VerificationFilesystemError):
    """The private evaluator bundle could not be removed safely."""


class EvaluatorInputError(VerificationFilesystemError):
    """Trusted evaluator input failed validation or changed while being read."""


def _overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _same_or_ancestor(candidate: Path, protected: Path) -> bool:
    try:
        protected.relative_to(candidate)
    except ValueError:
        return False
    return True


def _protected_locations() -> tuple[Path, ...]:
    home = Path.home().resolve(strict=False)
    project = Path(__file__).resolve().parents[2]
    return (
        project,
        home,
        home / ".ssh",
        home / ".docker",
        home / ".aws",
        home / ".config" / "gcloud",
        home / ".azure",
        home / ".netrc",
        home / "_netrc",
        home / ".npmrc",
        home / ".pypirc",
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
    )


def _is_protected_mount(path: Path) -> bool:
    locations = tuple(item.resolve(strict=False) for item in _protected_locations())
    project, home, *sensitive = locations
    return (
        _overlaps(path, project)
        or _same_or_ancestor(path, home)
        or any(_overlaps(path, item) for item in sensitive)
    )


def _ordinary_directory(path: Path) -> tuple[Path, os.stat_result]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise VerificationFilesystemError
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise VerificationFilesystemError from None
    if (
        path != resolved
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise VerificationFilesystemError
    return resolved, metadata


def _tree_identity(root: Path, root_metadata: os.stat_result) -> str:
    entries: list[tuple[str, int, int, int, int, int, int, int]] = []
    root_entry = (
        "",
        root_metadata.st_dev,
        root_metadata.st_ino,
        root_metadata.st_mode,
        root_metadata.st_size,
        root_metadata.st_mtime_ns,
        getattr(root_metadata, "st_reparse_tag", 0),
        root_metadata.st_nlink,
    )
    entries.append(root_entry)
    identities: set[str] = set()
    try:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names.sort(key=lambda value: (value.casefold(), value))
            file_names.sort(key=lambda value: (value.casefold(), value))
            for name in (*directory_names, *file_names):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                identity = relative.casefold()
                if identity in identities:
                    raise VerificationFilesystemError
                identities.add(identity)
                metadata = path.lstat()
                if (
                    path.is_symlink()
                    or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                    or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
                    or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
                ):
                    raise VerificationFilesystemError
                entries.append(
                    (
                        relative,
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mode,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        getattr(metadata, "st_reparse_tag", 0),
                        metadata.st_nlink,
                    )
                )
    except VerificationFilesystemError:
        raise
    except OSError:
        raise VerificationFilesystemError from None
    digest = hashlib.sha256(b"tracefix-candidate-identity\x00v1\x00")
    for entry in entries:
        for value in entry:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and getattr(left, "st_reparse_tag", 0) == getattr(right, "st_reparse_tag", 0)
        and left.st_nlink == right.st_nlink
    )


def _read_bounded_ordinary_file(path: Path, error_type: type[VerificationFilesystemError]) -> bytes:
    descriptor = -1
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or getattr(before, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise error_type
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_identity(before, opened):
            raise error_type
        chunks: list[bytes] = []
        remaining = _MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after_open = os.fstat(descriptor)
        after_path = path.lstat()
        if (
            len(data) > _MAX_FILE_BYTES
            or path.is_symlink()
            or getattr(after_path, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            or not _same_file_identity(before, after_open)
            or not _same_file_identity(before, after_path)
        ):
            raise error_type
        return data
    except error_type:
        raise
    except (OSError, RuntimeError):
        raise error_type from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


class LocalCandidateFilesystem:
    def inspect_candidate(self, repository: Path) -> CandidateInspection:
        resolved, metadata = _ordinary_directory(repository)
        if _is_protected_mount(resolved):
            raise VerificationFilesystemError
        first = _tree_identity(resolved, metadata)
        resolved_again, metadata_again = _ordinary_directory(repository)
        second = _tree_identity(resolved_again, metadata_again)
        if resolved != resolved_again or first != second:
            raise VerificationFilesystemError
        return CandidateInspection(repository=resolved, identity=first)

    def validate_visible_tests(
        self, repository: Path, declared_paths: tuple[str, ...]
    ) -> tuple[str, ...]:
        if type(declared_paths) is not tuple or not declared_paths:
            raise VerificationFilesystemError
        try:
            normalized = normalize_repository_paths(declared_paths)
        except UnifiedDiffError:
            raise VerificationFilesystemError from None
        if normalized != declared_paths or len(normalized) > _MAX_TEST_FILES:
            raise VerificationFilesystemError
        for value in normalized:
            path = PurePosixPath(value)
            if (
                path.parts[:1] != ("tests",)
                or not path.name.startswith("test_")
                or (path.suffix != ".py")
            ):
                raise VerificationFilesystemError
            try:
                target = repository.joinpath(*path.parts)
                metadata = target.lstat()
                resolved = target.resolve(strict=True)
                resolved.relative_to(repository)
            except (OSError, RuntimeError, ValueError):
                raise VerificationFilesystemError from None
            if (
                target != resolved
                or target.is_symlink()
                or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise VerificationFilesystemError
        return normalized

    def validate_workspace(
        self, workspace: Path, candidate: Path, trusted_case: TrustedCase
    ) -> Path:
        resolved, metadata = _ordinary_directory(workspace)
        protected = (
            candidate,
            trusted_case.case_path,
            trusted_case.repository_path,
            trusted_case.reference_patch,
            trusted_case.reference_patch.parent,
        )
        if (
            (os.name != "nt" and stat.S_IMODE(metadata.st_mode) & 0o077 != 0)
            or _is_protected_mount(resolved)
            or any(_overlaps(resolved, path.resolve(strict=False)) for path in protected)
        ):
            raise VerificationFilesystemError
        return resolved


class LocalEvaluatorBundleFilesystem:
    def __init__(self) -> None:
        self._identities: dict[Path, tuple[int, int, int, int]] = {}

    def create_bundle(
        self, workspace: Path, trusted_case: TrustedCase, identifier: str
    ) -> HiddenTestBundle:
        if type(identifier) is not str or _IDENTIFIER.fullmatch(identifier) is None:
            raise VerificationFilesystemError
        workspace_resolved, workspace_metadata = _ordinary_directory(workspace)
        if type(trusted_case) is not TrustedCase:
            raise EvaluatorInputError
        if (
            (os.name != "nt" and stat.S_IMODE(workspace_metadata.st_mode) & 0o077 != 0)
            or _is_protected_mount(workspace_resolved)
            or any(
                _overlaps(workspace_resolved, path.resolve(strict=False))
                for path in (
                    trusted_case.case_path,
                    trusted_case.repository_path,
                    trusted_case.reference_patch,
                    trusted_case.reference_patch.parent,
                )
            )
        ):
            raise VerificationFilesystemError
        root = workspace_resolved / f"{HIDDEN_BUNDLE_PREFIX}{identifier}"
        if root.exists():
            raise VerificationFilesystemError
        created = False
        try:
            root.mkdir(mode=0o755)
            created = True
            metadata = root.lstat()
            if root.resolve(strict=True) != root or not stat.S_ISDIR(metadata.st_mode):
                raise VerificationFilesystemError
            try:
                hidden_root, _ = _ordinary_directory(
                    trusted_case.case_path / "evaluator" / "hidden_tests"
                )
            except VerificationFilesystemError:
                raise EvaluatorInputError from None
            manifest_paths = trusted_case.manifest.hidden_tests
            if not manifest_paths or len(manifest_paths) != len(trusted_case.hidden_tests):
                raise EvaluatorInputError
            try:
                normalized_manifest_paths = normalize_repository_paths(manifest_paths)
            except UnifiedDiffError:
                raise EvaluatorInputError from None
            if normalized_manifest_paths != manifest_paths:
                raise EvaluatorInputError
            source_identities: set[tuple[int, int]] = set()
            generated: list[str] = []
            for index, (declared, source) in enumerate(
                zip(manifest_paths, trusted_case.hidden_tests, strict=True)
            ):
                relative = PurePosixPath(declared)
                if relative.parts[:2] != ("evaluator", "hidden_tests"):
                    raise EvaluatorInputError
                expected = trusted_case.case_path.joinpath(*relative.parts)
                try:
                    source_resolved = source.resolve(strict=True)
                    source_resolved.relative_to(hidden_root)
                except (OSError, RuntimeError, ValueError):
                    raise EvaluatorInputError from None
                if expected != source or source != source_resolved:
                    raise EvaluatorInputError
                try:
                    before = source.lstat()
                except OSError:
                    raise EvaluatorInputError from None
                identity = (before.st_dev, before.st_ino)
                if (
                    source.is_symlink()
                    or getattr(before, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or identity in source_identities
                    or before.st_size > _MAX_FILE_BYTES
                ):
                    raise EvaluatorInputError
                source_identities.add(identity)
                source_bytes = _read_bounded_ordinary_file(source, EvaluatorInputError)
                name = f"test_{index:04d}.py"
                destination = root / name
                with destination.open("xb") as handle:
                    handle.write(source_bytes)
                if (
                    _read_bounded_ordinary_file(destination, VerificationFilesystemError)
                    != source_bytes
                ):
                    raise VerificationFilesystemError
                if os.name != "nt":
                    destination.chmod(0o444)
                generated.append(name)
            final = root.lstat()
            self._identities[root] = (
                final.st_dev,
                final.st_ino,
                final.st_mode,
                getattr(final, "st_reparse_tag", 0),
            )
            return HiddenTestBundle(
                workspace_root=workspace_resolved,
                root=root,
                test_files=tuple(generated),
            )
        except (KeyboardInterrupt, SystemExit):
            if created:
                with suppress(Exception):
                    self._remove_unchecked(root)
            raise
        except Exception as error:
            if created:
                try:
                    self._remove_unchecked(root)
                except Exception:
                    raise BundleCleanupError from error
            if isinstance(error, VerificationFilesystemError):
                raise
            raise VerificationFilesystemError from None

    def remove_bundle(self, bundle: HiddenTestBundle) -> None:
        if type(bundle) is not HiddenTestBundle:
            raise BundleCleanupError
        root = bundle.root
        expected = self._identities.get(root)
        if expected is None:
            raise BundleCleanupError
        try:
            metadata = root.lstat()
            actual = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                getattr(metadata, "st_reparse_tag", 0),
            )
            if (
                actual != expected
                or root.resolve(strict=True) != root
                or root.parent != (bundle.workspace_root)
            ):
                raise BundleCleanupError
            self._remove_unchecked(root)
            self._identities.pop(root, None)
        except BundleCleanupError:
            raise
        except Exception:
            raise BundleCleanupError from None

    @staticmethod
    def _remove_unchecked(root: Path) -> None:
        metadata = root.lstat()
        if (
            root.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise BundleCleanupError
        for path in root.iterdir():
            child = path.lstat()
            if (
                _GENERATED_TEST.fullmatch(path.name) is None
                or path.is_symlink()
                or getattr(child, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                or not stat.S_ISREG(child.st_mode)
                or child.st_nlink != 1
            ):
                raise BundleCleanupError
        if os.name != "nt":
            root.chmod(0o700)
        shutil.rmtree(root)


TrustedCaseLoader = Callable[[str], TrustedCase]


def _diagnostic(code: VerificationDiagnosticCode) -> VerificationDiagnostic:
    return VerificationDiagnostic(code=code, message=_DIAGNOSTIC_MESSAGES[code])


def _not_run(stage: VerificationStage) -> VerificationStageResult:
    return VerificationStageResult(
        stage=stage,
        status=VerificationStageStatus.NOT_RUN,
        diagnostic=_diagnostic(VerificationDiagnosticCode.PREREQUISITE_FAILED),
    )


def _blank_stages() -> list[VerificationStageResult]:
    return [_not_run(stage) for stage in VerificationStage]


def _infrastructure_result(
    stages: list[VerificationStageResult],
    index: int,
    code: VerificationDiagnosticCode,
    evidence: VerificationSandboxEvidence | None = None,
) -> CandidateVerificationResult:
    values: dict[str, object] = {
        "stage": tuple(VerificationStage)[index],
        "status": VerificationStageStatus.INFRASTRUCTURE_ERROR,
        "diagnostic": _diagnostic(code),
    }
    if evidence is not None:
        values.update(evidence.model_dump())
    stages[index] = VerificationStageResult.model_validate(values, strict=True)
    for later in range(index + 1, len(stages)):
        stages[later] = _not_run(tuple(VerificationStage)[later])
    return CandidateVerificationResult(stages=tuple(stages))


def _stage_from_evidence(
    stage: VerificationStage, evidence: VerificationSandboxEvidence
) -> VerificationStageResult:
    values = evidence.model_dump()
    values["stage"] = stage
    if evidence.completion is SandboxCompletion.TIMED_OUT:
        values["status"] = VerificationStageStatus.INFRASTRUCTURE_ERROR
        values["diagnostic"] = _diagnostic(VerificationDiagnosticCode.SANDBOX_TIMED_OUT)
    elif evidence.completion is SandboxCompletion.MEMORY_LIMIT:
        values["status"] = VerificationStageStatus.INFRASTRUCTURE_ERROR
        values["diagnostic"] = _diagnostic(VerificationDiagnosticCode.SANDBOX_MEMORY_LIMIT)
    elif evidence.exit_code == 0:
        values["status"] = VerificationStageStatus.PASSED
    else:
        values["status"] = VerificationStageStatus.FAILED
    return VerificationStageResult.model_validate(values, strict=True)


def _runner_diagnostic(error: VerificationRunnerError) -> VerificationDiagnosticCode:
    return {
        VerificationRunnerErrorCode.INVALID_INPUT: (
            VerificationDiagnosticCode.SANDBOX_VALIDATION_FAILED
        ),
        VerificationRunnerErrorCode.DAEMON_SECURITY: (
            VerificationDiagnosticCode.SANDBOX_VALIDATION_FAILED
        ),
        VerificationRunnerErrorCode.DOCKER_FAILED: (
            VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED
        ),
        VerificationRunnerErrorCode.TIMED_OUT: VerificationDiagnosticCode.SANDBOX_TIMED_OUT,
        VerificationRunnerErrorCode.MEMORY_LIMIT: (VerificationDiagnosticCode.SANDBOX_MEMORY_LIMIT),
        VerificationRunnerErrorCode.MALFORMED_RESULT: (
            VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT
        ),
        VerificationRunnerErrorCode.CLEANUP_FAILED: (
            VerificationDiagnosticCode.SANDBOX_CLEANUP_FAILED
        ),
    }[error.code]


class VerificationService:
    def __init__(
        self,
        *,
        loader: TrustedCaseLoader,
        fingerprinter: CandidateFingerprinter,
        candidate_filesystem: CandidateFilesystem,
        bundle_filesystem: EvaluatorBundleFilesystem,
        sandbox: VerificationSandboxBoundary,
        identifier_factory: Callable[[], str],
    ) -> None:
        self._loader = loader
        self._fingerprinter = fingerprinter
        self._candidate_filesystem = candidate_filesystem
        self._bundle_filesystem = bundle_filesystem
        self._sandbox = sandbox
        self._identifier_factory = identifier_factory

    def verify(self, request: CandidateVerificationRequest) -> CandidateVerificationResult:
        stages = _blank_stages()
        try:
            validated = self._revalidate_request(request)
        except _StaticPrerequisiteError:
            return _infrastructure_result(
                stages, 0, VerificationDiagnosticCode.STATIC_PREREQUISITE_FAILED
            )
        except Exception:
            return _infrastructure_result(stages, 0, VerificationDiagnosticCode.INVALID_INPUT)

        candidate = validated.static_result.candidate
        if candidate is None:
            return _infrastructure_result(
                stages, 0, VerificationDiagnosticCode.STATIC_PREREQUISITE_FAILED
            )
        try:
            trusted_case = self._revalidate_case(self._loader(validated.case_id))
        except Exception:
            return _infrastructure_result(stages, 0, VerificationDiagnosticCode.CASE_LOAD_FAILED)
        if trusted_case.manifest.case_id != validated.case_id:
            return _infrastructure_result(stages, 0, VerificationDiagnosticCode.CASE_MISMATCH)
        try:
            workspace = self._candidate_filesystem.validate_workspace(
                validated.verification_workspace_root,
                candidate.repository,
                trusted_case,
            )
        except Exception:
            return _infrastructure_result(stages, 0, VerificationDiagnosticCode.WORKSPACE_INVALID)
        try:
            visible_paths = self._candidate_filesystem.validate_visible_tests(
                candidate.repository,
                trusted_case.manifest.visible_tests,
            )
        except Exception:
            return _infrastructure_result(
                stages, 0, VerificationDiagnosticCode.VISIBLE_TEST_INVALID
            )
        try:
            initial_identity = self._candidate_state(candidate)
        except Exception:
            return _infrastructure_result(stages, 0, VerificationDiagnosticCode.CANDIDATE_INVALID)

        visible_operations: tuple[Callable[[], VerificationSandboxEvidence], ...] = (
            lambda: self._sandbox.execute_targeted(candidate.repository, visible_paths),
            lambda: self._sandbox.execute_visible_suite(candidate.repository),
        )
        for index, operation in enumerate(visible_operations):
            evidence, error_code = self._execute_stage(operation)
            integrity_ok = self._integrity_matches(candidate, initial_identity)
            if not integrity_ok:
                return _infrastructure_result(
                    stages, index, VerificationDiagnosticCode.INTEGRITY_FAILED, evidence
                )
            if error_code is not None:
                return _infrastructure_result(stages, index, error_code, evidence)
            if evidence is None:
                return _infrastructure_result(
                    stages, index, VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT
                )
            stages[index] = _stage_from_evidence(tuple(VerificationStage)[index], evidence)
            if stages[index].status is VerificationStageStatus.INFRASTRUCTURE_ERROR:
                return CandidateVerificationResult(stages=tuple(stages))
            if stages[index].status is VerificationStageStatus.FAILED:
                return CandidateVerificationResult(stages=tuple(stages))

        try:
            identifier = self._identifier_factory()
            if type(identifier) is not str or _IDENTIFIER.fullmatch(identifier) is None:
                raise VerificationFilesystemError
            bundle = self._bundle_filesystem.create_bundle(workspace, trusted_case, identifier)
            if type(bundle) is not HiddenTestBundle:
                raise VerificationFilesystemError
            bundle = HiddenTestBundle.model_validate(
                bundle.model_dump(warnings="none"), strict=True
            )
        except BundleCleanupError:
            self._integrity_matches(candidate, initial_identity)
            return _infrastructure_result(stages, 2, VerificationDiagnosticCode.CLEANUP_FAILED)
        except EvaluatorInputError:
            code = (
                VerificationDiagnosticCode.EVALUATOR_INVALID
                if self._integrity_matches(candidate, initial_identity)
                else VerificationDiagnosticCode.INTEGRITY_FAILED
            )
            return _infrastructure_result(stages, 2, code)
        except Exception:
            code = (
                VerificationDiagnosticCode.BUNDLE_FAILED
                if self._integrity_matches(candidate, initial_identity)
                else VerificationDiagnosticCode.INTEGRITY_FAILED
            )
            return _infrastructure_result(stages, 2, code)

        hidden_evidence: VerificationSandboxEvidence | None = None
        hidden_error_code: VerificationDiagnosticCode | None = None
        cleanup_failed = False
        interrupted = False
        try:
            if not self._integrity_matches(candidate, initial_identity):
                hidden_error_code = VerificationDiagnosticCode.INTEGRITY_FAILED
            else:
                hidden_evidence, hidden_error_code = self._execute_stage(
                    lambda: self._sandbox.execute_hidden(candidate.repository, bundle)
                )
        except (KeyboardInterrupt, SystemExit):
            interrupted = True
            with suppress(Exception):
                self._bundle_filesystem.remove_bundle(bundle)
            raise
        finally:
            if not interrupted:
                try:
                    self._bundle_filesystem.remove_bundle(bundle)
                except Exception:
                    cleanup_failed = True

        integrity_ok = self._integrity_matches(candidate, initial_identity)
        if cleanup_failed:
            return _infrastructure_result(
                stages, 2, VerificationDiagnosticCode.CLEANUP_FAILED, hidden_evidence
            )
        if not integrity_ok:
            return _infrastructure_result(
                stages, 2, VerificationDiagnosticCode.INTEGRITY_FAILED, hidden_evidence
            )
        if hidden_error_code is not None:
            return _infrastructure_result(stages, 2, hidden_error_code, hidden_evidence)
        if hidden_evidence is None:
            return _infrastructure_result(
                stages, 2, VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT
            )
        stages[2] = _stage_from_evidence(VerificationStage.HIDDEN_SUITE, hidden_evidence)
        return CandidateVerificationResult(stages=tuple(stages))

    @staticmethod
    def _revalidate_request(
        request: CandidateVerificationRequest,
    ) -> CandidateVerificationRequest:
        if type(request) is not CandidateVerificationRequest:
            raise ValueError
        if (
            type(request.case_id) is not str
            or _CASE_ID.fullmatch(request.case_id) is None
            or not isinstance(request.verification_workspace_root, Path)
            or type(request.static_result) is not StaticValidationResult
        ):
            raise ValueError
        static = request.static_result
        try:
            raw_static = static.model_dump(warnings="none")
            checked_static = StaticValidationResult.model_validate(raw_static, strict=True)
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise _StaticPrerequisiteError from None
        if (
            checked_static.outcome is not StaticValidationOutcome.PASSED
            or checked_static.candidate is None
            or type(checked_static.candidate) is not ValidatedCandidate
            or type(checked_static.candidate.fingerprint) is not RepositoryFingerprint
            or checked_static.diagnostic is not None
            or tuple(item.check for item in checked_static.checks) != tuple(StaticCheck)
        ):
            raise _StaticPrerequisiteError
        raw = request.model_dump(warnings="none")
        checked = CandidateVerificationRequest.model_validate(raw, strict=True)
        if checked.model_dump(warnings="none") != raw:
            raise ValueError
        return checked

    @staticmethod
    def _revalidate_case(case: TrustedCase) -> TrustedCase:
        if type(case) is not TrustedCase:
            raise ValueError
        raw = case.model_dump(warnings="none")
        checked = TrustedCase.model_validate(raw, strict=True)
        if checked.model_dump(warnings="none") != raw:
            raise ValueError
        return checked

    def _candidate_state(self, candidate: ValidatedCandidate) -> CandidateInspection:
        first = self._candidate_filesystem.inspect_candidate(candidate.repository)
        if type(first) is not CandidateInspection:
            raise VerificationFilesystemError
        first = CandidateInspection.model_validate(first.model_dump(warnings="none"), strict=True)
        fingerprint = self._fingerprinter.fingerprint(candidate.repository)
        if type(fingerprint) is not RepositoryFingerprint:
            raise VerificationFilesystemError
        fingerprint = RepositoryFingerprint.model_validate(
            fingerprint.model_dump(warnings="none"), strict=True
        )
        second = self._candidate_filesystem.inspect_candidate(candidate.repository)
        if type(second) is not CandidateInspection:
            raise VerificationFilesystemError
        second = CandidateInspection.model_validate(second.model_dump(warnings="none"), strict=True)
        if (
            first != second
            or first.repository != candidate.repository
            or fingerprint != candidate.fingerprint
        ):
            raise VerificationFilesystemError
        return first

    def _integrity_matches(
        self, candidate: ValidatedCandidate, expected: CandidateInspection
    ) -> bool:
        try:
            return self._candidate_state(candidate) == expected
        except Exception:
            return False

    @staticmethod
    def _execute_stage(
        operation: Callable[[], VerificationSandboxEvidence],
    ) -> tuple[VerificationSandboxEvidence | None, VerificationDiagnosticCode | None]:
        try:
            raw = operation()
            if type(raw) is not VerificationSandboxEvidence:
                raise ValueError
            dumped = raw.model_dump(warnings="none")
            evidence = VerificationSandboxEvidence.model_validate(dumped, strict=True)
            if evidence.model_dump(warnings="none") != dumped:
                raise ValueError
            return evidence, None
        except VerificationRunnerValidationError:
            return None, VerificationDiagnosticCode.SANDBOX_VALIDATION_FAILED
        except VerificationRunnerError as error:
            if (
                type(error) is not VerificationRunnerError
                or type(error.code) is not VerificationRunnerErrorCode
            ):
                return None, VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED
            return None, _runner_diagnostic(error)
        except (AttributeError, TypeError, ValueError, ValidationError):
            return None, VerificationDiagnosticCode.MALFORMED_SANDBOX_RESULT
        except Exception:
            return None, VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED


class _StaticPrerequisiteError(Exception):
    pass
