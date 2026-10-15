from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from app.agent.graph import OrchestrationState
from app.agent.policy import PatchPolicyResult, PatchPolicyViolation
from app.agent.routing import EXIT_CODES, TerminalDiagnostic, TerminalOutcome, terminal_summary
from app.agent.schemas import BaselineDiagnosticCode, BaselineOutcome, FailureCategory
from app.agent.static_checks import (
    StaticCheck,
    StaticCheckResult,
    StaticCheckStatus,
    StaticDiagnostic,
    StaticValidationOutcome,
)
from app.agent.verification import CandidateVerificationResult, VerificationOutcome
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationResult,
    CandidateFunctionalStatus,
)
from app.sandbox.results import NonNegativeFiniteFloat, SandboxCompletion

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_TEMP_ID = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_REPARSE_POINT = 0x400


class BaselineArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: BaselineOutcome
    diagnostic_code: BaselineDiagnosticCode | None
    completion: SandboxCompletion | None
    exit_code: int | None
    stdout_truncated: bool | None
    stderr_truncated: bool | None
    duration_seconds: NonNegativeFiniteFloat | None

    @model_validator(mode="after")
    def validate_outcome(self) -> BaselineArtifact:
        evidence = (
            self.completion,
            self.exit_code,
            self.stdout_truncated,
            self.stderr_truncated,
            self.duration_seconds,
        )
        if self.outcome is BaselineOutcome.REPRODUCED_FAILURE:
            if (
                self.diagnostic_code is not None
                or self.completion is not SandboxCompletion.NORMAL
                or self.exit_code in (None, 0)
                or any(value is None for value in evidence[2:])
            ):
                raise ValueError("reproduced baseline artifact is inconsistent")
        elif self.outcome is BaselineOutcome.NOT_REPRODUCIBLE:
            if (
                self.diagnostic_code is not BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE
                or self.completion is not SandboxCompletion.NORMAL
                or self.exit_code != 0
                or any(value is None for value in evidence[2:])
            ):
                raise ValueError("non-reproduced baseline artifact is inconsistent")
        elif self.diagnostic_code in {
            None,
            BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE,
        }:
            raise ValueError("infrastructure baseline artifact requires a diagnostic")
        elif self.diagnostic_code is BaselineDiagnosticCode.SANDBOX_TIMED_OUT:
            if self.completion is not SandboxCompletion.TIMED_OUT or self.exit_code is not None:
                raise ValueError("timeout baseline artifact is inconsistent")
        elif self.diagnostic_code is BaselineDiagnosticCode.SANDBOX_MEMORY_LIMIT:
            if self.completion is not SandboxCompletion.MEMORY_LIMIT:
                raise ValueError("memory-limit baseline artifact is inconsistent")
        elif any(value is not None for value in evidence):
            raise ValueError("non-execution baseline artifact cannot contain execution metadata")
        return self


class PolicyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    approved: bool
    violations: tuple[PatchPolicyViolation, ...]

    @model_validator(mode="after")
    def validate_policy(self) -> PolicyArtifact:
        checked = PatchPolicyResult(approved=self.approved, violations=self.violations)
        if checked.approved != self.approved or checked.violations != self.violations:
            raise ValueError("policy artifact is not stable")
        return self


class StaticArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: StaticValidationOutcome
    checks: tuple[StaticCheckResult, ...]
    diagnostic: StaticDiagnostic | None

    @model_validator(mode="after")
    def validate_static(self) -> StaticArtifact:
        if tuple(result.check for result in self.checks) != tuple(StaticCheck):
            raise ValueError("static artifact checks are incomplete or out of order")
        statuses = tuple(result.status for result in self.checks)
        if any(status is StaticCheckStatus.NOT_APPLICABLE for status in statuses[:2]):
            raise ValueError("required static checks cannot be not applicable")
        language_statuses = set(statuses[2:])
        if StaticCheckStatus.NOT_APPLICABLE in language_statuses and language_statuses != {
            StaticCheckStatus.NOT_APPLICABLE
        }:
            raise ValueError("language static checks must share applicability")
        if self.outcome is StaticValidationOutcome.PASSED:
            if (
                self.diagnostic is not None
                or any(
                    status not in {StaticCheckStatus.PASSED, StaticCheckStatus.NOT_APPLICABLE}
                    for status in statuses
                )
                or language_statuses
                not in (
                    {StaticCheckStatus.PASSED},
                    {StaticCheckStatus.NOT_APPLICABLE},
                )
            ):
                raise ValueError("passing static artifact contains blocking evidence")
        elif self.outcome is StaticValidationOutcome.CANDIDATE_FAILED:
            if (
                self.diagnostic is not None
                or StaticCheckStatus.FAILED not in statuses
                or StaticCheckStatus.INFRASTRUCTURE_ERROR in statuses
            ):
                raise ValueError("failed static artifact is inconsistent")
        elif self.diagnostic is None or StaticCheckStatus.INFRASTRUCTURE_ERROR not in statuses:
            raise ValueError("infrastructure static artifact is inconsistent")
        return self


class RunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    outcome: TerminalOutcome
    exit_code: int
    started_at: AwareDatetime
    completed_at: AwareDatetime
    attempt_number: int = Field(strict=True, ge=0, le=1)
    max_attempts: Literal[1] = 1
    provider_name: str | None
    model_name: str | None
    prompt_version: str | None
    baseline: BaselineArtifact | None
    failure_category: FailureCategory | None
    policy: PolicyArtifact | None
    static: StaticArtifact | None
    verification: CandidateVerificationResult | None
    candidate_evaluation: CandidateEvaluationResult | None
    terminal_diagnostic: TerminalDiagnostic | None

    @model_validator(mode="after")
    def validate_artifact(self) -> RunArtifact:
        if self.exit_code != EXIT_CODES[self.outcome]:
            raise ValueError("artifact exit code does not match outcome")
        if self.started_at.utcoffset() != UTC.utcoffset(
            self.started_at
        ) or self.completed_at.utcoffset() != UTC.utcoffset(self.completed_at):
            raise ValueError("artifact timestamps must be UTC")
        if self.completed_at < self.started_at:
            raise ValueError("artifact completion precedes start")
        configuration = (self.provider_name, self.model_name, self.prompt_version)
        if any(value is None for value in configuration) and any(
            value is not None for value in configuration
        ):
            raise ValueError("artifact model configuration is incomplete")
        expected = (
            None
            if self.outcome is TerminalOutcome.SUCCESS
            else TerminalDiagnostic.for_outcome(self.outcome)
        )
        if self.terminal_diagnostic != expected:
            raise ValueError("artifact diagnostic does not match outcome")
        if self.verification is None and self.candidate_evaluation is not None:
            raise ValueError("artifact evaluation requires verification")
        if self.policy is None and self.static is not None:
            raise ValueError("artifact static evidence requires policy evidence")
        if self.baseline is None and any(
            value is not None
            for value in (
                self.failure_category,
                self.policy,
                self.static,
                self.verification,
                self.candidate_evaluation,
            )
        ):
            raise ValueError("artifact evidence requires a baseline")
        if (
            self.baseline is not None
            and self.baseline.outcome is not BaselineOutcome.REPRODUCED_FAILURE
            and any(
                value is not None
                for value in (
                    self.failure_category,
                    self.policy,
                    self.static,
                    self.verification,
                    self.candidate_evaluation,
                )
            )
        ):
            raise ValueError("terminal baseline artifact cannot contain later evidence")
        if (
            self.policy is not None
            and not self.policy.approved
            and any(
                value is not None
                for value in (self.static, self.verification, self.candidate_evaluation)
            )
        ):
            raise ValueError("rejected policy artifact cannot contain candidate evidence")
        if (
            self.static is not None
            and self.static.outcome is not StaticValidationOutcome.PASSED
            and (self.verification is not None or self.candidate_evaluation is not None)
        ):
            raise ValueError("non-passing static artifact cannot contain verification evidence")
        if self.verification is not None and self.candidate_evaluation is not None:
            expected_status = {
                VerificationOutcome.PASSED: CandidateFunctionalStatus.PASSED,
                VerificationOutcome.CANDIDATE_FAILED: CandidateFunctionalStatus.FAILED,
                VerificationOutcome.INFRASTRUCTURE_DIAGNOSTIC: (
                    CandidateFunctionalStatus.NOT_EVALUATED
                ),
            }[self.verification.outcome]
            if (
                self.candidate_evaluation.case_id != self.case_id
                or self.candidate_evaluation.functional_status is not expected_status
            ):
                raise ValueError("artifact evaluation contradicts verification")
        self._validate_terminal_shape()
        return self

    def _validate_terminal_shape(self) -> None:
        reproduced = (
            self.baseline is not None
            and self.baseline.outcome is BaselineOutcome.REPRODUCED_FAILURE
        )
        approved = self.policy is not None and self.policy.approved
        static_passed = (
            self.static is not None and self.static.outcome is StaticValidationOutcome.PASSED
        )
        verification_outcome = None if self.verification is None else self.verification.outcome
        evaluation_status = (
            None
            if self.candidate_evaluation is None
            else self.candidate_evaluation.functional_status
        )
        classified = self.failure_category is not None
        configured = self.provider_name is not None
        if self.outcome is TerminalOutcome.SUCCESS:
            valid = (
                self.attempt_number == 1
                and reproduced
                and classified
                and configured
                and approved
                and static_passed
                and verification_outcome is VerificationOutcome.PASSED
                and evaluation_status is CandidateFunctionalStatus.PASSED
            )
        elif self.outcome is TerminalOutcome.CANDIDATE_FAILURE:
            early = self.policy is None and self.static is None and self.verification is None
            late = (
                approved
                and static_passed
                and verification_outcome is VerificationOutcome.CANDIDATE_FAILED
                and evaluation_status is CandidateFunctionalStatus.FAILED
            )
            valid = (
                self.attempt_number == 1
                and reproduced
                and classified
                and configured
                and (early or late)
            )
        elif self.outcome is TerminalOutcome.POLICY_VIOLATION:
            valid = (
                self.attempt_number == 1
                and reproduced
                and classified
                and configured
                and self.policy is not None
                and not self.policy.approved
                and self.static is None
                and self.verification is None
                and self.candidate_evaluation is None
            )
        elif self.outcome is TerminalOutcome.STATIC_CHECK_FAILURE:
            valid = (
                self.attempt_number == 1
                and reproduced
                and classified
                and configured
                and approved
                and self.static is not None
                and self.static.outcome is StaticValidationOutcome.CANDIDATE_FAILED
                and self.verification is None
                and self.candidate_evaluation is None
            )
        elif self.outcome is TerminalOutcome.BASELINE_NOT_REPRODUCIBLE:
            valid = (
                self.attempt_number == 0
                and self.baseline is not None
                and self.baseline.outcome is BaselineOutcome.NOT_REPRODUCIBLE
            )
        elif self.outcome is TerminalOutcome.MALFORMED_MODEL_OUTPUT:
            valid = (
                self.attempt_number == 1
                and reproduced
                and classified
                and configured
                and self.policy is None
                and self.static is None
                and self.verification is None
                and self.candidate_evaluation is None
            )
        else:
            valid = True
        if not valid:
            raise ValueError("artifact evidence does not match terminal outcome")


def artifact_bytes(artifact: RunArtifact) -> bytes:
    if type(artifact) is not RunArtifact:
        raise ValueError("artifact has the wrong type")
    raw = artifact.model_dump(warnings="none")
    checked = RunArtifact.model_validate(raw, strict=True)
    if checked.model_dump(warnings="none") != raw:
        raise ValueError("artifact is not stable")
    serialized = json.dumps(
        checked.model_dump(mode="json", warnings="none"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return bytes(f"{serialized}\n", "utf-8")


def _baseline_projection(state: OrchestrationState) -> BaselineArtifact | None:
    if state.baseline is None:
        return None
    evidence = state.baseline.evidence
    return BaselineArtifact(
        outcome=state.baseline.outcome,
        diagnostic_code=state.baseline.diagnostic_code,
        completion=None if evidence is None else evidence.completion,
        exit_code=None if evidence is None else evidence.exit_code,
        stdout_truncated=None if evidence is None else evidence.stdout_truncated,
        stderr_truncated=None if evidence is None else evidence.stderr_truncated,
        duration_seconds=None if evidence is None else evidence.duration_seconds,
    )


def project_artifact(state: OrchestrationState) -> RunArtifact:
    if type(state) is not OrchestrationState:
        raise ValueError("orchestration state has the wrong type")
    try:
        raw = state.model_dump(warnings="none")
        checked = OrchestrationState.model_validate(raw, strict=True)
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise ValueError("orchestration state is invalid") from None
    route = checked.terminal_route
    if route is None or checked.completed_at is None:
        raise ValueError("run has not reached a terminal route")
    configuration = checked.model_configuration
    policy = (
        None
        if checked.policy_result is None
        else PolicyArtifact(
            approved=checked.policy_result.approved,
            violations=checked.policy_result.violations,
        )
    )
    static = (
        None
        if checked.static_result is None
        else StaticArtifact(
            outcome=checked.static_result.outcome,
            checks=checked.static_result.checks,
            diagnostic=checked.static_result.diagnostic,
        )
    )
    return RunArtifact(
        run_id=checked.run_id,
        case_id=checked.case_id,
        outcome=route.outcome,
        exit_code=route.exit_code,
        started_at=checked.started_at,
        completed_at=checked.completed_at,
        attempt_number=checked.attempt_number,
        provider_name=None if configuration is None else configuration.provider,
        model_name=None if configuration is None else configuration.model_name,
        prompt_version=None if configuration is None else configuration.prompt_version,
        baseline=_baseline_projection(checked),
        failure_category=(
            None if checked.failure_analysis is None else checked.failure_analysis.category
        ),
        policy=policy,
        static=static,
        verification=checked.verification_result,
        candidate_evaluation=checked.evaluation_result,
        terminal_diagnostic=route.diagnostic,
    )


def _relative_artifact_path(value: PurePosixPath) -> None:
    rendered = value.as_posix()
    windows = PureWindowsPath(rendered)
    if (
        value.is_absolute()
        or windows.drive
        or windows.root
        or value.parts[:2] != (".tracefix", "artifacts")
        or len(value.parts) != 3
        or any(part in {"", ".", ".."} for part in value.parts)
        or "\\" in rendered
    ):
        raise ValueError("artifact path must be a safe invocation-relative path")


class ArtifactPublication(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    relative_path: PurePosixPath

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _relative_artifact_path(self.relative_path)
        return self


class ArtifactSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    outcome: TerminalOutcome
    summary: str
    exit_code: int
    artifact_path: PurePosixPath

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.summary != terminal_summary(self.outcome):
            raise ValueError("summary must be the safe terminal constant")
        if self.exit_code != EXIT_CODES[self.outcome]:
            raise ValueError("summary exit code does not match outcome")
        _relative_artifact_path(self.artifact_path)
        return self

    def render(self) -> str:
        return (
            f"case_id={self.case_id} outcome={self.outcome.value} "
            f"summary={self.summary} exit_code={self.exit_code} "
            f"artifact={self.artifact_path.as_posix()}"
        )


class ArtifactWriteError(Exception):
    """A safe artifact could not be published with certainty."""


class ArtifactWriter(Protocol):
    def write(self, artifact: RunArtifact, invocation_directory: Path) -> ArtifactPublication: ...


def _ordinary_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ArtifactWriteError("artifact output root is invalid")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ArtifactWriteError("artifact output root is invalid") from None
    if (
        path != resolved
        or not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise ArtifactWriteError("artifact output root is invalid")
    return resolved


def _directory_identity(path: Path) -> tuple[int, int, int, int]:
    checked = _ordinary_directory(path)
    metadata = checked.lstat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        getattr(metadata, "st_file_attributes", 0),
    )


def _require_directory_identity(path: Path, identity: tuple[int, int, int, int]) -> None:
    if _directory_identity(path) != identity:
        raise ArtifactWriteError("artifact output directory identity changed")


def _ensure_private_directory(parent: Path, name: str) -> Path:
    parent_identity = _directory_identity(parent)
    child = parent / name
    try:
        child.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError:
        raise ArtifactWriteError("artifact output directory could not be created") from None
    checked = _ordinary_directory(child)
    if checked.parent != parent:
        raise ArtifactWriteError("artifact output directory is aliased")
    _require_directory_identity(parent, parent_identity)
    return checked


def _safe_filename(case_id: str, run_id: str) -> str:
    if _CASE_ID.fullmatch(case_id) is None or _RUN_ID.fullmatch(run_id) is None:
        raise ArtifactWriteError("artifact identity is invalid")
    filename = f"{case_id}-{run_id}.json"
    if Path(filename).name != filename or PureWindowsPath(filename).name != filename:
        raise ArtifactWriteError("artifact filename is invalid")
    return filename


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _unlink_owned(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
        if (metadata.st_dev, metadata.st_ino) != identity or not stat.S_ISREG(metadata.st_mode):
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _cleanup_failed_publication(
    *,
    descriptor: int | None,
    temporary: Path,
    destination: Path,
    published: bool,
    file_identity: tuple[int, int] | None,
) -> bool:
    if descriptor is not None:
        with suppress(OSError):
            os.close(descriptor)
    temporary_removed = _unlink(temporary)
    destination_removed = (
        not published or file_identity is not None and _unlink_owned(destination, file_identity)
    )
    return temporary_removed and destination_removed


def _cleanup_preserving_exception(
    *,
    descriptor: int | None,
    temporary: Path,
    destination: Path,
    published: bool,
    file_identity: tuple[int, int] | None,
) -> None:
    with suppress(ArtifactWriteError, KeyboardInterrupt, OSError, SystemExit):
        _cleanup_failed_publication(
            descriptor=descriptor,
            temporary=temporary,
            destination=destination,
            published=published,
            file_identity=file_identity,
        )


def _sync_directory(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(temporary, destination)
        return
    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError):
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from None
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_current_working_directory = -100
    no_replace = 1
    result = renameat2(
        at_current_working_directory,
        os.fsencode(temporary),
        at_current_working_directory,
        os.fsencode(destination),
        no_replace,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


class LocalArtifactWriter:
    def __init__(self, identifier_factory: Callable[[], str] = lambda: uuid.uuid4().hex) -> None:
        self._identifier_factory = identifier_factory

    def write(self, artifact: RunArtifact, invocation_directory: Path) -> ArtifactPublication:
        try:
            payload = artifact_bytes(artifact)
            filename = _safe_filename(artifact.case_id, artifact.run_id)
            invocation = _ordinary_directory(invocation_directory)
            private_root = _ensure_private_directory(invocation, ".tracefix")
            output = _ensure_private_directory(private_root, "artifacts")
            output_identity = _directory_identity(output)
            for existing in output.iterdir():
                if existing.name.casefold() == filename.casefold():
                    raise ArtifactWriteError("artifact destination already exists")
            destination = output / filename
            identifier = self._identifier_factory()
            if type(identifier) is not str or _TEMP_ID.fullmatch(identifier) is None:
                raise ArtifactWriteError("artifact temporary identity is invalid")
            temporary = output / f".{filename}.{identifier}.tmp"
        except ArtifactWriteError:
            raise
        except Exception:
            raise ArtifactWriteError("artifact output preparation failed") from None

        descriptor: int | None = None
        published = False
        file_identity: tuple[int, int] | None = None
        try:
            _require_directory_identity(output, output_identity)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            descriptor = os.open(temporary, flags, 0o600)
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError(errno.EIO, "artifact write made no progress")
                written += count
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            file_identity = (metadata.st_dev, metadata.st_ino)
            os.close(descriptor)
            descriptor = None
            _require_directory_identity(output, output_identity)
            _publish_no_replace(temporary, destination)
            published = True
            published_metadata = destination.lstat()
            if file_identity != (
                published_metadata.st_dev,
                published_metadata.st_ino,
            ) or not stat.S_ISREG(published_metadata.st_mode):
                raise OSError(errno.EIO, "published artifact identity changed")
            _require_directory_identity(output, output_identity)
            collisions = tuple(
                existing
                for existing in output.iterdir()
                if existing.name.casefold() == filename.casefold()
            )
            if collisions != (destination,):
                raise OSError(errno.EIO, "artifact destination collision appeared")
            _sync_directory(output)
            return ArtifactPublication(
                relative_path=PurePosixPath(".tracefix", "artifacts", filename)
            )
        except (ArtifactWriteError, KeyboardInterrupt, SystemExit):
            _cleanup_preserving_exception(
                descriptor=descriptor,
                temporary=temporary,
                destination=destination,
                published=published,
                file_identity=file_identity,
            )
            raise
        except (FileExistsError, OSError):
            cleaned = _cleanup_failed_publication(
                descriptor=descriptor,
                temporary=temporary,
                destination=destination,
                published=published,
                file_identity=file_identity,
            )
            if not cleaned:
                raise ArtifactWriteError("artifact publication state is uncertain") from None
            raise ArtifactWriteError("artifact publication failed") from None
