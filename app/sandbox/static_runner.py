from __future__ import annotations

import json
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.sandbox.results import SandboxLimits
from app.sandbox.runner import (
    DockerCommandAdapter,
    DockerCommandError,
    DockerCommandResult,
    DockerCommandTimeout,
)

STATIC_CHECK_IMAGE = "tracefix-static-checks:0.1.8"
STATIC_CONTAINER_NAME_PREFIX = "tracefix-static-checks-"

_CONTAINER_INPUT = "/tracefix/input"
_CONTAINER_WORKSPACE = "/tracefix/workspace"
_CONTAINER_TEMP = "/tmp"
_CONTAINER_USER = "10001:10001"
_MAX_PROTOCOL_BYTES = 65_536
_MAX_CHANGED_PATHS = 3
_MAX_RULE_CODE = 64
_WINDOWS_REPARSE_POINT = 0x400
_RULE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FIXED_ENVIRONMENT = (
    "HOME=/nonexistent",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONHASHSEED=0",
    "PYTHONNOUSERSITE=1",
    "PYTHONUNBUFFERED=1",
)
_DOCKER_INFO_ARGUMENTS = ("info", "--format", "{{json .}}")


class StaticToolCheck(StrEnum):
    PYTHON_COMPILE = "python_compile"
    AST_PARSE = "ast_parse"
    RUFF = "ruff"
    MYPY = "mypy"
    BANDIT = "bandit"


class StaticToolStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class StaticToolErrorCode(StrEnum):
    EXECUTABLE_MISSING = "executable_missing"
    VERSION_MISMATCH = "version_mismatch"
    INVOCATION_FAILED = "invocation_failed"
    OUTPUT_INVALID = "output_invalid"
    OUTPUT_LIMIT = "output_limit"
    SOURCE_READ_FAILED = "source_read_failed"


def _validate_path(value: str, *, require_python: bool = False) -> None:
    windows = PureWindowsPath(value)
    path = PurePosixPath(value)
    parts = value.split("/")
    if (
        not value
        or len(value.encode("utf-8")) > 1_024
        or value != value.strip()
        or "\\" in value
        or windows.drive
        or windows.root
        or path.is_absolute()
        or path.as_posix() != value
        or value.startswith(("-", "@"))
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        or (require_python and path.suffix.casefold() not in {".py", ".pyi"})
    ):
        raise ValueError("path must be a normalized relative Python path")


class StaticToolFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: Annotated[str, Field(min_length=1, max_length=1_024)]
    line: Annotated[int, Field(strict=True, ge=1, le=10_000_000)] | None = None
    column: Annotated[int, Field(strict=True, ge=1, le=10_000_000)] | None = None
    rule_code: Annotated[str, Field(min_length=1, max_length=_MAX_RULE_CODE)]

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        _validate_path(self.path, require_python=True)
        if self.column is not None and self.line is None:
            raise ValueError("finding column requires a line")
        if _RULE_CODE.fullmatch(self.rule_code) is None:
            raise ValueError("finding rule code is invalid")
        return self


def _finding_key(finding: StaticToolFinding) -> tuple[str, str, int, int, str]:
    return (
        finding.path.casefold(),
        finding.path,
        finding.line or 0,
        finding.column or 0,
        finding.rule_code,
    )


class StaticToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    check: StaticToolCheck
    status: StaticToolStatus
    findings: tuple[StaticToolFinding, ...] = ()
    error_code: StaticToolErrorCode | None = None

    @field_validator("findings", mode="before")
    @classmethod
    def freeze_findings(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if len(self.findings) > 1_000:
            raise ValueError("too many static findings")
        if self.findings != tuple(sorted(self.findings, key=_finding_key)):
            raise ValueError("static findings must be deterministically ordered")
        if len(set(self.findings)) != len(self.findings):
            raise ValueError("static findings must not contain duplicates")
        if self.status is StaticToolStatus.FAILED:
            if not self.findings or self.error_code is not None:
                raise ValueError("failed tool result requires findings only")
        elif self.status is StaticToolStatus.INFRASTRUCTURE_ERROR:
            if self.findings or self.error_code is None:
                raise ValueError("infrastructure tool result requires an error code only")
        elif self.findings or self.error_code is not None:
            raise ValueError("passing or inapplicable tool result cannot contain evidence")
        return self


class StaticContainerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Literal[1] = 1
    results: tuple[StaticToolResult, ...]

    @field_validator("results", mode="before")
    @classmethod
    def freeze_results(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_complete_order(self) -> Self:
        if tuple(item.check for item in self.results) != tuple(StaticToolCheck):
            raise ValueError("static tool results must contain the fixed complete order")
        statuses = {item.status for item in self.results}
        if StaticToolStatus.NOT_APPLICABLE in statuses and statuses != {
            StaticToolStatus.NOT_APPLICABLE
        }:
            raise ValueError("not-applicable status must cover every static tool")
        return self


class StaticContainerRunner(Protocol):
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult: ...


class StaticRunnerErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    DAEMON_SECURITY = "daemon_security"
    DOCKER_FAILED = "docker_failed"
    TIMED_OUT = "timed_out"
    MEMORY_LIMIT = "memory_limit"
    UNEXPECTED_EXIT = "unexpected_exit"
    MALFORMED_RESULT = "malformed_result"
    CLEANUP_FAILED = "cleanup_failed"


class StaticRunnerError(Exception):
    def __init__(self, code: StaticRunnerErrorCode) -> None:
        super().__init__(f"static_runner:{code.value}")
        self.code = code


class _DockerState(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)

    OOMKilled: bool
    Running: bool


class _DockerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)

    OSType: str
    SecurityOptions: tuple[str, ...]

    @field_validator("SecurityOptions", mode="before")
    @classmethod
    def freeze_security_options(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


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
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
    )


def _validated_repository(repository: Path) -> Path:
    if not isinstance(repository, Path) or not repository.is_absolute():
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    try:
        root_metadata = repository.lstat()
        resolved = repository.resolve(strict=True)
    except (OSError, RuntimeError):
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT) from None
    if repository != resolved or not stat.S_ISDIR(root_metadata.st_mode):
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    if (
        repository.is_symlink()
        or getattr(root_metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    if any(
        _same_or_ancestor(resolved, location.resolve(strict=False))
        for location in _protected_locations()
    ):
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    identities: set[str] = set()
    try:
        for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
            for name in (*directory_names, *file_names):
                path = Path(directory) / name
                relative = path.relative_to(resolved).as_posix()
                identity = relative.casefold()
                if identity in identities:
                    raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
                identities.add(identity)
                metadata = path.lstat()
                if (
                    path.is_symlink()
                    or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                    or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
                    or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
                ):
                    raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    except StaticRunnerError:
        raise
    except OSError:
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT) from None
    return resolved


def _validated_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    if type(paths) is not tuple or len(paths) > _MAX_CHANGED_PATHS:
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    identities: set[str] = set()
    for value in paths:
        if type(value) is not str:
            raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
        try:
            _validate_path(value, require_python=True)
        except (UnicodeError, ValueError):
            raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT) from None
        identity = value.casefold()
        if identity in identities:
            raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
        identities.add(identity)
    if paths != tuple(sorted(paths, key=lambda item: (item.casefold(), item))):
        raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
    return paths


class DockerStaticRunner:
    """Run the fixed static protocol in a fresh hardened Linux container."""

    def __init__(
        self,
        docker: DockerCommandAdapter,
        *,
        limits: SandboxLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
        identifier_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._docker = docker
        self._limits = limits or SandboxLimits(
            timeout_seconds=120.0,
            stdout_bytes=_MAX_PROTOCOL_BYTES,
            stderr_bytes=4_096,
        )
        self._clock = clock
        self._identifier_factory = identifier_factory

    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        prepared = _validated_repository(repository)
        paths = _validated_paths(changed_python_paths)
        started = self._clock()
        self._validate_daemon_security(started)
        try:
            identifier = self._identifier_factory()
        except Exception:
            raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT) from None
        if re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
            raise StaticRunnerError(StaticRunnerErrorCode.INVALID_INPUT)
        name = f"{STATIC_CONTAINER_NAME_PREFIX}{identifier}"
        primary: StaticRunnerError | None = None
        result: StaticContainerResult | None = None
        process_control = False
        try:
            self._run_lifecycle(
                self._create_arguments(name, prepared, paths, self._limits), started=started
            )
            try:
                attached = self._docker.run(
                    ("start", "--attach", name),
                    timeout_seconds=self._remaining_seconds(started),
                    stdout_bytes=self._limits.stdout_bytes,
                    stderr_bytes=self._limits.stderr_bytes,
                    check=False,
                )
            except DockerCommandTimeout:
                state = self._inspect(name, started)
                if state.Running:
                    self._run_lifecycle(("kill", "--signal", "KILL", name), started=started)
                raise StaticRunnerError(StaticRunnerErrorCode.TIMED_OUT) from None
            waited = self._run_lifecycle(("wait", name), started=started)
            exit_code = _parse_exit_code(waited.stdout)
            state = self._inspect(name, started)
            if state.OOMKilled:
                raise StaticRunnerError(StaticRunnerErrorCode.MEMORY_LIMIT)
            if attached.exit_code != 0 or exit_code != 0:
                raise StaticRunnerError(StaticRunnerErrorCode.UNEXPECTED_EXIT)
            if (
                attached.stdout_truncated
                or attached.stderr_truncated
                or attached.stderr
                or len(attached.stdout) > _MAX_PROTOCOL_BYTES
            ):
                raise StaticRunnerError(StaticRunnerErrorCode.MALFORMED_RESULT)
            try:
                result = StaticContainerResult.model_validate_json(attached.stdout, strict=True)
            except (ValidationError, ValueError):
                raise StaticRunnerError(StaticRunnerErrorCode.MALFORMED_RESULT) from None
            self._remaining_seconds(started)
        except (KeyboardInterrupt, SystemExit):
            process_control = True
            raise
        except StaticRunnerError as error:
            primary = error
        except DockerCommandTimeout:
            primary = StaticRunnerError(StaticRunnerErrorCode.TIMED_OUT)
        except (DockerCommandError, OSError, ValueError):
            primary = StaticRunnerError(StaticRunnerErrorCode.DOCKER_FAILED)
        finally:
            try:
                self._run_lifecycle(("rm", "--force", name))
            except Exception:
                if not process_control:
                    raise StaticRunnerError(StaticRunnerErrorCode.CLEANUP_FAILED) from primary
        if primary is not None:
            raise primary
        if result is None:
            raise StaticRunnerError(StaticRunnerErrorCode.MALFORMED_RESULT)
        return result

    def _validate_daemon_security(self, started: float) -> None:
        try:
            response = self._run_lifecycle(_DOCKER_INFO_ARGUMENTS, started=started)
            info = _DockerInfo.model_validate(json.loads(response.stdout), strict=True)
        except DockerCommandTimeout:
            raise StaticRunnerError(StaticRunnerErrorCode.TIMED_OUT) from None
        except StaticRunnerError as error:
            if error.code is StaticRunnerErrorCode.TIMED_OUT:
                raise
            raise StaticRunnerError(StaticRunnerErrorCode.DAEMON_SECURITY) from None
        except (
            DockerCommandError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
        ):
            raise StaticRunnerError(StaticRunnerErrorCode.DAEMON_SECURITY) from None
        normalized = tuple(option.casefold() for option in info.SecurityOptions)
        if info.OSType != "linux" or not any(
            option == "name=seccomp" or option.startswith("name=seccomp,") for option in normalized
        ):
            raise StaticRunnerError(StaticRunnerErrorCode.DAEMON_SECURITY)
        if any("seccomp=unconfined" in option for option in normalized):
            raise StaticRunnerError(StaticRunnerErrorCode.DAEMON_SECURITY)

    def _inspect(self, name: str, started: float) -> _DockerState:
        try:
            response = self._run_lifecycle(
                ("inspect", "--format", "{{json .State}}", name), started=started
            )
            return _DockerState.model_validate(json.loads(response.stdout), strict=True)
        except DockerCommandTimeout:
            raise
        except (
            DockerCommandError,
            StaticRunnerError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
        ):
            raise StaticRunnerError(StaticRunnerErrorCode.DOCKER_FAILED) from None

    def _run_lifecycle(
        self,
        arguments: tuple[str, ...],
        *,
        started: float | None = None,
        check: bool = True,
    ) -> DockerCommandResult:
        response = self._docker.run(
            arguments,
            timeout_seconds=(
                self._limits.timeout_seconds
                if started is None
                else self._remaining_seconds(started)
            ),
            stdout_bytes=self._limits.stdout_bytes,
            stderr_bytes=self._limits.stderr_bytes,
            check=check,
        )
        if (
            response.stdout_truncated
            or response.stderr_truncated
            or response.stderr
            or (check and response.exit_code != 0)
        ):
            raise StaticRunnerError(StaticRunnerErrorCode.DOCKER_FAILED)
        return response

    def _remaining_seconds(self, started: float) -> float:
        elapsed = self._clock() - started
        if elapsed < 0.0:
            raise StaticRunnerError(StaticRunnerErrorCode.DOCKER_FAILED)
        remaining = self._limits.timeout_seconds - elapsed
        if remaining <= 0.0:
            raise StaticRunnerError(StaticRunnerErrorCode.TIMED_OUT)
        return remaining

    @staticmethod
    def _create_arguments(
        name: str,
        repository: Path,
        paths: tuple[str, ...],
        limits: SandboxLimits,
    ) -> tuple[str, ...]:
        mount = f"type=bind,source={os.fspath(repository)},target={_CONTAINER_INPUT},readonly"
        log_size = limits.stdout_bytes + limits.stderr_bytes
        arguments = [
            "create",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--user",
            _CONTAINER_USER,
            "--cpus",
            format(limits.cpu_count, ".15g"),
            "--memory",
            str(limits.memory_bytes),
            "--pids-limit",
            str(limits.pids_limit),
            "--log-driver",
            "local",
            "--log-opt",
            f"max-size={log_size}b",
            "--log-opt",
            "max-file=1",
            "--log-opt",
            "compress=false",
            "--mount",
            mount,
            "--mount",
            f"type=tmpfs,destination={_CONTAINER_WORKSPACE},tmpfs-mode=1777",
            "--mount",
            f"type=tmpfs,destination={_CONTAINER_TEMP},tmpfs-mode=1777",
        ]
        for value in _FIXED_ENVIRONMENT:
            arguments.extend(("--env", value))
        arguments.extend((STATIC_CHECK_IMAGE, *paths))
        return tuple(arguments)


def _parse_exit_code(payload: bytes) -> int:
    try:
        value = payload.decode("ascii").strip()
        if not value or not value.isdecimal():
            raise ValueError
        return int(value)
    except (UnicodeError, ValueError):
        raise StaticRunnerError(StaticRunnerErrorCode.DOCKER_FAILED) from None
