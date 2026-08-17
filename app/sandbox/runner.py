from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import IO, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.sandbox.results import (
    TRUNCATION_MARKER,
    SandboxCompletion,
    SandboxLimits,
    SandboxResult,
)

SANDBOX_IMAGE = "tracefix-sandbox:0.1.3a"
CONTAINER_NAME_PREFIX = "tracefix-sandbox-"
_CONTAINER_INPUT = "/tracefix/input"
_CONTAINER_WORKSPACE = "/tracefix/workspace"
_CONTAINER_TEMP = "/tmp"
_CONTAINER_USER = "10001:10001"
_FIXED_ENVIRONMENT = ("PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1")
_INFRASTRUCTURE_OUTPUT_LIMIT = 65_536
_FORBIDDEN_PATH_CHARACTERS = frozenset("*?[]{};&|<>`$!()\"'")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_REPARSE_POINT = 0x400
_DOCKER_INFO_ARGUMENTS = (
    "info",
    "--format",
    "{{json .}}",
)


class SandboxError(Exception):
    """Base class for sandbox boundary failures."""


class SandboxValidationError(SandboxError):
    """Untrusted runner input failed closed before Docker interaction."""


class DockerCommandError(SandboxError):
    """The Docker CLI could not complete an infrastructure operation."""


class DockerCommandTimeout(DockerCommandError):
    """The trusted host deadline expired while waiting for Docker."""

    def __init__(self, message: str, *, result: DockerCommandResult) -> None:
        super().__init__(message)
        self.result = result


class SandboxExecutionError(SandboxError):
    """A container operation or result mapping failed."""


class SandboxCleanupError(SandboxError):
    """The generated container could not be removed."""

    def __init__(self, message: str, *, cleanup_cause: BaseException) -> None:
        super().__init__(message)
        self.cleanup_cause = cleanup_cause


@dataclass(frozen=True, slots=True)
class DockerCommandResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class DockerCommandAdapter(Protocol):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = _INFRASTRUCTURE_OUTPUT_LIMIT,
        stderr_bytes: int = _INFRASTRUCTURE_OUTPUT_LIMIT,
        check: bool = True,
    ) -> DockerCommandResult: ...


class SubprocessDockerCommandAdapter:
    """Invoke Docker without a shell while retaining bounded output in memory."""

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = _INFRASTRUCTURE_OUTPUT_LIMIT,
        stderr_bytes: int = _INFRASTRUCTURE_OUTPUT_LIMIT,
        check: bool = True,
    ) -> DockerCommandResult:
        command = ("docker", *arguments)
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
            )
        except OSError as error:
            raise DockerCommandError("failed to invoke the Docker CLI") from error
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise DockerCommandError("Docker CLI output pipes were not created")
        stdout, stdout_truncated, stdout_thread = _bounded_pipe(process.stdout, stdout_bytes)
        stderr, stderr_truncated, stderr_thread = _bounded_pipe(process.stderr, stderr_bytes)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            result = DockerCommandResult(
                exit_code=process.returncode,
                stdout=bytes(stdout),
                stderr=bytes(stderr),
                stdout_truncated=stdout_truncated[0],
                stderr_truncated=stderr_truncated[0],
            )
            raise DockerCommandTimeout(
                "Docker CLI operation exceeded the host deadline", result=result
            ) from error
        except (OSError, subprocess.SubprocessError) as error:
            process.kill()
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            raise DockerCommandError("failed to invoke the Docker CLI") from error
        stdout_thread.join()
        stderr_thread.join()
        result = DockerCommandResult(
            exit_code=return_code,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            stdout_truncated=stdout_truncated[0],
            stderr_truncated=stderr_truncated[0],
        )
        if check and result.exit_code != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise DockerCommandError(
                f"Docker CLI operation failed{f': {detail}' if detail else ''}"
            )
        return result


def _bounded_pipe(pipe: IO[bytes], limit: int) -> tuple[bytearray, list[bool], threading.Thread]:
    retained = bytearray()
    truncated = [False]

    def drain() -> None:
        while chunk := pipe.read(65_536):
            remaining = limit - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return retained, truncated, thread


@dataclass(frozen=True, slots=True, init=False)
class PytestCommand:
    tokens: tuple[str, ...]

    def __init__(self, tokens: Sequence[str]) -> None:
        approved = tuple(tokens)
        if approved[:3] != ("python", "-m", "pytest") or len(approved) < 3:
            raise SandboxValidationError("command must begin exactly with: python -m pytest")
        for value in approved[3:]:
            _validate_test_path(value)
        object.__setattr__(self, "tokens", approved)

    @property
    def test_paths(self) -> tuple[str, ...]:
        return self.tokens[3:]


def _validate_test_path(value: str) -> None:
    if (
        not value
        or value.startswith(("-", "@"))
        or "\\" in value
        or any(c.isspace() for c in value)
        or _CONTROL_CHARACTER.search(value)
        or any(c in _FORBIDDEN_PATH_CHARACTERS for c in value)
        or "://" in value
    ):
        raise SandboxValidationError("invalid pytest test path")
    selections = value.split("::")
    file_part, node_parts = selections[0], selections[1:]
    raw_parts = file_part.split("/")
    windows_path, path = PureWindowsPath(file_part), PurePosixPath(file_part)
    if (
        windows_path.drive
        or windows_path.root
        or file_part.startswith("/")
        or path.parts[:1] != ("tests",)
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path.suffix != ".py"
        or any(not part or not part.isidentifier() for part in node_parts)
    ):
        raise SandboxValidationError("invalid pytest test path")


@dataclass(frozen=True, slots=True)
class _FilesystemIdentity:
    relative_path: PurePath
    device: int
    inode: int
    mode: int
    reparse_tag: int


def _identity(path: Path, relative_path: PurePath) -> _FilesystemIdentity:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SandboxValidationError(
            "prepared repository could not be completely inspected"
        ) from error
    attributes = getattr(metadata, "st_file_attributes", 0)
    if path.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
        raise SandboxValidationError("prepared repository contains a filesystem alias")
    if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
        raise SandboxValidationError("prepared repository contains an unsupported filesystem entry")
    return _FilesystemIdentity(
        relative_path=relative_path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        reparse_tag=getattr(metadata, "st_reparse_tag", 0),
    )


def _tree_snapshot(repository: Path) -> tuple[_FilesystemIdentity, ...]:
    identities: list[_FilesystemIdentity] = []

    def inspect(directory: Path, relative_directory: PurePath) -> None:
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: os.path.normcase(entry.name))
        except OSError as error:
            raise SandboxValidationError(
                "prepared repository could not be completely inspected"
            ) from error
        for entry in ordered:
            relative = relative_directory / entry.name
            entry_path = Path(entry.path)
            identity = _identity(entry_path, relative)
            identities.append(identity)
            if stat.S_ISDIR(identity.mode):
                inspect(entry_path, relative)

    inspect(repository, PurePath())
    return tuple(identities)


def _same_or_ancestor(candidate: Path, protected: Path) -> bool:
    try:
        protected.relative_to(candidate)
    except ValueError:
        return False
    return True


def _protected_locations() -> tuple[Path, ...]:
    home = Path.home().resolve(strict=False)
    project_root = Path(__file__).resolve().parents[2]
    return (
        project_root,
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


def _is_windows_docker_pipe(path: Path) -> bool:
    normalized = os.path.normcase(os.fspath(path)).replace("/", "\\")
    return normalized == r"\\.\pipe\docker_engine"


def _validated_repository(repository: Path) -> Path:
    if not repository.is_absolute():
        raise SandboxValidationError("prepared repository path must be absolute")
    if _is_windows_docker_pipe(repository):
        raise SandboxValidationError("prepared repository is a protected host location")
    try:
        before = _identity(repository, PurePath())
        resolved = repository.resolve(strict=True)
    except SandboxValidationError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise SandboxValidationError("prepared repository is missing or unstable") from error
    if not resolved.is_dir():
        raise SandboxValidationError("prepared repository must be a directory")
    if any(
        _same_or_ancestor(resolved, protected.resolve(strict=False))
        for protected in _protected_locations()
    ):
        raise SandboxValidationError("prepared repository is a protected host location")
    first_snapshot = _tree_snapshot(resolved)
    second_snapshot = _tree_snapshot(resolved)
    try:
        after = repository.lstat()
    except OSError as error:
        raise SandboxValidationError("prepared repository changed during validation") from error
    if (
        resolved != repository.resolve()
        or (before.device, before.inode, before.mode)
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
        )
        or first_snapshot != second_snapshot
    ):
        raise SandboxValidationError("prepared repository changed during validation")
    return resolved


class _DockerState(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
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


class SandboxRunner:
    def __init__(
        self, docker: DockerCommandAdapter, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._docker, self._clock = docker, clock

    def execute(
        self, repository: Path, command_tokens: Sequence[str], limits: SandboxLimits | None = None
    ) -> SandboxResult:
        effective = limits if limits is not None else SandboxLimits()
        command, prepared = PytestCommand(command_tokens), _validated_repository(repository)
        self._validate_daemon_security()
        name = f"{CONTAINER_NAME_PREFIX}{uuid.uuid4().hex}"
        started = self._clock()
        execution_error: BaseException | None = None
        result: SandboxResult | None = None
        try:
            self._docker.run(self._create_arguments(name, prepared, command, effective))
            timed_out = False
            try:
                attached = self._docker.run(
                    ("start", "--attach", name),
                    timeout_seconds=effective.timeout_seconds,
                    stdout_bytes=effective.stdout_bytes,
                    stderr_bytes=effective.stderr_bytes,
                    check=False,
                )
                captured = attached
                waited = self._docker.run(("wait", name))
                exit_code: int | None = _parse_exit_code(waited.stdout)
            except DockerCommandTimeout as timeout:
                captured = timeout.result
                raced_state = self._inspect(name)
                if raced_state.Running:
                    timed_out = True
                    exit_code = None
                    self._docker.run(("kill", "--signal", "KILL", name))
                else:
                    waited = self._docker.run(("wait", name))
                    exit_code = _parse_exit_code(waited.stdout)
            state = self._inspect(name)
            duration = self._clock() - started
            if duration < 0.0:
                raise SandboxExecutionError("monotonic clock moved backwards")
            completion = (
                SandboxCompletion.TIMED_OUT
                if timed_out
                else (
                    SandboxCompletion.MEMORY_LIMIT if state.OOMKilled else SandboxCompletion.NORMAL
                )
            )
            stdout, stdout_cut = _bounded_text(
                captured.stdout, effective.stdout_bytes, captured.stdout_truncated
            )
            stderr, stderr_cut = _bounded_text(
                captured.stderr, effective.stderr_bytes, captured.stderr_truncated
            )
            result = SandboxResult(
                exit_code=exit_code,
                completion=completion,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_cut,
                stderr_truncated=stderr_cut,
                duration_seconds=duration,
            )
        except (
            DockerCommandError,
            SandboxExecutionError,
            ValidationError,
            ValueError,
            OSError,
        ) as error:
            execution_error = error
        finally:
            try:
                self._docker.run(("rm", "--force", name))
            except Exception as cleanup_error:
                message = f"failed to remove generated container {name}"
                if execution_error is not None:
                    raise SandboxCleanupError(
                        f"{message}; execution also failed", cleanup_cause=cleanup_error
                    ) from execution_error
                raise SandboxCleanupError(message, cleanup_cause=cleanup_error) from cleanup_error
        if execution_error is not None:
            raise SandboxExecutionError("sandbox execution failed") from execution_error
        if result is None:
            raise SandboxExecutionError("sandbox execution produced no result")
        return result

    def _inspect(self, name: str) -> _DockerState:
        inspected = self._docker.run(("inspect", "--format", "{{json .State}}", name))
        try:
            return _DockerState.model_validate(json.loads(inspected.stdout))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as error:
            raise SandboxExecutionError("Docker inspect returned malformed state") from error

    def _validate_daemon_security(self) -> None:
        try:
            response = self._docker.run(_DOCKER_INFO_ARGUMENTS)
            info = _DockerInfo.model_validate(json.loads(response.stdout))
        except (
            DockerCommandError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValidationError,
        ) as error:
            raise SandboxValidationError(
                "Docker daemon security posture could not be validated"
            ) from error
        if info.OSType != "linux":
            raise SandboxValidationError("Docker daemon must serve Linux containers")
        normalized = tuple(option.casefold() for option in info.SecurityOptions)
        if not any(
            option == "name=seccomp" or option.startswith("name=seccomp,") for option in normalized
        ):
            raise SandboxValidationError("Docker daemon must report seccomp support")
        if any("seccomp=unconfined" in option for option in normalized):
            raise SandboxValidationError("unconfined seccomp is prohibited")

    @staticmethod
    def _create_arguments(
        name: str, repository: Path, command: PytestCommand, limits: SandboxLimits
    ) -> tuple[str, ...]:
        mount = f"type=bind,source={os.fspath(repository)},target={_CONTAINER_INPUT},readonly"
        log_size = limits.stdout_bytes + limits.stderr_bytes
        return (
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
            "--env",
            _FIXED_ENVIRONMENT[0],
            "--env",
            _FIXED_ENVIRONMENT[1],
            SANDBOX_IMAGE,
            *command.test_paths,
        )


def _bounded_text(payload: bytes, limit: int, already_truncated: bool = False) -> tuple[str, bool]:
    truncated = already_truncated or len(payload) > limit
    if not truncated:
        return payload.decode("utf-8", errors="replace"), False
    budget = limit - len(TRUNCATION_MARKER)
    prefix = payload[:budget]
    while prefix:
        try:
            text = prefix.decode("utf-8")
            break
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data" and error.end == len(prefix):
                prefix = prefix[: error.start]
            else:
                text = prefix.decode("utf-8", errors="replace")
                while len(text.encode("utf-8")) > budget:
                    text = text[:-1]
                break
    else:
        text = ""
    return text + TRUNCATION_MARKER.decode("ascii"), True


def _parse_exit_code(payload: bytes) -> int:
    try:
        text = payload.decode("ascii").strip()
        if not text or not text.isdecimal():
            raise ValueError
        return int(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise SandboxExecutionError("Docker wait returned a malformed exit code") from error
