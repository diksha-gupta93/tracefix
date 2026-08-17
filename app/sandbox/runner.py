from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

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
_INFRASTRUCTURE_OUTPUT_LIMIT = 65_536
_FORBIDDEN_PATH_CHARACTERS = frozenset("*?[]{};&|<>`$!()\"'")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


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


def _validated_repository(repository: Path) -> Path:
    if not repository.is_absolute():
        raise SandboxValidationError("prepared repository path must be absolute")
    try:
        before = repository.lstat()
        if repository.is_symlink():
            raise SandboxValidationError("prepared repository must not be a symlink")
        resolved, after = repository.resolve(strict=True), repository.lstat()
    except SandboxValidationError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise SandboxValidationError("prepared repository is missing or unstable") from error
    if not resolved.is_dir():
        raise SandboxValidationError("prepared repository must be a directory")
    if resolved != repository.resolve() or (before.st_dev, before.st_ino, before.st_mode) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
    ):
        raise SandboxValidationError("prepared repository changed during validation")
    return resolved


class _DockerState(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    OOMKilled: bool
    Running: bool


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
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
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
