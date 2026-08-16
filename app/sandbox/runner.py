from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from app.sandbox.results import SandboxResult

SANDBOX_IMAGE = "tracefix-sandbox:0.1.3a"
CONTAINER_NAME_PREFIX = "tracefix-sandbox-"
_CONTAINER_INPUT = "/tracefix/input"
_CONTAINER_WORKSPACE = "/tracefix/workspace"
_CONTAINER_TEMP = "/tmp"
_CONTAINER_USER = "10001:10001"
_FORBIDDEN_PATH_CHARACTERS = frozenset("*?[]{};&|<>`$!()\"'")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


class SandboxError(Exception):
    """Base class for sandbox boundary failures."""


class SandboxValidationError(SandboxError):
    """Untrusted runner input failed closed before Docker interaction."""


class DockerCommandError(SandboxError):
    """The Docker CLI could not complete an infrastructure operation."""


class SandboxExecutionError(SandboxError):
    """A container operation or result mapping failed."""


class SandboxCleanupError(SandboxError):
    """The generated container could not be removed."""

    def __init__(self, message: str, *, cleanup_cause: BaseException) -> None:
        super().__init__(message)
        self.cleanup_cause = cleanup_cause


@dataclass(frozen=True, slots=True)
class DockerCommandResult:
    """Private-process details returned by the typed Docker boundary."""

    exit_code: int
    stdout: bytes
    stderr: bytes


class DockerCommandAdapter(Protocol):
    def run(self, arguments: Sequence[str]) -> DockerCommandResult: ...


class SubprocessDockerCommandAdapter:
    """Invoke the installed Docker CLI without a host shell."""

    def run(self, arguments: Sequence[str]) -> DockerCommandResult:
        command = ("docker", *arguments)
        try:
            completed = subprocess.run(command, check=False, capture_output=True, shell=False)
        except (OSError, subprocess.SubprocessError) as error:
            raise DockerCommandError("failed to invoke the Docker CLI") from error
        result = DockerCommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if result.exit_code != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            message = "Docker CLI operation failed"
            if detail:
                message = f"{message}: {detail}"
            raise DockerCommandError(message)
        return result


@dataclass(frozen=True, slots=True, init=False)
class PytestCommand:
    """An immutable command restricted to the Task 0.1.3a pytest grammar."""

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
        or value.startswith("-")
        or value.startswith("@")
        or "\\" in value
        or any(character.isspace() for character in value)
        or _CONTROL_CHARACTER.search(value)
        or any(character in _FORBIDDEN_PATH_CHARACTERS for character in value)
        or "://" in value
    ):
        raise SandboxValidationError("invalid pytest test path")
    selections = value.split("::")
    file_part = selections[0]
    node_parts = selections[1:]
    raw_parts = file_part.split("/")
    windows_path = PureWindowsPath(file_part)
    path = PurePosixPath(file_part)
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
        resolved = repository.resolve(strict=True)
        after = repository.lstat()
    except SandboxValidationError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise SandboxValidationError("prepared repository is missing or unstable") from error
    if not resolved.is_dir():
        raise SandboxValidationError("prepared repository must be a directory")
    identity_before = (before.st_dev, before.st_ino, before.st_mode)
    identity_after = (after.st_dev, after.st_ino, after.st_mode)
    if resolved != repository.resolve() or identity_before != identity_after:
        raise SandboxValidationError("prepared repository changed during validation")
    return resolved


class SandboxRunner:
    """Run one approved pytest invocation in a fresh disposable container."""

    def __init__(
        self,
        docker: DockerCommandAdapter,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._docker = docker
        self._clock = clock

    def execute(self, repository: Path, command_tokens: Sequence[str]) -> SandboxResult:
        command = PytestCommand(command_tokens)
        prepared_repository = _validated_repository(repository)
        container_name = f"{CONTAINER_NAME_PREFIX}{uuid.uuid4().hex}"
        started = self._clock()
        execution_error: BaseException | None = None
        result: SandboxResult | None = None
        try:
            self._docker.run(self._create_arguments(container_name, prepared_repository, command))
            self._docker.run(("start", container_name))
            wait_result = self._docker.run(("wait", container_name))
            exit_code = _parse_exit_code(wait_result.stdout)
            logs = self._docker.run(("logs", container_name))
            duration = self._clock() - started
            if duration < 0.0:
                raise SandboxExecutionError("monotonic clock moved backwards")
            result = SandboxResult(
                exit_code=exit_code,
                stdout=logs.stdout.decode("utf-8", errors="replace"),
                stderr=logs.stderr.decode("utf-8", errors="replace"),
                duration_seconds=duration,
            )
        except (DockerCommandError, SandboxExecutionError, ValueError, OSError) as error:
            execution_error = error
        finally:
            try:
                self._docker.run(("rm", "--force", container_name))
            except Exception as cleanup_error:
                message = f"failed to remove generated container {container_name}"
                if execution_error is not None:
                    message = f"{message}; execution also failed"
                    raise SandboxCleanupError(
                        message, cleanup_cause=cleanup_error
                    ) from execution_error
                raise SandboxCleanupError(message, cleanup_cause=cleanup_error) from cleanup_error
        if execution_error is not None:
            raise SandboxExecutionError("sandbox execution failed") from execution_error
        if result is None:
            raise SandboxExecutionError("sandbox execution produced no result")
        return result

    @staticmethod
    def _create_arguments(
        container_name: str, repository: Path, command: PytestCommand
    ) -> tuple[str, ...]:
        mount = f"type=bind,source={os.fspath(repository)},target={_CONTAINER_INPUT},readonly"
        return (
            "create",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--user",
            _CONTAINER_USER,
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


def _parse_exit_code(payload: bytes) -> int:
    try:
        text = payload.decode("ascii").strip()
        if not text or not text.isdecimal():
            raise ValueError
        return int(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise SandboxExecutionError("Docker wait returned a malformed exit code") from error
