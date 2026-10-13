from __future__ import annotations

import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.sandbox.results import NonNegativeFiniteFloat, SandboxCompletion, SandboxLimits
from app.sandbox.runner import (
    SANDBOX_IMAGE,
    DockerCommandAdapter,
    DockerCommandError,
    DockerCommandTimeout,
)

VERIFICATION_CONTAINER_NAME_PREFIX = "tracefix-verification-"
HIDDEN_BUNDLE_PREFIX = ".tracefix-verification-"
_CONTAINER_INPUT = "/tracefix/input"
_CONTAINER_WORKSPACE = "/tracefix/workspace"
_CONTAINER_HIDDEN = "/tracefix/hidden"
_CONTAINER_TEMP = "/tmp"
_CONTAINER_USER = "10001:10001"
_FIXED_ENVIRONMENT = (
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=/tracefix/workspace/repository/src",
)
_DOCKER_INFO_ARGUMENTS = ("info", "--format", "{{json .}}")
_IDENTIFIER = re.compile(r"^[0-9a-f]{32}$")
_GENERATED_TEST = re.compile(r"^test_[0-9]{4}\.py$")
_WINDOWS_REPARSE_POINT = 0x400
_MAX_TEST_FILES = 1_000
_MAX_EXIT_CODE = 2**31 - 1
_HIDDEN_ENTRYPOINT = (
    "import os,shutil,sys;"
    "shutil.copytree('/tracefix/input','/tracefix/workspace/repository');"
    "os.chdir('/tracefix/workspace/repository');"
    "os.execv(sys.executable,(sys.executable,'-m','pytest',*sys.argv[1:]))"
)


class VerificationRunnerErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    DAEMON_SECURITY = "daemon_security"
    DOCKER_FAILED = "docker_failed"
    TIMED_OUT = "timed_out"
    MEMORY_LIMIT = "memory_limit"
    MALFORMED_RESULT = "malformed_result"
    CLEANUP_FAILED = "cleanup_failed"


class VerificationRunnerError(Exception):
    def __init__(self, code: VerificationRunnerErrorCode) -> None:
        super().__init__(f"verification_runner:{code.value}")
        self.code = code


class VerificationRunnerValidationError(VerificationRunnerError):
    def __init__(self) -> None:
        super().__init__(VerificationRunnerErrorCode.INVALID_INPUT)


StrictExitCode = Annotated[int, Field(strict=True, ge=0, le=_MAX_EXIT_CODE)]


class VerificationSandboxEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    completion: SandboxCompletion
    exit_code: StrictExitCode | None
    stdout_truncated: Annotated[bool, Field(strict=True)]
    stderr_truncated: Annotated[bool, Field(strict=True)]
    duration_seconds: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.completion is SandboxCompletion.NORMAL and self.exit_code is None:
            raise ValueError("normal completion requires an exit code")
        if self.completion is SandboxCompletion.TIMED_OUT and self.exit_code is not None:
            raise ValueError("timed-out completion cannot contain an exit code")
        return self


class HiddenTestBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    workspace_root: Path
    root: Path
    test_files: tuple[str, ...]

    @field_validator("test_files", mode="before")
    @classmethod
    def freeze_test_files(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if not self.workspace_root.is_absolute() or not self.root.is_absolute():
            raise ValueError("hidden bundle paths must be absolute")
        if not self.test_files or len(self.test_files) > _MAX_TEST_FILES:
            raise ValueError("hidden bundle must contain a bounded test set")
        if any(
            type(value) is not str or not value or len(value.encode("utf-8")) > 128
            for value in self.test_files
        ):
            raise ValueError("hidden bundle names must be bounded strings")
        return self


class VerificationSandboxBoundary(Protocol):
    def execute_targeted(
        self, repository: Path, test_paths: tuple[str, ...]
    ) -> VerificationSandboxEvidence: ...

    def execute_visible_suite(self, repository: Path) -> VerificationSandboxEvidence: ...

    def execute_hidden(
        self, repository: Path, bundle: HiddenTestBundle
    ) -> VerificationSandboxEvidence: ...


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


def _ordinary_directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise VerificationRunnerValidationError
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise VerificationRunnerValidationError from None
    if (
        path != resolved
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise VerificationRunnerValidationError
    return resolved


def _validated_repository(repository: Path) -> Path:
    resolved = _ordinary_directory(repository)
    if _is_protected_mount(resolved):
        raise VerificationRunnerValidationError
    identities: set[str] = set()
    try:
        for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
            directory_names.sort(key=lambda value: (value.casefold(), value))
            file_names.sort(key=lambda value: (value.casefold(), value))
            for name in (*directory_names, *file_names):
                path = Path(directory) / name
                relative = path.relative_to(resolved).as_posix()
                identity = relative.casefold()
                if identity in identities:
                    raise VerificationRunnerValidationError
                identities.add(identity)
                metadata = path.lstat()
                if (
                    path.is_symlink()
                    or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                    or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
                    or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
                ):
                    raise VerificationRunnerValidationError
    except VerificationRunnerValidationError:
        raise
    except OSError:
        raise VerificationRunnerValidationError from None
    return resolved


def _validated_targeted_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    if type(paths) is not tuple or not paths or len(paths) > _MAX_TEST_FILES:
        raise VerificationRunnerValidationError
    identities: set[str] = set()
    for value in paths:
        if type(value) is not str:
            raise VerificationRunnerValidationError
        windows = PureWindowsPath(value)
        posix = PurePosixPath(value)
        parts = value.split("/")
        if (
            not value
            or len(value.encode("utf-8")) > 1_024
            or "\\" in value
            or windows.drive
            or windows.root
            or posix.is_absolute()
            or posix.parts[:1] != ("tests",)
            or not posix.name.startswith("test_")
            or posix.suffix != ".py"
            or any(part in {"", ".", ".."} for part in parts)
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
            or value.startswith(("-", "@"))
        ):
            raise VerificationRunnerValidationError
        identity = value.casefold()
        if identity in identities:
            raise VerificationRunnerValidationError
        identities.add(identity)
    return paths


def _validated_bundle(bundle: HiddenTestBundle, repository: Path) -> HiddenTestBundle:
    if type(bundle) is not HiddenTestBundle:
        raise VerificationRunnerValidationError
    try:
        raw = bundle.model_dump(warnings="none")
        validated = HiddenTestBundle.model_validate(raw, strict=True)
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise VerificationRunnerValidationError from None
    workspace = _ordinary_directory(validated.workspace_root)
    root = _ordinary_directory(validated.root)
    expected_names = tuple(f"test_{index:04d}.py" for index in range(len(validated.test_files)))
    try:
        workspace_is_private = (
            os.name == "nt" or stat.S_IMODE(workspace.lstat().st_mode) & 0o077 == 0
        )
    except OSError:
        raise VerificationRunnerValidationError from None
    if (
        root.parent != workspace
        or not workspace_is_private
        or not root.name.startswith(HIDDEN_BUNDLE_PREFIX)
        or _IDENTIFIER.fullmatch(root.name.removeprefix(HIDDEN_BUNDLE_PREFIX)) is None
        or _overlaps(root, repository)
        or validated.test_files != expected_names
        or _is_protected_mount(root)
    ):
        raise VerificationRunnerValidationError
    try:
        entries = tuple(sorted(root.iterdir(), key=lambda path: (path.name.casefold(), path.name)))
        if tuple(path.name for path in entries) != validated.test_files:
            raise VerificationRunnerValidationError
        for path in entries:
            metadata = path.lstat()
            if (
                _GENERATED_TEST.fullmatch(path.name) is None
                or path.is_symlink()
                or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise VerificationRunnerValidationError
    except VerificationRunnerValidationError:
        raise
    except OSError:
        raise VerificationRunnerValidationError from None
    return validated


class VerificationSandboxRunner:
    """Execute fixed candidate-verification stages in fresh hardened containers."""

    def __init__(
        self,
        docker: DockerCommandAdapter,
        *,
        limits: SandboxLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
        identifier_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._docker = docker
        self._limits = limits or SandboxLimits()
        self._clock = clock
        self._identifier_factory = identifier_factory

    def execute_targeted(
        self, repository: Path, test_paths: tuple[str, ...]
    ) -> VerificationSandboxEvidence:
        return self._execute(repository, _validated_targeted_paths(test_paths), None)

    def execute_visible_suite(self, repository: Path) -> VerificationSandboxEvidence:
        return self._execute(repository, (), None)

    def execute_hidden(
        self, repository: Path, bundle: HiddenTestBundle
    ) -> VerificationSandboxEvidence:
        prepared = _validated_repository(repository)
        validated_bundle = _validated_bundle(bundle, prepared)
        paths = tuple(f"{_CONTAINER_HIDDEN}/{name}" for name in validated_bundle.test_files)
        return self._execute_prevalidated(prepared, paths, validated_bundle)

    def _execute(
        self,
        repository: Path,
        paths: tuple[str, ...],
        bundle: HiddenTestBundle | None,
    ) -> VerificationSandboxEvidence:
        prepared = _validated_repository(repository)
        return self._execute_prevalidated(prepared, paths, bundle)

    def _execute_prevalidated(
        self,
        repository: Path,
        paths: tuple[str, ...],
        bundle: HiddenTestBundle | None,
    ) -> VerificationSandboxEvidence:
        started = self._clock()
        self._validate_daemon_security()
        try:
            identifier = self._identifier_factory()
        except Exception:
            raise VerificationRunnerValidationError from None
        if type(identifier) is not str or _IDENTIFIER.fullmatch(identifier) is None:
            raise VerificationRunnerValidationError
        name = f"{VERIFICATION_CONTAINER_NAME_PREFIX}{identifier}"
        primary: VerificationRunnerError | None = None
        evidence: VerificationSandboxEvidence | None = None
        process_control = False
        created = False
        try:
            self._docker.run(self._create_arguments(name, repository, paths, bundle, self._limits))
            created = True
            timed_out = False
            try:
                attached = self._docker.run(
                    ("start", "--attach", name),
                    timeout_seconds=self._limits.timeout_seconds,
                    stdout_bytes=self._limits.stdout_bytes,
                    stderr_bytes=self._limits.stderr_bytes,
                    check=False,
                )
                waited = self._docker.run(("wait", name))
                exit_code: int | None = _parse_exit_code(waited.stdout)
            except DockerCommandTimeout as error:
                attached = error.result
                state = self._inspect(name)
                if state.Running:
                    timed_out = True
                    exit_code = None
                    self._docker.run(("kill", "--signal", "KILL", name))
                else:
                    waited = self._docker.run(("wait", name))
                    exit_code = _parse_exit_code(waited.stdout)
            state = self._inspect(name)
            duration = self._clock() - started
            if not math.isfinite(duration) or duration < 0.0:
                raise VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT)
            completion = (
                SandboxCompletion.TIMED_OUT
                if timed_out
                else (
                    SandboxCompletion.MEMORY_LIMIT if state.OOMKilled else SandboxCompletion.NORMAL
                )
            )
            evidence = VerificationSandboxEvidence(
                completion=completion,
                exit_code=exit_code,
                stdout_truncated=attached.stdout_truncated,
                stderr_truncated=attached.stderr_truncated,
                duration_seconds=duration,
            )
        except (KeyboardInterrupt, SystemExit):
            process_control = True
            raise
        except VerificationRunnerError as error:
            primary = error
        except (AttributeError, TypeError, ValidationError, ValueError):
            primary = VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT)
        except Exception:
            primary = VerificationRunnerError(VerificationRunnerErrorCode.DOCKER_FAILED)
        finally:
            if created:
                try:
                    self._docker.run(("rm", "--force", name))
                except Exception:
                    if not process_control:
                        raise VerificationRunnerError(
                            VerificationRunnerErrorCode.CLEANUP_FAILED
                        ) from primary
        if primary is not None:
            raise primary
        if evidence is None:
            raise VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT)
        return evidence

    def _validate_daemon_security(self) -> None:
        try:
            response = self._docker.run(_DOCKER_INFO_ARGUMENTS)
            info = _DockerInfo.model_validate(json.loads(response.stdout), strict=True)
        except Exception:
            raise VerificationRunnerError(VerificationRunnerErrorCode.DAEMON_SECURITY) from None
        normalized = tuple(option.casefold() for option in info.SecurityOptions)
        if info.OSType != "linux" or not any(
            option == "name=seccomp" or option.startswith("name=seccomp,") for option in normalized
        ):
            raise VerificationRunnerError(VerificationRunnerErrorCode.DAEMON_SECURITY)
        if any("seccomp=unconfined" in option for option in normalized):
            raise VerificationRunnerError(VerificationRunnerErrorCode.DAEMON_SECURITY)

    def _inspect(self, name: str) -> _DockerState:
        try:
            response = self._docker.run(("inspect", "--format", "{{json .State}}", name))
            return _DockerState.model_validate(json.loads(response.stdout), strict=True)
        except (DockerCommandError, OSError):
            raise VerificationRunnerError(VerificationRunnerErrorCode.DOCKER_FAILED) from None
        except (AttributeError, TypeError, UnicodeError, json.JSONDecodeError, ValidationError):
            raise VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT) from None

    @staticmethod
    def _create_arguments(
        name: str,
        repository: Path,
        paths: tuple[str, ...],
        bundle: HiddenTestBundle | None,
        limits: SandboxLimits,
    ) -> tuple[str, ...]:
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
            f"type=bind,source={os.fspath(repository)},target={_CONTAINER_INPUT},readonly",
            "--mount",
            f"type=tmpfs,destination={_CONTAINER_WORKSPACE},tmpfs-mode=1777",
            "--mount",
            f"type=tmpfs,destination={_CONTAINER_TEMP},tmpfs-mode=1777",
        ]
        if bundle is not None:
            arguments.extend(
                (
                    "--mount",
                    f"type=bind,source={os.fspath(bundle.root)},target={_CONTAINER_HIDDEN},readonly",
                    "--entrypoint",
                    "python",
                )
            )
        for value in _FIXED_ENVIRONMENT:
            arguments.extend(("--env", value))
        arguments.append(SANDBOX_IMAGE)
        if bundle is not None:
            arguments.extend(("-c", _HIDDEN_ENTRYPOINT))
        arguments.extend(paths)
        return tuple(arguments)


def _parse_exit_code(payload: bytes) -> int:
    try:
        value = payload.decode("ascii").strip()
        if not value or not value.isdecimal():
            raise ValueError
        result = int(value)
        if result > _MAX_EXIT_CODE:
            raise ValueError
        return result
    except (AttributeError, TypeError, UnicodeError, ValueError):
        raise VerificationRunnerError(VerificationRunnerErrorCode.MALFORMED_RESULT) from None
