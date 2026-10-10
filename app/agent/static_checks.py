from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.agent.patches import (
    DiffLineKind,
    ParsedDiffFile,
    ParsedUnifiedDiff,
    UnifiedDiffError,
    normalize_repository_paths,
    parse_unified_diff,
)
from app.agent.policy import PatchPolicyResult
from app.agent.schemas import PatchProposal, RepositoryFingerprint, RepositoryPreparation
from app.sandbox.static_runner import (
    StaticContainerResult,
    StaticContainerRunner,
    StaticRunnerError,
    StaticRunnerErrorCode,
    StaticToolCheck,
    StaticToolResult,
    StaticToolStatus,
)

_WINDOWS_REPARSE_POINT = 0x400
_FINGERPRINT_HEADER = b"tracefix-repository-fingerprint\x00v1\x00"
_MAX_TREE_ENTRIES = 10_000
_MAX_FILE_BYTES = 67_108_864
_MAX_TREE_BYTES = 268_435_456
_IDENTIFIER = re.compile(r"^[0-9a-f]{32}$")
_RULE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class StaticCheck(StrEnum):
    PATCH_APPLY = "patch_apply"
    CHANGED_FILES = "changed_files"
    PYTHON_COMPILE = "python_compile"
    AST_PARSE = "ast_parse"
    RUFF = "ruff"
    MYPY = "mypy"
    BANDIT = "bandit"


class StaticCheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_RUN = "not_run"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class StaticValidationOutcome(StrEnum):
    PASSED = "passed"
    CANDIDATE_FAILED = "candidate_failed"
    INFRASTRUCTURE_DIAGNOSTIC = "infrastructure_diagnostic"


class StaticDiagnosticCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    PREPARATION_INVALID = "preparation_invalid"
    WORKSPACE_INVALID = "workspace_invalid"
    FILESYSTEM_FAILED = "filesystem_failed"
    STATIC_RUNNER_FAILED = "static_runner_failed"
    INTEGRITY_FAILED = "integrity_failed"
    CLEANUP_FAILED = "cleanup_failed"
    PREREQUISITE_FAILED = "prerequisite_failed"


_DIAGNOSTIC_MESSAGES: dict[StaticDiagnosticCode, str] = {
    StaticDiagnosticCode.INVALID_INPUT: "static validation input is invalid",
    StaticDiagnosticCode.PREPARATION_INVALID: "prepared repository validation failed",
    StaticDiagnosticCode.WORKSPACE_INVALID: "candidate workspace validation failed",
    StaticDiagnosticCode.FILESYSTEM_FAILED: "candidate filesystem operation failed",
    StaticDiagnosticCode.STATIC_RUNNER_FAILED: "static container execution failed",
    StaticDiagnosticCode.INTEGRITY_FAILED: "repository integrity verification failed",
    StaticDiagnosticCode.CLEANUP_FAILED: "candidate cleanup failed",
    StaticDiagnosticCode.PREREQUISITE_FAILED: "a required earlier static check did not pass",
}

_FINDING_MESSAGES: dict[StaticCheck, str] = {
    StaticCheck.PATCH_APPLY: "exact patch application failed",
    StaticCheck.CHANGED_FILES: "changed-file validation failed",
    StaticCheck.PYTHON_COMPILE: "python compilation failed",
    StaticCheck.AST_PARSE: "python AST parsing failed",
    StaticCheck.RUFF: "ruff reported a finding",
    StaticCheck.MYPY: "mypy reported a finding",
    StaticCheck.BANDIT: "bandit reported a finding",
}


def _validate_relative_path(value: str) -> None:
    try:
        normalized = normalize_repository_paths((value,))
    except UnifiedDiffError:
        raise ValueError("path is not a normalized relative repository path") from None
    if normalized != (value,) or len(value.encode("utf-8")) > 1_024:
        raise ValueError("path is not a normalized relative repository path")


class StaticDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: StaticDiagnosticCode
    message: Annotated[str, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_safe_message(self) -> Self:
        if self.message != _DIAGNOSTIC_MESSAGES[self.code]:
            raise ValueError("diagnostic message must be the safe constant for its code")
        return self


class StaticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    check: StaticCheck
    path: Annotated[str, Field(min_length=1, max_length=1_024)] | None = None
    line: Annotated[int, Field(strict=True, ge=1, le=10_000_000)] | None = None
    column: Annotated[int, Field(strict=True, ge=1, le=10_000_000)] | None = None
    rule_code: Annotated[str, Field(min_length=1, max_length=64)]
    message: Annotated[str, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_safe_finding(self) -> Self:
        if self.path is not None:
            _validate_relative_path(self.path)
        if self.line is not None and self.path is None:
            raise ValueError("finding line requires a path")
        if self.column is not None and self.line is None:
            raise ValueError("finding column requires a line")
        if _RULE_CODE.fullmatch(self.rule_code) is None:
            raise ValueError("finding rule code is invalid")
        if self.message != _FINDING_MESSAGES[self.check]:
            raise ValueError("finding message must be the safe constant for its check")
        return self


def _finding_key(finding: StaticFinding) -> tuple[str, str, int, int, str]:
    path = finding.path or ""
    return (
        path.casefold(),
        path,
        finding.line or 0,
        finding.column or 0,
        finding.rule_code,
    )


class StaticCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    check: StaticCheck
    status: StaticCheckStatus
    findings: tuple[StaticFinding, ...] = ()
    diagnostic: StaticDiagnostic | None = None

    @field_validator("findings", mode="before")
    @classmethod
    def freeze_findings(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if len(self.findings) > 1_000:
            raise ValueError("too many findings")
        if any(finding.check is not self.check for finding in self.findings):
            raise ValueError("finding check must match its result")
        if self.findings != tuple(sorted(self.findings, key=_finding_key)):
            raise ValueError("findings must be deterministically ordered")
        if len(set(self.findings)) != len(self.findings):
            raise ValueError("findings must not contain duplicates")
        if self.status is StaticCheckStatus.FAILED:
            if not self.findings or self.diagnostic is not None:
                raise ValueError("failed check requires findings only")
        elif self.status in {StaticCheckStatus.NOT_RUN, StaticCheckStatus.INFRASTRUCTURE_ERROR}:
            if self.findings or self.diagnostic is None:
                raise ValueError("non-executed check requires a diagnostic only")
        elif self.findings or self.diagnostic is not None:
            raise ValueError("passing or inapplicable check cannot contain evidence")
        return self


class ValidatedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    repository: Path
    fingerprint: RepositoryFingerprint

    @model_validator(mode="after")
    def validate_repository(self) -> Self:
        if not self.repository.is_absolute():
            raise ValueError("candidate repository path must be absolute")
        return self


class StaticValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    preparation: RepositoryPreparation
    proposal: PatchProposal
    policy_result: PatchPolicyResult
    candidate_workspace_root: Path


class StaticValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: StaticValidationOutcome
    checks: tuple[StaticCheckResult, ...]
    candidate: ValidatedCandidate | None = None
    diagnostic: StaticDiagnostic | None = None

    @field_validator("checks", mode="before")
    @classmethod
    def freeze_checks(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if tuple(item.check for item in self.checks) != tuple(StaticCheck):
            raise ValueError("static checks must contain the fixed complete order")
        statuses = tuple(item.status for item in self.checks)
        if any(status is StaticCheckStatus.NOT_APPLICABLE for status in statuses[:2]):
            raise ValueError("patch and changed-file checks are always applicable")
        language_statuses = set(statuses[2:])
        if StaticCheckStatus.NOT_APPLICABLE in language_statuses and language_statuses != {
            StaticCheckStatus.NOT_APPLICABLE
        }:
            raise ValueError("language checks must share not-applicable status")
        if self.outcome is StaticValidationOutcome.PASSED:
            if (
                self.candidate is None
                or self.diagnostic is not None
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
                raise ValueError("passing validation requires a candidate and no blocking check")
        elif self.outcome is StaticValidationOutcome.CANDIDATE_FAILED:
            if (
                self.candidate is not None
                or self.diagnostic is not None
                or StaticCheckStatus.FAILED not in statuses
                or StaticCheckStatus.INFRASTRUCTURE_ERROR in statuses
            ):
                raise ValueError("candidate failure requires only deterministic failed checks")
        elif (
            self.candidate is not None
            or self.diagnostic is None
            or StaticCheckStatus.INFRASTRUCTURE_ERROR not in statuses
        ):
            raise ValueError("infrastructure outcome requires a diagnostic and failed boundary")
        return self


class _EntryKind(StrEnum):
    DIRECTORY = "directory"
    FILE = "file"


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    path: str
    kind: _EntryKind
    data: bytes | None
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    reparse_tag: int
    link_count: int


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    entries: tuple[_TreeEntry, ...]

    @property
    def by_path(self) -> dict[str, _TreeEntry]:
        return {entry.path: entry for entry in self.entries}


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int
    reparse_tag: int


class StaticFilesystem(Protocol):
    def snapshot(
        self, root: Path, expected_root: _DirectoryIdentity | None = None
    ) -> _TreeSnapshot: ...

    def make_directory(self, path: Path) -> _DirectoryIdentity: ...

    def copy_snapshot(
        self, snapshot: _TreeSnapshot, destination: Path, expected_root: _DirectoryIdentity
    ) -> None: ...

    def write_file(
        self,
        root: Path,
        expected_root: _DirectoryIdentity,
        relative_path: str,
        data: bytes,
        *,
        create: bool,
    ) -> None: ...

    def delete_file(
        self, root: Path, expected_root: _DirectoryIdentity, relative_path: str
    ) -> None: ...

    def publish(
        self, staging: Path, destination: Path, expected_root: _DirectoryIdentity
    ) -> _DirectoryIdentity: ...

    def remove_tree(self, path: Path, expected_root: _DirectoryIdentity) -> None: ...


class RepositoryFingerprinter(Protocol):
    def fingerprint(self, repository: Path) -> RepositoryFingerprint: ...


def _entry_from_stat(path: str, metadata: os.stat_result, data: bytes | None) -> _TreeEntry:
    return _TreeEntry(
        path=path,
        kind=_EntryKind.DIRECTORY if stat.S_ISDIR(metadata.st_mode) else _EntryKind.FILE,
        data=data,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        reparse_tag=getattr(metadata, "st_reparse_tag", 0),
        link_count=metadata.st_nlink,
    )


def _directory_identity(metadata: os.stat_result) -> _DirectoryIdentity:
    return _DirectoryIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        reparse_tag=getattr(metadata, "st_reparse_tag", 0),
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and getattr(left, "st_reparse_tag", 0) == getattr(right, "st_reparse_tag", 0)
    )


class LocalStaticFilesystem:
    """Filesystem boundary for inert candidate byte operations."""

    @staticmethod
    def _assert_root(root: Path, expected: _DirectoryIdentity | None = None) -> _DirectoryIdentity:
        metadata = root.lstat()
        identity = _directory_identity(metadata)
        if (
            root.is_symlink()
            or identity.reparse_tag
            or not stat.S_ISDIR(metadata.st_mode)
            or (expected is not None and identity != expected)
        ):
            raise OSError
        return identity

    @staticmethod
    def _directory_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def _open_root(self, root: Path, expected: _DirectoryIdentity) -> int:
        descriptor = os.open(root, self._directory_flags())
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or _directory_identity(opened) != expected:
            os.close(descriptor)
            raise OSError
        return descriptor

    def _open_parent(
        self,
        root: Path,
        expected_root: _DirectoryIdentity,
        relative_path: str,
        *,
        create_directories: bool,
    ) -> tuple[int, str]:
        parts = PurePosixPath(relative_path).parts
        descriptor = self._open_root(root, expected_root)
        try:
            for part in parts[:-1]:
                try:
                    next_descriptor = os.open(part, self._directory_flags(), dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_directories:
                        raise
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    next_descriptor = os.open(part, self._directory_flags(), dir_fd=descriptor)
                opened = os.fstat(next_descriptor)
                if not stat.S_ISDIR(opened.st_mode) or getattr(opened, "st_reparse_tag", False):
                    os.close(next_descriptor)
                    raise OSError
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def snapshot(
        self, root: Path, expected_root: _DirectoryIdentity | None = None
    ) -> _TreeSnapshot:
        try:
            root_identity = self._assert_root(root, expected_root)
            entries: list[_TreeEntry] = []
            identities: set[str] = set()
            total_bytes = 0

            def record(relative: str) -> None:
                identity = relative.casefold()
                if identity in identities or len(entries) >= _MAX_TREE_ENTRIES:
                    raise OSError
                identities.add(identity)

            def walk_descriptor(directory: int, prefix: PurePosixPath) -> None:
                nonlocal total_bytes
                with os.scandir(directory) as scanned:
                    children = sorted(scanned, key=lambda item: item.name)
                for child in children:
                    relative = (prefix / child.name).as_posix()
                    _validate_relative_path(relative)
                    record(relative)
                    metadata = child.stat(follow_symlinks=False)
                    attributes = getattr(metadata, "st_file_attributes", 0)
                    if child.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
                        raise OSError
                    if stat.S_ISDIR(metadata.st_mode):
                        child_descriptor = os.open(
                            child.name, self._directory_flags(), dir_fd=directory
                        )
                        opened = os.fstat(child_descriptor)
                        if not _same_identity(opened, metadata):
                            os.close(child_descriptor)
                            raise OSError
                        entries.append(_entry_from_stat(relative, metadata, None))
                        try:
                            walk_descriptor(child_descriptor, PurePosixPath(relative))
                        finally:
                            os.close(child_descriptor)
                    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                        if metadata.st_size > _MAX_FILE_BYTES:
                            raise OSError
                        flags = (
                            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                        )
                        descriptor = os.open(child.name, flags, dir_fd=directory)
                        try:
                            opened = os.fstat(descriptor)
                            if (
                                opened.st_dev != metadata.st_dev
                                or opened.st_ino != metadata.st_ino
                                or opened.st_size != metadata.st_size
                                or opened.st_mtime_ns != metadata.st_mtime_ns
                                or not stat.S_ISREG(opened.st_mode)
                                or opened.st_nlink != 1
                            ):
                                raise OSError
                            with os.fdopen(descriptor, "rb") as stream:
                                descriptor = -1
                                data = stream.read(_MAX_FILE_BYTES + 1)
                            if len(data) > _MAX_FILE_BYTES:
                                raise OSError
                            total_bytes += len(data)
                            if total_bytes > _MAX_TREE_BYTES:
                                raise OSError
                            after = os.stat(child.name, dir_fd=directory, follow_symlinks=False)
                            if not _same_identity(after, metadata):
                                raise OSError
                            entries.append(_entry_from_stat(relative, metadata, data))
                        finally:
                            if descriptor >= 0:
                                os.close(descriptor)
                    else:
                        raise OSError

            def walk_windows(directory: Path, prefix: PurePosixPath) -> None:
                nonlocal total_bytes
                before = directory.lstat()
                if (
                    directory.is_symlink()
                    or getattr(before, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
                ):
                    raise OSError
                with os.scandir(directory) as scanned:
                    children = sorted(scanned, key=lambda item: item.name)
                for child in children:
                    relative = (prefix / child.name).as_posix()
                    _validate_relative_path(relative)
                    record(relative)
                    path = Path(child.path)
                    metadata = path.lstat()
                    attributes = getattr(metadata, "st_file_attributes", 0)
                    if path.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
                        raise OSError
                    if stat.S_ISDIR(metadata.st_mode):
                        entries.append(_entry_from_stat(relative, metadata, None))
                        walk_windows(path, PurePosixPath(relative))
                    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                        if metadata.st_size > _MAX_FILE_BYTES:
                            raise OSError
                        descriptor = os.open(
                            path,
                            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        )
                        try:
                            opened = os.fstat(descriptor)
                            if not _same_identity(opened, metadata) or opened.st_nlink != 1:
                                raise OSError
                            with os.fdopen(descriptor, "rb") as stream:
                                descriptor = -1
                                data = stream.read(_MAX_FILE_BYTES + 1)
                            total_bytes += len(data)
                            if len(data) > _MAX_FILE_BYTES or total_bytes > _MAX_TREE_BYTES:
                                raise OSError
                            if not _same_identity(path.lstat(), metadata):
                                raise OSError
                            entries.append(_entry_from_stat(relative, metadata, data))
                        finally:
                            if descriptor >= 0:
                                os.close(descriptor)
                    else:
                        raise OSError
                if not _same_identity(directory.lstat(), before):
                    raise OSError

            if os.name == "nt":
                walk_windows(root, PurePosixPath())
            else:
                root_descriptor = self._open_root(root, root_identity)
                try:
                    walk_descriptor(root_descriptor, PurePosixPath())
                    if _directory_identity(os.fstat(root_descriptor)) != root_identity:
                        raise OSError
                finally:
                    os.close(root_descriptor)
            self._assert_root(root, root_identity)
            return _TreeSnapshot(tuple(entries))
        except (OSError, RuntimeError, UnicodeError, ValueError):
            raise StaticFilesystemError from None

    def make_directory(self, path: Path) -> _DirectoryIdentity:
        try:
            parent_identity = self._assert_root(path.parent)
            if os.name == "nt":
                path.mkdir()
                identity = self._assert_root(path)
            else:
                parent = self._open_root(path.parent, parent_identity)
                try:
                    os.mkdir(path.name, 0o700, dir_fd=parent)
                    descriptor = os.open(path.name, self._directory_flags(), dir_fd=parent)
                    try:
                        identity = _directory_identity(os.fstat(descriptor))
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(parent)
                self._assert_root(path, identity)
            return identity
        except OSError:
            raise StaticFilesystemError from None

    def copy_snapshot(
        self, snapshot: _TreeSnapshot, destination: Path, expected_root: _DirectoryIdentity
    ) -> None:
        try:
            for entry in snapshot.entries:
                if os.name == "nt":
                    target = destination.joinpath(*PurePosixPath(entry.path).parts)
                    target.resolve(strict=False).relative_to(destination.resolve(strict=True))
                    if entry.kind is _EntryKind.DIRECTORY:
                        target.mkdir()
                    else:
                        if entry.data is None:
                            raise OSError
                        descriptor = os.open(
                            target,
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_BINARY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                        )
                        with os.fdopen(descriptor, "wb") as stream:
                            stream.write(entry.data)
                else:
                    parent, name = self._open_parent(
                        destination, expected_root, entry.path, create_directories=False
                    )
                    try:
                        if entry.kind is _EntryKind.DIRECTORY:
                            os.mkdir(name, 0o700, dir_fd=parent)
                        else:
                            if entry.data is None:
                                raise OSError
                            descriptor = os.open(
                                name,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                                0o600,
                                dir_fd=parent,
                            )
                            with os.fdopen(descriptor, "wb") as stream:
                                stream.write(entry.data)
                    finally:
                        os.close(parent)
                self._assert_root(destination, expected_root)
        except (OSError, RuntimeError, ValueError):
            raise StaticFilesystemError from None

    def write_file(
        self,
        root: Path,
        expected_root: _DirectoryIdentity,
        relative_path: str,
        data: bytes,
        *,
        create: bool,
    ) -> None:
        try:
            _validate_relative_path(relative_path)
            if os.name == "nt":
                target = root.joinpath(*PurePosixPath(relative_path).parts)
                target.resolve(strict=False).relative_to(root.resolve(strict=True))
                parent_path = root
                for part in PurePosixPath(relative_path).parts[:-1]:
                    parent_path /= part
                    if not parent_path.exists():
                        parent_path.mkdir()
                    self._assert_root(parent_path)
                flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                flags |= os.O_CREAT | os.O_EXCL if create else 0
                descriptor = os.open(target, flags, 0o600)
            else:
                parent, name = self._open_parent(
                    root, expected_root, relative_path, create_directories=True
                )
                flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
                flags |= os.O_CREAT | os.O_EXCL if create else 0
                try:
                    descriptor = os.open(name, flags, 0o600, dir_fd=parent)
                finally:
                    os.close(parent)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or getattr(opened, "st_reparse_tag", False)
                ):
                    raise OSError
                if not create:
                    os.ftruncate(descriptor, 0)
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(data)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            self._assert_root(root, expected_root)
        except (OSError, RuntimeError, ValueError):
            raise StaticFilesystemError from None

    def delete_file(
        self, root: Path, expected_root: _DirectoryIdentity, relative_path: str
    ) -> None:
        try:
            _validate_relative_path(relative_path)
            if os.name == "nt":
                target = root.joinpath(*PurePosixPath(relative_path).parts)
                target.resolve(strict=True).relative_to(root.resolve(strict=True))
                metadata = target.lstat()
                if (
                    target.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise OSError
                target.unlink()
            else:
                parent, name = self._open_parent(
                    root, expected_root, relative_path, create_directories=False
                )
                try:
                    metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise OSError
                    os.unlink(name, dir_fd=parent)
                finally:
                    os.close(parent)
            self._assert_root(root, expected_root)
        except (OSError, RuntimeError, ValueError):
            raise StaticFilesystemError from None

    def publish(
        self, staging: Path, destination: Path, expected_root: _DirectoryIdentity
    ) -> _DirectoryIdentity:
        try:
            self._assert_root(staging, expected_root)
            if staging.parent != destination.parent:
                raise OSError
            if destination.exists() or destination.is_symlink():
                raise OSError
            if os.name == "nt":
                staging.rename(destination)
            else:
                parent_identity = self._assert_root(staging.parent)
                parent = self._open_root(staging.parent, parent_identity)
                try:
                    os.rename(
                        staging.name,
                        destination.name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                    )
                finally:
                    os.close(parent)
            return expected_root
        except OSError:
            raise StaticFilesystemError from None

    def remove_tree(self, path: Path, expected_root: _DirectoryIdentity) -> None:
        try:
            self._assert_root(path, expected_root)
            if os.name == "nt":
                shutil.rmtree(path)
                return
            descriptor = self._open_root(path, expected_root)
            try:
                self._remove_descriptor_tree(descriptor)
            finally:
                os.close(descriptor)
            self._assert_root(path, expected_root)
            path.rmdir()
        except (OSError, RuntimeError):
            raise StaticFilesystemError from None

    def _remove_descriptor_tree(self, directory: int) -> None:
        with os.scandir(directory) as scanned:
            children = sorted(scanned, key=lambda item: item.name)
        for child in children:
            metadata = child.stat(follow_symlinks=False)
            if child.is_symlink() or getattr(metadata, "st_reparse_tag", False):
                raise OSError
            if stat.S_ISDIR(metadata.st_mode):
                descriptor = os.open(child.name, self._directory_flags(), dir_fd=directory)
                try:
                    opened = os.fstat(descriptor)
                    if not _same_identity(opened, metadata):
                        raise OSError
                    self._remove_descriptor_tree(descriptor)
                finally:
                    os.close(descriptor)
                after = os.stat(child.name, dir_fd=directory, follow_symlinks=False)
                if _directory_identity(after) != _directory_identity(metadata):
                    raise OSError
                os.rmdir(child.name, dir_fd=directory)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                after = os.stat(child.name, dir_fd=directory, follow_symlinks=False)
                if not _same_identity(after, metadata) or after.st_nlink != 1:
                    raise OSError
                os.unlink(child.name, dir_fd=directory)
            else:
                raise OSError


class StaticFilesystemError(Exception):
    """The host filesystem could not prove a safe inert-byte operation."""


class Sha256RepositoryFingerprinter:
    def __init__(self, filesystem: StaticFilesystem) -> None:
        self._filesystem = filesystem

    def fingerprint(self, repository: Path) -> RepositoryFingerprint:
        snapshot = self._filesystem.snapshot(repository)
        return _fingerprint_snapshot(snapshot)


def _fingerprint_snapshot(snapshot: _TreeSnapshot) -> RepositoryFingerprint:
    digest = hashlib.sha256(_FINGERPRINT_HEADER)
    for entry in snapshot.entries:
        path_bytes = entry.path.encode("utf-8")
        digest.update(b"D" if entry.kind is _EntryKind.DIRECTORY else b"F")
        digest.update(_length(len(path_bytes)))
        digest.update(path_bytes)
        if entry.kind is _EntryKind.FILE:
            if entry.data is None:
                raise StaticFilesystemError
            digest.update(_length(len(entry.data)))
            digest.update(entry.data)
    return RepositoryFingerprint(digest=digest.hexdigest())


def _length(value: int) -> bytes:
    if value < 0 or value >= 2**64:
        raise StaticFilesystemError
    return value.to_bytes(8, "big")


class _PatchApplicationError(Exception):
    pass


class _ChangedFilesError(Exception):
    pass


def _decode_source(data: bytes) -> tuple[list[str], bytes, bool]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _PatchApplicationError from None
    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise _PatchApplicationError
    has_crlf = "\r\n" in text
    if has_crlf and "\n" in without_crlf:
        raise _PatchApplicationError
    newline = b"\r\n" if has_crlf else b"\n"
    normalized = text.replace("\r\n", "\n")
    final_newline = normalized.endswith("\n")
    lines = normalized.split("\n")
    if final_newline:
        lines.pop()
    return lines, newline, final_newline


def _apply_file_patch(file: ParsedDiffFile, source_data: bytes | None) -> bytes | None:
    if file.is_added:
        if source_data is not None:
            raise _PatchApplicationError
        source_lines: list[str] = []
        newline = b"\n"
        final_newline = not file.new_no_final_newline
    else:
        if source_data is None:
            raise _PatchApplicationError
        source_lines, newline, final_newline = _decode_source(source_data)
        old_touches_eof = any(line.old_line == len(source_lines) for line in file.lines)
        if file.old_no_final_newline and final_newline:
            raise _PatchApplicationError
        if file.old_no_final_newline and file.old_no_final_newline_line != len(source_lines):
            raise _PatchApplicationError
        if old_touches_eof and not final_newline and not file.old_no_final_newline:
            raise _PatchApplicationError
        if file.new_no_final_newline and final_newline:
            raise _PatchApplicationError
    output: list[str] = []
    source_cursor = 0
    for hunk in file.hunks:
        old_index = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        new_index = hunk.new_start - 1 if hunk.new_count else hunk.new_start
        if old_index < source_cursor or old_index > len(source_lines):
            raise _PatchApplicationError
        output.extend(source_lines[source_cursor:old_index])
        if len(output) != new_index:
            raise _PatchApplicationError
        cursor = old_index
        for line in hunk.lines:
            if line.kind is DiffLineKind.CONTEXT:
                if cursor >= len(source_lines) or source_lines[cursor] != line.content:
                    raise _PatchApplicationError
                output.append(line.content)
                cursor += 1
            elif line.kind is DiffLineKind.REMOVED:
                if cursor >= len(source_lines) or source_lines[cursor] != line.content:
                    raise _PatchApplicationError
                cursor += 1
            else:
                output.append(line.content)
        source_cursor = cursor
    output.extend(source_lines[source_cursor:])
    if file.new_no_final_newline:
        if file.new_no_final_newline_line != len(output):
            raise _PatchApplicationError
        final_newline = False
    elif file.old_no_final_newline and not file.is_deleted:
        raise _PatchApplicationError
    encoded = newline.join(line.encode("utf-8") for line in output)
    if output and final_newline:
        encoded += newline
    if file.is_deleted:
        if encoded:
            raise _PatchApplicationError
        return None
    return encoded


def _apply_patch(
    filesystem: StaticFilesystem,
    staging: Path,
    staging_identity: _DirectoryIdentity,
    pristine: _TreeSnapshot,
    parsed: ParsedUnifiedDiff,
) -> None:
    entries = pristine.by_path
    for file in parsed.files:
        existing = entries.get(file.path)
        if existing is not None and existing.kind is not _EntryKind.FILE:
            raise _PatchApplicationError
        result = _apply_file_patch(file, None if existing is None else existing.data)
        if result is None:
            filesystem.delete_file(staging, staging_identity, file.path)
        else:
            filesystem.write_file(
                staging, staging_identity, file.path, result, create=existing is None
            )


def _changed_python_paths(
    pristine: _TreeSnapshot,
    candidate: _TreeSnapshot,
    expected_paths: tuple[str, ...],
) -> tuple[str, ...]:
    before = pristine.by_path
    after = candidate.by_path
    before_files = {path: entry for path, entry in before.items() if entry.kind is _EntryKind.FILE}
    after_files = {path: entry for path, entry in after.items() if entry.kind is _EntryKind.FILE}
    actual = {
        *{path for path in before_files.keys() - after_files.keys()},
        *{path for path in after_files.keys() - before_files.keys()},
        *{
            path
            for path in before_files.keys() & after_files.keys()
            if before_files[path].data != after_files[path].data
        },
    }
    if {path.casefold() for path in actual} != {path.casefold() for path in expected_paths}:
        raise _ChangedFilesError
    actual_identities = {path.casefold(): path for path in actual}
    if any(actual_identities[path.casefold()] != path for path in expected_paths):
        raise _ChangedFilesError
    before_directories = {
        path for path, entry in before.items() if entry.kind is _EntryKind.DIRECTORY
    }
    after_directories = {
        path for path, entry in after.items() if entry.kind is _EntryKind.DIRECTORY
    }
    if before_directories - after_directories:
        raise _ChangedFilesError
    allowed_new_directories = {
        PurePosixPath(path).parents[index].as_posix()
        for path in expected_paths
        for index in range(len(PurePosixPath(path).parents) - 1)
        if PurePosixPath(path).parents[index].as_posix() != "."
    }
    if (after_directories - before_directories) - allowed_new_directories:
        raise _ChangedFilesError
    return tuple(
        sorted(
            (
                path
                for path in expected_paths
                if path in after_files and PurePosixPath(path).suffix.casefold() in {".py", ".pyi"}
            ),
            key=lambda item: (item.casefold(), item),
        )
    )


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


def _credential_locations() -> tuple[Path, ...]:
    home = Path.home().resolve(strict=False)
    return (
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


def _validate_root(root: Path, *, source: Path | None = None) -> Path:
    if not root.is_absolute():
        raise StaticFilesystemError
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise StaticFilesystemError from None
    if root != resolved or root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise StaticFilesystemError
    if getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT:
        raise StaticFilesystemError
    project = Path(__file__).resolve().parents[2]
    home = Path.home().resolve(strict=False)
    if (
        _overlaps(resolved, project)
        or _same_or_ancestor(resolved, home)
        or any(_overlaps(resolved, item.resolve(strict=False)) for item in _credential_locations())
        or (source is not None and _overlaps(resolved, source))
    ):
        raise StaticFilesystemError
    return resolved


def _diagnostic(code: StaticDiagnosticCode) -> StaticDiagnostic:
    return StaticDiagnostic(code=code, message=_DIAGNOSTIC_MESSAGES[code])


def _passed(check: StaticCheck) -> StaticCheckResult:
    return StaticCheckResult(check=check, status=StaticCheckStatus.PASSED)


def _not_applicable(check: StaticCheck) -> StaticCheckResult:
    return StaticCheckResult(check=check, status=StaticCheckStatus.NOT_APPLICABLE)


def _not_run(check: StaticCheck) -> StaticCheckResult:
    return StaticCheckResult(
        check=check,
        status=StaticCheckStatus.NOT_RUN,
        diagnostic=_diagnostic(StaticDiagnosticCode.PREREQUISITE_FAILED),
    )


def _infrastructure(check: StaticCheck, code: StaticDiagnosticCode) -> StaticCheckResult:
    return StaticCheckResult(
        check=check,
        status=StaticCheckStatus.INFRASTRUCTURE_ERROR,
        diagnostic=_diagnostic(code),
    )


def _failed(check: StaticCheck, rule_code: str, path: str | None = None) -> StaticCheckResult:
    return StaticCheckResult(
        check=check,
        status=StaticCheckStatus.FAILED,
        findings=(
            StaticFinding(
                check=check,
                path=path,
                rule_code=rule_code,
                message=_FINDING_MESSAGES[check],
            ),
        ),
    )


def _blank_checks() -> list[StaticCheckResult]:
    return [_not_run(check) for check in StaticCheck]


def _infrastructure_result(
    checks: list[StaticCheckResult], code: StaticDiagnosticCode
) -> StaticValidationResult:
    if not any(item.status is StaticCheckStatus.INFRASTRUCTURE_ERROR for item in checks):
        first = next(
            (
                index
                for index, item in enumerate(checks)
                if item.status is StaticCheckStatus.NOT_RUN
            ),
            0,
        )
        checks[first] = _infrastructure(checks[first].check, code)
    return StaticValidationResult(
        outcome=StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC,
        checks=tuple(checks),
        diagnostic=_diagnostic(code),
    )


def _candidate_failed_result(checks: list[StaticCheckResult]) -> StaticValidationResult:
    return StaticValidationResult(
        outcome=StaticValidationOutcome.CANDIDATE_FAILED,
        checks=tuple(checks),
    )


def _revalidate_request(request: StaticValidationRequest) -> StaticValidationRequest:
    if type(request) is not StaticValidationRequest:
        raise ValueError
    if (
        type(request.preparation) is not RepositoryPreparation
        or type(request.proposal) is not PatchProposal
        or type(request.policy_result) is not PatchPolicyResult
        or not isinstance(request.candidate_workspace_root, Path)
    ):
        raise ValueError
    raw = request.model_dump(warnings="none")
    validated = StaticValidationRequest.model_validate(raw, strict=True)
    if validated.model_dump(warnings="none") != raw:
        raise ValueError
    if not validated.policy_result.approved or validated.policy_result.violations:
        raise ValueError
    return validated


_TOOL_TO_CHECK: dict[StaticToolCheck, StaticCheck] = {
    StaticToolCheck.PYTHON_COMPILE: StaticCheck.PYTHON_COMPILE,
    StaticToolCheck.AST_PARSE: StaticCheck.AST_PARSE,
    StaticToolCheck.RUFF: StaticCheck.RUFF,
    StaticToolCheck.MYPY: StaticCheck.MYPY,
    StaticToolCheck.BANDIT: StaticCheck.BANDIT,
}


def _map_tool_result(result: StaticToolResult) -> StaticCheckResult:
    check = _TOOL_TO_CHECK[result.check]
    if result.status is StaticToolStatus.PASSED:
        return _passed(check)
    if result.status is StaticToolStatus.NOT_APPLICABLE:
        return _not_applicable(check)
    if result.status is StaticToolStatus.INFRASTRUCTURE_ERROR:
        return _infrastructure(check, StaticDiagnosticCode.STATIC_RUNNER_FAILED)
    findings = tuple(
        StaticFinding(
            check=check,
            path=finding.path,
            line=finding.line,
            column=finding.column,
            rule_code=finding.rule_code,
            message=_FINDING_MESSAGES[check],
        )
        for finding in result.findings
    )
    return StaticCheckResult(check=check, status=StaticCheckStatus.FAILED, findings=findings)


def _revalidate_container_result(
    result: StaticContainerResult, *, python_paths_present: bool
) -> StaticContainerResult:
    try:
        if type(result) is not StaticContainerResult:
            raise ValueError
        raw = result.model_dump(warnings="none")
        validated = StaticContainerResult.model_validate(raw, strict=True)
        if validated.model_dump(warnings="none") != raw:
            raise ValueError
        statuses = {item.status for item in validated.results}
        if python_paths_present and StaticToolStatus.NOT_APPLICABLE in statuses:
            raise ValueError
        if not python_paths_present and statuses != {StaticToolStatus.NOT_APPLICABLE}:
            raise ValueError
        return validated
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise StaticRunnerError(StaticRunnerErrorCode.MALFORMED_RESULT) from None


class StaticValidationService:
    """Apply one approved inert patch and run the blocking Node 8 static gate."""

    def __init__(
        self,
        *,
        filesystem: StaticFilesystem,
        fingerprinter: RepositoryFingerprinter,
        runner: StaticContainerRunner,
        identifier_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._filesystem = filesystem
        self._fingerprinter = fingerprinter
        self._runner = runner
        self._identifier_factory = identifier_factory

    def validate(self, request: StaticValidationRequest) -> StaticValidationResult:
        checks = _blank_checks()
        staging: Path | None = None
        staging_identity: _DirectoryIdentity | None = None
        published: Path | None = None
        cleanup_target: tuple[Path, _DirectoryIdentity] | None = None
        try:
            try:
                validated = _revalidate_request(request)
                proposal_paths = normalize_repository_paths(validated.proposal.files_changed)
                parsed = parse_unified_diff(validated.proposal.unified_diff)
                if parsed.has_ignored_content or set(proposal_paths) != set(parsed.changed_paths):
                    raise ValueError
            except (AttributeError, TypeError, ValueError, ValidationError, UnifiedDiffError):
                return _infrastructure_result(checks, StaticDiagnosticCode.INVALID_INPUT)

            source = validated.preparation.prepared_repository
            try:
                source = _validate_root(source)
                pristine = self._filesystem.snapshot(source)
                if (
                    _fingerprint_snapshot(pristine) != validated.preparation.fingerprint
                    or self._fingerprinter.fingerprint(source) != validated.preparation.fingerprint
                ):
                    raise StaticFilesystemError
            except (OSError, RuntimeError, StaticFilesystemError):
                return _infrastructure_result(checks, StaticDiagnosticCode.PREPARATION_INVALID)

            try:
                workspace = _validate_root(validated.candidate_workspace_root, source=source)
                identifier = self._identifier_factory()
                if _IDENTIFIER.fullmatch(identifier) is None:
                    raise StaticFilesystemError
                staging = workspace / f".tracefix-static-{identifier}.staging"
                published = workspace / f"tracefix-candidate-{identifier}"
                if (
                    staging.exists()
                    or staging.is_symlink()
                    or published.exists()
                    or published.is_symlink()
                ):
                    raise StaticFilesystemError
                staging_identity = self._filesystem.make_directory(staging)
                cleanup_target = (staging, staging_identity)
                self._filesystem.copy_snapshot(pristine, staging, staging_identity)
                if self._filesystem.snapshot(source) != pristine:
                    raise StaticFilesystemError
                if self._fingerprinter.fingerprint(source) != validated.preparation.fingerprint:
                    raise StaticFilesystemError
            except (OSError, RuntimeError, StaticFilesystemError):
                result = _infrastructure_result(checks, StaticDiagnosticCode.WORKSPACE_INVALID)
                return self._cleanup_or_result(cleanup_target, result, checks)

            try:
                if staging_identity is None:
                    raise StaticFilesystemError
                _apply_patch(self._filesystem, staging, staging_identity, pristine, parsed)
                checks[0] = _passed(StaticCheck.PATCH_APPLY)
            except _PatchApplicationError:
                checks[0] = _failed(StaticCheck.PATCH_APPLY, "PATCH_CONFLICT")
                result = _candidate_failed_result(checks)
                return self._cleanup_or_result(cleanup_target, result, checks)
            except StaticFilesystemError:
                checks[0] = _infrastructure(
                    StaticCheck.PATCH_APPLY, StaticDiagnosticCode.FILESYSTEM_FAILED
                )
                result = _infrastructure_result(checks, StaticDiagnosticCode.FILESYSTEM_FAILED)
                return self._cleanup_or_result(cleanup_target, result, checks)

            try:
                if staging_identity is None:
                    raise StaticFilesystemError
                candidate_before = self._filesystem.snapshot(staging, staging_identity)
                python_paths = _changed_python_paths(pristine, candidate_before, proposal_paths)
                checks[1] = _passed(StaticCheck.CHANGED_FILES)
            except _ChangedFilesError:
                checks[1] = _failed(StaticCheck.CHANGED_FILES, "CHANGED_FILES")
                result = _candidate_failed_result(checks)
                return self._cleanup_or_result(cleanup_target, result, checks)
            except StaticFilesystemError:
                checks[1] = _infrastructure(
                    StaticCheck.CHANGED_FILES, StaticDiagnosticCode.FILESYSTEM_FAILED
                )
                result = _infrastructure_result(checks, StaticDiagnosticCode.FILESYSTEM_FAILED)
                return self._cleanup_or_result(cleanup_target, result, checks)

            if not python_paths:
                for index in range(2, 7):
                    checks[index] = _not_applicable(checks[index].check)
            else:
                try:
                    container_result = _revalidate_container_result(
                        self._runner.execute(staging, python_paths),
                        python_paths_present=True,
                    )
                    for index, tool_result in enumerate(container_result.results, start=2):
                        checks[index] = _map_tool_result(tool_result)
                except StaticRunnerError:
                    for index in range(2, 7):
                        checks[index] = _infrastructure(
                            checks[index].check, StaticDiagnosticCode.STATIC_RUNNER_FAILED
                        )
                    result = _infrastructure_result(
                        checks, StaticDiagnosticCode.STATIC_RUNNER_FAILED
                    )
                    return self._cleanup_or_result(cleanup_target, result, checks)

            try:
                if staging_identity is None:
                    raise StaticFilesystemError
                if self._filesystem.snapshot(staging, staging_identity) != candidate_before:
                    raise StaticFilesystemError
                if self._filesystem.snapshot(source) != pristine:
                    raise StaticFilesystemError
                if self._fingerprinter.fingerprint(source) != validated.preparation.fingerprint:
                    raise StaticFilesystemError
            except StaticFilesystemError:
                checks[1] = _infrastructure(
                    StaticCheck.CHANGED_FILES, StaticDiagnosticCode.INTEGRITY_FAILED
                )
                result = _infrastructure_result(checks, StaticDiagnosticCode.INTEGRITY_FAILED)
                return self._cleanup_or_result(cleanup_target, result, checks)

            if any(item.status is StaticCheckStatus.INFRASTRUCTURE_ERROR for item in checks):
                result = _infrastructure_result(checks, StaticDiagnosticCode.STATIC_RUNNER_FAILED)
                return self._cleanup_or_result(cleanup_target, result, checks)
            if any(item.status is StaticCheckStatus.FAILED for item in checks):
                result = _candidate_failed_result(checks)
                return self._cleanup_or_result(cleanup_target, result, checks)

            try:
                if staging_identity is None:
                    raise StaticFilesystemError
                published_identity = self._filesystem.publish(staging, published, staging_identity)
                cleanup_target = (published, published_identity)
                staging = None
                published_snapshot = self._filesystem.snapshot(published, published_identity)
                if published_snapshot != candidate_before:
                    raise StaticFilesystemError
                fingerprint = _fingerprint_snapshot(published_snapshot)
                if self._filesystem.snapshot(source) != pristine:
                    raise StaticFilesystemError
                if self._fingerprinter.fingerprint(source) != validated.preparation.fingerprint:
                    raise StaticFilesystemError
            except StaticFilesystemError:
                result = _infrastructure_result(checks, StaticDiagnosticCode.INTEGRITY_FAILED)
                return self._cleanup_or_result(cleanup_target, result, checks)
            cleanup_target = None
            return StaticValidationResult(
                outcome=StaticValidationOutcome.PASSED,
                checks=tuple(checks),
                candidate=ValidatedCandidate(repository=published, fingerprint=fingerprint),
            )
        except (KeyboardInterrupt, SystemExit):
            if cleanup_target is not None:
                with suppress(Exception):
                    self._filesystem.remove_tree(*cleanup_target)
            raise
        except Exception:
            result = _infrastructure_result(checks, StaticDiagnosticCode.FILESYSTEM_FAILED)
            return self._cleanup_or_result(cleanup_target, result, checks)

    def _cleanup_or_result(
        self,
        target: tuple[Path, _DirectoryIdentity] | None,
        result: StaticValidationResult,
        checks: list[StaticCheckResult],
    ) -> StaticValidationResult:
        if target is None:
            return result
        try:
            self._filesystem.remove_tree(*target)
        except Exception:
            failing_check = next(
                (
                    index
                    for index, item in enumerate(checks)
                    if item.status in {StaticCheckStatus.NOT_RUN, StaticCheckStatus.PASSED}
                ),
                0,
            )
            checks[failing_check] = _infrastructure(
                checks[failing_check].check, StaticDiagnosticCode.CLEANUP_FAILED
            )
            return _infrastructure_result(checks, StaticDiagnosticCode.CLEANUP_FAILED)
        return result
