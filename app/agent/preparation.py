from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Protocol

from app.agent.schemas import RepositoryFingerprint, RepositoryPreparation, RevisionKind
from benchmarks.loader import BenchmarkLoadError, TrustedCase

_WINDOWS_REPARSE_POINT = 0x400
_DESTINATION_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_FINGERPRINT_HEADER = b"tracefix-repository-fingerprint\x00v1\x00"

TrustedCaseLoader = Callable[[str], TrustedCase]


class PreparationErrorCode(StrEnum):
    CASE_LOAD_FAILED = "case_load_failed"
    UNSUPPORTED_REVISION = "unsupported_revision"
    INVALID_WORKSPACE_ROOT = "invalid_workspace_root"
    UNSAFE_DESTINATION = "unsafe_destination"
    DESTINATION_EXISTS = "destination_exists"
    UNSAFE_SOURCE = "unsafe_source"
    COPY_FAILED = "copy_failed"
    SOURCE_MUTATED = "source_mutated"
    FINGERPRINT_FAILED = "fingerprint_failed"
    CLEANUP_FAILED = "cleanup_failed"


class PreparationError(Exception):
    def __init__(
        self,
        code: PreparationErrorCode,
        message: str,
        *,
        cleanup_cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cleanup_cause = cleanup_cause


@dataclass(frozen=True, slots=True)
class _EntryIdentity:
    path: PurePosixPath
    mode: int
    device: int
    inode: int
    size: int
    modified_ns: int
    reparse_tag: int
    link_count: int


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


def _reject_path_aliases(path: Path) -> None:
    """Reject aliases in every existing component without resolving through them."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT,
                "workspace root component could not be inspected",
            ) from error
        attributes = getattr(metadata, "st_file_attributes", 0)
        if current.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT,
                "workspace root contains a filesystem alias",
            )


def _identity(path: Path, relative: PurePosixPath) -> _EntryIdentity:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PreparationError(
            PreparationErrorCode.UNSAFE_SOURCE, "repository entry could not be inspected"
        ) from error
    attributes = getattr(metadata, "st_file_attributes", 0)
    if path.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
        raise PreparationError(
            PreparationErrorCode.UNSAFE_SOURCE, "repository contains a filesystem alias"
        )
    if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
        raise PreparationError(
            PreparationErrorCode.UNSAFE_SOURCE, "repository contains an unsupported entry"
        )
    if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
        raise PreparationError(
            PreparationErrorCode.UNSAFE_SOURCE, "repository contains a hard-linked file"
        )
    return _EntryIdentity(
        path=relative,
        mode=metadata.st_mode,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        reparse_tag=getattr(metadata, "st_reparse_tag", 0),
        link_count=metadata.st_nlink,
    )


def _snapshot(root: Path) -> tuple[_EntryIdentity, ...]:
    identities: list[_EntryIdentity] = []
    casefolded: set[str] = set()

    def walk(directory: Path, relative: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as scanned:
                entries = sorted(scanned, key=lambda item: item.name)
        except OSError as error:
            raise PreparationError(
                PreparationErrorCode.UNSAFE_SOURCE, "repository could not be enumerated"
            ) from error
        for entry in entries:
            child_relative = relative / entry.name
            key = child_relative.as_posix().casefold()
            if key in casefolded:
                raise PreparationError(
                    PreparationErrorCode.UNSAFE_SOURCE,
                    "repository contains case-insensitive path aliases",
                )
            casefolded.add(key)
            identity = _identity(Path(entry.path), child_relative)
            identities.append(identity)
            if stat.S_ISDIR(identity.mode):
                walk(Path(entry.path), child_relative)

    walk(root, PurePosixPath())
    return tuple(identities)


def _length(value: int) -> bytes:
    if value < 0 or value >= 2**64:
        raise PreparationError(
            PreparationErrorCode.FINGERPRINT_FAILED, "fingerprint entry length is unsupported"
        )
    return value.to_bytes(8, "big")


def _identity_from_stat(metadata: os.stat_result, relative: PurePosixPath) -> _EntryIdentity:
    return _EntryIdentity(
        path=relative,
        mode=metadata.st_mode,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        reparse_tag=getattr(metadata, "st_reparse_tag", 0),
        link_count=metadata.st_nlink,
    )


class PreparationFilesystem(Protocol):
    """Typed host-filesystem boundary for snapshot materialization."""

    def snapshot(self, root: Path) -> tuple[_EntryIdentity, ...]: ...

    def make_directory(
        self, path: Path, workspace_root: Path, workspace_identity: _EntryIdentity
    ) -> None: ...

    def open_source(self, path: Path, expected: _EntryIdentity) -> BinaryIO: ...

    def copy_file(
        self,
        source: Path,
        destination: Path,
        destination_root: Path,
        expected: _EntryIdentity,
    ) -> None: ...

    def read_file(self, path: Path, expected: _EntryIdentity) -> bytes: ...

    def remove_tree(self, path: Path, expected: _EntryIdentity) -> None: ...


class LocalPreparationFilesystem:
    """Standard-library implementation with opened-handle identity validation."""

    def snapshot(self, root: Path) -> tuple[_EntryIdentity, ...]:
        return _snapshot(root)

    @staticmethod
    def _validate_workspace_identity(root: Path, expected: _EntryIdentity) -> None:
        try:
            _reject_path_aliases(root)
            actual = _identity(root, PurePosixPath())
        except PreparationError as error:
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT,
                "workspace root changed during preparation",
            ) from error
        if (
            not stat.S_ISDIR(actual.mode)
            or actual.device != expected.device
            or actual.inode != expected.inode
            or actual.reparse_tag != expected.reparse_tag
        ):
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT,
                "workspace root changed during preparation",
            )

    def make_directory(
        self, path: Path, workspace_root: Path, workspace_identity: _EntryIdentity
    ) -> None:
        self._validate_workspace_identity(workspace_root, workspace_identity)
        self._validate_destination_parent(path, workspace_root)
        try:
            if os.mkdir in os.supports_dir_fd:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
                parent = os.open(workspace_root, flags)
                try:
                    opened_root = os.fstat(parent)
                    if (
                        not stat.S_ISDIR(opened_root.st_mode)
                        or opened_root.st_dev != workspace_identity.device
                        or opened_root.st_ino != workspace_identity.inode
                        or getattr(opened_root, "st_reparse_tag", 0)
                        != workspace_identity.reparse_tag
                    ):
                        raise PreparationError(
                            PreparationErrorCode.INVALID_WORKSPACE_ROOT,
                            "opened workspace root does not match the validated directory",
                        )
                    relative_parent = path.parent.relative_to(workspace_root)
                    for part in relative_parent.parts:
                        next_parent = os.open(part, flags, dir_fd=parent)
                        os.close(parent)
                        parent = next_parent
                    os.mkdir(path.name, dir_fd=parent)
                finally:
                    os.close(parent)
            else:
                path.mkdir()
        except OSError as error:
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION,
                "workspace directory could not be created safely",
            ) from error
        self._validate_workspace_identity(workspace_root, workspace_identity)
        self._validate_destination_parent(path / "child", workspace_root)

    def open_source(self, path: Path, expected: _EntryIdentity) -> BinaryIO:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise PreparationError(
                PreparationErrorCode.SOURCE_MUTATED, "source changed before it could be opened"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _identity_from_stat(opened, expected.path) != expected
            ):
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED,
                    "opened source does not match the inspected regular file",
                )
            return os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _validate_destination_parent(destination: Path, root: Path) -> None:
        try:
            relative_parent = destination.parent.relative_to(root)
            current = root
            for part in relative_parent.parts:
                current /= part
                metadata = current.lstat()
                attributes = getattr(metadata, "st_file_attributes", 0)
                if (
                    current.is_symlink()
                    or attributes & _WINDOWS_REPARSE_POINT
                    or not stat.S_ISDIR(metadata.st_mode)
                ):
                    raise PreparationError(
                        PreparationErrorCode.UNSAFE_DESTINATION,
                        "workspace destination contains a filesystem alias",
                    )
            destination.parent.resolve(strict=True).relative_to(root.resolve(strict=True))
        except PreparationError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION,
                "workspace destination parent is unsafe",
            ) from error

    def _open_destination(self, destination: Path, root: Path) -> BinaryIO:
        expected_root = _identity(root, PurePosixPath())
        self._validate_destination_parent(destination, root)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if os.name != "nt":
            directory_flags = (
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(root, directory_flags)
            try:
                opened_root = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(opened_root.st_mode)
                    or opened_root.st_dev != expected_root.device
                    or opened_root.st_ino != expected_root.inode
                    or getattr(opened_root, "st_reparse_tag", 0) != expected_root.reparse_tag
                ):
                    raise PreparationError(
                        PreparationErrorCode.UNSAFE_DESTINATION,
                        "opened destination root does not match the validated directory",
                    )
                relative = destination.relative_to(root)
                for part in relative.parts[:-1]:
                    next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = next_descriptor
                output_descriptor = os.open(relative.name, flags, 0o600, dir_fd=descriptor)
            finally:
                os.close(descriptor)
        else:
            output_descriptor = os.open(destination, flags, 0o600)
        try:
            self._validate_destination_parent(destination, root)
            return os.fdopen(output_descriptor, "wb")
        except BaseException:
            os.close(output_descriptor)
            raise

    def copy_file(
        self,
        source: Path,
        destination: Path,
        destination_root: Path,
        expected: _EntryIdentity,
    ) -> None:
        with (
            self.open_source(source, expected) as source_file,
            self._open_destination(destination, destination_root) as output,
        ):
            shutil.copyfileobj(source_file, output)
            if _identity_from_stat(os.fstat(source_file.fileno()), expected.path) != expected:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED, "source changed while being copied"
                )

    def read_file(self, path: Path, expected: _EntryIdentity) -> bytes:
        with self.open_source(path, expected) as source_file:
            data = source_file.read()
            if _identity_from_stat(os.fstat(source_file.fileno()), expected.path) != expected:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED, "repository changed while fingerprinting"
                )
            return data

    def remove_tree(self, path: Path, expected: _EntryIdentity) -> None:
        actual = _identity(path, expected.path)
        if (
            not stat.S_ISDIR(actual.mode)
            or actual.device != expected.device
            or actual.inode != expected.inode
            or actual.reparse_tag != expected.reparse_tag
        ):
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION,
                "failed workspace no longer matches the created directory",
            )
        metadata = path.lstat()
        if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT:
            path.rmdir() if stat.S_ISDIR(metadata.st_mode) else path.unlink()
        else:
            shutil.rmtree(path)


class RepositoryPreparer:
    """Materialize one validated benchmark snapshot into a caller-owned workspace."""

    def __init__(
        self, loader: TrustedCaseLoader, *, filesystem: PreparationFilesystem | None = None
    ) -> None:
        self._loader = loader
        self._filesystem = filesystem or LocalPreparationFilesystem()

    def prepare(
        self,
        case_id: str,
        requested_revision: str | None,
        workspace_root: Path,
        *,
        destination_name: str | None = None,
    ) -> RepositoryPreparation:
        try:
            case = self._loader(case_id)
        except (BenchmarkLoadError, OSError, ValueError) as error:
            raise PreparationError(
                PreparationErrorCode.CASE_LOAD_FAILED, "trusted benchmark case could not be loaded"
            ) from error
        if not requested_revision or requested_revision != case.manifest.failing_revision:
            raise PreparationError(
                PreparationErrorCode.UNSUPPORTED_REVISION,
                "requested revision must exactly match the declared failing revision",
            )
        root = self._workspace_root(workspace_root, case.repository_path)
        root_identity = _identity(root, PurePosixPath())
        name = (
            f"{case.manifest.case_id}-{uuid.uuid4().hex}"
            if destination_name is None
            else destination_name
        )
        if not self._safe_destination_name(name):
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION, "workspace destination name is unsafe"
            )
        destination = root / name
        try:
            resolved_destination = destination.resolve(strict=False)
            resolved_destination.relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION, "workspace destination escapes its root"
            ) from error
        if destination.exists() or destination.is_symlink():
            raise PreparationError(
                PreparationErrorCode.DESTINATION_EXISTS, "workspace destination already exists"
            )
        source = case.repository_path.resolve(strict=True)
        if _overlaps(source, resolved_destination):
            raise PreparationError(
                PreparationErrorCode.UNSAFE_DESTINATION, "source and destination overlap"
            )
        created = False
        destination_identity: _EntryIdentity | None = None
        try:
            before = self._filesystem.snapshot(source)
            source_fingerprint = self.fingerprint(source)
            if self._filesystem.snapshot(source) != before:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED,
                    "source changed before preparation",
                )
            self._filesystem.make_directory(destination, root, root_identity)
            created = True
            destination_identity = _identity(destination, PurePosixPath())
            self._copy(source, destination, root, root_identity, before)
            if (
                self._filesystem.snapshot(source) != before
                or self.fingerprint(source) != source_fingerprint
            ):
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED, "source changed during preparation"
                )
            copied = self._filesystem.snapshot(destination)
            fingerprint = self.fingerprint(destination)
            if fingerprint != source_fingerprint:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED,
                    "prepared repository content does not match the source snapshot",
                )
            if (
                self._filesystem.snapshot(destination) != copied
                or self.fingerprint(destination) != fingerprint
            ):
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED,
                    "prepared repository changed during validation",
                )
            return RepositoryPreparation(
                case_id=case.manifest.case_id,
                requested_revision=requested_revision,
                revision_kind=RevisionKind.SNAPSHOT,
                prepared_repository=destination.resolve(strict=True),
                fingerprint=fingerprint,
            )
        except PreparationError as preparation_error:
            self._cleanup_failed(destination, created, destination_identity, preparation_error)
            raise
        except (OSError, RuntimeError, ValueError) as error:
            copy_error = PreparationError(PreparationErrorCode.COPY_FAILED, "snapshot copy failed")
            self._cleanup_failed(destination, created, destination_identity, copy_error)
            raise copy_error from error

    def fingerprint(self, repository: Path) -> RepositoryFingerprint:
        try:
            root = repository.resolve(strict=True)
            entries = self._filesystem.snapshot(root)
            digest = hashlib.sha256(_FINGERPRINT_HEADER)
            for entry in entries:
                path_bytes = entry.path.as_posix().encode("utf-8")
                if stat.S_ISDIR(entry.mode):
                    digest.update(b"D")
                    digest.update(_length(len(path_bytes)))
                    digest.update(path_bytes)
                    continue
                file_path = root.joinpath(*entry.path.parts)
                data = self._filesystem.read_file(file_path, entry)
                digest.update(b"F")
                digest.update(_length(len(path_bytes)))
                digest.update(path_bytes)
                digest.update(_length(len(data)))
                digest.update(data)
            if self._filesystem.snapshot(root) != entries:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED, "repository changed while fingerprinting"
                )
            return RepositoryFingerprint(digest=digest.hexdigest())
        except PreparationError:
            raise
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            raise PreparationError(
                PreparationErrorCode.FINGERPRINT_FAILED, "repository fingerprint failed"
            ) from error

    @staticmethod
    def _safe_destination_name(name: str) -> bool:
        windows = PureWindowsPath(name)
        windows_basename = name.rstrip(" .").split(".", maxsplit=1)[0].upper()
        return bool(
            name
            and _DESTINATION_NAME.fullmatch(name)
            and name not in {".", ".."}
            and not windows.drive
            and not windows.root
            and "/" not in name
            and "\\" not in name
            and name == name.rstrip(" .")
            and windows_basename not in _WINDOWS_RESERVED_NAMES
        )

    @staticmethod
    def _workspace_root(workspace_root: Path, source: Path) -> Path:
        if not workspace_root.is_absolute():
            workspace_root = workspace_root.absolute()
        try:
            _reject_path_aliases(workspace_root)
            identity = _identity(workspace_root, PurePosixPath())
            root = workspace_root.resolve(strict=True)
        except (PreparationError, OSError, RuntimeError) as error:
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT, "workspace root is invalid"
            ) from error
        if not stat.S_ISDIR(identity.mode):
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT, "workspace root is not a directory"
            )
        project = Path(__file__).resolve().parents[2]
        home = Path.home().resolve(strict=False)
        protected = (
            _overlaps(root, source.resolve(strict=True))
            or _overlaps(root, project)
            or _same_or_ancestor(root, home)
            or any(
                _overlaps(root, location.resolve(strict=False))
                for location in _credential_locations()
            )
        )
        if protected:
            raise PreparationError(
                PreparationErrorCode.INVALID_WORKSPACE_ROOT, "workspace root is protected"
            )
        return root

    def _copy(
        self,
        source: Path,
        destination: Path,
        workspace_root: Path,
        workspace_identity: _EntryIdentity,
        entries: tuple[_EntryIdentity, ...],
    ) -> None:
        for entry in entries:
            source_path = source.joinpath(*entry.path.parts)
            destination_path = destination.joinpath(*entry.path.parts)
            if _identity(source_path, entry.path) != entry:
                raise PreparationError(
                    PreparationErrorCode.SOURCE_MUTATED, "source changed during copy"
                )
            if stat.S_ISDIR(entry.mode):
                self._filesystem.make_directory(
                    destination_path, workspace_root, workspace_identity
                )
            else:
                self._filesystem.copy_file(source_path, destination_path, destination, entry)
                if _identity(source_path, entry.path) != entry:
                    raise PreparationError(
                        PreparationErrorCode.SOURCE_MUTATED, "source changed during copy"
                    )

    def _cleanup_failed(
        self,
        destination: Path,
        created: bool,
        destination_identity: _EntryIdentity | None,
        primary: PreparationError,
    ) -> None:
        if not created or destination_identity is None:
            return
        try:
            self._filesystem.remove_tree(destination, destination_identity)
        except Exception as cleanup_error:
            raise PreparationError(
                PreparationErrorCode.CLEANUP_FAILED,
                "failed workspace cleanup also failed",
                cleanup_cause=cleanup_error,
            ) from primary
