from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

from app.agent.preparation import (
    LocalPreparationFilesystem,
    PreparationError,
    PreparationErrorCode,
    RepositoryPreparer,
    _EntryIdentity,
)
from app.agent.schemas import RevisionKind
from benchmarks.loader import TrustedCase, load_trusted_case


def _loader(case_id: str) -> TrustedCase:
    return load_trusted_case(case_id)


def test_prepares_exact_snapshot_with_stable_fingerprint(tmp_path: Path) -> None:
    preparer = RepositoryPreparer(_loader)
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first = preparer.prepare(
        "incorrect_conditional",
        "incorrect_conditional-failing-v1",
        tmp_path / "one",
    )
    second = preparer.prepare(
        "incorrect_conditional",
        "incorrect_conditional-failing-v1",
        tmp_path / "two",
    )

    assert first.revision_kind is RevisionKind.SNAPSHOT
    assert first.prepared_repository.is_absolute()
    assert first.fingerprint == second.fingerprint
    assert first.prepared_repository != second.prepared_repository
    source = load_trusted_case("incorrect_conditional").repository_path
    for copied in first.prepared_repository.rglob("*"):
        if copied.is_file():
            relative = copied.relative_to(first.prepared_repository)
            assert copied.read_bytes() == (source / relative).read_bytes()
    assert not (first.prepared_repository / "evaluator").exists()


@pytest.mark.parametrize("revision", [None, "", "incorrect_conditional-base-v1"])
def test_rejects_missing_or_unsupported_revision(tmp_path: Path, revision: str | None) -> None:
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader).prepare("incorrect_conditional", revision, tmp_path)
    assert raised.value.code is PreparationErrorCode.UNSUPPORTED_REVISION


def test_rejects_missing_workspace_root_parent(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "root"
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader).prepare(
            "incorrect_conditional", "incorrect_conditional-failing-v1", missing
        )
    assert raised.value.code is PreparationErrorCode.INVALID_WORKSPACE_ROOT


def test_rejects_workspace_root_beneath_directory_alias(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual"
    workspace = actual_parent / "workspace"
    workspace.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is not permitted")

    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            alias / "workspace",
        )

    assert raised.value.code is PreparationErrorCode.INVALID_WORKSPACE_ROOT
    assert list(workspace.iterdir()) == []


def test_rejects_existing_destination(tmp_path: Path) -> None:
    (tmp_path / "fixed").mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            tmp_path,
            destination_name="fixed",
        )
    assert raised.value.code is PreparationErrorCode.DESTINATION_EXISTS


def test_rejects_unsafe_destination_names(tmp_path: Path) -> None:
    for name in (
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "C:escape",
        "CON",
        "con.txt",
        "NUL",
        "COM1",
        "LPT9",
        "valid.",
    ):
        with pytest.raises(PreparationError):
            RepositoryPreparer(_loader).prepare(
                "incorrect_conditional",
                "incorrect_conditional-failing-v1",
                tmp_path,
                destination_name=name,
            )


def test_fingerprint_changes_with_path_or_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.bin").write_bytes(b"same")
    (second / "b.bin").write_bytes(b"same")
    preparer = RepositoryPreparer(_loader)
    assert preparer.fingerprint(first) != preparer.fingerprint(second)
    (second / "b.bin").write_bytes(b"changed")
    changed = preparer.fingerprint(second)
    (second / "b.bin").write_bytes(b"same")
    assert changed != preparer.fingerprint(second)


def test_fingerprint_is_independent_of_file_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root, names in ((first, ("z.py", "a.py")), (second, ("a.py", "z.py"))):
        for name in names:
            (root / name).write_bytes(name.encode("ascii"))
    preparer = RepositoryPreparer(_loader)
    assert preparer.fingerprint(first) == preparer.fingerprint(second)


class FingerprintReadFailingFilesystem(LocalPreparationFilesystem):
    def read_file(self, path: Path, expected: object) -> bytes:
        del path, expected
        raise OSError("injected fingerprint read failure")


def test_fingerprint_read_failure_is_typed(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "file.py").write_text("content", encoding="utf-8")
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader, filesystem=FingerprintReadFailingFilesystem()).fingerprint(
            repository
        )
    assert raised.value.code is PreparationErrorCode.FINGERPRINT_FAILED


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_rejects_special_file_entry(tmp_path: Path) -> None:
    source_case = load_trusted_case("incorrect_conditional")
    case_copy = tmp_path / "case"
    shutil.copytree(source_case.case_path, case_copy)
    os.mkfifo(case_copy / "repository" / "unsafe-fifo")

    def loader(_: str) -> TrustedCase:
        loaded = load_trusted_case("incorrect_conditional")
        return loaded.model_copy(update={"repository_path": case_copy / "repository"})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(loader).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.UNSAFE_SOURCE
    assert not (workspace / "partial").exists()


def test_rejects_source_alias_and_removes_partial_destination(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    source_case = load_trusted_case("incorrect_conditional")
    case_copy = tmp_path / "case"
    shutil.copytree(source_case.case_path, case_copy)
    target = case_copy / "repository" / "alias.py"
    try:
        target.symlink_to(case_copy / "repository" / "tests" / "test_behavior.py")
    except OSError:
        pytest.skip("symlink creation is not permitted")

    def loader(_: str) -> TrustedCase:
        loaded = load_trusted_case("incorrect_conditional")
        return loaded.model_copy(update={"repository_path": case_copy / "repository"})

    with pytest.raises(PreparationError):
        RepositoryPreparer(loader).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            tmp_path / "workspace",
            destination_name="partial",
        )
    assert not (tmp_path / "workspace" / "partial").exists()


def test_rejects_hard_link_to_file_outside_repository(tmp_path: Path) -> None:
    source_case = load_trusted_case("incorrect_conditional")
    case_copy = tmp_path / "case"
    shutil.copytree(source_case.case_path, case_copy)
    external = tmp_path / "external-secret"
    external.write_text("must not be copied", encoding="utf-8")
    linked = case_copy / "repository" / "linked-secret"
    try:
        os.link(external, linked)
    except OSError:
        pytest.skip("hard-link creation is not permitted")

    def loader(_: str) -> TrustedCase:
        loaded = load_trusted_case("incorrect_conditional")
        return loaded.model_copy(update={"repository_path": case_copy / "repository"})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(loader).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.UNSAFE_SOURCE
    assert not (workspace / "partial").exists()


class ReplacingFilesystem(LocalPreparationFilesystem):
    def __init__(self, replacement: Path) -> None:
        self._replacement = replacement

    def open_source(self, path: Path, expected: object) -> BinaryIO:
        path.unlink()
        path.symlink_to(self._replacement)
        return super().open_source(path, expected)


def test_rejects_source_replaced_by_alias_before_open(tmp_path: Path) -> None:
    source_case = load_trusted_case("incorrect_conditional")
    case_copy = tmp_path / "case"
    shutil.copytree(source_case.case_path, case_copy)
    external = tmp_path / "external-secret"
    external.write_text("must not be copied", encoding="utf-8")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(external)
        probe.unlink()
    except OSError:
        pytest.skip("symlink creation is not permitted")

    def loader(_: str) -> TrustedCase:
        loaded = load_trusted_case("incorrect_conditional")
        return loaded.model_copy(update={"repository_path": case_copy / "repository"})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(loader, filesystem=ReplacingFilesystem(external)).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.SOURCE_MUTATED
    assert not (workspace / "partial").exists()


class SameMetadataMutationFilesystem(LocalPreparationFilesystem):
    def __init__(self) -> None:
        self._mutated = False

    def copy_file(
        self, source: Path, destination: Path, destination_root: Path, expected: object
    ) -> None:
        if not self._mutated and source.is_file() and source.stat().st_size > 0:
            original = source.read_bytes()
            metadata = source.stat()
            replacement = bytes(byte ^ 0xFF for byte in original)
            source.write_bytes(replacement)
            os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            try:
                super().copy_file(source, destination, destination_root, expected)
            finally:
                source.write_bytes(original)
                os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            self._mutated = True
            return
        super().copy_file(source, destination, destination_root, expected)


def test_rejects_same_size_source_mutation_with_restored_timestamp(tmp_path: Path) -> None:
    source_case = load_trusted_case("incorrect_conditional")
    case_copy = tmp_path / "case"
    shutil.copytree(source_case.case_path, case_copy)

    def loader(_: str) -> TrustedCase:
        loaded = load_trusted_case("incorrect_conditional")
        return loaded.model_copy(update={"repository_path": case_copy / "repository"})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(loader, filesystem=SameMetadataMutationFilesystem()).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )

    assert raised.value.code is PreparationErrorCode.SOURCE_MUTATED
    assert not (workspace / "partial").exists()


class FailingFilesystem(LocalPreparationFilesystem):
    def __init__(self, fail: Callable[[], None]) -> None:
        self._fail = fail

    def copy_file(
        self, source: Path, destination: Path, destination_root: Path, expected: object
    ) -> None:
        del source, destination, destination_root, expected
        self._fail()


def test_injected_filesystem_copy_failure_is_typed_and_cleaned(tmp_path: Path) -> None:
    def fail() -> None:
        raise OSError("injected copy failure")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader, filesystem=FailingFilesystem(fail)).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.COPY_FAILED
    assert not (workspace / "partial").exists()


class CleanupFailingFilesystem(FailingFilesystem):
    def remove_tree(self, path: Path, expected: _EntryIdentity) -> None:
        del path, expected
        raise OSError("injected cleanup failure")


class RuntimeCleanupFailingFilesystem(FailingFilesystem):
    def remove_tree(self, path: Path, expected: _EntryIdentity) -> None:
        del path, expected
        raise RuntimeError("cleanup failed")


def test_injected_cleanup_failure_preserves_primary_and_cleanup_errors(tmp_path: Path) -> None:
    def fail() -> None:
        raise OSError("injected copy failure")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader, filesystem=CleanupFailingFilesystem(fail)).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.CLEANUP_FAILED
    assert isinstance(raised.value.cleanup_cause, OSError)
    assert isinstance(raised.value.__cause__, PreparationError)
    assert raised.value.__cause__.code is PreparationErrorCode.COPY_FAILED


def test_non_os_cleanup_failure_is_typed_and_preserves_both_causes(tmp_path: Path) -> None:
    def fail() -> None:
        raise OSError("injected copy failure")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader, filesystem=RuntimeCleanupFailingFilesystem(fail)).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.CLEANUP_FAILED
    assert str(raised.value) == "failed workspace cleanup also failed"
    assert isinstance(raised.value.cleanup_cause, RuntimeError)
    assert isinstance(raised.value.__cause__, PreparationError)
    assert raised.value.__cause__.code is PreparationErrorCode.COPY_FAILED


class DestinationReplacingFilesystem(LocalPreparationFilesystem):
    def __init__(self, external: Path) -> None:
        self._external = external
        self._replaced = False

    def copy_file(
        self, source: Path, destination: Path, destination_root: Path, expected: object
    ) -> None:
        if not self._replaced and destination.parent != destination.parents[1]:
            replaced_parent = destination.parent
            if replaced_parent.exists():
                replaced_parent.rmdir()
                replaced_parent.symlink_to(self._external, target_is_directory=True)
                self._replaced = True
        super().copy_file(source, destination, destination_root, expected)


def test_rejects_destination_parent_replaced_by_alias_before_write(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    probe = tmp_path / "junction-probe"
    try:
        probe.symlink_to(external, target_is_directory=True)
        probe.unlink()
    except OSError:
        pytest.skip("directory symlink creation is not permitted")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PreparationError) as raised:
        RepositoryPreparer(_loader, filesystem=DestinationReplacingFilesystem(external)).prepare(
            "incorrect_conditional",
            "incorrect_conditional-failing-v1",
            workspace,
            destination_name="partial",
        )
    assert raised.value.code is PreparationErrorCode.UNSAFE_DESTINATION
    assert list(external.iterdir()) == []


class WorkspaceRootReplacingFilesystem(LocalPreparationFilesystem):
    def __init__(self, external: Path, original: Path) -> None:
        self._external = external
        self._original = original

    def make_directory(
        self, path: Path, workspace_root: Path, workspace_identity: _EntryIdentity
    ) -> None:
        workspace_root.rename(self._original)
        workspace_root.symlink_to(self._external, target_is_directory=True)
        super().make_directory(path, workspace_root, workspace_identity)


class WorkspaceRootReplacingAfterValidationFilesystem(LocalPreparationFilesystem):
    def __init__(self, external: Path, original: Path) -> None:
        self._external = external
        self._original = original
        self._replaced = False

    def _validate_destination_parent(self, destination: Path, root: Path) -> None:
        super()._validate_destination_parent(destination, root)
        if not self._replaced and destination.parent == root:
            root.rename(self._original)
            self._external.rename(root)
            self._replaced = True


class DestinationRootReplacingAfterValidationFilesystem(LocalPreparationFilesystem):
    def __init__(self, external: Path, original: Path) -> None:
        self._external = external
        self._original = original
        self._replaced = False

    def _validate_destination_parent(self, destination: Path, root: Path) -> None:
        super()._validate_destination_parent(destination, root)
        if not self._replaced and root.name == "partial" and destination.parent != root:
            root.rename(self._original)
            self._external.rename(root)
            self._replaced = True


@pytest.mark.skipif(os.name == "nt", reason="requires Linux directory-relative primitives")
def test_linux_workspace_root_swap_after_validation_cannot_modify_attacker_directory(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "original-workspace"

    try:
        with pytest.raises(PreparationError):
            RepositoryPreparer(
                _loader,
                filesystem=WorkspaceRootReplacingAfterValidationFilesystem(external, original),
            ).prepare(
                "incorrect_conditional",
                "incorrect_conditional-failing-v1",
                workspace,
                destination_name="partial",
            )
        assert list(workspace.iterdir()) == []
        assert list(original.iterdir()) == []
    finally:
        if workspace.exists() and original.exists():
            workspace.rename(external)
        if original.exists():
            original.rename(workspace)


@pytest.mark.skipif(os.name == "nt", reason="requires Linux directory-relative primitives")
def test_linux_destination_root_swap_cannot_write_to_or_remove_attacker_directory(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker"
    marker.write_text("must survive", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "original-destination"

    try:
        with pytest.raises(PreparationError):
            RepositoryPreparer(
                _loader,
                filesystem=DestinationRootReplacingAfterValidationFilesystem(external, original),
            ).prepare(
                "incorrect_conditional",
                "incorrect_conditional-failing-v1",
                workspace,
                destination_name="partial",
            )
        assert (workspace / "partial" / "marker").read_text(encoding="utf-8") == "must survive"
        assert list((workspace / "partial").iterdir()) == [workspace / "partial" / "marker"]
    finally:
        attacker = workspace / "partial"
        if attacker.exists() and original.exists():
            attacker.rename(external)
        if original.exists():
            original.rename(attacker)


def test_rejects_workspace_root_replaced_before_destination_creation(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "original-workspace"
    probe = tmp_path / "alias-probe"
    try:
        probe.symlink_to(external, target_is_directory=True)
        probe.unlink()
    except OSError:
        pytest.skip("directory alias creation is not permitted")

    try:
        with pytest.raises(PreparationError) as raised:
            RepositoryPreparer(
                _loader,
                filesystem=WorkspaceRootReplacingFilesystem(external, original),
            ).prepare(
                "incorrect_conditional",
                "incorrect_conditional-failing-v1",
                workspace,
                destination_name="partial",
            )
        assert raised.value.code is PreparationErrorCode.INVALID_WORKSPACE_ROOT
        assert list(external.iterdir()) == []
        assert list(original.iterdir()) == []
    finally:
        if workspace.is_symlink():
            workspace.unlink()
        if original.exists():
            original.rename(workspace)
