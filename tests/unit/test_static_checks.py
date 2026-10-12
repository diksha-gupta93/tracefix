from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent.policy import PatchPolicyResult
from app.agent.schemas import (
    PatchProposal,
    RepositoryFingerprint,
    RepositoryPreparation,
    RevisionKind,
)
from app.agent.static_checks import (
    LocalStaticFilesystem,
    Sha256RepositoryFingerprinter,
    StaticCheck,
    StaticCheckResult,
    StaticCheckStatus,
    StaticDiagnostic,
    StaticDiagnosticCode,
    StaticFilesystemError,
    StaticFinding,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationResult,
    StaticValidationService,
    ValidatedCandidate,
    _DirectoryIdentity,
    _revalidate_container_result,
)
from app.sandbox.static_runner import (
    StaticContainerResult,
    StaticContainerRunner,
    StaticRunnerError,
    StaticRunnerErrorCode,
    StaticToolCheck,
    StaticToolFinding,
    StaticToolResult,
    StaticToolStatus,
)


class PassingStaticRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        self.calls.append((repository, changed_python_paths))
        results = tuple(
            StaticToolResult(check=check, status=StaticToolStatus.PASSED, findings=())
            for check in StaticToolCheck
        )
        return StaticContainerResult(version=1, results=results)


class FindingStaticRunner:
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        del repository
        path = changed_python_paths[0]
        results = []
        for check in StaticToolCheck:
            if check in {StaticToolCheck.RUFF, StaticToolCheck.MYPY, StaticToolCheck.BANDIT}:
                results.append(
                    StaticToolResult(
                        check=check,
                        status=StaticToolStatus.FAILED,
                        findings=(
                            StaticToolFinding(
                                path=path,
                                line=2,
                                column=1,
                                rule_code={
                                    StaticToolCheck.RUFF: "F401",
                                    StaticToolCheck.MYPY: "return-value",
                                    StaticToolCheck.BANDIT: "B404",
                                }[check],
                            ),
                        ),
                    )
                )
            else:
                results.append(
                    StaticToolResult(check=check, status=StaticToolStatus.PASSED, findings=())
                )
        return StaticContainerResult(results=tuple(results))


class SingleFindingStaticRunner:
    def __init__(self, failed_check: StaticToolCheck) -> None:
        self.failed_check = failed_check

    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        del repository
        results = tuple(
            StaticToolResult(
                check=check,
                status=(
                    StaticToolStatus.FAILED
                    if check is self.failed_check
                    else StaticToolStatus.PASSED
                ),
                findings=(
                    (
                        StaticToolFinding(
                            path=changed_python_paths[0],
                            line=1,
                            column=1,
                            rule_code={
                                StaticToolCheck.PYTHON_COMPILE: "PY_COMPILE",
                                StaticToolCheck.AST_PARSE: "AST_PARSE",
                                StaticToolCheck.RUFF: "F401",
                                StaticToolCheck.MYPY: "return-value",
                                StaticToolCheck.BANDIT: "B404",
                            }[check],
                        ),
                    )
                    if check is self.failed_check
                    else ()
                ),
            )
            for check in StaticToolCheck
        )
        return StaticContainerResult(results=results)


class FailingStaticRunner:
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        del repository, changed_python_paths
        raise StaticRunnerError(StaticRunnerErrorCode.MALFORMED_RESULT)


class MutatingStaticRunner(PassingStaticRunner):
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        result = super().execute(repository, changed_python_paths)
        (repository / "src" / "example.py").write_text("mutated = True\n", encoding="utf-8")
        return result


class InterruptingStaticRunner:
    def __init__(self, interrupt: BaseException) -> None:
        self.interrupt = interrupt

    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        del repository, changed_python_paths
        raise self.interrupt


class ExtraChangeFilesystem(LocalStaticFilesystem):
    def write_file(
        self,
        root: Path,
        expected_root: _DirectoryIdentity,
        relative_path: str,
        data: bytes,
        *,
        create: bool,
    ) -> None:
        super().write_file(root, expected_root, relative_path, data, create=create)
        super().write_file(root, expected_root, "undeclared.txt", b"unexpected\n", create=True)


class CleanupFailingFilesystem(LocalStaticFilesystem):
    def remove_tree(self, path: Path, expected_root: _DirectoryIdentity) -> None:
        del path, expected_root
        raise OSError("synthetic cleanup failure")


class NotApplicableStaticRunner:
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        del repository, changed_python_paths
        return StaticContainerResult(
            results=tuple(
                StaticToolResult(check=check, status=StaticToolStatus.NOT_APPLICABLE)
                for check in StaticToolCheck
            )
        )


class MutatedEvidenceStaticRunner(PassingStaticRunner):
    def execute(
        self, repository: Path, changed_python_paths: tuple[str, ...]
    ) -> StaticContainerResult:
        result = super().execute(repository, changed_python_paths)
        object.__setattr__(result.results[0], "findings", ("contradictory",))
        return result


class PublishingMutationFilesystem(LocalStaticFilesystem):
    def publish(
        self, staging: Path, destination: Path, expected_root: _DirectoryIdentity
    ) -> _DirectoryIdentity:
        identity = super().publish(staging, destination, expected_root)
        (destination / "src" / "example.py").write_bytes(b"publish race\n")
        return identity


def _proposal(diff: str, paths: list[str] | None = None) -> PatchProposal:
    return PatchProposal(
        summary="repair",
        root_cause="defect",
        files_changed=["src/example.py"] if paths is None else paths,
        unified_diff=diff,
        expected_effect="correct behavior",
        risks=[],
        confidence=0.9,
    )


def _approved() -> PatchPolicyResult:
    return PatchPolicyResult(approved=True, violations=())


def _request(
    source: Path,
    workspace: Path,
    proposal: PatchProposal,
) -> tuple[StaticValidationRequest, LocalStaticFilesystem, Sha256RepositoryFingerprinter]:
    filesystem = LocalStaticFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprinter.fingerprint(source.resolve()),
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=proposal,
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )
    return request, filesystem, fingerprinter


def _service(
    filesystem: LocalStaticFilesystem,
    fingerprinter: Sha256RepositoryFingerprinter,
    runner: PassingStaticRunner,
) -> StaticValidationService:
    return StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=runner,
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    )


def _write_source(root: Path, content: bytes = b"value = 1\n") -> None:
    path = root / "src" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)


def test_static_contracts_are_strict_frozen_and_outcome_consistent(tmp_path: Path) -> None:
    finding = StaticFinding(
        check=StaticCheck.RUFF,
        path="src/example.py",
        line=1,
        column=1,
        rule_code="F401",
        message="ruff reported a finding",
    )
    check = StaticCheckResult(
        check=StaticCheck.RUFF,
        status=StaticCheckStatus.FAILED,
        findings=(finding,),
    )
    with pytest.raises(ValidationError):
        StaticCheckResult(
            check=StaticCheck.RUFF,
            status=StaticCheckStatus.PASSED,
            findings=(finding,),
        )
    with pytest.raises(ValidationError):
        StaticFinding(
            check=StaticCheck.RUFF,
            path="../escape.py",
            line=1,
            column=1,
            rule_code="F401",
            message="ruff reported a finding",
        )
    with pytest.raises(ValidationError):
        StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=(check,),
            candidate=None,
            diagnostic=None,
        )
    diagnostic = StaticDiagnostic(
        code=StaticDiagnosticCode.INVALID_INPUT,
        message="static validation input is invalid",
    )
    with pytest.raises(ValidationError):
        diagnostic.message = "changed"  # type: ignore[misc]
    assert tmp_path.exists()


def test_exact_modify_passes_in_fresh_candidate_and_preserves_source(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.PASSED
    assert result.candidate is not None
    assert result.candidate.repository != source
    assert (result.candidate.repository / "src" / "example.py").read_bytes() == b"value = 2\n"
    assert (source / "src" / "example.py").read_bytes() == b"value = 1\n"
    assert runner.calls[0][1] == ("src/example.py",)
    assert tuple(item.check for item in result.checks) == tuple(StaticCheck)


@pytest.mark.parametrize(
    ("source_bytes", "diff", "expected"),
    [
        (
            b"value = 1\r\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
            b"value = 2\r\n",
        ),
        (
            b"value = 1",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n\\ No newline at end of file\n+value = 2\n\\ No newline at end of file\n",
            b"value = 2",
        ),
    ],
)
def test_modify_preserves_consistent_newline_and_final_newline(
    tmp_path: Path, source_bytes: bytes, diff: str, expected: bytes
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, source_bytes)
    request, filesystem, fingerprinter = _request(source, workspace, _proposal(diff))

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.candidate is not None
    assert (result.candidate.repository / "src" / "example.py").read_bytes() == expected


def test_exact_add_and_delete_are_supported(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "src").mkdir()
    (source / "src" / "old.py").write_bytes(b"old = True\n")
    proposal = _proposal(
        "--- /dev/null\n+++ b/src/new.py\n@@ -0,0 +1 @@\n+new = True\n"
        "--- a/src/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old = True\n",
        ["src/new.py", "src/old.py"],
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.candidate is not None
    assert (result.candidate.repository / "src" / "new.py").read_bytes() == b"new = True\n"
    assert not (result.candidate.repository / "src" / "old.py").exists()


def test_multiple_exact_hunks_apply_without_offset_search(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, b"one\ntwo\nthree\nfour\n")
    proposal = _proposal(
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1 +1 @@\n"
        "-one\n"
        "+ONE\n"
        "@@ -4 +4 @@\n"
        "-four\n"
        "+FOUR\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.candidate is not None
    assert (result.candidate.repository / "src" / "example.py").read_bytes() == (
        b"ONE\ntwo\nthree\nFOUR\n"
    )


def test_zero_count_insertion_applies_at_exact_coordinate(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, b"one\nthree\n")
    request, filesystem, fingerprinter = _request(
        source,
        workspace,
        _proposal("--- a/src/example.py\n+++ b/src/example.py\n@@ -1,0 +2 @@\n+two\n"),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.candidate is not None
    assert (result.candidate.repository / "src" / "example.py").read_bytes() == (
        b"one\ntwo\nthree\n"
    )


@pytest.mark.parametrize(
    ("source_bytes", "diff"),
    [
        (
            b"one\ntwo\n",
            "--- a/src/example.py\n+++ b/src/example.py\n"
            "@@ -1 +1 @@\n-one\n+ONE\n@@ -1 +1 @@\n-one\n+AGAIN\n",
        ),
        (
            b"one\rbreak\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-one\n+ONE\n",
        ),
        (
            b"one\ntwo\n",
            "--- a/src/example.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-one\n",
        ),
    ],
    ids=("overlapping-hunks", "lone-cr", "nonempty-deletion"),
)
def test_ambiguous_or_partial_patch_shapes_fail_atomically(
    tmp_path: Path, source_bytes: bytes, diff: str
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, source_bytes)
    request, filesystem, fingerprinter = _request(source, workspace, _proposal(diff))

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert (source / "src" / "example.py").read_bytes() == source_bytes
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize(
    ("diff", "paths"),
    [
        (
            "--- /dev/null\n+++ b/src/example.py\n@@ -0,0 +1 @@\n+new\n",
            ["src/example.py"],
        ),
        (
            "--- a/src/missing.py\n+++ b/src/missing.py\n@@ -1 +1 @@\n-old\n+new\n",
            ["src/missing.py"],
        ),
    ],
    ids=("added-target-exists", "modified-target-missing"),
)
def test_patch_target_type_mismatch_fails_atomically(
    tmp_path: Path, diff: str, paths: list[str]
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    request, filesystem, fingerprinter = _request(source, workspace, _proposal(diff, paths))

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert (source / "src" / "example.py").read_bytes() == b"value = 1\n"
    assert list(workspace.iterdir()) == []


def test_late_multi_file_conflict_does_not_publish_partial_application(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "a.py").write_bytes(b"a = 1\n")
    (source / "b.py").write_bytes(b"b = 1\n")
    request, filesystem, fingerprinter = _request(
        source,
        workspace,
        _proposal(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-wrong\n+b = 2\n",
            ["a.py", "b.py"],
        ),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert (source / "a.py").read_bytes() == b"a = 1\n"
    assert (source / "b.py").read_bytes() == b"b = 1\n"
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize(
    ("content", "diff"),
    [
        (
            b"value = 1\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-wrong\n+value = 2\n",
        ),
        (
            b"value = 1\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 1\n",
        ),
        (
            b"first\r\nsecond\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-first\n+changed\n",
        ),
        (
            b"\xff\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n",
        ),
    ],
)
def test_patch_mismatch_noop_mixed_newline_and_invalid_utf8_fail_atomically(
    tmp_path: Path, content: bytes, diff: str
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, content)
    request, filesystem, fingerprinter = _request(source, workspace, _proposal(diff))

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert list(workspace.iterdir()) == []
    assert (source / "src" / "example.py").read_bytes() == content


@pytest.mark.parametrize(
    "metadata",
    [
        "diff --git a/src/example.py b/src/example.py\nold mode 100644\nnew mode 100755\n",
        "similarity index 100%\nrename from src/old.py\nrename to src/example.py\n",
        "similarity index 100%\ncopy from src/original.py\ncopy to src/example.py\n",
        "GIT binary patch\n",
    ],
    ids=("mode", "rename", "copy", "binary"),
)
def test_unsupported_diff_metadata_is_rejected_before_staging(
    tmp_path: Path, metadata: str
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    original = b"value = 1\n"
    _write_source(source, original)
    proposal = _proposal(
        metadata
        + "--- a/src/example.py\n"
        + "+++ b/src/example.py\n"
        + "@@ -1 +1 @@\n"
        + "-value = 1\n"
        + "+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.INVALID_INPUT
    assert result.candidate is None
    assert (source / "src" / "example.py").read_bytes() == original
    assert list(workspace.iterdir()) == []
    assert runner.calls == []


def test_rejected_policy_blocks_before_workspace_mutation(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    object.__setattr__(request.policy_result, "approved", False)
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.INVALID_INPUT
    assert list(workspace.iterdir()) == []
    assert runner.calls == []


def test_hard_link_in_prepared_tree_is_rejected(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    filesystem = LocalStaticFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    fingerprint = fingerprinter.fingerprint(source.resolve())
    os.link(source / "src" / "example.py", source / "src" / "alias.py")
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprint,
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=proposal,
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.candidate is None


def test_non_python_patch_marks_all_language_checks_not_applicable(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "README.md").write_text("old\n", encoding="utf-8")
    proposal = _proposal(
        "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n",
        ["README.md"],
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.PASSED
    assert {item.status for item in result.checks[2:]} == {StaticCheckStatus.NOT_APPLICABLE}
    assert runner.calls == []


def test_deletion_only_python_patch_is_not_applicable_to_language_tools(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    request, filesystem, fingerprinter = _request(
        source,
        workspace,
        _proposal("--- a/src/example.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-value = 1\n"),
    )
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.PASSED
    assert {item.status for item in result.checks[2:]} == {StaticCheckStatus.NOT_APPLICABLE}
    assert runner.calls == []


def test_multiple_independent_tool_findings_are_aggregated_and_block(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)

    result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=FindingStaticRunner(),
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    ).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert tuple(
        item.check for item in result.checks if item.status is StaticCheckStatus.FAILED
    ) == (
        StaticCheck.RUFF,
        StaticCheck.MYPY,
        StaticCheck.BANDIT,
    )
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("failed_tool", list(StaticToolCheck))
def test_each_container_check_can_fail_independently(
    tmp_path: Path, failed_tool: StaticToolCheck
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=SingleFindingStaticRunner(failed_tool),
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    ).validate(request)

    expected = StaticCheck(failed_tool.value)
    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert tuple(
        item.check for item in result.checks if item.status is StaticCheckStatus.FAILED
    ) == (expected,)


def test_runner_failure_is_distinct_infrastructure_diagnostic_and_cleans(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)

    result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=FailingStaticRunner(),
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    ).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.STATIC_RUNNER_FAILED
    assert list(workspace.iterdir()) == []


def test_candidate_and_pristine_mutation_are_detected(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)

    result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=MutatingStaticRunner(),
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    ).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.INTEGRITY_FAILED
    assert list(workspace.iterdir()) == []


def test_undeclared_actual_change_is_candidate_failure(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    filesystem = ExtraChangeFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprinter.fingerprint(source.resolve()),
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=_proposal(
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        ),
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.checks[1].status is StaticCheckStatus.FAILED
    assert list(workspace.iterdir()) == []


def test_cleanup_failure_is_visible_and_blocking(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    filesystem = CleanupFailingFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprinter.fingerprint(source.resolve()),
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=_proposal(
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-wrong\n+value = 2\n"
        ),
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.CLEANUP_FAILED


def test_mutated_preparation_fingerprint_blocks_before_staging(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    (source / "src" / "example.py").write_text("changed = True\n", encoding="utf-8")
    runner = PassingStaticRunner()

    result = _service(filesystem, fingerprinter, runner).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.PREPARATION_INVALID
    assert runner.calls == []
    assert list(workspace.iterdir()) == []


def test_staging_collision_and_unsafe_identifier_fail_without_deleting_existing_data(
    tmp_path: Path,
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    identifier = "0123456789abcdef0123456789abcdef"
    collision = workspace / f".tracefix-static-{identifier}.staging"
    collision.mkdir()
    marker = collision / "owned.txt"
    marker.write_text("caller data", encoding="utf-8")

    collision_result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)
    unsafe_result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=PassingStaticRunner(),
        identifier_factory=lambda: "../unsafe",
    ).validate(request)

    assert collision_result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert unsafe_result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert marker.read_text(encoding="utf-8") == "caller data"
    assert list(workspace.iterdir()) == [collision]


def test_protected_project_source_and_workspace_are_rejected(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "app"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    filesystem = LocalStaticFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    proposal = _proposal(
        "--- a/__init__.py\n+++ b/__init__.py\n@@ -0,0 +1 @@\n+value = 1\n",
        ["__init__.py"],
    )
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprinter.fingerprint(source.resolve()),
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=proposal,
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.PREPARATION_INVALID


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(2)])
def test_process_control_exceptions_are_not_swallowed(
    tmp_path: Path, interrupt: BaseException
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    proposal = _proposal(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    request, filesystem, fingerprinter = _request(source, workspace, proposal)
    service = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=InterruptingStaticRunner(interrupt),
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    )

    with pytest.raises(type(interrupt)):
        service.validate(request)


def test_static_result_rejects_impossible_applicability_topologies(tmp_path: Path) -> None:
    candidate = ValidatedCandidate(
        repository=tmp_path.resolve(),
        fingerprint=RepositoryFingerprint(digest="0" * 64),
    )
    passed = tuple(
        StaticCheckResult(check=check, status=StaticCheckStatus.PASSED) for check in StaticCheck
    )
    all_not_applicable = (
        *passed[:2],
        *(
            StaticCheckResult(check=check, status=StaticCheckStatus.NOT_APPLICABLE)
            for check in tuple(StaticCheck)[2:]
        ),
    )

    assert (
        StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=passed,
            candidate=candidate,
        ).outcome
        is StaticValidationOutcome.PASSED
    )
    assert (
        StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=all_not_applicable,
            candidate=candidate,
        ).outcome
        is StaticValidationOutcome.PASSED
    )
    with pytest.raises(ValidationError):
        StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=(
                *passed[:2],
                StaticCheckResult(
                    check=StaticCheck.PYTHON_COMPILE,
                    status=StaticCheckStatus.NOT_APPLICABLE,
                ),
                *passed[3:],
            ),
            candidate=candidate,
        )
    failed_finding = StaticFinding(
        check=StaticCheck.RUFF,
        path="example.py",
        line=1,
        column=1,
        rule_code="F401",
        message="ruff reported a finding",
    )
    with pytest.raises(ValidationError):
        StaticValidationResult(
            outcome=StaticValidationOutcome.CANDIDATE_FAILED,
            checks=(
                *passed[:2],
                StaticCheckResult(
                    check=StaticCheck.PYTHON_COMPILE,
                    status=StaticCheckStatus.NOT_APPLICABLE,
                ),
                passed[3],
                StaticCheckResult(
                    check=StaticCheck.RUFF,
                    status=StaticCheckStatus.FAILED,
                    findings=(failed_finding,),
                ),
                *passed[5:],
            ),
        )
    with pytest.raises(ValidationError):
        StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=(
                StaticCheckResult(
                    check=StaticCheck.PATCH_APPLY,
                    status=StaticCheckStatus.NOT_APPLICABLE,
                ),
                *passed[1:],
            ),
            candidate=candidate,
        )


def test_valid_injected_runner_evidence_survives_strict_revalidation() -> None:
    result = PassingStaticRunner().execute(Path.cwd(), ("src/example.py",))

    assert (
        _revalidate_container_result(result, python_paths_present=True).model_dump()
        == result.model_dump()
    )


@pytest.mark.parametrize(
    "runner",
    [NotApplicableStaticRunner(), MutatedEvidenceStaticRunner()],
    ids=("not-applicable-with-python", "mutated-evidence"),
)
def test_malformed_injected_runner_evidence_is_blocking(
    tmp_path: Path, runner: StaticContainerRunner
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    request, filesystem, fingerprinter = _request(
        source,
        workspace,
        _proposal(
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        ),
    )

    result = StaticValidationService(
        filesystem=filesystem,
        fingerprinter=fingerprinter,
        runner=runner,
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    ).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.STATIC_RUNNER_FAILED
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize(
    ("source_bytes", "diff"),
    [
        (
            b"old",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n"
            "-old\n\\ No newline at end of file\n+new\n",
        ),
        (
            b"old\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n"
            "-old\n+new\n\\ No newline at end of file\n",
        ),
        (
            b"one\ntwo\n",
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1,2 +1,2 @@\n"
            "-one\n\\ No newline at end of file\n+ONE\n two\n",
        ),
    ],
)
def test_unsupported_or_non_eof_newline_marker_changes_fail_atomically(
    tmp_path: Path, source_bytes: bytes, diff: str
) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, source_bytes)
    request, filesystem, fingerprinter = _request(source, workspace, _proposal(diff))

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.CANDIDATE_FAILED
    assert result.candidate is None
    assert (source / "src" / "example.py").read_bytes() == source_bytes
    assert list(workspace.iterdir()) == []


def test_patch_before_unterminated_eof_preserves_existing_newline_state(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source, b"one\ntwo")
    request, filesystem, fingerprinter = _request(
        source,
        workspace,
        _proposal("--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-one\n+ONE\n"),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.candidate is not None
    assert (result.candidate.repository / "src" / "example.py").read_bytes() == b"ONE\ntwo"


def test_cleanup_refuses_a_replaced_task_directory(tmp_path: Path) -> None:
    filesystem = LocalStaticFilesystem()
    target = tmp_path / "candidate"
    identity = filesystem.make_directory(target)
    original = tmp_path / "original"
    target.rename(original)
    target.mkdir()

    with pytest.raises(StaticFilesystemError):
        filesystem.remove_tree(target, identity)

    assert target.is_dir()
    target.rmdir()
    shutil.rmtree(original)


def test_local_filesystem_identity_preserves_normal_copy_write_publish_flow(
    tmp_path: Path,
) -> None:
    filesystem = LocalStaticFilesystem()
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    pristine = filesystem.snapshot(source)
    staging = workspace / "staging"
    published = workspace / "published"

    identity = filesystem.make_directory(staging)
    filesystem.copy_snapshot(pristine, staging, identity)
    filesystem.write_file(staging, identity, "src/example.py", b"value = 2\n", create=False)
    candidate = filesystem.snapshot(staging, identity)
    published_identity = filesystem.publish(staging, published, identity)

    assert filesystem.snapshot(published, published_identity) == candidate
    filesystem.remove_tree(published, published_identity)
    assert not published.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission behavior")
def test_local_filesystem_candidate_is_readable_by_non_owner_container(tmp_path: Path) -> None:
    filesystem = LocalStaticFilesystem()
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    staging = workspace / "staging"

    identity = filesystem.make_directory(staging)
    filesystem.copy_snapshot(filesystem.snapshot(source), staging, identity)
    filesystem.write_file(staging, identity, "new/package.py", b"value = 1\n", create=True)

    for directory in (staging, staging / "src", staging / "new"):
        mode = directory.stat().st_mode
        assert mode & stat.S_IROTH
        assert mode & stat.S_IXOTH
        assert not mode & stat.S_IWOTH
    for file in (staging / "src" / "example.py", staging / "new" / "package.py"):
        mode = file.stat().st_mode
        assert mode & stat.S_IROTH
        assert not mode & stat.S_IWOTH


def test_write_rejects_root_replaced_by_filesystem_alias(tmp_path: Path) -> None:
    filesystem = LocalStaticFilesystem()
    target = tmp_path / "candidate"
    identity = filesystem.make_directory(target)
    original = tmp_path / "original"
    outside = tmp_path / "outside"
    outside.mkdir()
    target.rename(original)
    try:
        target.symlink_to(outside, target_is_directory=True)
    except OSError:
        shutil.rmtree(original)
        pytest.skip("directory symlink creation is unavailable on this host")

    with pytest.raises(StaticFilesystemError):
        filesystem.write_file(target, identity, "escaped.py", b"bad\n", create=True)

    assert not (outside / "escaped.py").exists()
    target.unlink()
    shutil.rmtree(original)


def test_snapshot_rejects_symlink_hardlink_case_alias_and_special_entries(
    tmp_path: Path,
) -> None:
    filesystem = LocalStaticFilesystem()

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    (symlink_root / "target.py").write_bytes(b"value = 1\n")
    try:
        (symlink_root / "alias.py").symlink_to(symlink_root / "target.py")
    except OSError:
        pass
    else:
        with pytest.raises(StaticFilesystemError):
            filesystem.snapshot(symlink_root)

    hardlink_root = tmp_path / "hardlink"
    hardlink_root.mkdir()
    (hardlink_root / "target.py").write_bytes(b"value = 1\n")
    os.link(hardlink_root / "target.py", hardlink_root / "alias.py")
    with pytest.raises(StaticFilesystemError):
        filesystem.snapshot(hardlink_root)

    if os.name != "nt":
        case_root = tmp_path / "case"
        case_root.mkdir()
        (case_root / "A.py").write_bytes(b"a = 1\n")
        (case_root / "a.py").write_bytes(b"a = 2\n")
        with pytest.raises(StaticFilesystemError):
            filesystem.snapshot(case_root)

        fifo_root = tmp_path / "fifo"
        fifo_root.mkdir()
        os.mkfifo(fifo_root / "pipe")
        with pytest.raises(StaticFilesystemError):
            filesystem.snapshot(fifo_root)


def test_publish_time_mutation_is_rejected_and_removed(tmp_path: Path) -> None:
    source, workspace = tmp_path / "prepared", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    _write_source(source)
    filesystem = PublishingMutationFilesystem()
    fingerprinter = Sha256RepositoryFingerprinter(filesystem)
    preparation = RepositoryPreparation(
        case_id="case",
        requested_revision="revision",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=source.resolve(),
        fingerprint=fingerprinter.fingerprint(source.resolve()),
    )
    request = StaticValidationRequest(
        preparation=preparation,
        proposal=_proposal(
            "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        ),
        policy_result=_approved(),
        candidate_workspace_root=workspace.resolve(),
    )

    result = _service(filesystem, fingerprinter, PassingStaticRunner()).validate(request)

    assert result.outcome is StaticValidationOutcome.INFRASTRUCTURE_DIAGNOSTIC
    assert result.diagnostic is not None
    assert result.diagnostic.code is StaticDiagnosticCode.INTEGRITY_FAILED
    assert list(workspace.iterdir()) == []
