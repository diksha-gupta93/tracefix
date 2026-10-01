from __future__ import annotations

import json
import traceback
import warnings
from pathlib import PurePosixPath

import pytest
from pydantic import ValidationError

from app.agent.policy import (
    MAX_CHANGED_FILES,
    PatchPolicy,
    PatchPolicyInputDiagnosticCode,
    PatchPolicyInputError,
    PatchPolicyResult,
    PatchPolicyViolation,
    PatchPolicyViolationCode,
    validate_patch_policy,
)
from app.agent.schemas import DEFAULT_PROTECTED_PATHS, PatchProposal, ProtectedPathPolicy

PROTECTED_PATHS = DEFAULT_PROTECTED_PATHS


def policy(
    *,
    forbidden: tuple[str, ...] = (),
    allowlist: tuple[str, ...] = (),
) -> PatchPolicy:
    return PatchPolicy(
        protected_path_policy=ProtectedPathPolicy(protected_paths=PROTECTED_PATHS),
        forbidden_changed_files=tuple(PurePosixPath(value) for value in forbidden),
        dependency_allowlist=allowlist,
    )


def diff_section(
    path: str,
    *,
    removed: tuple[str, ...] = ("old = 1",),
    added: tuple[str, ...] = ("new = 2",),
    deleted: bool = False,
    created: bool = False,
) -> str:
    old_header = "/dev/null" if created else f"a/{path}"
    new_header = "/dev/null" if deleted else f"b/{path}"
    old_start = 0 if created else 1
    new_start = 0 if deleted else 1
    old_count = 0 if created else len(removed)
    new_count = 0 if deleted else len(added)
    body = tuple(f"-{line}" for line in (() if created else removed)) + tuple(
        f"+{line}" for line in (() if deleted else added)
    )
    return "\n".join(
        (
            f"--- {old_header}",
            f"+++ {new_header}",
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@",
            *body,
        )
    )


def proposal(
    *,
    files: tuple[str, ...] = ("src/example.py",),
    diff: str | None = None,
    summary: str = "Correct the defect",
    root_cause: str = "The boundary was wrong",
    expected_effect: str = "The boundary is corrected",
    risks: tuple[str, ...] = (),
) -> PatchProposal:
    rendered = diff or "\n".join(diff_section(path) for path in files)
    return PatchProposal(
        summary=summary,
        root_cause=root_cause,
        files_changed=list(files),
        unified_diff=rendered,
        expected_effect=expected_effect,
        risks=list(risks),
        confidence=0.9,
    )


def codes(result: PatchPolicyResult) -> tuple[PatchPolicyViolationCode, ...]:
    return tuple(item.code for item in result.violations)


def test_valid_single_file_patch_passes_without_mutation() -> None:
    candidate = proposal()
    before = candidate.model_dump()

    result = validate_patch_policy(candidate, policy())

    assert result == PatchPolicyResult(approved=True, violations=())
    assert candidate.model_dump() == before


def test_valid_three_file_patch_passes_and_serializes_deterministically() -> None:
    files = ("src/a.py", "src/b.py", "src/c.py")
    result = validate_patch_policy(proposal(files=files), policy())

    assert result.approved is True
    assert result.model_dump_json() == '{"approved":true,"violations":[]}'


def test_policy_contracts_are_strict_frozen_and_extra_forbidding() -> None:
    current = policy()
    result = validate_patch_policy(proposal(), current)
    with pytest.raises(ValidationError):
        current.max_changed_files = 4
    with pytest.raises(ValidationError):
        result.approved = False
    with pytest.raises(ValidationError):
        PatchPolicyResult.model_validate({"approved": True, "violations": (), "extra": 1})
    with pytest.raises(ValidationError):
        PatchPolicyResult(approved=True, violations=())
        PatchPolicyViolation(
            code=PatchPolicyViolationCode.SECRET_DETECTED,
            message="unsafe caller text",
        )


@pytest.mark.parametrize("protected", PROTECTED_PATHS)
def test_protected_path_modified_rejects_every_architecture_default(
    protected: PurePosixPath,
) -> None:
    path = protected.as_posix()
    changed = path if protected.suffix == ".lock" else f"{path}/nested.py"
    result = validate_patch_policy(proposal(files=(changed,)), policy())

    assert PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED in codes(result)


def test_protected_path_matching_is_component_aware_case_insensitive_and_trusted() -> None:
    trusted = policy(forbidden=("generated/forbidden.py",))
    exact = validate_patch_policy(proposal(files=("GENERATED/FORBIDDEN.PY",)), trusted)
    below = validate_patch_policy(proposal(files=("generated/forbidden.py/child",)), trusted)
    near = validate_patch_policy(proposal(files=("dockerized/module.py",)), policy())

    assert PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED in codes(exact)
    assert PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED in codes(below)
    assert PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED not in codes(near)


def test_too_many_files_rejects_four_but_not_three() -> None:
    three = tuple(f"src/file_{index}.py" for index in range(MAX_CHANGED_FILES))
    four = three + ("src/file_3.py",)

    assert PatchPolicyViolationCode.TOO_MANY_FILES not in codes(
        validate_patch_policy(proposal(files=three), policy())
    )
    assert PatchPolicyViolationCode.TOO_MANY_FILES in codes(
        validate_patch_policy(proposal(files=four), policy())
    )


def test_line_change_limit_rejects_201_but_not_200_and_counts_replacements_twice() -> None:
    two_hundred = diff_section(
        "src/example.py",
        removed=tuple(f"old_{index}" for index in range(100)),
        added=tuple(f"new_{index}" for index in range(100)),
    )
    two_hundred_one = diff_section(
        "src/example.py",
        removed=tuple(f"old_{index}" for index in range(100)),
        added=tuple(f"new_{index}" for index in range(101)),
    )

    assert PatchPolicyViolationCode.LINE_CHANGE_LIMIT_EXCEEDED not in codes(
        validate_patch_policy(proposal(diff=two_hundred), policy())
    )
    assert PatchPolicyViolationCode.LINE_CHANGE_LIMIT_EXCEEDED in codes(
        validate_patch_policy(proposal(diff=two_hundred_one), policy())
    )


def test_context_headers_and_no_newline_marker_do_not_count_as_changes() -> None:
    body = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,101 +1,101 @@\n"
        " unchanged context\n"
        + "\n".join(f"-old_{index}" for index in range(100))
        + "\n"
        + "\n".join(f"+new_{index}" for index in range(100))
        + "\n\\ No newline at end of file"
    )
    result = validate_patch_policy(proposal(diff=body), policy())

    assert PatchPolicyViolationCode.LINE_CHANGE_LIMIT_EXCEEDED not in codes(result)


@pytest.mark.parametrize(
    ("path", "diff"),
    [
        (
            "src/image.bin",
            "--- a/src/image.bin\n+++ b/src/image.bin\nGIT binary patch\nliteral 1\nA",
        ),
        (
            "src/image.bin",
            "--- a/src/image.bin\n+++ b/src/image.bin\n"
            "Binary files a/src/image.bin and b/src/image.bin differ",
        ),
        ("src/example.py", diff_section("src/example.py").replace("old = 1", "old = \x00")),
        ("src/example.py", diff_section("src/example.py").replace("old = 1", "old = \x07")),
    ],
)
def test_binary_content_rejects_markers_nul_and_control_data(path: str, diff: str) -> None:
    result = validate_patch_policy(proposal(files=(path,), diff=diff), policy())

    assert result.approved is False
    assert PatchPolicyViolationCode.BINARY_CONTENT in codes(result)


def test_binary_content_does_not_hide_other_detectable_violations() -> None:
    diff = diff_section("src/example.py", added=("import subprocess\x00",))

    result = validate_patch_policy(proposal(diff=diff), policy())

    assert PatchPolicyViolationCode.BINARY_CONTENT in codes(result)
    assert PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED in codes(result)


def test_binary_marker_still_requires_consistent_safe_headers() -> None:
    diff = "--- a/src/image.bin\n+++ b/src/image.bin\nGIT binary patch\nliteral 1\nA"

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(proposal(files=("src/other.bin",), diff=diff), policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INCONSISTENT_PATHS


@pytest.mark.parametrize(
    "secret",
    [
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "AKIAABCDEFGHIJKLMNOP",
        "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "gho_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "ghu_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "ghs_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "ghr_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "github_pat_" + "A" * 70,
        "client_secret=synthetic-secret-value",
        "password: synthetic-password-value",
        "private_key=synthetic-private-value",
    ],
)
def test_secret_detected_rejects_each_supported_synthetic_signature(secret: str) -> None:
    result = validate_patch_policy(proposal(summary=secret), policy())

    assert codes(result) == (PatchPolicyViolationCode.SECRET_DETECTED,)
    rendered = result.model_dump_json() + repr(result)
    assert secret not in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_cause", "api_key=synthetic-api-key-value"),
        ("expected_effect", "access_token=synthetic-access-token"),
        ("risks", ("password=synthetic-password-value",)),
        (
            "diff",
            diff_section("src/example.py", added=("private_key=synthetic-private-value",)),
        ),
    ],
)
def test_secret_scanning_covers_all_proposal_prose(field: str, value: object) -> None:
    arguments: dict[str, object] = {field: value}
    result = validate_patch_policy(proposal(**arguments), policy())  # type: ignore[arg-type]
    assert PatchPolicyViolationCode.SECRET_DETECTED in codes(result)


@pytest.mark.parametrize(
    "placeholder",
    ["api_key=placeholder", "password=changeme", "access_token=${TOKEN}", "private_key=<key>"],
)
def test_secret_placeholders_and_near_misses_pass(placeholder: str) -> None:
    assert validate_patch_policy(proposal(summary=placeholder), policy()).approved is True


@pytest.mark.parametrize(
    "secret",
    [
        '{"password": "synthetic-secret-value"}',
        "'client_secret': 'synthetic-secret-value'",
        'password = (\n    "synthetic-secret-value"\n)',
    ],
)
def test_quoted_and_multiline_secret_assignments_are_rejected(secret: str) -> None:
    assert codes(validate_patch_policy(proposal(summary=secret), policy())) == (
        PatchPolicyViolationCode.SECRET_DETECTED,
    )


def test_secret_bearing_path_is_never_returned_as_metadata() -> None:
    secret_path = "security/password=synthetic-secret-value.py"

    result = validate_patch_policy(
        proposal(
            files=(secret_path,), diff=diff_section(secret_path, added=("import subprocess",))
        ),
        policy(),
    )
    rendered = result.model_dump_json() + repr(result)

    assert PatchPolicyViolationCode.SECRET_DETECTED in codes(result)
    assert all(item.path is None and item.line is None for item in result.violations)
    assert secret_path not in rendered


def test_test_disabled_rejects_deleted_test_file() -> None:
    path = "tests/test_example.py"
    diff = diff_section(path, removed=("def test_value():", "    pass"), deleted=True)
    result = validate_patch_policy(proposal(files=(path,), diff=diff), policy())

    assert PatchPolicyViolationCode.TEST_DISABLED in codes(result)


@pytest.mark.parametrize("path", ["test_example.py", "src/example_test.py"])
def test_test_disabled_recognizes_python_test_filename_forms(path: str) -> None:
    diff = diff_section(path, removed=("def test_value():", "    pass"), deleted=True)
    result = validate_patch_policy(proposal(files=(path,), diff=diff), policy())
    assert PatchPolicyViolationCode.TEST_DISABLED in codes(result)


def test_test_disabled_rejects_removed_test_definition() -> None:
    diff = diff_section(
        "src/example.py",
        removed=("def test_hidden_behavior():",),
        added=("def helper():",),
    )
    result = validate_patch_policy(proposal(diff=diff), policy())
    assert PatchPolicyViolationCode.TEST_DISABLED in codes(result)


@pytest.mark.parametrize(
    "added",
    [
        'pytest.skip("later")',
        "pytestmark = pytest.mark.skip",
        "pytestmark = pytest.mark.xfail(reason='known')",
        "@pytest.mark.skip",
        "@pytest.mark.skipif(True, reason='known')",
        "@pytest.mark.xfail",
        "@unittest.skip('known')",
        "@unittest.skipIf(True, 'known')",
    ],
)
def test_test_disabled_rejects_each_skip_xfail_construct(added: str) -> None:
    result = validate_patch_policy(
        proposal(
            files=("tests/test_example.py",),
            diff=diff_section("tests/test_example.py", added=(added,)),
        ),
        policy(),
    )
    assert PatchPolicyViolationCode.TEST_DISABLED in codes(result)


@pytest.mark.parametrize(
    "added",
    [
        "# pytest.skip('comment')",
        'message = "pytest.mark.xfail"',
        "ordinary_pytest_skip_identifier = True",
    ],
)
def test_test_disable_comments_strings_and_identifiers_do_not_trigger(added: str) -> None:
    result = validate_patch_policy(
        proposal(
            files=("tests/test_example.py",),
            diff=diff_section("tests/test_example.py", added=(added,)),
        ),
        policy(),
    )
    assert PatchPolicyViolationCode.TEST_DISABLED not in codes(result)


def test_multiline_test_disable_is_rejected_but_multiline_string_is_not() -> None:
    disabled = diff_section(
        "tests/test_example.py",
        added=("@pytest.mark.skipif(", "    True,", "    reason='known',", ")"),
    )
    harmless = diff_section(
        "tests/test_example.py",
        added=("message = (", '    "pytest.mark.skipif("', ")"),
    )

    assert PatchPolicyViolationCode.TEST_DISABLED in codes(
        validate_patch_policy(proposal(files=("tests/test_example.py",), diff=disabled), policy())
    )
    assert PatchPolicyViolationCode.TEST_DISABLED not in codes(
        validate_patch_policy(proposal(files=("tests/test_example.py",), diff=harmless), policy())
    )


@pytest.mark.parametrize(
    "removed",
    [
        "assert value == expected",
        "with pytest.raises(ValueError):",
        "with pytest.warns(RuntimeWarning):",
        "self.assertEqual(value, expected)",
        "with self.assertRaises(ValueError):",
    ],
)
def test_assertion_removed_rejects_supported_assertions(removed: str) -> None:
    result = validate_patch_policy(
        proposal(
            files=("tests/test_example.py",),
            diff=diff_section("tests/test_example.py", removed=(removed,)),
        ),
        policy(),
    )
    assert PatchPolicyViolationCode.ASSERTION_REMOVED in codes(result)


@pytest.mark.parametrize(
    "removed",
    [
        "self.assertLogs(logger)",
        "self.assertNoLogs(logger)",
        "self.assertStartsWith(value, prefix)",
        "self.assertNotStartsWith(value, prefix)",
        "self.assertEndsWith(value, suffix)",
        "self.assertNotEndsWith(value, suffix)",
        "self.assertHasAttr(value, name)",
        "self.assertNotHasAttr(value, name)",
        "self.assertIsSubclass(value, expected)",
        "self.assertNotIsSubclass(value, expected)",
        "unittest.TestCase.assertEqual(self, value, expected)",
    ],
)
def test_assertion_removed_covers_python314_unittest_assertions(removed: str) -> None:
    result = validate_patch_policy(
        proposal(
            files=("tests/test_example.py",),
            diff=diff_section("tests/test_example.py", removed=(removed,)),
        ),
        policy(),
    )

    assert PatchPolicyViolationCode.ASSERTION_REMOVED in codes(result)


def test_multiline_assertion_removal_is_rejected_but_similar_identifier_is_not() -> None:
    removed = diff_section(
        "tests/test_example.py",
        removed=("assert (", "    value == expected", ")"),
        added=("value = (", "    actual", ")"),
    )
    similar = diff_section(
        "tests/test_example.py",
        removed=("assertLogs_result = (", "    value", ")"),
        added=("value = (", "    actual", ")"),
    )

    assert PatchPolicyViolationCode.ASSERTION_REMOVED in codes(
        validate_patch_policy(proposal(files=("tests/test_example.py",), diff=removed), policy())
    )
    assert PatchPolicyViolationCode.ASSERTION_REMOVED not in codes(
        validate_patch_policy(proposal(files=("tests/test_example.py",), diff=similar), policy())
    )


@pytest.mark.parametrize(
    ("removed", "added"),
    [
        ("value = 1", "assert value"),
        ('text = "assert value"', "value = 2"),
        ("helper.assertion(value)", "value = 2"),
    ],
)
def test_added_context_and_similarly_named_assertions_do_not_trigger(
    removed: str, added: str
) -> None:
    result = validate_patch_policy(
        proposal(
            files=("tests/test_example.py",),
            diff=diff_section("tests/test_example.py", removed=(removed,), added=(added,)),
        ),
        policy(),
    )
    assert PatchPolicyViolationCode.ASSERTION_REMOVED not in codes(result)


def test_unchanged_assertion_context_does_not_trigger() -> None:
    diff = (
        "--- a/tests/test_example.py\n"
        "+++ b/tests/test_example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " assert value\n"
        "-old = 1\n"
        "+new = 2"
    )
    result = validate_patch_policy(proposal(files=("tests/test_example.py",), diff=diff), policy())
    assert PatchPolicyViolationCode.ASSERTION_REMOVED not in codes(result)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        ".github/actions/check/action.yml",
        ".gitlab-ci.yml",
        ".circleci/config.yml",
        "azure-pipelines.yml",
        "Jenkinsfile",
        "Dockerfile",
        "docker/runtime.yml",
        "security/policy.yml",
        "profiles/seccomp.json",
        ".bandit",
        "SECURITY.md",
    ],
)
def test_ci_or_security_config_modified_rejects_every_required_path_class(path: str) -> None:
    result = validate_patch_policy(proposal(files=(path,)), policy())
    assert PatchPolicyViolationCode.CI_OR_SECURITY_CONFIG_MODIFIED in codes(result)


def test_ci_security_path_near_misses_do_not_trigger() -> None:
    for path in ("src/dockerized.py", "docs/security-notes.txt", "src/jenkinsfile.py"):
        result = validate_patch_policy(proposal(files=(path,)), policy())
        assert PatchPolicyViolationCode.CI_OR_SECURITY_CONFIG_MODIFIED not in codes(result)


@pytest.mark.parametrize(
    "added",
    [
        "import subprocess",
        "from subprocess import run",
        "from os import system",
        "os.system('true')",
        "os.popen('true')",
        "os.spawnvp(os.P_WAIT, 'x', ['x'])",
        "subprocess.run(['true'])",
        "runner(command, shell=True)",
    ],
)
def test_unrestricted_subprocess_introduced_rejects_direct_signatures(added: str) -> None:
    result = validate_patch_policy(
        proposal(diff=diff_section("src/example.py", added=(added,))), policy()
    )
    assert PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED in codes(result)


@pytest.mark.parametrize(
    "added",
    [
        "# subprocess.run(['true'])",
        'message = "os.system"',
        "subprocess_result = value",
        "runner(command, shell=False)",
    ],
)
def test_subprocess_comments_strings_and_identifiers_do_not_trigger(added: str) -> None:
    result = validate_patch_policy(
        proposal(diff=diff_section("src/example.py", added=(added,))), policy()
    )
    assert PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED not in codes(result)


def test_multiline_subprocess_is_rejected_but_multiline_string_is_not() -> None:
    process = diff_section("src/example.py", added=("subprocess.run(", "    ['true'],", ")"))
    harmless = diff_section("src/example.py", added=("message = (", '    "subprocess.run("', ")"))

    assert PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED in codes(
        validate_patch_policy(proposal(diff=process), policy())
    )
    assert PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED not in codes(
        validate_patch_policy(proposal(diff=harmless), policy())
    )


@pytest.mark.parametrize(
    ("path", "added"),
    [
        ("requirements.txt", "Requests>=2"),
        ("requirements-dev.txt", "pytest==8.4"),
        ("pyproject.toml", 'dependencies = ["pydantic>=2"]'),
        ("setup.py", 'install_requires=["pydantic>=2"]'),
        ("setup.cfg", "pydantic>=2"),
        ("Pipfile", 'requests = "*"'),
        ("Pipfile.lock", '"requests": {'),
        ("poetry.lock", 'name = "requests"'),
        ("uv.lock", 'name = "requests"'),
    ],
)
def test_unapproved_dependency_added_rejects_supported_files(path: str, added: str) -> None:
    result = validate_patch_policy(
        proposal(files=(path,), diff=diff_section(path, added=(added,))), policy()
    )
    assert PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED in codes(result)


def test_dependency_names_use_pep503_normalization_and_allowlist() -> None:
    path = "requirements.txt"
    diff = diff_section(path, added=("My_Package.Name>=1",))

    rejected = validate_patch_policy(proposal(files=(path,), diff=diff), policy())
    approved = validate_patch_policy(
        proposal(files=(path,), diff=diff), policy(allowlist=("my-package-name",))
    )

    assert PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED in codes(rejected)
    assert approved.approved is True


def test_dependency_comments_whitespace_pass_and_ambiguous_syntax_fails_closed() -> None:
    path = "requirements.txt"
    harmless = diff_section(path, removed=("# old",), added=("# comment",))
    ambiguous = diff_section(path, added=("git+https://example.invalid/project.git",))

    assert validate_patch_policy(proposal(files=(path,), diff=harmless), policy()).approved is True
    assert PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED in codes(
        validate_patch_policy(proposal(files=(path,), diff=ambiguous), policy())
    )


def test_poetry_direct_dependency_table_is_rejected_but_project_metadata_is_not() -> None:
    path = "pyproject.toml"
    dependency = diff_section(path, added=('evil = {git = "https://example.invalid/evil"}',))
    metadata = diff_section(path, added=('authors = [{name = "Example Person"}]',))

    assert PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED in codes(
        validate_patch_policy(proposal(files=(path,), diff=dependency), policy())
    )
    assert validate_patch_policy(proposal(files=(path,), diff=metadata), policy()).approved is True


@pytest.mark.parametrize(
    "path",
    [
        "evaluator/test_answer.py",
        "case/hidden_tests/test_regression.py",
        "case/hidden-tests/test_regression.py",
        "case/hidden/tests/test_regression.py",
    ],
)
def test_hidden_test_modified_rejects_evaluator_and_hidden_test_variants(path: str) -> None:
    result = validate_patch_policy(proposal(files=(path,)), policy())
    assert PatchPolicyViolationCode.HIDDEN_TEST_MODIFIED in codes(result)


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/development/case/manifest.json",
        "benchmarks/development/case/metadata.json",
        "tools/benchmarks/case/evaluator/data.json",
    ],
)
def test_benchmark_metadata_modified_rejects_manifest_and_benchmark_tree(path: str) -> None:
    result = validate_patch_policy(proposal(files=(path,)), policy())
    assert PatchPolicyViolationCode.BENCHMARK_METADATA_MODIFIED in codes(result)


def test_benchmark_repository_source_path_is_not_metadata() -> None:
    path = "benchmarks/development/case/repository/src/module.py"
    result = validate_patch_policy(proposal(files=(path,)), policy())
    assert PatchPolicyViolationCode.BENCHMARK_METADATA_MODIFIED not in codes(result)


@pytest.mark.parametrize(
    "path",
    [
        "reference.patch",
        "evaluator/reference.patch",
        "case/evaluator/reference-patch.diff",
    ],
)
def test_reference_patch_modified_rejects_reference_variants(path: str) -> None:
    result = validate_patch_policy(proposal(files=(path,)), policy())
    assert PatchPolicyViolationCode.REFERENCE_PATCH_MODIFIED in codes(result)


def test_combined_findings_are_complete_distinct_deduplicated_and_stably_ordered() -> None:
    paths = (
        "security/policy.py",
        "tests/test_example.py",
        "evaluator/reference.patch",
        "requirements.txt",
    )
    diff = "\n".join(
        (
            diff_section(paths[0], added=("import subprocess",)),
            diff_section(paths[1], removed=("assert value",), added=("pytest.skip('x')",)),
            diff_section(paths[2]),
            diff_section(paths[3], added=("unsafe-package>=1",)),
        )
    )
    first = validate_patch_policy(proposal(files=paths, diff=diff), policy())
    second = validate_patch_policy(proposal(files=tuple(reversed(paths)), diff=diff), policy())

    assert first == second
    assert len(first.violations) == len(set(first.violations))
    assert codes(first) == tuple(sorted(codes(first), key=list(PatchPolicyViolationCode).index))
    assert {
        PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED,
        PatchPolicyViolationCode.TOO_MANY_FILES,
        PatchPolicyViolationCode.TEST_DISABLED,
        PatchPolicyViolationCode.ASSERTION_REMOVED,
        PatchPolicyViolationCode.CI_OR_SECURITY_CONFIG_MODIFIED,
        PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED,
        PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED,
        PatchPolicyViolationCode.HIDDEN_TEST_MODIFIED,
        PatchPolicyViolationCode.REFERENCE_PATCH_MODIFIED,
    } <= set(codes(first))


def test_lf_crlf_and_lone_cr_produce_equivalent_decisions() -> None:
    lf = proposal(diff=diff_section("src/example.py", added=("import subprocess",)))
    crlf = proposal(diff=lf.unified_diff.replace("\n", "\r\n"))
    cr = proposal(diff=lf.unified_diff.replace("\n", "\r"))

    assert validate_patch_policy(lf, policy()) == validate_patch_policy(crlf, policy())
    assert validate_patch_policy(lf, policy()) == validate_patch_policy(cr, policy())


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "src//example.py",
        "src/./example.py",
        "src/../example.py",
        "src\\example.py",
        "/absolute.py",
        "C:drive.py",
        "C:/drive.py",
        "//server/share.py",
        "src/control\n.py",
        "src/control\u0085.py",
    ],
)
def test_malformed_paths_fail_with_typed_safe_input_diagnostic(path: str) -> None:
    candidate = proposal()
    object.__setattr__(candidate, "files_changed", [path])

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(candidate, policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PATH
    if path:
        assert path not in repr(raised.value)


def test_case_aliased_and_diff_disagreement_paths_fail_closed() -> None:
    aliased = proposal(files=("src/example.py", "SRC/EXAMPLE.PY"))
    with pytest.raises(PatchPolicyInputError) as alias_error:
        validate_patch_policy(aliased, policy())
    assert alias_error.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PATH

    mismatched = proposal(files=("src/other.py",), diff=diff_section("src/example.py"))
    with pytest.raises(PatchPolicyInputError) as mismatch_error:
        validate_patch_policy(mismatched, policy())
    assert mismatch_error.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INCONSISTENT_PATHS


def test_printable_unicode_path_next_to_c1_range_remains_valid() -> None:
    path = "src/printable\u00a0name.py"
    assert validate_patch_policy(proposal(files=(path,)), policy()).approved is True


@pytest.mark.parametrize(
    "diff",
    [
        "not a diff",
        "--- a/src/example.py\n+++ b/src/example.py",
        "--- a/src/example.py\n+++ b/src/example.py\n@@ malformed @@\n-old\n+new",
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1,2 +1,1 @@\n-old\n+new",
    ],
)
def test_malformed_diff_fails_with_distinct_typed_diagnostic(diff: str) -> None:
    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(proposal(diff=diff), policy())
    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_DIFF
    assert diff not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize(
    "header",
    [
        "@@ -" + "9" * 5000 + ",1 +1,1 @@",
        "@@ -0,1 +1,1 @@",
        "@@ -1,1 +0,1 @@",
    ],
)
def test_adversarial_hunk_coordinates_fail_with_typed_diagnostic(header: str) -> None:
    diff = f"--- a/src/example.py\n+++ b/src/example.py\n{header}\n-old\n+new"

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(proposal(diff=diff), policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_DIFF


def test_zero_start_with_zero_count_remains_a_valid_created_file_hunk() -> None:
    path = "src/created.py"
    candidate = proposal(files=(path,), diff=diff_section(path, added=("value = 1",), created=True))

    assert validate_patch_policy(candidate, policy()).approved is True


def test_adversarial_python_expression_fails_with_typed_diagnostic() -> None:
    diff = diff_section("src/example.py", added=("+" * 5000 + "1",))

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(proposal(diff=diff), policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_DIFF


def test_wrong_mutated_proposal_and_policy_types_fail_closed() -> None:
    with pytest.raises(PatchPolicyInputError) as wrong_proposal:
        validate_patch_policy(object(), policy())  # type: ignore[arg-type]
    assert wrong_proposal.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL

    candidate = proposal()
    object.__setattr__(candidate, "confidence", 2.0)
    with pytest.raises(PatchPolicyInputError) as mutated_proposal:
        validate_patch_policy(candidate, policy())
    assert mutated_proposal.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL

    current = policy()
    object.__setattr__(current, "max_changed_files", 4)
    with pytest.raises(PatchPolicyInputError) as mutated_policy:
        validate_patch_policy(proposal(), current)
    assert mutated_policy.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_POLICY


def test_mutated_whitespace_path_is_rejected_instead_of_rewritten() -> None:
    candidate = proposal()
    object.__setattr__(candidate, "files_changed", [" src/example.py "])

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(candidate, policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PATH


def test_model_subclasses_are_rejected_without_invoking_overridden_serializers() -> None:
    called = False

    class UnsafeProposal(PatchProposal):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            nonlocal called
            called = True
            raise RuntimeError("serializer must not run")

    class UnsafePolicy(PatchPolicy):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            nonlocal called
            called = True
            raise RuntimeError("serializer must not run")

    unsafe = UnsafeProposal.model_validate(proposal().model_dump())
    unsafe_policy = UnsafePolicy.model_validate(policy().model_dump())

    with pytest.raises(PatchPolicyInputError) as raised:
        validate_patch_policy(unsafe, policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL
    assert called is False

    with pytest.raises(PatchPolicyInputError) as policy_error:
        validate_patch_policy(proposal(), unsafe_policy)

    assert policy_error.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_POLICY
    assert called is False


def test_mutated_secret_value_is_rejected_without_serializer_warning_or_leak() -> None:
    secret = "synthetic-secret-value"
    candidate = proposal()
    object.__setattr__(candidate, "confidence", secret)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(PatchPolicyInputError) as raised:
            validate_patch_policy(candidate, policy())

    assert raised.value.diagnostic.code is PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL
    assert secret not in repr(raised.value)
    assert all(secret not in str(item.message) for item in captured)


def test_invalid_policy_order_names_and_exceptions_fail_at_construction() -> None:
    protected = ProtectedPathPolicy(protected_paths=PROTECTED_PATHS)
    with pytest.raises(ValidationError):
        PatchPolicy(
            protected_path_policy=protected,
            forbidden_changed_files=(PurePosixPath("z.py"), PurePosixPath("a.py")),
        )
    with pytest.raises(ValidationError):
        PatchPolicy(protected_path_policy=protected, dependency_allowlist=("Not_Normalized",))
    with pytest.raises(ValidationError):
        PatchPolicy(protected_path_policy=protected, dependency_allowlist=("z", "a"))
    with pytest.raises(ValidationError):
        PatchPolicy(protected_path_policy=protected, dependency_allowlist=("a", "a"))
    with pytest.raises(ValidationError):
        PatchPolicy(
            protected_path_policy=protected,
            forbidden_changed_files=(PurePosixPath("a.py"), PurePosixPath("a.py")),
        )
    with pytest.raises(ValidationError):
        ProtectedPathPolicy(protected_paths=PROTECTED_PATHS, exceptions=(PurePosixPath("src"),))
    with pytest.raises(ValidationError):
        ProtectedPathPolicy(protected_paths=())


def test_patch_policy_requires_the_complete_architecture_protected_set() -> None:
    incomplete = ProtectedPathPolicy(protected_paths=(PurePosixPath("docker"),))

    with pytest.raises(ValidationError):
        PatchPolicy(protected_path_policy=incomplete)

    assert policy().protected_path_policy.protected_paths == DEFAULT_PROTECTED_PATHS


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit()])
def test_process_control_exceptions_are_not_converted(
    monkeypatch: pytest.MonkeyPatch, interrupt: BaseException
) -> None:
    def interrupted(value: str) -> object:
        del value
        raise interrupt

    monkeypatch.setattr("app.agent.policy.parse_unified_diff", interrupted)
    with pytest.raises(type(interrupt)):
        validate_patch_policy(proposal(), policy())


def test_policy_validation_uses_no_host_or_execution_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*values: object, **named: object) -> None:
        del values, named
        raise AssertionError("forbidden host capability was used")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("os.getenv", forbidden)

    assert validate_patch_policy(proposal(), policy()).approved is True


def test_result_json_contains_only_safe_typed_metadata() -> None:
    secret = "password=synthetic-secret-value"
    result = validate_patch_policy(proposal(summary=secret), policy())
    payload = json.loads(result.model_dump_json())

    assert payload == {
        "approved": False,
        "violations": [
            {
                "code": "secret_detected",
                "message": "credential or private-key material is not allowed",
                "path": None,
                "line": None,
            }
        ],
    }
    assert secret not in result.model_dump_json()
