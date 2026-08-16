from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from benchmarks.loader import BugCategory, load_development_cases


def _patched_files(patch: str) -> set[PurePosixPath]:
    old_paths = {
        PurePosixPath(match.group(1))
        for match in re.finditer(r"^--- a/(.+)$", patch, flags=re.MULTILINE)
    }
    new_paths = {
        PurePosixPath(match.group(1))
        for match in re.finditer(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
    }
    assert old_paths == new_paths
    return old_paths


def test_fixture_assets_are_isolated_and_patches_match_policy() -> None:
    for case in load_development_cases():
        assert not (case.repository_path / "evaluator").exists()
        patch = case.reference_patch.read_text(encoding="utf-8")
        assert _patched_files(patch) == {
            PurePosixPath(item) for item in case.manifest.expected_changed_files
        }
        assert not {
            PurePosixPath(item) for item in case.manifest.forbidden_changed_files
        } & _patched_files(patch)


def test_visible_and_hidden_expectations_are_meaningfully_distinct() -> None:
    for case in load_development_cases():
        visible = {path.read_text(encoding="utf-8") for path in case.visible_tests}
        hidden = {path.read_text(encoding="utf-8") for path in case.hidden_tests}
        assert visible.isdisjoint(hidden)
        assert all("test_" in content and "assert" in content for content in visible | hidden)


def test_each_fixture_contains_its_expected_localized_defect_and_repair() -> None:
    expected_evidence = {
        BugCategory.incorrect_conditional: ("return age > 18", "return age >= 18"),
        BugCategory.boundary_condition: ("items // size + 1", "(items + size - 1) // size"),
        BugCategory.incorrect_return_value: ("return name.upper()", "return name.strip().lower()"),
        BugCategory.exception_handling: (
            "except TypeError:",
            "except (TypeError, ValueError):",
        ),
        BugCategory.fixture_or_mocking: ("client.fetch_users()", "client.fetch_user()"),
    }
    for case in load_development_cases():
        failing, repaired = expected_evidence[case.manifest.bug_category]
        changed_file = case.repository_path / case.manifest.expected_changed_files[0]
        patch = case.reference_patch.read_text(encoding="utf-8")
        assert failing in changed_file.read_text(encoding="utf-8")
        assert failing in patch
        assert repaired in patch
        assert len(case.manifest.expected_changed_files) == 1


def test_fixture_filenames_are_portable_and_unique_case_insensitively() -> None:
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    paths = [
        path.relative_to("benchmarks/development")
        for path in Path("benchmarks/development").rglob("*")
    ]
    folded = [path.as_posix().casefold() for path in paths]
    assert len(folded) == len(set(folded))
    assert all(path.stem.casefold() not in reserved for path in paths)


def test_fixture_text_is_utf8_lf() -> None:
    for path in Path("benchmarks/development").rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            content.decode("utf-8")
            assert b"\r\n" not in content
