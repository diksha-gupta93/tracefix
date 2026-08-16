from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.loader import (
    BenchmarkLoadError,
    BugCategory,
    ModelSafeCase,
    TrustedCase,
    load_development_cases,
    load_model_safe_case,
    load_trusted_case,
)

BENCHMARK_ROOT = Path("benchmarks/development")


def _copy_case(tmp_path: Path, case_id: str = "incorrect_conditional") -> Path:
    shutil.copytree(BENCHMARK_ROOT / case_id, tmp_path / case_id)
    return tmp_path / case_id


def _manifest(case: Path) -> dict[str, object]:
    value = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(case: Path, manifest: dict[str, object]) -> None:
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_loads_complete_typed_corpus() -> None:
    cases = load_development_cases()

    assert len(cases) == 5
    assert all(isinstance(case, TrustedCase) for case in cases)
    assert {case.manifest.case_id for case in cases} == {
        "incorrect_conditional",
        "boundary_condition",
        "incorrect_return_value",
        "exception_handling",
        "fixture_or_mocking",
    }
    assert {case.manifest.bug_category for case in cases} == set(BugCategory)
    for case in cases:
        assert case.manifest.base_revision != case.manifest.failing_revision
        assert case.visible_tests
        assert case.hidden_tests
        assert case.reference_patch.is_file()


def test_safe_view_excludes_all_evaluator_and_policy_fields() -> None:
    safe = load_model_safe_case("incorrect_conditional")

    assert isinstance(safe, ModelSafeCase)
    excluded = {
        "hidden_tests",
        "reference_patch",
        "expected_changed_files",
        "forbidden_changed_files",
        "case_path",
    }
    assert excluded.isdisjoint(type(safe).model_fields)
    assert all("evaluator" not in item.path.parts for item in safe.repository_files)
    assert all(path.parts[0] == "tests" for path in safe.visible_tests)


def test_safe_loading_does_not_access_evaluator_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(tmp_path)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if "evaluator" in path.parts:
            raise AssertionError("model-safe loading accessed evaluator content")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    safe = load_model_safe_case(case.name, benchmark_root=tmp_path)

    assert safe.case_id == case.name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_tests", ["evaluator/hidden_tests/missing.py"]),
        ("reference_patch", "evaluator/missing.patch"),
        (
            "hidden_tests",
            [
                "evaluator/hidden_tests/test_regression.py",
                "evaluator/hidden_tests/test_regression.py",
            ],
        ),
    ],
)
def test_safe_loading_validates_evaluator_declarations_without_reading_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    _write_manifest(case, manifest)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if "evaluator" in path.parts:
            raise AssertionError("model-safe loading accessed evaluator content")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    with pytest.raises(BenchmarkLoadError):
        load_model_safe_case(case.name, benchmark_root=tmp_path)


def test_safe_loading_wraps_invalid_repository_text(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    (case / "repository/invalid.py").write_bytes(b"\xff")

    with pytest.raises(BenchmarkLoadError, match="repository file is not UTF-8 text"):
        load_model_safe_case(case.name, benchmark_root=tmp_path)


@pytest.mark.parametrize(
    "case_id", ["missing", "../escape", "a/b", "C:\\escape", "//server/share", "/tmp/x", ""]
)
def test_invalid_or_missing_case_fails_closed(case_id: str) -> None:
    with pytest.raises(BenchmarkLoadError):
        load_trusted_case(case_id)


@pytest.mark.parametrize("content", ["[]", "{", '"text"', "null"])
def test_malformed_manifest_fails_closed(tmp_path: Path, content: str) -> None:
    case = tmp_path / "broken"
    case.mkdir()
    (case / "manifest.json").write_text(content, encoding="utf-8")

    with pytest.raises(BenchmarkLoadError, match="case broken: invalid manifest"):
        load_trusted_case("broken", benchmark_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issue_description", " "),
        ("visible_tests", "tests/test_behavior.py"),
        ("bug_category", "unknown"),
        ("difficulty", "hard"),
        ("risk_level", "high"),
    ],
)
def test_invalid_manifest_fields_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="invalid manifest"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_unknown_manifest_field_fails_closed(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest["unexpected"] = True
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="invalid manifest"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository_path", "/tmp/repository"),
        ("repository_path", "C:/repository"),
        ("visible_tests", ["../evaluator/hidden_tests/test_regression.py"]),
        ("hidden_tests", ["//server/share/test_hidden.py"]),
        ("reference_patch", "evaluator/../reference.patch"),
        ("expected_changed_files", ["tests/test_behavior.py"]),
        ("forbidden_changed_files", [".github/workflows/ci.yml"]),
    ],
)
def test_adversarial_manifest_paths_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError):
        load_trusted_case(case.name, benchmark_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visible_tests", ["tests/missing.py"]),
        ("hidden_tests", ["evaluator/hidden_tests/missing.py"]),
        ("reference_patch", "evaluator/missing.patch"),
    ],
)
def test_missing_declared_artifacts_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_declared_directory_cannot_stand_in_for_file(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest["reference_patch"] = "evaluator/hidden_tests"
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError):
        load_trusted_case(case.name, benchmark_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visible_tests", ["tests/test_behavior.py", "tests/test_behavior.py"]),
        (
            "hidden_tests",
            [
                "evaluator/hidden_tests/test_regression.py",
                "evaluator/hidden_tests/test_regression.py",
            ],
        ),
        (
            "expected_changed_files",
            ["src/eligibility/eligibility.py", "src/eligibility/eligibility.py"],
        ),
        ("forbidden_changed_files", ["README.md", "README.md"]),
    ],
)
def test_duplicate_paths_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_overlapping_changed_file_policy_fails_closed(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest["forbidden_changed_files"] = ["src/eligibility/eligibility.py"]
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="invalid changed-file policy"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_case_id_and_directory_must_match(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest["case_id"] = "different"
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="case_id does not match"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_revisions_must_differ(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    manifest = _manifest(case)
    manifest["failing_revision"] = manifest["base_revision"]
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="revision identifiers must differ"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_evaluator_directory_inside_repository_fails_closed(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    (case / "repository/evaluator").mkdir()

    with pytest.raises(BenchmarkLoadError, match="evaluator-only content"):
        load_trusted_case(case.name, benchmark_root=tmp_path)


def test_corpus_rejects_duplicate_category(tmp_path: Path) -> None:
    for source in BENCHMARK_ROOT.iterdir():
        if source.is_dir():
            shutil.copytree(source, tmp_path / source.name)
    case = tmp_path / "boundary_condition"
    manifest = _manifest(case)
    manifest["bug_category"] = "incorrect_conditional"
    _write_manifest(case, manifest)

    with pytest.raises(BenchmarkLoadError, match="exactly one case per category"):
        load_development_cases(benchmark_root=tmp_path)


def test_repository_escape_through_symlink_fails_closed(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    link = case / "repository/escape.py"
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n", encoding="utf-8")
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("this platform does not permit test-created symlinks")

    with pytest.raises(BenchmarkLoadError, match="repository path escapes root"):
        load_model_safe_case(case.name, benchmark_root=tmp_path)
