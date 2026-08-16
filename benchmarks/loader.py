from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CaseId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]

_CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_PROTECTED_NAMES = frozenset(
    {
        ".git",
        ".github",
        ".gitlab-ci.yml",
        "agents.md",
        "benchmarks",
        "dockerfile",
        "manifest.json",
        "evaluator",
        "pyproject.toml",
        "tox.ini",
    }
)


class BugCategory(StrEnum):
    incorrect_conditional = "incorrect_conditional"
    boundary_condition = "boundary_condition"
    incorrect_return_value = "incorrect_return_value"
    exception_handling = "exception_handling"
    fixture_or_mocking = "fixture_or_mocking"


class Difficulty(StrEnum):
    easy = "easy"
    medium = "medium"


class RiskLevel(StrEnum):
    low = "low"
    medium = "medium"


class BenchmarkLoadError(Exception):
    """A benchmark case failed closed during metadata loading."""


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: CaseId
    repository_path: NonBlank
    base_revision: NonBlank
    failing_revision: NonBlank
    visible_tests: tuple[NonBlank, ...]
    hidden_tests: tuple[NonBlank, ...]
    issue_description: NonBlank
    expected_behavior: NonBlank
    reference_patch: NonBlank
    expected_changed_files: tuple[NonBlank, ...]
    forbidden_changed_files: tuple[NonBlank, ...]
    bug_category: BugCategory
    difficulty: Difficulty
    risk_level: RiskLevel


class TrustedCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    manifest: Manifest
    case_path: Path
    repository_path: Path
    visible_tests: tuple[Path, ...]
    hidden_tests: tuple[Path, ...]
    reference_patch: Path


class ModelSafeFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: PurePosixPath
    content: str


class ModelSafeCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str
    issue_description: str
    expected_behavior: str
    bug_category: BugCategory
    difficulty: Difficulty
    risk_level: RiskLevel
    repository_files: tuple[ModelSafeFile, ...]
    visible_tests: tuple[PurePosixPath, ...]


def _relative_path(value: str, *, field: str, case_id: str) -> PurePosixPath:
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or windows_path.drive
        or windows_path.root
        or value.startswith("/")
    ):
        raise BenchmarkLoadError(f"case {case_id}: invalid {field} path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise BenchmarkLoadError(f"case {case_id}: invalid {field} path")
    return path


def _contained(root: Path, relative: PurePosixPath, *, field: str, case_id: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = root.joinpath(*relative.parts).resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise BenchmarkLoadError(f"case {case_id}: invalid or missing {field}") from error
    return resolved


def _file(root: Path, value: str, *, field: str, case_id: str) -> Path:
    resolved = _contained(
        root, _relative_path(value, field=field, case_id=case_id), field=field, case_id=case_id
    )
    if not resolved.is_file():
        raise BenchmarkLoadError(f"case {case_id}: {field} is not a file")
    return resolved


def _directory(root: Path, value: str, *, field: str, case_id: str) -> Path:
    resolved = _contained(
        root, _relative_path(value, field=field, case_id=case_id), field=field, case_id=case_id
    )
    if not resolved.is_dir():
        raise BenchmarkLoadError(f"case {case_id}: {field} is not a directory")
    return resolved


def _manifest(case_path: Path) -> Manifest:
    case_id = case_path.name
    try:
        manifest = Manifest.model_validate_json(
            (case_path / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as error:
        raise BenchmarkLoadError(f"case {case_id}: invalid manifest") from error
    if manifest.case_id != case_id:
        raise BenchmarkLoadError(f"case {case_id}: case_id does not match directory")
    if manifest.base_revision == manifest.failing_revision:
        raise BenchmarkLoadError(f"case {case_id}: revision identifiers must differ")
    return manifest


def _case_path(case_id: str, benchmark_root: Path | None) -> Path:
    if not isinstance(case_id, str) or not _CASE_ID_PATTERN.fullmatch(case_id):
        raise BenchmarkLoadError(f"case {case_id!r}: invalid case_id")
    root = (benchmark_root or Path(__file__).parent / "development").resolve()
    try:
        case_path = (root / case_id).resolve(strict=True)
        case_path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise BenchmarkLoadError(f"case {case_id}: case directory is missing") from error
    if not case_path.is_dir():
        raise BenchmarkLoadError(f"case {case_id}: case directory is missing")
    return case_path


def _validate_test_path(path: PurePosixPath, *, hidden: bool, case_id: str) -> None:
    expected_prefix = ("evaluator", "hidden_tests") if hidden else ("tests",)
    if path.parts[: len(expected_prefix)] != expected_prefix or not path.name.startswith("test_"):
        field = "hidden_tests" if hidden else "visible_tests"
        raise BenchmarkLoadError(f"case {case_id}: invalid {field} path")
    if path.suffix != ".py":
        field = "hidden_tests" if hidden else "visible_tests"
        raise BenchmarkLoadError(f"case {case_id}: invalid {field} path")


def _validate_repository(repository: Path, *, case_id: str) -> tuple[Path, ...]:
    if (repository / "evaluator").exists():
        raise BenchmarkLoadError(f"case {case_id}: evaluator-only content inside repository")
    files: list[Path] = []
    for candidate in repository.rglob("*"):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repository)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise BenchmarkLoadError(f"case {case_id}: repository path escapes root") from error
        if resolved.is_file():
            files.append(resolved)
    return tuple(sorted(set(files), key=lambda item: item.as_posix().casefold()))


def _validate_policy(manifest: Manifest) -> None:
    case_id = manifest.case_id
    expected = tuple(
        _relative_path(item, field="expected_changed_files", case_id=case_id)
        for item in manifest.expected_changed_files
    )
    forbidden = tuple(
        _relative_path(item, field="forbidden_changed_files", case_id=case_id)
        for item in manifest.forbidden_changed_files
    )
    expected_keys = [path.as_posix().casefold() for path in expected]
    forbidden_keys = [path.as_posix().casefold() for path in forbidden]
    if (
        not expected
        or len(expected_keys) != len(set(expected_keys))
        or len(forbidden_keys) != len(set(forbidden_keys))
        or set(expected_keys) & set(forbidden_keys)
    ):
        raise BenchmarkLoadError(f"case {case_id}: invalid changed-file policy")
    for path in (*expected, *forbidden):
        lowered = tuple(part.casefold() for part in path.parts)
        if "tests" in lowered or any(part in _PROTECTED_NAMES for part in lowered):
            raise BenchmarkLoadError(f"case {case_id}: protected changed-file path")


def _validate_manifest_paths(manifest: Manifest) -> None:
    case_id = manifest.case_id
    _relative_path(manifest.repository_path, field="repository_path", case_id=case_id)
    for item in manifest.visible_tests:
        relative = _relative_path(item, field="visible_tests", case_id=case_id)
        _validate_test_path(relative, hidden=False, case_id=case_id)
    for item in manifest.hidden_tests:
        relative = _relative_path(item, field="hidden_tests", case_id=case_id)
        _validate_test_path(relative, hidden=True, case_id=case_id)
    patch = _relative_path(manifest.reference_patch, field="reference_patch", case_id=case_id)
    if patch.parts[:1] != ("evaluator",) or patch.suffix != ".patch":
        raise BenchmarkLoadError(f"case {case_id}: invalid reference_patch path")
    _validate_policy(manifest)


def _load_common(
    case_id: str, benchmark_root: Path | None
) -> tuple[Path, Manifest, Path, tuple[Path, ...]]:
    case_path = _case_path(case_id, benchmark_root)
    manifest = _manifest(case_path)
    _validate_manifest_paths(manifest)
    repository = _directory(
        case_path, manifest.repository_path, field="repository_path", case_id=case_id
    )
    _validate_repository(repository, case_id=case_id)
    visible: list[Path] = []
    seen: set[str] = set()
    for item in manifest.visible_tests:
        relative = _relative_path(item, field="visible_tests", case_id=case_id)
        _validate_test_path(relative, hidden=False, case_id=case_id)
        key = relative.as_posix().casefold()
        if key in seen:
            raise BenchmarkLoadError(f"case {case_id}: duplicate visible_tests path")
        seen.add(key)
        visible.append(_file(repository, item, field="visible_tests", case_id=case_id))
    if not visible:
        raise BenchmarkLoadError(f"case {case_id}: visible_tests must not be empty")
    hidden_seen: set[str] = set()
    for item in manifest.hidden_tests:
        relative = _relative_path(item, field="hidden_tests", case_id=case_id)
        key = relative.as_posix().casefold()
        if key in hidden_seen:
            raise BenchmarkLoadError(f"case {case_id}: duplicate hidden_tests path")
        hidden_seen.add(key)
        _file(case_path, item, field="hidden_tests", case_id=case_id)
    if not hidden_seen:
        raise BenchmarkLoadError(f"case {case_id}: hidden_tests must not be empty")
    _file(case_path, manifest.reference_patch, field="reference_patch", case_id=case_id)
    return case_path, manifest, repository, tuple(visible)


def load_trusted_case(case_id: str, *, benchmark_root: Path | None = None) -> TrustedCase:
    case_path, manifest, repository, visible = _load_common(case_id, benchmark_root)
    hidden: list[Path] = []
    seen_hidden: set[str] = set()
    for item in manifest.hidden_tests:
        relative = _relative_path(item, field="hidden_tests", case_id=case_id)
        _validate_test_path(relative, hidden=True, case_id=case_id)
        key = relative.as_posix().casefold()
        if key in seen_hidden:
            raise BenchmarkLoadError(f"case {case_id}: duplicate hidden_tests path")
        seen_hidden.add(key)
        hidden.append(_file(case_path, item, field="hidden_tests", case_id=case_id))
    if not hidden:
        raise BenchmarkLoadError(f"case {case_id}: hidden_tests must not be empty")
    patch = _file(case_path, manifest.reference_patch, field="reference_patch", case_id=case_id)
    return TrustedCase(
        manifest=manifest,
        case_path=case_path,
        repository_path=repository,
        visible_tests=visible,
        hidden_tests=tuple(hidden),
        reference_patch=patch,
    )


def load_model_safe_case(case_id: str, *, benchmark_root: Path | None = None) -> ModelSafeCase:
    _, manifest, repository, visible = _load_common(case_id, benchmark_root)
    files: list[ModelSafeFile] = []
    for path in _validate_repository(repository, case_id=case_id):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkLoadError(
                f"case {case_id}: repository file is not UTF-8 text"
            ) from error
        files.append(
            ModelSafeFile(
                path=PurePosixPath(path.relative_to(repository).as_posix()),
                content=content,
            )
        )
    return ModelSafeCase(
        case_id=manifest.case_id,
        issue_description=manifest.issue_description,
        expected_behavior=manifest.expected_behavior,
        bug_category=manifest.bug_category,
        difficulty=manifest.difficulty,
        risk_level=manifest.risk_level,
        repository_files=tuple(files),
        visible_tests=tuple(
            PurePosixPath(path.relative_to(repository).as_posix()) for path in visible
        ),
    )


def load_development_cases(*, benchmark_root: Path | None = None) -> tuple[TrustedCase, ...]:
    root = (benchmark_root or Path(__file__).parent / "development").resolve()
    try:
        children = tuple(sorted(path for path in root.iterdir() if path.is_dir()))
    except OSError as error:
        raise BenchmarkLoadError("development corpus is missing") from error
    cases = tuple(load_trusted_case(path.name, benchmark_root=root) for path in children)
    ids = [case.manifest.case_id.casefold() for case in cases]
    categories = [case.manifest.bug_category for case in cases]
    if len(cases) != 5 or len(set(ids)) != 5 or set(categories) != set(BugCategory):
        raise BenchmarkLoadError("development corpus must contain exactly one case per category")
    return cases
