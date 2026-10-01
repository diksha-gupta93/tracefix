from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath

_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>0|[1-9][0-9]*)(?:,(?P<old_count>[0-9]+))? "
    r"\+(?P<new_start>0|[1-9][0-9]*)(?:,(?P<new_count>[0-9]+))? @@(?: .*)?$"
)
_BINARY_MARKER = re.compile(r"^(?:GIT binary patch|Binary files [^\r\n]+ differ)$")
_MAX_HUNK_NUMBER_DIGITS = 18


class DiffLineKind(StrEnum):
    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class ParsedDiffLine:
    kind: DiffLineKind
    content: str
    old_line: int | None
    new_line: int | None


@dataclass(frozen=True, slots=True)
class ParsedDiffFile:
    path: str
    old_path: str | None
    new_path: str | None
    lines: tuple[ParsedDiffLine, ...]

    @property
    def is_added(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None


@dataclass(frozen=True, slots=True)
class ParsedUnifiedDiff:
    normalized_text: str
    files: tuple[ParsedDiffFile, ...]

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(file.path for file in self.files)

    @property
    def changed_line_count(self) -> int:
        return sum(
            1
            for file in self.files
            for line in file.lines
            if line.kind in {DiffLineKind.ADDED, DiffLineKind.REMOVED}
        )


class UnifiedDiffError(ValueError):
    """An inert unified diff could not be interpreted safely."""


@dataclass(slots=True)
class _FileBuilder:
    path: str
    old_path: str | None
    new_path: str | None
    lines: list[ParsedDiffLine] = field(default_factory=list)
    has_hunk: bool = False

    def freeze(self) -> ParsedDiffFile:
        return ParsedDiffFile(
            path=self.path,
            old_path=self.old_path,
            new_path=self.new_path,
            lines=tuple(self.lines),
        )


def normalize_repository_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if not paths:
        raise UnifiedDiffError("at least one repository path is required")
    normalized: list[str] = []
    identities: set[str] = set()
    for value in paths:
        if type(value) is not str:
            raise UnifiedDiffError("repository path must be a string")
        windows = PureWindowsPath(value)
        candidate = PurePosixPath(value)
        parts = value.split("/")
        if (
            not value
            or "\x00" in value
            or "\\" in value
            or value != value.strip()
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
            or windows.drive
            or windows.root
            or candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or candidate.as_posix() != value
        ):
            raise UnifiedDiffError("repository path is not normalized and relative")
        identity = value.casefold()
        if identity in identities:
            raise UnifiedDiffError("repository paths contain a case-fold alias")
        identities.add(identity)
        normalized.append(value)
    return tuple(normalized)


def _diff_header_path(value: str, prefix: str) -> str | None:
    if value == "/dev/null":
        return None
    if not value.startswith(prefix):
        raise UnifiedDiffError("unified diff header has an unsupported prefix")
    path = value[len(prefix) :]
    return normalize_repository_paths((path,))[0]


def _changed_path_from_headers(
    old_value: str, new_value: str
) -> tuple[str, str | None, str | None]:
    old_path = _diff_header_path(old_value, "a/")
    new_path = _diff_header_path(new_value, "b/")
    if old_path is None and new_path is None:
        raise UnifiedDiffError("unified diff cannot pair two null files")
    if old_path is not None and new_path is not None and old_path != new_path:
        raise UnifiedDiffError("unified diff file headers disagree")
    changed_path = old_path if old_path is not None else new_path
    if changed_path is None:
        raise UnifiedDiffError("unified diff has no changed path")
    return changed_path, old_path, new_path


def _add_unique_identity(path: str, identities: set[str]) -> None:
    identity = path.casefold()
    if identity in identities:
        raise UnifiedDiffError("unified diff contains a duplicate file identity")
    identities.add(identity)


def parse_binary_diff_paths(unified_diff: str) -> tuple[str, ...]:
    """Extract and validate path headers from a recognized header-only binary diff."""
    if type(unified_diff) is not str:
        raise UnifiedDiffError("unified diff must be text")
    normalized = unified_diff.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    paths: list[str] = []
    identities: set[str] = set()
    marker_found = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise UnifiedDiffError("unified diff file headers are not paired")
            changed_path, _, _ = _changed_path_from_headers(line[4:], lines[index + 1][4:])
            _add_unique_identity(changed_path, identities)
            paths.append(changed_path)
            index += 2
            continue
        if _BINARY_MARKER.fullmatch(line):
            marker_found = True
        elif line.startswith("+++ ") or line.startswith("---") or line.startswith("+++"):
            raise UnifiedDiffError("unified diff contains an unsupported header")
        elif line.startswith(("@@", " ", "-", "+")):
            raise UnifiedDiffError("binary diff contains malformed text hunk content")
        index += 1
    if not marker_found or not paths:
        raise UnifiedDiffError("binary diff must contain a recognized marker and file headers")
    return tuple(paths)


def _parse_hunk_number(value: str) -> int:
    if len(value) > _MAX_HUNK_NUMBER_DIGITS:
        raise UnifiedDiffError("unified diff hunk coordinate is too large")
    try:
        return int(value)
    except ValueError:
        raise UnifiedDiffError("unified diff contains a malformed hunk") from None


def parse_unified_diff(unified_diff: str) -> ParsedUnifiedDiff:
    if type(unified_diff) is not str:
        raise UnifiedDiffError("unified diff must be text")
    if "\x00" in unified_diff:
        raise UnifiedDiffError("unified diff contains NUL")
    normalized = unified_diff.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    files: list[_FileBuilder] = []
    identities: set[str] = set()
    current: _FileBuilder | None = None
    old_remaining: int | None = None
    new_remaining: int | None = None
    old_line = 0
    new_line = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if old_remaining is not None and new_remaining is not None:
            if old_remaining == 0 and new_remaining == 0:
                old_remaining = None
                new_remaining = None
                if line == r"\ No newline at end of file":
                    index += 1
                continue
            if line == r"\ No newline at end of file":
                index += 1
                continue
            if current is None:
                raise UnifiedDiffError("unified diff contains malformed hunk content")
            if line.startswith(" "):
                current.lines.append(
                    ParsedDiffLine(DiffLineKind.CONTEXT, line[1:], old_line, new_line)
                )
                old_remaining -= 1
                new_remaining -= 1
                old_line += 1
                new_line += 1
            elif line.startswith("-"):
                current.lines.append(ParsedDiffLine(DiffLineKind.REMOVED, line[1:], old_line, None))
                old_remaining -= 1
                old_line += 1
            elif line.startswith("+"):
                current.lines.append(ParsedDiffLine(DiffLineKind.ADDED, line[1:], None, new_line))
                new_remaining -= 1
                new_line += 1
            else:
                raise UnifiedDiffError("unified diff contains malformed hunk content")
            if old_remaining < 0 or new_remaining < 0:
                raise UnifiedDiffError("unified diff hunk line counts disagree")
            index += 1
            continue
        if line.startswith("--- "):
            paired = index + 1 < len(lines) and lines[index + 1].startswith("+++ ")
            if current is not None and not current.has_hunk:
                raise UnifiedDiffError("unified diff file has no hunk")
            if not paired:
                raise UnifiedDiffError("unified diff file headers are not paired")
            changed_path, old_path, new_path = _changed_path_from_headers(
                line[4:], lines[index + 1][4:]
            )
            _add_unique_identity(changed_path, identities)
            current = _FileBuilder(changed_path, old_path, new_path)
            files.append(current)
            index += 2
            continue
        if line.startswith("+++ ") or line.startswith("---") or line.startswith("+++"):
            raise UnifiedDiffError("unified diff contains an unsupported header")
        if line.startswith("@@"):
            match = _HUNK_HEADER.fullmatch(line)
            if current is None or match is None:
                raise UnifiedDiffError("unified diff contains a malformed hunk")
            old_count = match.group("old_count")
            new_count = match.group("new_count")
            old_remaining = 1 if old_count is None else _parse_hunk_number(old_count)
            new_remaining = 1 if new_count is None else _parse_hunk_number(new_count)
            old_line = _parse_hunk_number(match.group("old_start"))
            new_line = _parse_hunk_number(match.group("new_start"))
            if (old_remaining > 0 and old_line == 0) or (new_remaining > 0 and new_line == 0):
                raise UnifiedDiffError("unified diff hunk has an invalid zero start")
            current.has_hunk = True
        elif line.startswith((" ", "-", "+")) or line == r"\ No newline at end of file":
            raise UnifiedDiffError("unified diff contains content outside a hunk")
        index += 1
    if old_remaining == 0 and new_remaining == 0:
        old_remaining = None
        new_remaining = None
    if old_remaining is not None or new_remaining is not None:
        raise UnifiedDiffError("unified diff hunk line counts disagree")
    if current is None or not current.has_hunk:
        raise UnifiedDiffError("unified diff must contain a file pair and hunk")
    return ParsedUnifiedDiff(normalized, tuple(file.freeze() for file in files))
