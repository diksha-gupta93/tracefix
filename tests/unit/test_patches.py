from __future__ import annotations

import pytest

from app.agent.patches import DiffLineKind, UnifiedDiffError, parse_unified_diff


def test_parser_retains_immutable_hunk_coordinates_and_lines() -> None:
    parsed = parse_unified_diff(
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -2,2 +2,3 @@\n keep\n-old\n+new\n+extra\n"
    )

    file = parsed.files[0]
    assert len(file.hunks) == 1
    hunk = file.hunks[0]
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (2, 2, 2, 3)
    assert tuple(line.kind for line in hunk.lines) == (
        DiffLineKind.CONTEXT,
        DiffLineKind.REMOVED,
        DiffLineKind.ADDED,
        DiffLineKind.ADDED,
    )
    assert file.lines == hunk.lines
    assert parsed.has_ignored_content is False
    with pytest.raises(AttributeError):
        hunk.old_start = 3  # type: ignore[misc]


def test_parser_retains_multiple_and_zero_count_hunks() -> None:
    parsed = parse_unified_diff(
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -0,0 +1 @@\n"
        "+first\n"
        "@@ -2,0 +4 @@\n"
        "+fourth\n"
    )

    assert tuple(
        (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count)
        for hunk in parsed.files[0].hunks
    ) == ((0, 0, 1, 1), (2, 0, 4, 1))


def test_parser_retains_no_final_newline_metadata() -> None:
    parsed = parse_unified_diff(
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )

    file = parsed.files[0]
    assert file.old_no_final_newline is True
    assert file.new_no_final_newline is True


@pytest.mark.parametrize(
    "diff",
    [
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n\\ No newline at end of file\n-x\n+y\n",
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n\\ No newline at end of file\n\\ No newline at end of file\n+y\n",
    ],
)
def test_parser_rejects_misplaced_or_repeated_no_newline_marker(diff: str) -> None:
    with pytest.raises(UnifiedDiffError):
        parse_unified_diff(diff)


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
def test_parser_retains_ignored_non_envelope_content(metadata: str) -> None:
    parsed = parse_unified_diff(
        metadata
        + "--- a/src/example.py\n"
        + "+++ b/src/example.py\n"
        + "@@ -1 +1 @@\n"
        + "-old\n"
        + "+new\n"
    )

    assert parsed.has_ignored_content is True
