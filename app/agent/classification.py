from __future__ import annotations

import re
from enum import StrEnum

from app.agent.schemas import BaselineOutcome, BaselineResult, FailureAnalysis, FailureCategory


class ClassificationErrorCode(StrEnum):
    BASELINE_NOT_REPRODUCED = "baseline_not_reproduced"
    MALFORMED_BASELINE = "malformed_baseline"


class ClassificationError(Exception):
    def __init__(self, code: ClassificationErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


_SIGNALS: tuple[tuple[FailureCategory, tuple[re.Pattern[str], ...]], ...] = (
    (
        FailureCategory.syntax_failure,
        (re.compile(r"\b(?:SyntaxError|IndentationError|TabError)\b", re.IGNORECASE),),
    ),
    (
        FailureCategory.import_or_dependency_failure,
        (
            re.compile(r"\b(?:ModuleNotFoundError|ImportError)\b", re.IGNORECASE),
            re.compile(r"(?:no module named|could not import)", re.IGNORECASE),
        ),
    ),
    (
        FailureCategory.incorrect_exception_behaviour,
        (
            re.compile(r"\bDID NOT RAISE\b", re.IGNORECASE),
            re.compile(r"\braised unexpected exception\b", re.IGNORECASE),
            re.compile(r"\bdoes not raise\b", re.IGNORECASE),
        ),
    ),
    (
        FailureCategory.type_related_failure,
        (
            re.compile(r"\bTypeError\b", re.IGNORECASE),
            re.compile(r"\btype mismatch\b", re.IGNORECASE),
        ),
    ),
    (
        FailureCategory.timeout,
        (
            re.compile(r"\bTimeout(?:Error)?\b", re.IGNORECASE),
            re.compile(r"\btimed out\b", re.IGNORECASE),
        ),
    ),
    (
        FailureCategory.test_environment_failure,
        (
            re.compile(r"fixture ['\"][^'\"]+['\"] not found", re.IGNORECASE),
            re.compile(r"(?:pytest|test) configuration error", re.IGNORECASE),
            re.compile(r"error (?:at setup|in setup)", re.IGNORECASE),
        ),
    ),
    (
        FailureCategory.assertion_failure,
        (
            re.compile(r"(?:^|\n)E\s+(?:assert\b|AssertionError\b)", re.IGNORECASE),
            re.compile(r"\bAssertionError\b", re.IGNORECASE),
        ),
    ),
)

_HOST_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|/(?:home|users|tmp|var|private)/)\S+")


def _summary(category: FailureCategory, evidence: str) -> str:
    for line in evidence.splitlines():
        cleaned = _HOST_PATH.sub("<host-path>", line).strip()
        if cleaned:
            bounded = cleaned.encode("utf-8")[:240].decode("utf-8", errors="ignore")
            return f"{category.value}: {bounded}"
    return f"{category.value}: baseline evidence was empty or not safely recognized"


def classify_failure(baseline: BaselineResult) -> FailureAnalysis:
    if baseline.outcome is not BaselineOutcome.REPRODUCED_FAILURE:
        raise ClassificationError(ClassificationErrorCode.BASELINE_NOT_REPRODUCED)
    evidence = baseline.evidence
    if evidence is None or evidence.exit_code in (None, 0):
        raise ClassificationError(ClassificationErrorCode.MALFORMED_BASELINE)
    combined = "\n".join((evidence.stdout, evidence.stderr))
    if evidence.stdout_truncated or evidence.stderr_truncated:
        category = FailureCategory.unsupported_or_ambiguous
        return FailureAnalysis(category=category, summary=_summary(category, combined))

    matched = tuple(
        category for category, patterns in _SIGNALS if any(p.search(combined) for p in patterns)
    )
    if FailureCategory.syntax_failure in matched:
        category = FailureCategory.syntax_failure
    elif len(matched) == 1:
        category = matched[0]
    else:
        category = FailureCategory.unsupported_or_ambiguous
    return FailureAnalysis(category=category, summary=_summary(category, combined))
