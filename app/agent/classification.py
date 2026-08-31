from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from app.agent.schemas import (
    BaselineEvidence,
    BaselineOutcome,
    BaselineResult,
    FailureAnalysis,
    FailureCategory,
    RepositoryPreparation,
)
from app.sandbox.results import SandboxCompletion


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

_HOST_PATHS = (
    re.compile(
        r"(?i)(?:\"(?:[a-z]:[\\/]|\\\\(?:[?.][\\/]|[^\\/\s]+[\\/])|/)[^\"\r\n]+\"|"
        r"'(?:[a-z]:[\\/]|\\\\(?:[?.][\\/]|[^\\/\s]+[\\/])|/)[^'\r\n]+')"
    ),
    re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\(?:[?.][\\/]|[^\\/\s]+[\\/]))[^\s\"']*"),
    re.compile(r"(?<![\w.])/(?:[^/\s\"']+/)*[^\s\"']*"),
)
_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?"
        r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(
        r"(?im)^(\s*(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*=\s*)"
        r'(?:"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\r\n#]+)'
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def redact_host_paths(value: str, repository_root: str | None = None) -> str:
    redacted = value
    if repository_root:
        variants = {
            repository_root,
            repository_root.replace("\\", "/"),
            repository_root.replace("/", "\\"),
        }
        for variant in sorted(variants, key=len, reverse=True):
            redacted = re.sub(re.escape(variant), "<repository>", redacted, flags=re.IGNORECASE)
    for pattern in _HOST_PATHS:
        redacted = pattern.sub("<host-path>", redacted)
    return redacted


def redact_untrusted_evidence(value: str, repository_root: str | None = None) -> str:
    redacted = redact_host_paths(value, repository_root)
    for index, pattern in enumerate(_SECRET_PATTERNS):
        replacement = r"\1<redacted-secret>" if index == 1 else "<redacted-secret>"
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _summary(category: FailureCategory, evidence: str, repository_root: str) -> str:
    redacted = redact_untrusted_evidence(evidence, repository_root)
    for line in redacted.splitlines():
        cleaned = line.strip()
        if cleaned:
            bounded = cleaned.encode("utf-8")[:240].decode("utf-8", errors="ignore")
            return f"{category.value}: {bounded}"
    return f"{category.value}: baseline evidence was empty or not safely recognized"


def classify_failure(baseline: BaselineResult) -> FailureAnalysis:
    try:
        outcome = baseline.outcome
        evidence = baseline.evidence
        preparation = baseline.preparation
        diagnostic_code = baseline.diagnostic_code
        message = baseline.message
        terminal = baseline.terminal
        if outcome is not BaselineOutcome.REPRODUCED_FAILURE:
            raise ClassificationError(ClassificationErrorCode.BASELINE_NOT_REPRODUCED)
        if not isinstance(evidence, BaselineEvidence) or not isinstance(
            preparation, RepositoryPreparation
        ):
            raise ClassificationError(ClassificationErrorCode.MALFORMED_BASELINE)
        if (
            type(evidence.exit_code) is not int
            or type(evidence.stdout) is not str
            or type(evidence.stderr) is not str
            or type(evidence.stdout_truncated) is not bool
            or type(evidence.stderr_truncated) is not bool
        ):
            raise ClassificationError(ClassificationErrorCode.MALFORMED_BASELINE)
        evidence = BaselineEvidence.model_validate(evidence.model_dump())
        preparation = RepositoryPreparation.model_validate(preparation.model_dump())
        if (
            evidence.completion is not SandboxCompletion.NORMAL
            or evidence.exit_code == 0
            or not isinstance(preparation.prepared_repository, Path)
            or diagnostic_code is not None
            or message is not None
            or terminal is not True
        ):
            raise ClassificationError(ClassificationErrorCode.MALFORMED_BASELINE)
        combined = "\n".join((evidence.stdout, evidence.stderr))
        repository_root = str(preparation.prepared_repository)
    except ClassificationError:
        raise
    except (AttributeError, TypeError, ValidationError, ValueError):
        raise ClassificationError(ClassificationErrorCode.MALFORMED_BASELINE) from None
    if evidence.stdout_truncated or evidence.stderr_truncated:
        category = FailureCategory.unsupported_or_ambiguous
        return FailureAnalysis(
            category=category,
            summary=_summary(category, combined, repository_root),
        )

    matched = tuple(
        category for category, patterns in _SIGNALS if any(p.search(combined) for p in patterns)
    )
    category = matched[0] if len(matched) == 1 else FailureCategory.unsupported_or_ambiguous
    return FailureAnalysis(
        category=category,
        summary=_summary(category, combined, repository_root),
    )
