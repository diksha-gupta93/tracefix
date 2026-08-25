from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.classification import ClassificationError, classify_failure
from app.agent.schemas import (
    BaselineDiagnosticCode,
    BaselineEvidence,
    BaselineOutcome,
    BaselineResult,
    FailureCategory,
    RepositoryFingerprint,
    RepositoryPreparation,
    RevisionKind,
)
from app.sandbox import SandboxCompletion


def _result(text: str, *, truncated: bool = False) -> BaselineResult:
    return BaselineResult(
        preparation=RepositoryPreparation(
            case_id="case",
            requested_revision="failing-v1",
            revision_kind=RevisionKind.SNAPSHOT,
            prepared_repository=Path("C:/prepared"),
            fingerprint=RepositoryFingerprint(digest="0" * 64),
        ),
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=BaselineEvidence(
            completion=SandboxCompletion.NORMAL,
            exit_code=1,
            stdout=text,
            stderr="",
            stdout_truncated=truncated,
            stderr_truncated=False,
            duration_seconds=0.1,
        ),
    )


@pytest.mark.parametrize(
    ("evidence", "category"),
    [
        ("E   SyntaxError: invalid syntax", FailureCategory.syntax_failure),
        (
            "ModuleNotFoundError: No module named 'missing'",
            FailureCategory.import_or_dependency_failure,
        ),
        ("E   assert 1 == 2", FailureCategory.assertion_failure),
        (
            "E   Failed: DID NOT RAISE <class 'ValueError'>",
            FailureCategory.incorrect_exception_behaviour,
        ),
        ("TypeError: unsupported operand type(s)", FailureCategory.type_related_failure),
        ("FAILED test_x.py::test_x - Timeout (>1.0s)", FailureCategory.timeout),
        ("fixture 'client' not found", FailureCategory.test_environment_failure),
        ("pytest ended in an unfamiliar way", FailureCategory.unsupported_or_ambiguous),
    ],
)
def test_classifies_runtime_evidence(evidence: str, category: FailureCategory) -> None:
    analysis = classify_failure(_result(evidence))
    assert analysis.category is category
    assert analysis.summary
    assert "C:" not in analysis.summary


def test_syntax_has_precedence_and_conflicting_non_syntax_signals_are_ambiguous() -> None:
    assert (
        classify_failure(_result("SyntaxError: bad\nassert False")).category
        is FailureCategory.syntax_failure
    )
    assert (
        classify_failure(_result("ModuleNotFoundError: x\nE   assert False")).category
        is FailureCategory.unsupported_or_ambiguous
    )


def test_materially_truncated_evidence_is_ambiguous() -> None:
    assert (
        classify_failure(_result("assert False", truncated=True)).category
        is FailureCategory.unsupported_or_ambiguous
    )


def test_non_reproduced_result_is_refused() -> None:
    reproduced = _result("assert False")
    result = BaselineResult(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.NOT_REPRODUCIBLE,
        evidence=reproduced.evidence.model_copy(update={"exit_code": 0}),
        diagnostic_code=BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE,
        message="not reproduced",
    )
    with pytest.raises(ClassificationError, match="baseline_not_reproduced"):
        classify_failure(result)
