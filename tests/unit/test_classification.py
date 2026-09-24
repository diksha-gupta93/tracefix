from __future__ import annotations

from itertools import combinations
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

_RELIABLE_SIGNALS = (
    ("SyntaxError: invalid syntax", FailureCategory.syntax_failure),
    (
        "ModuleNotFoundError: No module named 'missing'",
        FailureCategory.import_or_dependency_failure,
    ),
    ("E   assert 1 == 2", FailureCategory.assertion_failure),
    ("Failed: DID NOT RAISE <class 'ValueError'>", FailureCategory.incorrect_exception_behaviour),
    ("TypeError: unsupported operand type(s)", FailureCategory.type_related_failure),
    ("FAILED test_x.py::test_x - Timeout (>1.0s)", FailureCategory.timeout),
    ("fixture 'client' not found", FailureCategory.test_environment_failure),
)


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
        *_RELIABLE_SIGNALS,
        ("pytest ended in an unfamiliar way", FailureCategory.unsupported_or_ambiguous),
    ],
)
def test_classifies_runtime_evidence(evidence: str, category: FailureCategory) -> None:
    analysis = classify_failure(_result(evidence))
    assert analysis.category is category
    assert analysis.summary
    assert "C:" not in analysis.summary


def test_single_syntax_signal_is_recognized_and_conflicting_signals_are_ambiguous() -> None:
    assert classify_failure(_result("SyntaxError: bad")).category is FailureCategory.syntax_failure
    assert (
        classify_failure(_result("SyntaxError: bad\nE   assert False")).category
        is FailureCategory.unsupported_or_ambiguous
    )
    assert (
        classify_failure(_result("ModuleNotFoundError: x\nE   assert False")).category
        is FailureCategory.unsupported_or_ambiguous
    )


@pytest.mark.parametrize(
    ("evidence", "category"),
    [
        ("IndentationError: unexpected indent", FailureCategory.syntax_failure),
        ("TabError: inconsistent use of tabs", FailureCategory.syntax_failure),
        ("ImportError: cannot import name 'missing'", FailureCategory.import_or_dependency_failure),
        ("could not import module", FailureCategory.import_or_dependency_failure),
        ("raised unexpected exception", FailureCategory.incorrect_exception_behaviour),
        ("does not raise ValueError", FailureCategory.incorrect_exception_behaviour),
        ("type mismatch", FailureCategory.type_related_failure),
        ("TimeoutError: operation expired", FailureCategory.timeout),
        ("operation timed out", FailureCategory.timeout),
        ("pytest configuration error", FailureCategory.test_environment_failure),
        ("ERROR at setup of test_client", FailureCategory.test_environment_failure),
        ("E   AssertionError: mismatch", FailureCategory.assertion_failure),
    ],
)
def test_every_supported_signal_branch_is_recognized(
    evidence: str, category: FailureCategory
) -> None:
    assert classify_failure(_result(evidence)).category is category


@pytest.mark.parametrize(
    ("first", "second"),
    combinations((signal for signal, _ in _RELIABLE_SIGNALS), 2),
)
def test_every_cross_category_signal_conflict_is_ambiguous(first: str, second: str) -> None:
    assert (
        classify_failure(_result(f"{first}\n{second}")).category
        is FailureCategory.unsupported_or_ambiguous
    )


def test_materially_truncated_evidence_is_ambiguous() -> None:
    assert (
        classify_failure(_result("assert False", truncated=True)).category
        is FailureCategory.unsupported_or_ambiguous
    )


@pytest.mark.parametrize(
    "host_path",
    [
        "C:\\Users\\alice\\repo\\test_x.py",
        "\\\\server\\share\\test_x.py",
        "\\\\?\\C:\\repo\\test_x.py",
        "/opt/private/repo/test_x.py",
    ],
)
def test_summary_redacts_cross_platform_absolute_paths(host_path: str) -> None:
    analysis = classify_failure(_result(f"{host_path}:2:\nE   assert False"))
    assert analysis.category is FailureCategory.assertion_failure
    assert host_path not in analysis.summary
    assert "<host-path>" in analysis.summary


@pytest.mark.parametrize(
    "host_path",
    [
        '"C:\\Users\\Alice Smith\\repo\\test_x.py"',
        '"\\\\server\\shared folder\\test_x.py"',
        "'/opt/private folder/repo/test_x.py'",
    ],
)
def test_summary_redacts_quoted_absolute_paths_containing_spaces(host_path: str) -> None:
    analysis = classify_failure(_result(f"File {host_path}, line 2\nE   assert False"))
    assert analysis.category is FailureCategory.assertion_failure
    assert host_path.strip("\"'") not in analysis.summary
    assert "<host-path>" in analysis.summary


def test_summary_redacts_recognized_credentials() -> None:
    analysis = classify_failure(_result("api_key=super-secret-value\nE   assert False"))
    assert analysis.category is FailureCategory.assertion_failure
    assert "super-secret-value" not in analysis.summary
    assert "<redacted-secret>" in analysis.summary


@pytest.mark.parametrize(
    "secret",
    [
        'password="top secret value"',
        "client_secret='another secret value'",
        'api_key="value#fragment secret"',
        "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----",
    ],
)
def test_summary_fully_redacts_multiline_and_quoted_secrets(secret: str) -> None:
    analysis = classify_failure(_result(f"{secret}\nE   assert False"))

    assert analysis.category is FailureCategory.assertion_failure
    assert secret not in analysis.summary
    assert "top secret value" not in analysis.summary
    assert "another secret value" not in analysis.summary
    assert "PRIVATE KEY" not in analysis.summary
    assert "private-material" not in analysis.summary
    assert "<redacted-secret>" in analysis.summary


@pytest.mark.parametrize(
    ("near_miss", "category"),
    [
        ("documentation mentions syntax errors", FailureCategory.syntax_failure),
        ("a module was not found useful", FailureCategory.import_or_dependency_failure),
        ("the assertion completed successfully", FailureCategory.assertion_failure),
        ("the function may raise later", FailureCategory.incorrect_exception_behaviour),
        ("the operation has a type", FailureCategory.type_related_failure),
        ("elapsed time was long", FailureCategory.timeout),
        ("fixture 'client' was found", FailureCategory.test_environment_failure),
    ],
)
def test_category_near_misses_remain_ambiguous(near_miss: str, category: FailureCategory) -> None:
    assert category is not FailureCategory.unsupported_or_ambiguous
    assert classify_failure(_result(near_miss)).category is FailureCategory.unsupported_or_ambiguous


@pytest.mark.parametrize(
    "denied_evidence",
    [
        "evaluator/hidden_tests/test_secret.py:7: AssertionError: SECRET_REFERENCE",
        "hidden_tests/test_secret.py:7: TypeError: SECRET_REFERENCE",
        "reference.patch: SECRET_REFERENCE\nE   assert False",
        "Evaluator\\Hidden_Tests\\test_secret.py:7: TimeoutError: SECRET_REFERENCE",
    ],
)
def test_denied_evaluator_evidence_cannot_classify_or_reach_summary(
    denied_evidence: str,
) -> None:
    analysis = classify_failure(_result(denied_evidence))

    assert analysis.category is FailureCategory.unsupported_or_ambiguous
    assert "SECRET_REFERENCE" not in analysis.summary
    assert "evaluator" not in analysis.summary.casefold()
    assert "hidden_tests" not in analysis.summary.casefold()
    assert "reference.patch" not in analysis.summary.casefold()


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


def test_infrastructure_and_malformed_reproduced_results_are_refused() -> None:
    reproduced = _result("E   assert False")
    infrastructure = BaselineResult.model_construct(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC,
        evidence=reproduced.evidence,
        terminal=True,
    )
    with pytest.raises(ClassificationError, match="baseline_not_reproduced"):
        classify_failure(infrastructure)
    malformed = BaselineResult.model_construct(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=None,
        terminal=True,
    )
    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(malformed)

    missing_preparation = BaselineResult.model_construct(
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=reproduced.evidence,
        terminal=True,
    )
    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(missing_preparation)

    malformed_evidence = BaselineResult.model_construct(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=object(),
        terminal=True,
    )
    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(malformed_evidence)

    contradictory_completion = BaselineResult.model_construct(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=reproduced.evidence.model_copy(update={"completion": SandboxCompletion.TIMED_OUT}),
        terminal=True,
    )
    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(contradictory_completion)


@pytest.mark.parametrize(
    "malformed_evidence",
    [
        BaselineEvidence.model_construct(),
        BaselineEvidence.model_construct(
            completion=SandboxCompletion.NORMAL,
            exit_code=1,
            stdout=1,
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        ),
        BaselineEvidence.model_construct(
            completion=SandboxCompletion.NORMAL,
            exit_code=1,
            stdout="E   assert False",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
    ],
)
def test_malformed_nested_evidence_is_a_typed_failure(
    malformed_evidence: BaselineEvidence,
) -> None:
    reproduced = _result("E   assert False")
    malformed = BaselineResult.model_construct(
        preparation=reproduced.preparation,
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=malformed_evidence,
        diagnostic_code=None,
        message=None,
        terminal=True,
    )

    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(malformed)


def test_malformed_nested_preparation_is_a_typed_failure() -> None:
    reproduced = _result("E   assert False")
    malformed = BaselineResult.model_construct(
        preparation=RepositoryPreparation.model_construct(),
        outcome=BaselineOutcome.REPRODUCED_FAILURE,
        evidence=reproduced.evidence,
        diagnostic_code=None,
        message=None,
        terminal=True,
    )

    with pytest.raises(ClassificationError, match="malformed_baseline"):
        classify_failure(malformed)
