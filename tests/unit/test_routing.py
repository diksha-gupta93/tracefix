from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent.policy import PatchPolicyResult
from app.agent.routing import (
    EXIT_CODES,
    RoutingEvidence,
    TerminalDiagnostic,
    TerminalOutcome,
    TerminalRoute,
    route_terminal,
    terminal_summary,
)
from app.agent.schemas import (
    BaselineDiagnosticCode,
    BaselineEvidence,
    BaselineOutcome,
    BaselineResult,
    RepositoryFingerprint,
    RepositoryPreparation,
    RevisionKind,
)
from app.sandbox.results import SandboxCompletion


def test_terminal_outcomes_and_exit_codes_are_exact() -> None:
    assert tuple(item.value for item in TerminalOutcome) == (
        "success",
        "candidate_failure",
        "policy_violation",
        "static_check_failure",
        "baseline_not_reproducible",
        "infrastructure_failure",
        "malformed_model_output",
    )
    assert EXIT_CODES == {
        TerminalOutcome.SUCCESS: 0,
        TerminalOutcome.CANDIDATE_FAILURE: 1,
        TerminalOutcome.POLICY_VIOLATION: 3,
        TerminalOutcome.STATIC_CHECK_FAILURE: 4,
        TerminalOutcome.BASELINE_NOT_REPRODUCIBLE: 5,
        TerminalOutcome.INFRASTRUCTURE_FAILURE: 6,
        TerminalOutcome.MALFORMED_MODEL_OUTPUT: 7,
    }


def test_routes_require_exact_constant_diagnostics() -> None:
    success = TerminalRoute(outcome=TerminalOutcome.SUCCESS, exit_code=0, diagnostic=None)
    assert success.diagnostic is None
    for outcome in tuple(TerminalOutcome)[1:]:
        diagnostic = TerminalDiagnostic.for_outcome(outcome)
        route = TerminalRoute(
            outcome=outcome,
            exit_code=EXIT_CODES[outcome],
            diagnostic=diagnostic,
        )
        assert route.diagnostic == diagnostic
        assert terminal_summary(outcome) == diagnostic.message


def _baseline(outcome: BaselineOutcome) -> BaselineResult:
    preparation = RepositoryPreparation(
        case_id="case_one",
        requested_revision="failing-v1",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=Path("prepared"),
        fingerprint=RepositoryFingerprint(digest="a" * 64),
    )
    if outcome is BaselineOutcome.REPRODUCED_FAILURE:
        return BaselineResult(
            preparation=preparation,
            outcome=outcome,
            evidence=BaselineEvidence(
                completion=SandboxCompletion.NORMAL,
                exit_code=1,
                stdout="",
                stderr="failure",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=0.1,
            ),
        )
    if outcome is BaselineOutcome.NOT_REPRODUCIBLE:
        return BaselineResult(
            preparation=preparation,
            outcome=outcome,
            evidence=BaselineEvidence(
                completion=SandboxCompletion.NORMAL,
                exit_code=0,
                stdout="passed",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=0.1,
            ),
            diagnostic_code=BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE,
            message="baseline did not reproduce",
        )
    return BaselineResult(
        preparation=preparation,
        outcome=outcome,
        evidence=None,
        diagnostic_code=BaselineDiagnosticCode.SANDBOX_EXECUTION_FAILED,
        message="sandbox execution failed",
    )


def test_baseline_routes_are_evidence_derived() -> None:
    assert (
        route_terminal(
            RoutingEvidence(baseline=_baseline(BaselineOutcome.NOT_REPRODUCIBLE))
        ).outcome
        is TerminalOutcome.BASELINE_NOT_REPRODUCIBLE
    )
    assert (
        route_terminal(
            RoutingEvidence(baseline=_baseline(BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC))
        ).outcome
        is TerminalOutcome.INFRASTRUCTURE_FAILURE
    )


def test_explicit_internal_failures_route_without_later_evidence() -> None:
    for outcome in (
        TerminalOutcome.CANDIDATE_FAILURE,
        TerminalOutcome.INFRASTRUCTURE_FAILURE,
        TerminalOutcome.MALFORMED_MODEL_OUTPUT,
    ):
        route = route_terminal(RoutingEvidence(failure_override=outcome))
        assert route.outcome is outcome
        assert route.exit_code == EXIT_CODES[outcome]


def test_missing_or_incomplete_evidence_can_never_route_to_success() -> None:
    assert route_terminal(RoutingEvidence()).outcome is TerminalOutcome.INFRASTRUCTURE_FAILURE
    assert (
        route_terminal(
            RoutingEvidence(baseline=_baseline(BaselineOutcome.REPRODUCED_FAILURE))
        ).outcome
        is TerminalOutcome.INFRASTRUCTURE_FAILURE
    )


def test_contradictory_later_evidence_is_rejected_and_cannot_override_baseline() -> None:
    raw = RoutingEvidence.model_construct(
        baseline=_baseline(BaselineOutcome.NOT_REPRODUCIBLE),
        policy=PatchPolicyResult(approved=True, violations=()),
    )
    assert route_terminal(raw).outcome is TerminalOutcome.INFRASTRUCTURE_FAILURE
    with pytest.raises(ValidationError, match="terminal baseline"):
        RoutingEvidence(
            baseline=_baseline(BaselineOutcome.NOT_REPRODUCIBLE),
            policy=PatchPolicyResult(approved=True, violations=()),
        )


def test_candidate_override_cannot_hide_later_evidence() -> None:
    raw = RoutingEvidence.model_construct(
        baseline=_baseline(BaselineOutcome.REPRODUCED_FAILURE),
        policy=PatchPolicyResult(approved=True, violations=()),
        failure_override=TerminalOutcome.CANDIDATE_FAILURE,
    )
    assert route_terminal(raw).outcome is TerminalOutcome.INFRASTRUCTURE_FAILURE
