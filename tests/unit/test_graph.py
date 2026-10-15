from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from app.agent.classification import classify_failure
from app.agent.graph import (
    NODE_NAMES,
    GraphDependencies,
    LocalWorkspaceIntegrityBoundary,
    OrchestrationState,
    RunWorkspaces,
    build_repair_graph,
    invoke_repair_graph,
    new_orchestration_state,
)
from app.agent.model import ModelProvider
from app.agent.policy import PatchPolicy, PatchPolicyResult, validate_patch_policy
from app.agent.routing import TerminalOutcome
from app.agent.schemas import (
    DEFAULT_PROTECTED_PATHS,
    BaselineDiagnosticCode,
    BaselineEvidence,
    BaselineOutcome,
    BaselineResult,
    ContextItem,
    ContextItemKind,
    ContextPackage,
    FailureAnalysis,
    ModelConfiguration,
    ModelProviderRequest,
    ModelProviderResponse,
    PatchProposal,
    ProtectedPathPolicy,
    RepairPlan,
    RepositoryFingerprint,
    RepositoryPreparation,
    RevisionKind,
)
from app.agent.static_checks import (
    StaticCheck,
    StaticCheckResult,
    StaticCheckStatus,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationResult,
    ValidatedCandidate,
)
from app.agent.verification import (
    CandidateVerificationRequest,
    CandidateVerificationResult,
    VerificationStage,
    VerificationStageResult,
    VerificationStageStatus,
)
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationRequest,
    CandidateEvaluationResult,
    evaluate_candidate,
)
from app.sandbox.results import SandboxCompletion
from benchmarks.loader import load_trusted_case


class PreparingStub:
    def prepare(
        self,
        case_id: str,
        requested_revision: str | None,
        workspace_root: Path,
        *,
        destination_name: str | None = None,
    ) -> RepositoryPreparation:
        repository = workspace_root / (destination_name or "repository")
        repository.mkdir()
        assert requested_revision is not None
        return RepositoryPreparation(
            case_id=case_id,
            requested_revision=requested_revision,
            revision_kind=RevisionKind.SNAPSHOT,
            prepared_repository=repository.resolve(),
            fingerprint=RepositoryFingerprint(digest="a" * 64),
        )


class NonReproducingBaselineStub:
    def run(self, preparation: RepositoryPreparation) -> BaselineResult:
        return BaselineResult(
            preparation=preparation,
            outcome=BaselineOutcome.NOT_REPRODUCIBLE,
            evidence=BaselineEvidence(
                completion=SandboxCompletion.NORMAL,
                exit_code=0,
                stdout="untrusted raw output",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=0.25,
            ),
            diagnostic_code=BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE,
            message="declared visible failure was not reproducible",
        )


class FingerprinterStub:
    def fingerprint(self, repository: Path) -> RepositoryFingerprint:
        del repository
        return RepositoryFingerprint(digest="a" * 64)


class UnexpectedBoundary:
    def select(
        self, baseline: object, analysis: object, *, limit_utf8_bytes: int | None = None
    ) -> object:
        raise AssertionError("later boundary must not run")

    def validate(self, request: object) -> object:
        raise AssertionError("later boundary must not run")

    def verify(self, request: object) -> object:
        raise AssertionError("later boundary must not run")


def _workspaces(tmp_path: Path, run_id: str) -> RunWorkspaces:
    root = (tmp_path / "run").resolve()
    root.mkdir()
    children = tuple(root / name for name in ("preparation", "candidate", "verification"))
    for child in children:
        child.mkdir()
    return RunWorkspaces(
        root=root,
        preparation=children[0],
        candidate=children[1],
        verification=children[2],
        ownership_token=run_id,
    )


def test_graph_has_exact_eleven_nodes_and_no_retry_edges(tmp_path: Path) -> None:
    boundary = UnexpectedBoundary()
    graph = build_repair_graph(
        GraphDependencies(
            preparer=PreparingStub(),
            baseline=NonReproducingBaselineStub(),
            context_selector=boundary,
            static_validation=boundary,
            verification=boundary,
            workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    drawable = graph.get_graph()
    assert set(drawable.nodes) == {"__start__", "__end__", *NODE_NAMES}
    assert {(edge.source, edge.target, edge.conditional) for edge in drawable.edges} == {
        ("__start__", "prepare_repository", False),
        ("prepare_repository", "run_baseline", True),
        ("prepare_repository", "route_result", True),
        ("run_baseline", "classify_failure", True),
        ("run_baseline", "route_result", True),
        ("classify_failure", "select_relevant_context", True),
        ("classify_failure", "route_result", True),
        ("select_relevant_context", "plan_repair", True),
        ("select_relevant_context", "route_result", True),
        ("plan_repair", "generate_patch", True),
        ("plan_repair", "route_result", True),
        ("generate_patch", "validate_patch_policy", True),
        ("generate_patch", "route_result", True),
        ("validate_patch_policy", "run_static_checks", True),
        ("validate_patch_policy", "route_result", True),
        ("run_static_checks", "verify_in_sandbox", True),
        ("run_static_checks", "route_result", True),
        ("verify_in_sandbox", "evaluate_candidate", True),
        ("verify_in_sandbox", "route_result", True),
        ("evaluate_candidate", "route_result", False),
        ("route_result", "__end__", False),
    }
    assert not any(edge.source == edge.target for edge in drawable.edges)
    assert not any(
        edge.source == "route_result" and edge.target != "__end__" for edge in drawable.edges
    )

    initial = new_orchestration_state(
        run_id="a" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, "a" * 32),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = invoke_repair_graph(graph, initial)
    assert result.completed_nodes == (
        "prepare_repository",
        "run_baseline",
        "route_result",
    )
    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.BASELINE_NOT_REPRODUCIBLE
    assert result.attempt_number == 0


def test_state_and_workspace_contracts_are_strict_and_frozen(tmp_path: Path) -> None:
    workspaces = _workspaces(tmp_path, "c" * 32)
    state = new_orchestration_state(
        run_id="c" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=workspaces,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        state.run_id = "d" * 32
    with pytest.raises(ValidationError):
        type(state).model_validate({**state.model_dump(), "extra": True}, strict=True)
    with pytest.raises(ValidationError):
        type(state).model_validate(
            {
                **state.model_dump(),
                "completed_at": state.started_at - timedelta(seconds=1),
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="execution prefix"):
        OrchestrationState.model_validate(
            {
                **state.model_dump(),
                "completed_nodes": ("prepare_repository", "classify_failure"),
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="ownership"):
        OrchestrationState.model_validate(
            {
                **state.model_dump(),
                "workspaces": {
                    **workspaces.model_dump(),
                    "ownership_token": "d" * 32,
                },
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="missing its evidence"):
        OrchestrationState.model_validate(
            {**state.model_dump(), "completed_nodes": ("prepare_repository",)},
            strict=True,
        )
    with pytest.raises(ValidationError, match="cannot record terminal"):
        OrchestrationState.model_validate(
            {**state.model_dump(), "completed_nodes": ("route_result",)},
            strict=True,
        )


def test_workspace_integrity_rejects_outside_paths_and_fingerprint_changes(
    tmp_path: Path,
) -> None:
    workspaces = _workspaces(tmp_path, "e" * 32)
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    preparation = RepositoryPreparation(
        case_id="boundary_condition",
        requested_revision="failing-v1",
        revision_kind=RevisionKind.SNAPSHOT,
        prepared_repository=outside,
        fingerprint=RepositoryFingerprint(digest="a" * 64),
    )
    boundary = LocalWorkspaceIntegrityBoundary(FingerprinterStub())
    with pytest.raises(ValueError, match="path is invalid"):
        boundary.validate_preparation(preparation, workspaces.preparation, "failing-v1")

    repository = workspaces.preparation / "repository"
    repository.mkdir()
    contained = preparation.model_copy(update={"prepared_repository": repository})

    class ChangedFingerprinter:
        def fingerprint(self, path: Path) -> RepositoryFingerprint:
            del path
            return RepositoryFingerprint(digest="f" * 64)

    with pytest.raises(ValueError, match="fingerprint changed"):
        LocalWorkspaceIntegrityBoundary(ChangedFingerprinter()).validate_preparation(
            contained, workspaces.preparation, "failing-v1"
        )


def test_workspace_integrity_accepts_owned_candidate_and_rejects_outside_candidate(
    tmp_path: Path,
) -> None:
    workspaces = _workspaces(tmp_path, "f" * 32)
    checks = tuple(
        StaticCheckResult(check=check, status=StaticCheckStatus.PASSED) for check in StaticCheck
    )
    candidate = workspaces.candidate / "candidate"
    candidate.mkdir()
    result = StaticValidationResult(
        outcome=StaticValidationOutcome.PASSED,
        checks=checks,
        candidate=ValidatedCandidate(
            repository=candidate,
            fingerprint=RepositoryFingerprint(digest="a" * 64),
        ),
    )
    boundary = LocalWorkspaceIntegrityBoundary(FingerprinterStub())
    boundary.validate_candidate(result, workspaces.candidate)
    outside = (tmp_path / "outside-candidate").resolve()
    outside.mkdir()
    with pytest.raises(ValueError, match="path is invalid"):
        boundary.validate_candidate(
            result.model_copy(
                update={"candidate": result.candidate.model_copy(update={"repository": outside})}
            ),
            workspaces.candidate,
        )


def test_workspace_integrity_rejects_candidate_symlink(tmp_path: Path) -> None:
    workspaces = _workspaces(tmp_path, "1" * 32)
    target = workspaces.candidate / "target"
    target.mkdir()
    alias = workspaces.candidate / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    result = StaticValidationResult(
        outcome=StaticValidationOutcome.PASSED,
        checks=tuple(
            StaticCheckResult(check=check, status=StaticCheckStatus.PASSED) for check in StaticCheck
        ),
        candidate=ValidatedCandidate(
            repository=alias.absolute(),
            fingerprint=RepositoryFingerprint(digest="a" * 64),
        ),
    )
    with pytest.raises(ValueError, match="path is invalid"):
        LocalWorkspaceIntegrityBoundary(FingerprinterStub()).validate_candidate(
            result, workspaces.candidate
        )


class UnusedProvider:
    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        del request
        raise AssertionError("the injected node operations must not call the provider")


def _model_configuration() -> ModelConfiguration:
    return ModelConfiguration(
        provider="stub",
        model_name="stub-model",
        temperature=0.0,
        max_output_tokens=1024,
        prompt_version="repair-v1.0",
    )


def _context_package(case_id: str, analysis: FailureAnalysis) -> ContextPackage:
    policy = ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS)
    policy_text = "\n".join(path.as_posix() for path in policy.protected_paths)
    classification = f"{analysis.category.value}\n{analysis.summary}"
    issue = "repair the boundary defect"
    items = (
        ContextItem(
            path=PurePosixPath("tracefix/issue.txt"),
            kind=ContextItemKind.ISSUE,
            content=issue,
            priority=0,
            original_utf8_bytes=len(issue.encode()),
        ),
        ContextItem(
            path=PurePosixPath("tracefix/classification.txt"),
            kind=ContextItemKind.CLASSIFICATION,
            content=classification,
            priority=1,
            original_utf8_bytes=len(classification.encode()),
        ),
        ContextItem(
            path=PurePosixPath("tracefix/protected-paths.txt"),
            kind=ContextItemKind.PROTECTED_PATH_POLICY,
            content=policy_text,
            priority=2,
            original_utf8_bytes=len(policy_text.encode()),
        ),
    )
    limit = 32_768
    return ContextPackage(
        case_id=case_id,
        failure_analysis=analysis,
        protected_path_policy=policy,
        items=items,
        limit_utf8_bytes=limit,
        actual_utf8_bytes=ContextPackage.downstream_utf8_size(
            case_id=case_id,
            items=items,
            omissions=(),
            limit_utf8_bytes=limit,
            safe_material_omitted=False,
        ),
        safe_material_omitted=False,
    )


def _repair_plan() -> RepairPlan:
    return RepairPlan(
        suspected_root_cause="off by one",
        files_expected_to_change=["src/pagination/pagination.py"],
        intended_behavioural_correction="use ceiling division",
        risks=[],
        validation_strategy="run tests",
        autonomous_repair_suitable=True,
    )


def _patch_proposal() -> PatchProposal:
    return PatchProposal(
        summary="repair page count",
        root_cause="off by one",
        files_changed=["src/pagination/pagination.py"],
        unified_diff=(
            "--- a/src/pagination/pagination.py\n"
            "+++ b/src/pagination/pagination.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def page_count(items: int, size: int) -> int:\n"
            "-    return items // size + 1\n"
            "+    return (items + size - 1) // size\n"
        ),
        expected_effect="correct exact multiples",
        risks=[],
        confidence=1.0,
    )


class TransitionBoundaries:
    def __init__(self, failure_node: str | None, calls: list[str]) -> None:
        self._failure_node = failure_node
        self._calls = calls

    def _record(self, node: str) -> None:
        self._calls.append(node)
        if node == self._failure_node:
            raise RuntimeError("injected boundary failure")

    def prepare(
        self,
        case_id: str,
        requested_revision: str | None,
        workspace_root: Path,
        *,
        destination_name: str | None = None,
    ) -> RepositoryPreparation:
        self._record("prepare_repository")
        repository = workspace_root / (destination_name or "repository")
        repository.mkdir()
        assert requested_revision is not None
        return RepositoryPreparation(
            case_id=case_id,
            requested_revision=requested_revision,
            revision_kind=RevisionKind.SNAPSHOT,
            prepared_repository=repository.resolve(),
            fingerprint=RepositoryFingerprint(digest="a" * 64),
        )

    def run(self, preparation: RepositoryPreparation) -> BaselineResult:
        self._record("run_baseline")
        return BaselineResult(
            preparation=preparation,
            outcome=BaselineOutcome.REPRODUCED_FAILURE,
            evidence=BaselineEvidence(
                completion=SandboxCompletion.NORMAL,
                exit_code=1,
                stdout="",
                stderr="AssertionError: expected 2",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=0.1,
            ),
        )

    def classify(self, baseline: BaselineResult) -> FailureAnalysis:
        self._record("classify_failure")
        return classify_failure(baseline)

    def select(
        self,
        baseline: BaselineResult,
        analysis: FailureAnalysis,
        *,
        limit_utf8_bytes: int | None = None,
    ) -> ContextPackage:
        del limit_utf8_bytes
        self._record("select_relevant_context")
        return _context_package(baseline.preparation.case_id, analysis)

    def plan(
        self,
        context: ContextPackage,
        configuration: ModelConfiguration,
        provider: ModelProvider,
    ) -> RepairPlan:
        del context, configuration, provider
        self._record("plan_repair")
        return _repair_plan()

    def generate(
        self,
        context: ContextPackage,
        plan: RepairPlan,
        configuration: ModelConfiguration,
        provider: ModelProvider,
    ) -> PatchProposal:
        del context, plan, configuration, provider
        self._record("generate_patch")
        return _patch_proposal()

    def policy(self, proposal: PatchProposal, policy: PatchPolicy) -> PatchPolicyResult:
        self._record("validate_patch_policy")
        return validate_patch_policy(proposal, policy)

    def validate(self, request: StaticValidationRequest) -> StaticValidationResult:
        self._record("run_static_checks")
        repository = request.candidate_workspace_root / "candidate-result"
        repository.mkdir()
        return StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=tuple(
                StaticCheckResult(check=check, status=StaticCheckStatus.PASSED)
                for check in StaticCheck
            ),
            candidate=ValidatedCandidate(
                repository=repository.resolve(),
                fingerprint=RepositoryFingerprint(digest="a" * 64),
            ),
        )

    def verify(self, request: CandidateVerificationRequest) -> CandidateVerificationResult:
        del request
        self._record("verify_in_sandbox")
        return CandidateVerificationResult(
            stages=tuple(
                VerificationStageResult(
                    stage=stage,
                    status=VerificationStageStatus.PASSED,
                    completion=SandboxCompletion.NORMAL,
                    exit_code=0,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_seconds=0.1,
                )
                for stage in VerificationStage
            )
        )

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluationResult:
        self._record("evaluate_candidate")
        return evaluate_candidate(request)


def _transition_dependencies(failure_node: str | None, calls: list[str]) -> GraphDependencies:
    boundaries = TransitionBoundaries(failure_node, calls)
    return GraphDependencies(
        preparer=boundaries,
        baseline=boundaries,
        context_selector=boundaries,
        static_validation=boundaries,
        verification=boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
        model_provider=UnusedProvider(),
        classifier=boundaries.classify,
        planner=boundaries.plan,
        patch_generator=boundaries.generate,
        policy_validator=boundaries.policy,
        evaluator=boundaries.evaluate,
    )


def _invoke_transition_graph(
    tmp_path: Path, failure_node: str | None, calls: list[str]
) -> OrchestrationState:
    run_id = "9" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )
    return invoke_repair_graph(
        build_repair_graph(_transition_dependencies(failure_node, calls)), initial
    )


def test_normal_graph_executes_every_forward_transition_once(tmp_path: Path) -> None:
    calls: list[str] = []
    result = _invoke_transition_graph(tmp_path, None, calls)
    assert tuple(calls) == NODE_NAMES[:-1]
    assert result.completed_nodes == NODE_NAMES
    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.SUCCESS


@pytest.mark.parametrize("failure_node", NODE_NAMES[:-1])
def test_each_node_failure_executes_its_terminal_conditional_edge_once(
    tmp_path: Path, failure_node: str
) -> None:
    calls: list[str] = []
    result = _invoke_transition_graph(tmp_path, failure_node, calls)
    failure_index = NODE_NAMES.index(failure_node)
    expected_prefix = NODE_NAMES[: failure_index + 1]
    assert tuple(calls) == expected_prefix
    assert result.completed_nodes == (*expected_prefix, "route_result")
    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.INFRASTRUCTURE_FAILURE
