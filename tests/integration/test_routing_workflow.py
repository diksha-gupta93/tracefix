from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path, PurePosixPath

import pytest

from app.agent.classification import classify_failure
from app.agent.graph import (
    GraphDependencies,
    LocalWorkspaceIntegrityBoundary,
    RunWorkspaces,
    build_repair_graph,
    invoke_repair_graph,
    new_orchestration_state,
)
from app.agent.model import ModelDiagnosticError, ModelProvider
from app.agent.policy import (
    PatchPolicyResult,
    PatchPolicyViolation,
    PatchPolicyViolationCode,
    validate_patch_policy,
)
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
    ModelDiagnostic,
    ModelDiagnosticCode,
    ModelDiagnosticStage,
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
    StaticFinding,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationResult,
    ValidatedCandidate,
)
from app.agent.verification import (
    CandidateVerificationRequest,
    CandidateVerificationResult,
    VerificationDiagnostic,
    VerificationDiagnosticCode,
    VerificationStage,
    VerificationStageResult,
    VerificationStageStatus,
)
from app.artifacts import (
    ArtifactPublication,
    ArtifactWriteError,
    LocalArtifactWriter,
    RunArtifact,
)
from app.cli import (
    ApplicationDependencies,
    LocalWorkspaceManager,
    ProviderRegistry,
    RepairApplication,
    main,
)
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationRequest,
    CandidateEvaluationResult,
    evaluate_candidate,
)
from app.sandbox.results import SandboxCompletion
from benchmarks.loader import load_trusted_case


class DummyProvider:
    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        del request
        raise AssertionError("operation stubs must prevent provider access")


def _configuration() -> ModelConfiguration:
    return ModelConfiguration(
        provider="stub",
        model_name="stub-model",
        temperature=0.0,
        max_output_tokens=1024,
        prompt_version="repair-v1.0",
    )


def _context(case_id: str, analysis: FailureAnalysis) -> ContextPackage:
    policy = ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS)
    policy_text = "\n".join(path.as_posix() for path in policy.protected_paths)
    items = tuple(
        sorted(
            (
                ContextItem(
                    path=PurePosixPath("tracefix/issue.txt"),
                    kind=ContextItemKind.ISSUE,
                    content="repair the boundary defect",
                    priority=0,
                    original_utf8_bytes=26,
                ),
                ContextItem(
                    path=PurePosixPath("tracefix/classification.txt"),
                    kind=ContextItemKind.CLASSIFICATION,
                    content=f"{analysis.category.value}\n{analysis.summary}",
                    priority=1,
                    original_utf8_bytes=len(
                        f"{analysis.category.value}\n{analysis.summary}".encode()
                    ),
                ),
                ContextItem(
                    path=PurePosixPath("tracefix/protected-paths.txt"),
                    kind=ContextItemKind.PROTECTED_PATH_POLICY,
                    content=policy_text,
                    priority=2,
                    original_utf8_bytes=len(policy_text.encode()),
                ),
            ),
            key=lambda item: (
                item.priority,
                item.path.as_posix().casefold(),
                item.source_line,
                item.kind.value,
            ),
        )
    )
    limit = 32_768
    actual = ContextPackage.downstream_utf8_size(
        case_id=case_id,
        items=items,
        omissions=(),
        limit_utf8_bytes=limit,
        safe_material_omitted=False,
    )
    return ContextPackage(
        case_id=case_id,
        failure_analysis=analysis,
        protected_path_policy=policy,
        items=items,
        limit_utf8_bytes=limit,
        actual_utf8_bytes=actual,
        safe_material_omitted=False,
    )


def _plan() -> RepairPlan:
    return RepairPlan(
        suspected_root_cause="off by one",
        files_expected_to_change=["src/pagination/pagination.py"],
        intended_behavioural_correction="use ceiling division",
        risks=[],
        validation_strategy="run tests",
        autonomous_repair_suitable=True,
    )


def _proposal() -> PatchProposal:
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


def _protected_proposal() -> PatchProposal:
    return PatchProposal(
        summary="modify protected documentation",
        root_cause="incorrect documentation",
        files_changed=["README.md"],
        unified_diff=("--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"),
        expected_effect="change documentation",
        risks=[],
        confidence=1.0,
    )


def _verification(outcome: str) -> CandidateVerificationResult:
    if outcome == "passed":
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
    prerequisite = VerificationDiagnostic(
        code=VerificationDiagnosticCode.PREREQUISITE_FAILED,
        message="a required earlier verification stage did not pass",
    )
    if outcome == "candidate_failed":
        first = VerificationStageResult(
            stage=VerificationStage.TARGETED,
            status=VerificationStageStatus.FAILED,
            completion=SandboxCompletion.NORMAL,
            exit_code=1,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        )
    else:
        first = VerificationStageResult(
            stage=VerificationStage.TARGETED,
            status=VerificationStageStatus.INFRASTRUCTURE_ERROR,
            diagnostic=VerificationDiagnostic(
                code=VerificationDiagnosticCode.SANDBOX_EXECUTION_FAILED,
                message="sandbox execution failed",
            ),
        )
    return CandidateVerificationResult(
        stages=(
            first,
            VerificationStageResult(
                stage=VerificationStage.VISIBLE_SUITE,
                status=VerificationStageStatus.NOT_RUN,
                diagnostic=prerequisite,
            ),
            VerificationStageResult(
                stage=VerificationStage.HIDDEN_SUITE,
                status=VerificationStageStatus.NOT_RUN,
                diagnostic=prerequisite,
            ),
        )
    )


def _static_result(request: StaticValidationRequest, outcome: str) -> StaticValidationResult:
    if outcome == "passed":
        repository = request.candidate_workspace_root / ("tracefix-candidate-" + "c" * 32)
        repository.mkdir()
        return StaticValidationResult(
            outcome=StaticValidationOutcome.PASSED,
            checks=tuple(
                StaticCheckResult(check=check, status=StaticCheckStatus.PASSED)
                for check in StaticCheck
            ),
            candidate=ValidatedCandidate(
                repository=repository.resolve(),
                fingerprint=RepositoryFingerprint(digest="b" * 64),
            ),
        )
    finding = StaticFinding(
        check=StaticCheck.PATCH_APPLY,
        rule_code="PATCH_CONFLICT",
        message="exact patch application failed",
    )
    return StaticValidationResult(
        outcome=StaticValidationOutcome.CANDIDATE_FAILED,
        checks=(
            StaticCheckResult(
                check=StaticCheck.PATCH_APPLY,
                status=StaticCheckStatus.FAILED,
                findings=(finding,),
            ),
            *(
                StaticCheckResult(check=check, status=StaticCheckStatus.PASSED)
                for check in tuple(StaticCheck)[1:]
            ),
        ),
    )


class ScenarioBoundaries:
    def __init__(self, scenario: str, calls: list[str]) -> None:
        self.scenario = scenario
        self.calls = calls

    def prepare(
        self,
        case_id: str,
        requested_revision: str | None,
        workspace_root: Path,
        *,
        destination_name: str | None = None,
    ) -> RepositoryPreparation:
        self.calls.append("prepare_repository")
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
        self.calls.append("run_baseline")
        if self.scenario == "baseline_not_reproducible":
            return BaselineResult(
                preparation=preparation,
                outcome=BaselineOutcome.NOT_REPRODUCIBLE,
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
                message="declared visible failure was not reproducible",
            )
        if self.scenario == "infrastructure_failure":
            return BaselineResult(
                preparation=preparation,
                outcome=BaselineOutcome.INFRASTRUCTURE_DIAGNOSTIC,
                evidence=None,
                diagnostic_code=BaselineDiagnosticCode.SANDBOX_EXECUTION_FAILED,
                message="sandbox execution failed",
            )
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
        self.calls.append("classify_failure")
        return classify_failure(baseline)

    def select(
        self,
        baseline: BaselineResult,
        analysis: FailureAnalysis,
        *,
        limit_utf8_bytes: int | None = None,
    ) -> ContextPackage:
        del limit_utf8_bytes
        self.calls.append("select_relevant_context")
        return _context(baseline.preparation.case_id, analysis)

    def plan(
        self,
        context: ContextPackage,
        configuration: ModelConfiguration,
        provider: ModelProvider,
    ) -> RepairPlan:
        del context, configuration, provider
        self.calls.append("plan_repair")
        if self.scenario == "malformed_model_output":
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    stage=ModelDiagnosticStage.PLANNING,
                    code=ModelDiagnosticCode.MALFORMED_STRUCTURED_OUTPUT,
                    message="model provider returned malformed structured output",
                )
            )
        if self.scenario == "unsuitable_repair":
            return _plan().model_copy(update={"autonomous_repair_suitable": False})
        if self.scenario == "policy_violation":
            return _plan().model_copy(update={"files_expected_to_change": ["README.md"]})
        return _plan()

    def generate(
        self,
        context: ContextPackage,
        plan: RepairPlan,
        configuration: ModelConfiguration,
        provider: ModelProvider,
    ) -> PatchProposal:
        del context, plan, configuration, provider
        self.calls.append("generate_patch")
        if self.scenario == "unsuitable_repair":
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    stage=ModelDiagnosticStage.PATCH_GENERATION,
                    code=ModelDiagnosticCode.UNSUITABLE_AUTONOMOUS_REPAIR,
                    message="repair plan is unsuitable for autonomous generation",
                )
            )
        if self.scenario == "policy_violation":
            return _protected_proposal()
        return _proposal()

    def policy(self, proposal: PatchProposal, policy: object) -> PatchPolicyResult:
        self.calls.append("validate_patch_policy")
        if self.scenario == "policy_mismatch":
            return PatchPolicyResult(
                approved=False,
                violations=(
                    PatchPolicyViolation(
                        code=PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED,
                        message="protected path modification is not allowed",
                        path=PurePosixPath("README.md"),
                    ),
                ),
            )
        return validate_patch_policy(proposal, policy)

    def validate(self, request: StaticValidationRequest) -> StaticValidationResult:
        self.calls.append("run_static_checks")
        return _static_result(
            request,
            "candidate_failed" if self.scenario == "static_check_failure" else "passed",
        )

    def verify(self, request: CandidateVerificationRequest) -> CandidateVerificationResult:
        del request
        self.calls.append("verify_in_sandbox")
        return _verification(
            "candidate_failed"
            if self.scenario == "candidate_failure"
            else "infrastructure"
            if self.scenario == "verification_infrastructure"
            else "passed"
        )

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluationResult:
        self.calls.append("evaluate_candidate")
        if self.scenario == "evaluation_failure":
            raise RuntimeError("untrusted evaluator detail")
        result = evaluate_candidate(request)
        if self.scenario == "evaluation_mismatch":
            return result.model_copy(update={"lines_added": result.lines_added + 1})
        return result


def _dependencies(scenario: str, calls: list[str]) -> GraphDependencies:
    boundaries = ScenarioBoundaries(scenario, calls)
    return GraphDependencies(
        preparer=boundaries,
        baseline=boundaries,
        context_selector=boundaries,
        static_validation=boundaries,
        verification=boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_configuration(),
        model_provider=DummyProvider(),
        classifier=boundaries.classify,
        planner=boundaries.plan,
        patch_generator=boundaries.generate,
        policy_validator=boundaries.policy,
        evaluator=boundaries.evaluate,
    )


class FingerprinterStub:
    def fingerprint(self, repository: Path) -> RepositoryFingerprint:
        digest = "a" * 64 if repository.name == "repository" else "b" * 64
        return RepositoryFingerprint(digest=digest)


def _workspaces(tmp_path: Path, run_id: str) -> RunWorkspaces:
    root = (tmp_path / "graph-run").resolve()
    root.mkdir()
    values = tuple(root / name for name in ("preparation", "candidate", "verification"))
    for value in values:
        value.mkdir()
    return RunWorkspaces(
        root=root,
        preparation=values[0],
        candidate=values[1],
        verification=values[2],
        ownership_token=run_id,
    )


@pytest.mark.parametrize(
    ("scenario", "outcome", "expected_calls"),
    [
        (
            "success",
            TerminalOutcome.SUCCESS,
            tuple(
                name
                for name in (
                    "prepare_repository",
                    "run_baseline",
                    "classify_failure",
                    "select_relevant_context",
                    "plan_repair",
                    "generate_patch",
                    "validate_patch_policy",
                    "run_static_checks",
                    "verify_in_sandbox",
                    "evaluate_candidate",
                )
            ),
        ),
        (
            "candidate_failure",
            TerminalOutcome.CANDIDATE_FAILURE,
            tuple(
                name
                for name in (
                    "prepare_repository",
                    "run_baseline",
                    "classify_failure",
                    "select_relevant_context",
                    "plan_repair",
                    "generate_patch",
                    "validate_patch_policy",
                    "run_static_checks",
                    "verify_in_sandbox",
                    "evaluate_candidate",
                )
            ),
        ),
        (
            "policy_violation",
            TerminalOutcome.POLICY_VIOLATION,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
            ),
        ),
        (
            "static_check_failure",
            TerminalOutcome.STATIC_CHECK_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
                "run_static_checks",
            ),
        ),
        (
            "baseline_not_reproducible",
            TerminalOutcome.BASELINE_NOT_REPRODUCIBLE,
            (
                "prepare_repository",
                "run_baseline",
            ),
        ),
        (
            "infrastructure_failure",
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
            ),
        ),
        (
            "malformed_model_output",
            TerminalOutcome.MALFORMED_MODEL_OUTPUT,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
            ),
        ),
        (
            "unsuitable_repair",
            TerminalOutcome.CANDIDATE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
            ),
        ),
        (
            "verification_infrastructure",
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
                "run_static_checks",
                "verify_in_sandbox",
                "evaluate_candidate",
            ),
        ),
        (
            "evaluation_failure",
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
                "run_static_checks",
                "verify_in_sandbox",
                "evaluate_candidate",
            ),
        ),
        (
            "policy_mismatch",
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
            ),
        ),
        (
            "evaluation_mismatch",
            TerminalOutcome.INFRASTRUCTURE_FAILURE,
            (
                "prepare_repository",
                "run_baseline",
                "classify_failure",
                "select_relevant_context",
                "plan_repair",
                "generate_patch",
                "validate_patch_policy",
                "run_static_checks",
                "verify_in_sandbox",
                "evaluate_candidate",
            ),
        ),
    ],
)
def test_compiled_graph_routes_every_terminal_branch_once(
    tmp_path: Path,
    scenario: str,
    outcome: TerminalOutcome,
    expected_calls: tuple[str, ...],
) -> None:
    calls: list[str] = []
    graph = build_repair_graph(_dependencies(scenario, calls))
    initial = new_orchestration_state(
        run_id="e" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, "e" * 32),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_configuration(),
    )
    result = invoke_repair_graph(graph, initial)
    assert tuple(calls) == expected_calls
    assert result.completed_nodes == (*expected_calls, "route_result")
    assert result.terminal_route is not None
    assert result.terminal_route.outcome is outcome
    assert len(calls) == len(set(calls))
    assert result.attempt_number == (1 if "plan_repair" in expected_calls else 0)
    if "generate_patch" in expected_calls and result.legacy_state.candidate_patch is not None:
        assert result.legacy_state.updated_at == datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("scenario", "expected_outcome", "exit_code"),
    [
        ("success", TerminalOutcome.SUCCESS, 0),
        ("candidate_failure", TerminalOutcome.CANDIDATE_FAILURE, 1),
        ("policy_violation", TerminalOutcome.POLICY_VIOLATION, 3),
        ("static_check_failure", TerminalOutcome.STATIC_CHECK_FAILURE, 4),
        ("baseline_not_reproducible", TerminalOutcome.BASELINE_NOT_REPRODUCIBLE, 5),
        ("infrastructure_failure", TerminalOutcome.INFRASTRUCTURE_FAILURE, 6),
        ("malformed_model_output", TerminalOutcome.MALFORMED_MODEL_OUTPUT, 7),
    ],
)
def test_application_reports_every_terminal_route(
    tmp_path: Path,
    scenario: str,
    expected_outcome: TerminalOutcome,
    exit_code: int,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    calls: list[str] = []

    def factory(
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock,
    ) -> GraphDependencies:
        del clock
        dependencies = _dependencies(scenario, calls)
        return replace(
            dependencies,
            model_configuration=configuration,
            model_provider=provider,
        )

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=lambda environment: _configuration(),
            provider_registry=ProviderRegistry({"stub": lambda configuration: DummyProvider()}),
            workspace_manager=LocalWorkspaceManager(workspace_base),
            artifact_writer=LocalArtifactWriter(lambda: "9" * 32),
            graph_dependencies_factory=factory,
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            run_id_factory=lambda: "a" * 32,
            invocation_directory=lambda: invocation,
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )
    assert application.repair("boundary_condition") == exit_code
    assert stderr.getvalue() == ""
    assert f"outcome={expected_outcome.value}" in stdout.getvalue()
    assert f"exit_code={exit_code}" in stdout.getvalue()
    artifact_path = (
        invocation / ".tracefix" / "artifacts" / ("boundary_condition-" + "a" * 32 + ".json")
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["outcome"] == expected_outcome
    assert not tuple(workspace_base.iterdir())


def test_application_writes_summary_artifact_and_cleans_workspace(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    calls: list[str] = []

    def factory(
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock,
    ) -> GraphDependencies:
        assert configuration == _configuration()
        assert provider is not None
        del clock
        return _dependencies("baseline_not_reproducible", calls)

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=lambda environment: _configuration(),
            provider_registry=ProviderRegistry({"stub": lambda configuration: DummyProvider()}),
            workspace_manager=LocalWorkspaceManager(workspace_base),
            artifact_writer=LocalArtifactWriter(lambda: "1" * 32),
            graph_dependencies_factory=factory,
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            run_id_factory=lambda: "2" * 32,
            invocation_directory=lambda: invocation,
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )
    assert application.repair("boundary_condition") == 5
    assert stderr.getvalue() == ""
    assert "outcome=baseline_not_reproducible" in stdout.getvalue()
    artifact = invocation / ".tracefix" / "artifacts" / ("boundary_condition-" + "2" * 32 + ".json")
    assert artifact.is_file()
    assert not tuple(workspace_base.iterdir())
    assert calls == ["prepare_repository", "run_baseline"]


def test_application_empty_provider_registry_fails_safely_without_fallback(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    calls: list[str] = []

    def factory(
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock,
    ) -> GraphDependencies:
        assert configuration == _configuration()
        assert provider is None
        del clock
        return replace(
            _dependencies("success", calls),
            model_configuration=configuration,
            model_provider=provider,
        )

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=lambda environment: _configuration(),
            provider_registry=ProviderRegistry(),
            workspace_manager=LocalWorkspaceManager(workspace_base),
            artifact_writer=LocalArtifactWriter(lambda: "3" * 32),
            graph_dependencies_factory=factory,
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            run_id_factory=lambda: "4" * 32,
            invocation_directory=lambda: invocation,
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )
    assert application.repair("boundary_condition") == 6
    assert stderr.getvalue() == ""
    assert "outcome=infrastructure_failure" in stdout.getvalue()
    artifact_path = (
        invocation / ".tracefix" / "artifacts" / ("boundary_condition-" + "4" * 32 + ".json")
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "infrastructure_failure"
    assert payload["attempt_number"] == 1
    assert payload["provider_name"] == "stub"
    assert not tuple(workspace_base.iterdir())
    assert calls == [
        "prepare_repository",
        "run_baseline",
        "classify_failure",
        "select_relevant_context",
    ]


class FailingArtifactWriter:
    def write(self, artifact: RunArtifact, invocation_directory: Path) -> ArtifactPublication:
        del artifact, invocation_directory
        raise ArtifactWriteError("untrusted publication detail")


class MalformedArtifactWriter:
    def write(self, artifact: RunArtifact, invocation_directory: Path) -> ArtifactPublication:
        del artifact, invocation_directory
        return ArtifactPublication.model_construct(relative_path=PurePosixPath("../invalid.json"))


class FailingOutput(StringIO):
    def write(self, value: str) -> int:
        del value
        raise RuntimeError("untrusted output detail")


class CleanupFailingWorkspaceManager(LocalWorkspaceManager):
    def cleanup(self, workspaces: RunWorkspaces) -> None:
        super().cleanup(workspaces)
        raise OSError("untrusted cleanup detail")


def _baseline_application(
    *,
    tmp_path: Path,
    workspace_manager: LocalWorkspaceManager,
    artifact_writer: LocalArtifactWriter | FailingArtifactWriter,
    stdout: StringIO,
    stderr: StringIO,
) -> RepairApplication:
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    calls: list[str] = []

    def factory(
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock,
    ) -> GraphDependencies:
        del configuration, provider, clock
        return _dependencies("baseline_not_reproducible", calls)

    return RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=lambda environment: _configuration(),
            provider_registry=ProviderRegistry(),
            workspace_manager=workspace_manager,
            artifact_writer=artifact_writer,
            graph_dependencies_factory=factory,
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            run_id_factory=lambda: "5" * 32,
            invocation_directory=lambda: invocation,
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )


def test_artifact_failure_returns_six_without_claiming_publication(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    application = _baseline_application(
        tmp_path=tmp_path,
        workspace_manager=LocalWorkspaceManager(workspace_base),
        artifact_writer=FailingArtifactWriter(),
        stdout=stdout,
        stderr=stderr,
    )

    assert application.repair("boundary_condition") == 6
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "tracefix: artifact publication failed\n"
    assert not tuple(workspace_base.iterdir())
    assert not (tmp_path / "invocation" / ".tracefix").exists()


def test_malformed_artifact_writer_result_is_rejected(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    application = _baseline_application(
        tmp_path=tmp_path,
        workspace_manager=LocalWorkspaceManager(workspace_base),
        artifact_writer=MalformedArtifactWriter(),
        stdout=stdout,
        stderr=stderr,
    )
    assert application.repair("boundary_condition") == 6
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "tracefix: artifact publication failed\n"


def test_stdout_failure_returns_six_after_safe_artifact_publication(tmp_path: Path) -> None:
    stdout = FailingOutput()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    application = _baseline_application(
        tmp_path=tmp_path,
        workspace_manager=LocalWorkspaceManager(workspace_base),
        artifact_writer=LocalArtifactWriter(lambda: "b" * 32),
        stdout=stdout,
        stderr=stderr,
    )
    assert application.repair("boundary_condition") == 6
    assert stderr.getvalue() == "tracefix: console output failed\n"
    assert (
        tmp_path
        / "invocation"
        / ".tracefix"
        / "artifacts"
        / ("boundary_condition-" + "5" * 32 + ".json")
    ).is_file()


def test_cleanup_failure_replaces_terminal_result_and_is_artifacted(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    application = _baseline_application(
        tmp_path=tmp_path,
        workspace_manager=CleanupFailingWorkspaceManager(workspace_base),
        artifact_writer=LocalArtifactWriter(lambda: "6" * 32),
        stdout=stdout,
        stderr=stderr,
    )

    assert application.repair("boundary_condition") == 6
    assert "outcome=infrastructure_failure" in stdout.getvalue()
    assert stderr.getvalue() == ""
    assert not tuple(workspace_base.iterdir())
    artifact_path = (
        tmp_path
        / "invocation"
        / ".tracefix"
        / "artifacts"
        / ("boundary_condition-" + "5" * 32 + ".json")
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "infrastructure_failure"
    assert payload["terminal_diagnostic"]["code"] == "infrastructure_failure"
    assert "untrusted cleanup detail" not in artifact_path.read_text(encoding="utf-8")


def test_keyboard_interrupt_still_cleans_owned_workspace(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    workspace_base = (tmp_path / "workspace-base").resolve()
    workspace_base.mkdir()
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()

    def interrupting_factory(
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock,
    ) -> GraphDependencies:
        del configuration, provider, clock
        raise KeyboardInterrupt

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=lambda environment: _configuration(),
            provider_registry=ProviderRegistry(),
            workspace_manager=LocalWorkspaceManager(workspace_base),
            artifact_writer=LocalArtifactWriter(lambda: "7" * 32),
            graph_dependencies_factory=interrupting_factory,
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            run_id_factory=lambda: "8" * 32,
            invocation_directory=lambda: invocation,
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert main(["repair", "boundary_condition"], lambda: application) == 130
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""
    assert not tuple(workspace_base.iterdir())
    assert not (invocation / ".tracefix").exists()
