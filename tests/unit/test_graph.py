from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path, PurePosixPath

import pytest
import requests
from langsmith import Client as LangSmithClient
from pydantic import ValidationError

import app.observability as observability
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
from app.agent.model import ModelDiagnosticError, ModelProvider
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
    CandidateModelUsage,
    evaluate_candidate,
)
from app.evaluation.live_model import LiveBenchmarkConfiguration
from app.evaluation.reporting import LocalBenchmarkReportWriter
from app.observability import (
    LANGSMITH_API_ENDPOINT,
    LANGSMITH_DIAGNOSTIC_PROJECT,
    NOOP_TRACER,
    SAFE_METADATA_ALLOW_LIST,
    DiagnosticRun,
    DiagnosticSpan,
    DiagnosticTracer,
    LangSmithDiagnosticTracer,
    SafeMetadata,
    create_diagnostic_tracer,
    safe_metadata,
)
from app.sandbox.results import SandboxCompletion
from benchmarks.loader import TrustedCase, load_development_cases, load_trusted_case


class GraphRecordingSpan:
    def __init__(self, record: dict[str, object]) -> None:
        self._record = record

    def add_metadata(self, metadata: SafeMetadata) -> None:
        target = self._record["metadata"]
        assert isinstance(target, dict)
        target.update(safe_metadata(metadata))


class GraphRecordingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self._stack: list[GraphRecordingSpan] = []

    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def span(self, name: str, metadata: SafeMetadata | None = None) -> Iterator[DiagnosticSpan]:
        record: dict[str, object] = {
            "name": name,
            "parent": None if not self._stack else self._stack[-1]._record["name"],
            "metadata": safe_metadata(metadata),
        }
        span = GraphRecordingSpan(record)
        self.records.append(record)
        self._stack.append(span)
        try:
            yield span
        finally:
            self._stack.pop()

    def add_metadata(self, metadata: SafeMetadata) -> None:
        if self._stack:
            self._stack[-1].add_metadata(metadata)


class CapturedDiagnosticRun:
    def __init__(
        self,
        name: str,
        metadata: SafeMetadata,
        parent: DiagnosticRun | None,
    ) -> None:
        self.payload: dict[str, object] = {
            "name": name,
            "metadata": safe_metadata(metadata),
            "parent": None if parent is None else id(parent),
            "inputs": {},
            "outputs": {},
            "error": None,
            "tags": [],
        }

    def add_metadata(self, metadata: SafeMetadata) -> None:
        target = self.payload["metadata"]
        assert isinstance(target, dict)
        target.update(safe_metadata(metadata))

    def close(self) -> None:
        self.payload["closed"] = True


class CapturingDiagnosticExporter:
    def __init__(self) -> None:
        self.runs: list[CapturedDiagnosticRun] = []

    def start_run(
        self,
        name: str,
        metadata: SafeMetadata,
        parent: DiagnosticRun | None,
    ) -> DiagnosticRun:
        run = CapturedDiagnosticRun(name, metadata, parent)
        self.runs.append(run)
        return run


class NoNetworkLangSmithSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.intercepted: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> requests.Response:
        self.intercepted.append({"method": method, "url": url, "kwargs": kwargs})
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = url
        response.headers = requests.structures.CaseInsensitiveDict()
        response.request = requests.Request(method, url).prepare()
        return response


def _production_no_network_tracer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    credential: str,
    environment_sentinel: str,
) -> tuple[DiagnosticTracer, NoNetworkLangSmithSession, LangSmithClient]:
    session = NoNetworkLangSmithSession()
    clients: list[LangSmithClient] = []

    def client_factory(**values: object) -> LangSmithClient:
        actual = dict(values)
        actual["auto_batch_tracing"] = False
        actual["session"] = session
        client = LangSmithClient(**actual)  # type: ignore[arg-type]
        clients.append(client)
        return client

    monkeypatch.setattr(observability, "Client", client_factory)
    tracer = create_diagnostic_tracer(
        {
            "TRACEFIX_LANGSMITH_DIAGNOSTICS": "true",
            "LANGSMITH_API_KEY": credential,
            "LANGSMITH_PROJECT": LANGSMITH_DIAGNOSTIC_PROJECT,
            "LANGSMITH_HIDE_INPUTS": "true",
            "LANGSMITH_HIDE_OUTPUTS": "true",
            "LANGSMITH_TRACING": "false",
            "TRACEFIX_SECURITY_SENTINEL": environment_sentinel,
        }
    )
    assert len(clients) == 1
    return tracer, session, clients[0]


def _intercepted_trace_payloads(session: NoNetworkLangSmithSession) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for request in session.intercepted:
        kwargs = request["kwargs"]
        assert isinstance(kwargs, dict)
        raw = kwargs.get("data")
        assert isinstance(raw, bytes | str)
        value = json.loads(raw)
        assert isinstance(value, dict)
        payloads.append(value)
    return payloads


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


class QueueGraphProvider:
    def __init__(self, *responses: ModelProviderResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelProviderRequest] = []

    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        self.requests.append(request)
        return self.responses.pop(0)


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


def _sentinel_context_package(
    case_id: str,
    analysis: FailureAnalysis,
    *,
    source: str,
    visible_test: str,
    rendered_prompt: str,
) -> ContextPackage:
    policy = ProtectedPathPolicy(protected_paths=DEFAULT_PROTECTED_PATHS)
    values = (
        ("tracefix/issue.txt", ContextItemKind.ISSUE, rendered_prompt),
        (
            "tracefix/classification.txt",
            ContextItemKind.CLASSIFICATION,
            f"{analysis.category.value}\n{analysis.summary}",
        ),
        (
            "tracefix/protected-paths.txt",
            ContextItemKind.PROTECTED_PATH_POLICY,
            "\n".join(path.as_posix() for path in policy.protected_paths),
        ),
        ("tests/test_sentinel.py", ContextItemKind.VISIBLE_TEST, visible_test),
        ("src/sentinel.py", ContextItemKind.SOURCE, source),
    )
    items = tuple(
        ContextItem(
            path=PurePosixPath(path),
            kind=kind,
            content=content,
            priority=priority,
            original_utf8_bytes=len(content.encode()),
        )
        for priority, (path, kind, content) in enumerate(values)
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


def test_graph_explicitly_disables_environment_driven_langchain_tracing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.tracers.langchain import LangChainTracer

    installed = False

    def unexpected_initialization(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        nonlocal installed
        installed = True

    monkeypatch.setattr(LangChainTracer, "__init__", unexpected_initialization)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    calls: list[str] = []
    result = _invoke_transition_graph(tmp_path, None, calls)

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.SUCCESS
    assert installed is False


def test_deeply_nested_json_keeps_graph_benchmark_and_exit_semantics_with_tracing(
    tmp_path: Path,
) -> None:
    from app.evaluation.benchmark import BenchmarkCaseExecution, run_benchmark_cases
    from benchmarks.loader import load_development_cases

    raw_json = "[" * 20_000 + "0" + "]" * 20_000

    def execute(tracer: DiagnosticTracer, suffix: str) -> OrchestrationState:
        calls: list[str] = []
        boundaries = TransitionBoundaries(None, calls)
        dependencies = GraphDependencies(
            preparer=boundaries,
            baseline=boundaries,
            context_selector=boundaries,
            static_validation=boundaries,
            verification=boundaries,
            workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            model_configuration=_model_configuration(),
            model_provider=QueueGraphProvider(ModelProviderResponse(raw_json=raw_json)),
            classifier=boundaries.classify,
            policy_validator=boundaries.policy,
            evaluator=boundaries.evaluate,
            tracer=tracer,
        )
        run_id = suffix * 32
        workspace_parent = tmp_path / suffix
        workspace_parent.mkdir()
        initial = new_orchestration_state(
            run_id=run_id,
            trusted_case=load_trusted_case("boundary_condition"),
            workspaces=_workspaces(workspace_parent, run_id),
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            model_configuration=_model_configuration(),
        )
        return invoke_repair_graph(build_repair_graph(dependencies), initial)

    untraced = execute(NOOP_TRACER, "2")
    traced = execute(GraphRecordingTracer(), "3")

    assert untraced.terminal_route is not None
    assert traced.terminal_route is not None
    assert traced.terminal_route == untraced.terminal_route
    assert traced.terminal_route.outcome is TerminalOutcome.MALFORMED_MODEL_OUTPUT
    assert traced.legacy_state.status == untraced.legacy_state.status
    assert traced.failure_override == untraced.failure_override

    def report(state: OrchestrationState) -> object:
        route = state.terminal_route
        assert route is not None
        return run_benchmark_cases(
            load_development_cases(),
            lambda case: BenchmarkCaseExecution(
                case_id=case.manifest.case_id,
                terminal_outcome=route.outcome,
                model_usage=None,
                artifact_path=PurePosixPath(
                    ".tracefix", "artifacts", f"{case.manifest.case_id}-{'a' * 32}.json"
                ),
            ),
        )

    assert report(traced) == report(untraced)


class MatrixScenarioProvider:
    def __init__(self, scenario: str) -> None:
        self._scenario = scenario
        self.requests: list[ModelProviderRequest] = []

    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        self.requests.append(request)
        if request.operation.value == "repair_planning":
            if self._scenario == "repair_plan_malformed":
                raw_json = "not-json"
            elif self._scenario == "repair_plan_deeply_nested":
                raw_json = "[" * 20_000 + "0" + "]" * 20_000
            elif self._scenario == "repair_plan_schema_invalid":
                raw_json = "{}"
            else:
                raw_json = _repair_plan().model_dump_json()
        elif self._scenario == "patch_proposal_malformed":
            raw_json = "not-json"
        elif self._scenario == "patch_proposal_deeply_nested":
            raw_json = "[" * 20_000 + "0" + "]" * 20_000
        elif self._scenario == "patch_proposal_schema_invalid":
            raw_json = "{}"
        else:
            raw_json = _patch_proposal().model_dump_json()
        return ModelProviderResponse(raw_json=raw_json)

    def snapshot_usage(self) -> CandidateModelUsage:
        calls = len(self.requests)
        return CandidateModelUsage(
            model_latency_seconds=float(calls),
            input_tokens=10 * calls,
            output_tokens=5 * calls,
            estimated_cost_usd=Decimal(0),
        )


class MatrixGraphRepairApplication:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._next_run = 1
        self.observations: list[tuple[str, str, str | None]] = []

    def execute_trusted_case(
        self,
        case: TrustedCase,
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        *,
        tracer: DiagnosticTracer = NOOP_TRACER,
    ) -> object:
        from app.cli import RepairRunExecution

        assert configuration is not None
        assert isinstance(provider, MatrixScenarioProvider)
        calls: list[str] = []
        boundaries = TransitionBoundaries(None, calls)
        dependencies = GraphDependencies(
            preparer=boundaries,
            baseline=boundaries,
            context_selector=boundaries,
            static_validation=boundaries,
            verification=boundaries,
            workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
            clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            model_configuration=configuration,
            model_provider=provider,
            model_usage_reader=provider,
            classifier=boundaries.classify,
            policy_validator=boundaries.policy,
            evaluator=boundaries.evaluate,
            tracer=tracer,
        )
        run_id = f"{self._next_run:032x}"
        self._next_run += 1
        case_root = self._root / run_id
        case_root.mkdir()
        initial = new_orchestration_state(
            run_id=run_id,
            trusted_case=case,
            workspaces=_workspaces(case_root, run_id),
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            model_configuration=configuration,
        )
        state = invoke_repair_graph(build_repair_graph(dependencies), initial)
        route = state.terminal_route
        assert route is not None
        self.observations.append(
            (
                route.outcome.value,
                state.legacy_state.status.value,
                None if route.diagnostic is None else route.diagnostic.code.value,
            )
        )
        return RepairRunExecution(
            state=state,
            artifact_path=PurePosixPath(
                ".tracefix", "artifacts", f"{case.manifest.case_id}-{run_id}.json"
            ),
        )


def _run_matrix_benchmark(
    root: Path,
    scenario: str,
    tracer: DiagnosticTracer,
) -> tuple[int, object, list[tuple[str, str, str | None]]]:
    from app.agent.schemas import ModelConfiguration
    from app.cli import BenchmarkApplication, BenchmarkApplicationDependencies, main
    from app.evaluation.benchmark import BenchmarkReport

    root.mkdir()
    runtime_root = root / "runtime"
    runtime_root.mkdir()
    repair = MatrixGraphRepairApplication(runtime_root)
    configuration = ModelConfiguration(
        provider="groq",
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
    )
    live_configuration = LiveBenchmarkConfiguration(
        reasoning_effort="low",
        request_timeout_seconds=120,
        input_price_usd_per_million_tokens=Decimal(0),
        output_price_usd_per_million_tokens=Decimal(0),
    )

    def provider_factory(
        checked: ModelConfiguration,
        live: LiveBenchmarkConfiguration,
        environment: Mapping[str, str],
    ) -> ModelProvider:
        del checked, live, environment
        return MatrixScenarioProvider(scenario)

    application = BenchmarkApplication(
        BenchmarkApplicationDependencies(
            repair_application=repair,  # type: ignore[arg-type]
            development_case_loader=load_development_cases,
            configuration_loader=lambda environment=None: configuration,
            live_configuration_loader=lambda environment: live_configuration,
            live_provider_factory=provider_factory,
            report_writer=LocalBenchmarkReportWriter(identifier_factory=lambda: "d" * 32),
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            run_id_factory=lambda: "e" * 32,
            invocation_directory=lambda: root,
            environment={},
            stdout=StringIO(),
            stderr=StringIO(),
            tracer=tracer,
        )
    )
    exit_code = main(
        ["benchmark", "--split", "development", "--provider", "groq"],
        application_factory=lambda: (_ for _ in ()).throw(AssertionError()),
        benchmark_application_factory=lambda: application,
    )
    report_path = root / ".tracefix" / "benchmarks" / ("e" * 32) / "report.json"
    report = BenchmarkReport.model_validate_json(
        report_path.read_text(encoding="utf-8"), strict=True
    )
    return exit_code, report, repair.observations


@pytest.mark.parametrize(
    "scenario",
    [
        "repair_plan_malformed",
        "repair_plan_deeply_nested",
        "repair_plan_valid",
        "repair_plan_schema_invalid",
        "patch_proposal_malformed",
        "patch_proposal_deeply_nested",
        "patch_proposal_valid",
        "patch_proposal_schema_invalid",
    ],
)
def test_full_benchmark_cli_semantics_are_identical_with_tracing(
    tmp_path: Path,
    scenario: str,
) -> None:
    untraced_exit, untraced_report, untraced_observations = _run_matrix_benchmark(
        tmp_path / "untraced", scenario, NOOP_TRACER
    )
    traced_exit, traced_report, traced_observations = _run_matrix_benchmark(
        tmp_path / "traced", scenario, GraphRecordingTracer()
    )

    assert traced_exit == untraced_exit == 0
    assert traced_observations == untraced_observations
    assert traced_report.model_dump(warnings="none") == untraced_report.model_dump(warnings="none")
    assert traced_report.counts.attempted == 5
    assert traced_report.counts.not_run == 0
    assert [item.classification for item in traced_report.cases] == [
        item.classification for item in untraced_report.cases
    ]
    assert [item.terminal_outcome for item in traced_report.cases] == [
        item.terminal_outcome for item in untraced_report.cases
    ]


def test_graph_emits_only_reached_stage_hierarchy_with_verification_and_evaluation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    tracer = GraphRecordingTracer()
    dependencies = replace(_transition_dependencies(None, calls), tracer=tracer)
    run_id = "7" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )

    with (
        tracer.span("tracefix.live_benchmark", safe_metadata(benchmark_run_id="6" * 32)),
        tracer.span("case.boundary_condition", safe_metadata(case_id="boundary_condition")),
    ):
        result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.SUCCESS
    assert [record["name"] for record in tracer.records] == [
        "tracefix.live_benchmark",
        "case.boundary_condition",
        "prepare_repository",
        "run_baseline",
        "classify_failure",
        "build_context",
        "plan_repair",
        "generate_patch",
        "patch_policy",
        "static_checks",
        "verification",
        "evaluation",
    ]
    assert all(record["parent"] == "case.boundary_condition" for record in tracer.records[2:])
    verification = next(record for record in tracer.records if record["name"] == "verification")
    evaluation = next(record for record in tracer.records if record["name"] == "evaluation")
    assert isinstance(verification["metadata"], dict)
    assert verification["metadata"]["verification_result"] == "passed"
    assert isinstance(evaluation["metadata"], dict)
    assert evaluation["metadata"]["evaluation_result"] == "passed"


def test_production_graph_tracing_excludes_separate_security_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sentinel = "SENTINEL_ACTUAL_REPOSITORY_SOURCE"
    visible_test_sentinel = "SENTINEL_VISIBLE_TEST_CONTENT"
    hidden_test_sentinel = "SENTINEL_HIDDEN_TEST_CONTENT"
    prompt_sentinel = "SENTINEL_RENDERED_PROMPT_CONTENT"
    model_sentinel = "SENTINEL_PROVIDER_MODEL_RESPONSE"
    patch_sentinel = "SENTINEL_GENERATED_PATCH_CONTENT"
    diff_sentinel = "SENTINEL_UNIFIED_DIFF_CONTENT"
    environment_sentinel = "SENTINEL_CREDENTIAL_ENVIRONMENT_VALUE"
    monkeypatch.setenv("TRACEFIX_SECURITY_SENTINEL", environment_sentinel)

    calls: list[str] = []

    class SentinelBoundaries(TransitionBoundaries):
        def select(
            self,
            baseline: BaselineResult,
            analysis: FailureAnalysis,
            *,
            limit_utf8_bytes: int | None = None,
        ) -> ContextPackage:
            del limit_utf8_bytes
            self._record("select_relevant_context")
            return _sentinel_context_package(
                baseline.preparation.case_id,
                analysis,
                source=source_sentinel,
                visible_test=visible_test_sentinel,
                rendered_prompt=prompt_sentinel,
            )

    boundaries = SentinelBoundaries(None, calls)
    plan = _repair_plan().model_copy(update={"suspected_root_cause": model_sentinel})
    proposal = _patch_proposal().model_copy(
        update={
            "summary": patch_sentinel,
            "unified_diff": _patch_proposal().unified_diff.replace(
                "+    return (items + size - 1) // size",
                f"+    return (items + size - 1) // size  # {diff_sentinel}",
            ),
        }
    )
    provider = QueueGraphProvider(
        ModelProviderResponse(raw_json=plan.model_dump_json()),
        ModelProviderResponse(raw_json=proposal.model_dump_json()),
    )
    exporter = CapturingDiagnosticExporter()
    tracer = LangSmithDiagnosticTracer(exporter)
    dependencies = GraphDependencies(
        preparer=boundaries,
        baseline=boundaries,
        context_selector=boundaries,
        static_validation=boundaries,
        verification=boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
        model_provider=provider,
        classifier=boundaries.classify,
        policy_validator=boundaries.policy,
        evaluator=boundaries.evaluate,
        tracer=tracer,
    )
    run_id = "4" * 32
    hidden_test = tmp_path / "hidden_test.py"
    hidden_test.write_text(hidden_test_sentinel, encoding="utf-8")
    trusted_case = load_trusted_case("boundary_condition").model_copy(
        update={"hidden_tests": (hidden_test.resolve(),)}
    )
    workspace_parent = tmp_path / "security"
    workspace_parent.mkdir()
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=trusted_case,
        workspaces=_workspaces(workspace_parent, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )

    with (
        tracer.span("tracefix.live_benchmark", safe_metadata(benchmark_run_id="8" * 32)),
        tracer.span("case.boundary_condition", safe_metadata(case_id="boundary_condition")),
    ):
        result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.SUCCESS
    assert len(provider.requests) == 2
    assert source_sentinel in provider.requests[0].rendered_prompt
    assert visible_test_sentinel in provider.requests[0].rendered_prompt
    assert prompt_sentinel in provider.requests[0].rendered_prompt
    captured = json.dumps([run.payload for run in exporter.runs], sort_keys=True)
    for sentinel in (
        source_sentinel,
        visible_test_sentinel,
        hidden_test_sentinel,
        prompt_sentinel,
        model_sentinel,
        patch_sentinel,
        diff_sentinel,
        environment_sentinel,
    ):
        assert sentinel not in captured
    names = [run.payload["name"] for run in exporter.runs]
    assert names == [
        "tracefix.live_benchmark",
        "case.boundary_condition",
        "prepare_repository",
        "run_baseline",
        "classify_failure",
        "build_context",
        "plan_repair",
        "provider_request",
        "parse_repair_plan",
        "validate_repair_plan",
        "generate_patch",
        "provider_request",
        "parse_patch_proposal",
        "validate_patch_proposal",
        "validate_patch_envelope",
        "patch_policy",
        "static_checks",
        "verification",
        "evaluation",
    ]
    assert all(run.payload["inputs"] == {} for run in exporter.runs)
    assert all(run.payload["outputs"] == {} for run in exporter.runs)
    assert all(run.payload["error"] is None for run in exporter.runs)
    assert all(run.payload["tags"] == [] for run in exporter.runs)


def test_compiled_graph_real_langsmith_serialization_is_content_free_and_rest_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sentinel = "SENTINEL_REAL_BOUNDARY_REPOSITORY_SOURCE"
    visible_test_sentinel = "SENTINEL_REAL_BOUNDARY_VISIBLE_TEST"
    hidden_test_sentinel = "SENTINEL_REAL_BOUNDARY_HIDDEN_TEST"
    prompt_sentinel = "SENTINEL_REAL_BOUNDARY_RENDERED_PROMPT"
    model_sentinel = "SENTINEL_REAL_BOUNDARY_MODEL_RESPONSE"
    patch_sentinel = "SENTINEL_REAL_BOUNDARY_GENERATED_PATCH"
    diff_sentinel = "SENTINEL_REAL_BOUNDARY_UNIFIED_DIFF"
    credential_sentinel = "SENTINEL_REAL_BOUNDARY_CREDENTIAL"
    environment_sentinel = "SENTINEL_REAL_BOUNDARY_ENVIRONMENT"
    validation_sentinel = "SENTINEL_REAL_BOUNDARY_VALIDATION_INPUT"
    hidden_test = tmp_path / "consumed_hidden_test.py"
    hidden_test.write_text(hidden_test_sentinel, encoding="utf-8")
    hidden_consumed = False

    class ProductionSentinelBoundaries(TransitionBoundaries):
        def select(
            self,
            baseline: BaselineResult,
            analysis: FailureAnalysis,
            *,
            limit_utf8_bytes: int | None = None,
        ) -> ContextPackage:
            del limit_utf8_bytes
            self._record("select_relevant_context")
            return _sentinel_context_package(
                baseline.preparation.case_id,
                analysis,
                source=source_sentinel,
                visible_test=visible_test_sentinel,
                rendered_prompt=prompt_sentinel,
            )

        def verify(self, request: CandidateVerificationRequest) -> CandidateVerificationResult:
            nonlocal hidden_consumed
            assert hidden_test.read_text(encoding="utf-8") == hidden_test_sentinel
            hidden_consumed = True
            return super().verify(request)

    tracer, session, client = _production_no_network_tracer(
        monkeypatch,
        credential=credential_sentinel,
        environment_sentinel=environment_sentinel,
    )
    assert client.tracing_mode == "langsmith"
    assert client.api_url == LANGSMITH_API_ENDPOINT
    assert client.otel_exporter is None

    calls: list[str] = []
    boundaries = ProductionSentinelBoundaries(None, calls)
    plan = _repair_plan().model_copy(update={"suspected_root_cause": model_sentinel})
    proposal = _patch_proposal().model_copy(
        update={
            "summary": patch_sentinel,
            "unified_diff": _patch_proposal().unified_diff.replace(
                "+    return (items + size - 1) // size",
                f"+    return (items + size - 1) // size  # {diff_sentinel}",
            ),
        }
    )
    provider = QueueGraphProvider(
        ModelProviderResponse(raw_json=plan.model_dump_json()),
        ModelProviderResponse(raw_json=proposal.model_dump_json()),
    )
    dependencies = GraphDependencies(
        preparer=boundaries,
        baseline=boundaries,
        context_selector=boundaries,
        static_validation=boundaries,
        verification=boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
        model_provider=provider,
        classifier=boundaries.classify,
        policy_validator=boundaries.policy,
        evaluator=boundaries.evaluate,
        tracer=tracer,
    )
    real_boundary_root = tmp_path / "real-boundary"
    real_boundary_root.mkdir()
    initial = new_orchestration_state(
        run_id="a" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(real_boundary_root, "a" * 32),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )
    invalid_values = _repair_plan().model_dump()
    invalid_values[validation_sentinel] = validation_sentinel
    invalid_provider = QueueGraphProvider(
        ModelProviderResponse(raw_json=json.dumps(invalid_values))
    )
    invalid_boundaries = TransitionBoundaries(None, [])
    invalid_dependencies = GraphDependencies(
        preparer=invalid_boundaries,
        baseline=invalid_boundaries,
        context_selector=invalid_boundaries,
        static_validation=invalid_boundaries,
        verification=invalid_boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
        model_provider=invalid_provider,
        classifier=invalid_boundaries.classify,
        policy_validator=invalid_boundaries.policy,
        evaluator=invalid_boundaries.evaluate,
        tracer=tracer,
    )
    validation_boundary_root = tmp_path / "validation-boundary"
    validation_boundary_root.mkdir()
    invalid_initial = new_orchestration_state(
        run_id="b" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(validation_boundary_root, "b" * 32),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )

    with tracer.span("tracefix.live_benchmark", safe_metadata(benchmark_run_id="c" * 32)):
        with tracer.span("case.boundary_condition", safe_metadata(case_id="boundary_condition")):
            result = invoke_repair_graph(build_repair_graph(dependencies), initial)
        with tracer.span("case.schema_validation", safe_metadata(case_id="schema_validation")):
            invalid_result = invoke_repair_graph(
                build_repair_graph(invalid_dependencies), invalid_initial
            )

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.SUCCESS
    assert invalid_result.terminal_route is not None
    assert invalid_result.terminal_route.outcome is TerminalOutcome.MALFORMED_MODEL_OUTPUT
    assert hidden_consumed is True
    assert source_sentinel in provider.requests[0].rendered_prompt
    assert visible_test_sentinel in provider.requests[0].rendered_prompt
    assert prompt_sentinel in provider.requests[0].rendered_prompt
    assert session.intercepted
    assert all(
        request["url"].startswith(LANGSMITH_API_ENDPOINT)
        for request in session.intercepted
        if isinstance(request["url"], str)
    )
    assert all(not str(request["url"]).endswith("/info") for request in session.intercepted)

    payloads = _intercepted_trace_payloads(session)
    rendered = json.dumps(payloads, sort_keys=True, default=str)
    for sentinel in (
        source_sentinel,
        visible_test_sentinel,
        hidden_test_sentinel,
        prompt_sentinel,
        model_sentinel,
        patch_sentinel,
        diff_sentinel,
        credential_sentinel,
        environment_sentinel,
        validation_sentinel,
    ):
        assert sentinel not in rendered
    for payload in payloads:
        assert payload.get("inputs", {}) == {}
        assert payload.get("outputs", {}) == {}
        assert payload.get("events", []) == []
        assert payload.get("tags", []) == []
        assert payload.get("error") is None
        assert payload.get("serialized", {}) == {}
        extra = payload.get("extra", {})
        assert isinstance(extra, dict)
        assert "runtime" not in extra
        metadata = extra.get("metadata", {})
        assert isinstance(metadata, dict)
        assert set(metadata).issubset(SAFE_METADATA_ALLOW_LIST)


def test_production_graph_sanitizes_model_controlled_validation_location(
    tmp_path: Path,
) -> None:
    validation_sentinel = "SENTINEL_RAW_VALIDATION_INPUT_KEY"
    values = _repair_plan().model_dump()
    values[validation_sentinel] = validation_sentinel
    provider = QueueGraphProvider(ModelProviderResponse(raw_json=json.dumps(values)))
    calls: list[str] = []
    boundaries = TransitionBoundaries(None, calls)
    exporter = CapturingDiagnosticExporter()
    tracer = LangSmithDiagnosticTracer(exporter)
    dependencies = GraphDependencies(
        preparer=boundaries,
        baseline=boundaries,
        context_selector=boundaries,
        static_validation=boundaries,
        verification=boundaries,
        workspace_integrity=LocalWorkspaceIntegrityBoundary(FingerprinterStub()),
        clock=lambda: datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
        model_provider=provider,
        classifier=boundaries.classify,
        policy_validator=boundaries.policy,
        evaluator=boundaries.evaluate,
        tracer=tracer,
    )
    run_id = "6" * 32
    workspace_parent = tmp_path / "validation"
    workspace_parent.mkdir()
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(workspace_parent, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )

    with (
        tracer.span("tracefix.live_benchmark"),
        tracer.span("case.boundary_condition"),
    ):
        result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.MALFORMED_MODEL_OUTPUT
    captured = json.dumps([run.payload for run in exporter.runs], sort_keys=True)
    assert validation_sentinel not in captured
    validation_run = next(
        run for run in exporter.runs if run.payload["name"] == "validate_repair_plan"
    )
    metadata = validation_run.payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["validation_field_path"] == "$.unknown"
    assert metadata["validation_error_type"] == "extra_forbidden"


def test_graph_does_not_trace_unreached_stages(tmp_path: Path) -> None:
    calls: list[str] = []
    tracer = GraphRecordingTracer()
    dependencies = replace(_transition_dependencies("generate_patch", calls), tracer=tracer)
    run_id = "5" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )

    result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.terminal_route is not None
    names = [record["name"] for record in tracer.records]
    assert names[-1] == "generate_patch"
    assert "patch_policy" not in names
    assert "static_checks" not in names
    assert "verification" not in names
    assert "evaluation" not in names


def test_graph_snapshots_live_usage_and_hands_it_to_candidate_evaluation(
    tmp_path: Path,
) -> None:
    usage = CandidateModelUsage(
        model_latency_seconds=0.75,
        input_tokens=120,
        output_tokens=40,
        estimated_cost_usd=Decimal("0.000042"),
    )

    class UsageReader:
        def snapshot_usage(self) -> CandidateModelUsage:
            return usage

    calls: list[str] = []
    dependencies = replace(_transition_dependencies(None, calls), model_usage_reader=UsageReader())
    run_id = "8" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )
    result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.model_usage == usage
    assert result.evaluation_result is not None
    assert result.evaluation_result.model_latency_seconds == 0.75
    assert result.evaluation_result.input_tokens == 120
    assert result.evaluation_result.output_tokens == 40
    assert result.evaluation_result.estimated_cost_usd == Decimal("0.000042")


def test_state_rejects_usage_before_model_execution(tmp_path: Path) -> None:
    run_id = "6" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )
    payload = initial.model_dump(warnings="none")
    payload["model_usage"] = CandidateModelUsage(
        model_latency_seconds=0.1,
        input_tokens=1,
        output_tokens=1,
        estimated_cost_usd=Decimal("0.01"),
    ).model_dump(warnings="none")
    with pytest.raises(ValidationError, match="without complete model execution"):
        OrchestrationState.model_validate(payload, strict=True)


def test_state_rejects_evaluation_usage_that_disagrees_with_run(tmp_path: Path) -> None:
    calls: list[str] = []
    result = _invoke_transition_graph(tmp_path, None, calls)
    assert result.evaluation_result is not None
    payload = result.model_dump(warnings="none")
    evaluation = dict(payload["evaluation_result"])
    evaluation["input_tokens"] = 1
    payload["evaluation_result"] = evaluation
    with pytest.raises(ValidationError, match="evaluation usage"):
        OrchestrationState.model_validate(payload, strict=True)


def test_early_malformed_model_failure_retains_completed_call_usage(
    tmp_path: Path,
) -> None:
    usage = CandidateModelUsage(
        model_latency_seconds=0.25,
        input_tokens=50,
        output_tokens=3,
        estimated_cost_usd=Decimal("0.0000093"),
    )

    class UsageReader:
        def snapshot_usage(self) -> CandidateModelUsage:
            return usage

    def malformed_planner(
        context: ContextPackage,
        configuration: ModelConfiguration,
        provider: ModelProvider,
    ) -> RepairPlan:
        del context, configuration, provider
        raise ModelDiagnosticError(
            ModelDiagnostic(
                stage=ModelDiagnosticStage.PLANNING,
                code=ModelDiagnosticCode.MALFORMED_STRUCTURED_OUTPUT,
                message="model provider returned malformed structured output",
            )
        )

    calls: list[str] = []
    dependencies = replace(
        _transition_dependencies(None, calls),
        planner=malformed_planner,
        model_usage_reader=UsageReader(),
    )
    run_id = "7" * 32
    initial = new_orchestration_state(
        run_id=run_id,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=_workspaces(tmp_path, run_id),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        model_configuration=_model_configuration(),
    )
    result = invoke_repair_graph(build_repair_graph(dependencies), initial)

    assert result.terminal_route is not None
    assert result.terminal_route.outcome is TerminalOutcome.MALFORMED_MODEL_OUTPUT
    assert result.model_usage == usage
    assert result.evaluation_result is None


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
