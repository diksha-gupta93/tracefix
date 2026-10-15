from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agent.classification import classify_failure
from app.agent.context import apply_failure_analysis
from app.agent.model import ModelDiagnosticError, ModelProvider
from app.agent.planning import (
    generate_patch,
    plan_repair,
    record_patch_proposal,
    record_repair_plan,
)
from app.agent.policy import PatchPolicy, PatchPolicyResult, validate_patch_policy
from app.agent.routing import (
    RoutingEvidence,
    TerminalOutcome,
    TerminalRoute,
    route_terminal,
)
from app.agent.schemas import (
    DEFAULT_PROTECTED_PATHS,
    BaselineOutcome,
    BaselineResult,
    ContextPackage,
    FailureAnalysis,
    LocalRepairCaseState,
    ModelConfiguration,
    ModelDiagnosticCode,
    PatchProposal,
    RepairPlan,
    RepairStatus,
    RepositoryPreparation,
)
from app.agent.static_checks import (
    RepositoryFingerprinter,
    StaticValidationOutcome,
    StaticValidationRequest,
    StaticValidationResult,
)
from app.agent.verification import (
    CandidateVerificationRequest,
    CandidateVerificationResult,
)
from app.evaluation.deterministic.candidate import (
    CandidateEvaluationRequest,
    CandidateEvaluationResult,
    CandidateModelUsage,
    evaluate_candidate,
)
from benchmarks.loader import TrustedCase

if TYPE_CHECKING:
    from langgraph.graph._node import StateNode

NodeName = Literal[
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
    "route_result",
]

NODE_NAMES: tuple[NodeName, ...] = (
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
    "route_result",
)

_RUN_ID_PATTERN = r"^[0-9a-f]{32}$"
_CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,127}$"
_MODEL_MALFORMED_CODES = frozenset(
    {
        ModelDiagnosticCode.OVERSIZED_RESPONSE,
        ModelDiagnosticCode.MALFORMED_STRUCTURED_OUTPUT,
        ModelDiagnosticCode.INVALID_PATCH_ENVELOPE,
    }
)
_WINDOWS_REPARSE_POINT = 0x400


class RunWorkspaces(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    root: Path
    preparation: Path
    candidate: Path
    verification: Path
    ownership_token: str = Field(pattern=_RUN_ID_PATTERN)

    @model_validator(mode="after")
    def validate_layout(self) -> RunWorkspaces:
        values = (self.root, self.preparation, self.candidate, self.verification)
        if any(not value.is_absolute() for value in values):
            raise ValueError("run workspaces must be absolute")
        if len({str(value).casefold() for value in values}) != len(values):
            raise ValueError("run workspaces must be separate")
        if any(value.parent != self.root for value in values[1:]):
            raise ValueError("run workspace children must be direct descendants")
        return self


def _proposal_digest(proposal: PatchProposal) -> str:
    payload = json.dumps(
        proposal.model_dump(mode="json", warnings="none"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OrchestrationState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    started_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    attempt_number: int = Field(strict=True, ge=0, le=1)
    max_attempts: Literal[1] = 1
    requested_revision: str = Field(min_length=1)
    trusted_case: TrustedCase
    legacy_state: LocalRepairCaseState
    workspaces: RunWorkspaces
    model_configuration: ModelConfiguration | None = None
    model_usage: CandidateModelUsage | None = None
    preparation: RepositoryPreparation | None = None
    baseline: BaselineResult | None = None
    failure_analysis: FailureAnalysis | None = None
    context: ContextPackage | None = None
    repair_plan: RepairPlan | None = None
    proposal: PatchProposal | None = None
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_result: PatchPolicyResult | None = None
    static_result: StaticValidationResult | None = None
    verification_result: CandidateVerificationResult | None = None
    evaluation_result: CandidateEvaluationResult | None = None
    failure_override: TerminalOutcome | None = None
    terminal_route: TerminalRoute | None = None
    completed_nodes: tuple[NodeName, ...] = ()

    @model_validator(mode="after")
    def validate_invariants(self) -> OrchestrationState:
        if self.started_at.utcoffset() != UTC.utcoffset(self.started_at):
            raise ValueError("run timestamp must be UTC")
        if self.completed_at is not None:
            if self.completed_at.utcoffset() != UTC.utcoffset(self.completed_at):
                raise ValueError("completion timestamp must be UTC")
            if self.completed_at < self.started_at:
                raise ValueError("completion timestamp precedes start")
        if self.trusted_case.manifest.case_id != self.case_id:
            raise ValueError("trusted case does not match run")
        if self.trusted_case.manifest.failing_revision != self.requested_revision:
            raise ValueError("trusted revision does not match run")
        if self.workspaces.ownership_token != self.run_id:
            raise ValueError("workspace ownership does not match run")
        if (
            self.legacy_state.case_id != self.case_id
            or self.legacy_state.max_attempts != 1
            or self.legacy_state.attempt_number != self.attempt_number
        ):
            raise ValueError("legacy state identity does not match run")
        before_route = (
            self.completed_nodes[:-1]
            if self.completed_nodes and self.completed_nodes[-1] == "route_result"
            else self.completed_nodes
        )
        if before_route != NODE_NAMES[: len(before_route)]:
            raise ValueError("graph nodes are not a valid execution prefix")
        if "route_result" in self.completed_nodes[:-1]:
            raise ValueError("terminal route must be the final graph node")
        evidence_nodes = (
            (self.preparation, "prepare_repository"),
            (self.baseline, "run_baseline"),
            (self.failure_analysis, "classify_failure"),
            (self.context, "select_relevant_context"),
            (self.repair_plan, "plan_repair"),
            (self.proposal, "generate_patch"),
            (self.policy_result, "validate_patch_policy"),
            (self.static_result, "run_static_checks"),
            (self.verification_result, "verify_in_sandbox"),
            (self.evaluation_result, "evaluate_candidate"),
        )
        if any(value is not None and node not in before_route for value, node in evidence_nodes):
            raise ValueError("graph evidence exists for a node that did not execute")
        if any(
            value is None
            and node in before_route
            and not (
                self.failure_override is not None and before_route and node == before_route[-1]
            )
            for value, node in evidence_nodes
        ):
            raise ValueError("completed graph node is missing its evidence")
        if self.preparation is not None and self.preparation.case_id != self.case_id:
            raise ValueError("preparation does not match run")
        if self.baseline is not None:
            if self.preparation is None or self.baseline.preparation != self.preparation:
                raise ValueError("baseline does not match preparation")
            if self.legacy_state.baseline_result != self.baseline:
                raise ValueError("baseline does not match legacy state")
        if self.failure_analysis is not None and (
            self.baseline is None or self.legacy_state.failure_analysis != self.failure_analysis
        ):
            raise ValueError("failure analysis does not match legacy state")
        if self.context is not None and (
            self.failure_analysis is None
            or self.context.case_id != self.case_id
            or self.context.failure_analysis != self.failure_analysis
        ):
            raise ValueError("context does not match failure analysis")
        if self.repair_plan is not None and self.legacy_state.repair_plan != self.repair_plan:
            raise ValueError("repair plan does not match legacy state")
        if self.proposal is None:
            if self.proposal_digest is not None or self.policy_result is not None:
                raise ValueError("proposal-dependent evidence is present without a proposal")
        elif (
            self.legacy_state.candidate_patch != self.proposal
            or self.proposal_digest != _proposal_digest(self.proposal)
        ):
            raise ValueError("proposal identity does not match active run")
        if self.static_result is not None and self.policy_result is None:
            raise ValueError("static result is missing policy evidence")
        if self.verification_result is not None and (
            self.static_result is None
            or self.static_result.outcome is not StaticValidationOutcome.PASSED
        ):
            raise ValueError("verification is missing passing static evidence")
        if self.evaluation_result is not None:
            if self.proposal is None or self.verification_result is None:
                raise ValueError("evaluation is missing proposal or verification evidence")
            if self.evaluation_result.case_id != self.case_id:
                raise ValueError("evaluation does not match run")
        if self.failure_override in {
            TerminalOutcome.SUCCESS,
            TerminalOutcome.POLICY_VIOLATION,
            TerminalOutcome.STATIC_CHECK_FAILURE,
            TerminalOutcome.BASELINE_NOT_REPRODUCIBLE,
        }:
            raise ValueError("invalid terminal override")
        if self.failure_override in {
            TerminalOutcome.CANDIDATE_FAILURE,
            TerminalOutcome.MALFORMED_MODEL_OUTPUT,
        } and any(
            value is not None
            for value in (
                self.policy_result,
                self.static_result,
                self.verification_result,
                self.evaluation_result,
            )
        ):
            raise ValueError("terminal override contradicts later graph evidence")
        if self.terminal_route is None:
            if self.completed_at is not None:
                raise ValueError("incomplete run cannot have a completion timestamp")
            if "route_result" in self.completed_nodes:
                raise ValueError("incomplete run cannot record terminal routing")
        else:
            if self.completed_at is None or not self.completed_nodes:
                raise ValueError("terminal run is missing completion evidence")
            if self.completed_nodes[-1] != "route_result":
                raise ValueError("terminal route must be the final graph node")
            if self.terminal_route != route_state(self):
                raise ValueError("terminal route disagrees with run evidence")
        return self


class GraphTransport(TypedDict):
    state: OrchestrationState


class RepositoryPreparationBoundary(Protocol):
    def prepare(
        self,
        case_id: str,
        requested_revision: str | None,
        workspace_root: Path,
        *,
        destination_name: str | None = None,
    ) -> RepositoryPreparation: ...


class BaselineBoundary(Protocol):
    def run(self, preparation: RepositoryPreparation) -> BaselineResult: ...


class ContextBoundary(Protocol):
    def select(
        self,
        baseline: BaselineResult,
        analysis: FailureAnalysis,
        *,
        limit_utf8_bytes: int | None = None,
    ) -> ContextPackage: ...


class StaticValidationBoundary(Protocol):
    def validate(self, request: StaticValidationRequest) -> StaticValidationResult: ...


class VerificationBoundary(Protocol):
    def verify(self, request: CandidateVerificationRequest) -> CandidateVerificationResult: ...


class WorkspaceIntegrityBoundary(Protocol):
    def validate_preparation(
        self,
        preparation: RepositoryPreparation,
        workspace_root: Path,
        requested_revision: str,
    ) -> None: ...

    def validate_candidate(
        self,
        result: StaticValidationResult,
        workspace_root: Path,
    ) -> None: ...


def _ordinary_owned_directory(path: Path, parent: Path, expected_name: str | None = None) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("workspace result path is invalid") from None
    if (
        not path.is_absolute()
        or path != resolved
        or path.parent != parent
        or (expected_name is not None and path.name != expected_name)
        or not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise ValueError("workspace result path is invalid")


class LocalWorkspaceIntegrityBoundary:
    def __init__(self, fingerprinter: RepositoryFingerprinter) -> None:
        self._fingerprinter = fingerprinter

    def validate_preparation(
        self,
        preparation: RepositoryPreparation,
        workspace_root: Path,
        requested_revision: str,
    ) -> None:
        if preparation.requested_revision != requested_revision:
            raise ValueError("prepared revision does not match run")
        _ordinary_owned_directory(
            preparation.prepared_repository,
            workspace_root,
            "repository",
        )
        if (
            self._fingerprinter.fingerprint(preparation.prepared_repository)
            != preparation.fingerprint
        ):
            raise ValueError("prepared repository fingerprint changed")

    def validate_candidate(
        self,
        result: StaticValidationResult,
        workspace_root: Path,
    ) -> None:
        if result.outcome is not StaticValidationOutcome.PASSED:
            return
        if result.candidate is None:
            raise ValueError("passing static result is missing a candidate")
        _ordinary_owned_directory(result.candidate.repository, workspace_root)
        if (
            self._fingerprinter.fingerprint(result.candidate.repository)
            != result.candidate.fingerprint
        ):
            raise ValueError("candidate repository fingerprint changed")


Classifier = Callable[[BaselineResult], FailureAnalysis]
Planner = Callable[[ContextPackage, ModelConfiguration, ModelProvider], RepairPlan]
PatchGenerator = Callable[
    [ContextPackage, RepairPlan, ModelConfiguration, ModelProvider], PatchProposal
]
PolicyValidator = Callable[[PatchProposal, PatchPolicy], PatchPolicyResult]
CandidateEvaluator = Callable[[CandidateEvaluationRequest], CandidateEvaluationResult]
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    preparer: RepositoryPreparationBoundary
    baseline: BaselineBoundary
    context_selector: ContextBoundary
    static_validation: StaticValidationBoundary
    verification: VerificationBoundary
    workspace_integrity: WorkspaceIntegrityBoundary
    clock: Clock
    model_configuration: ModelConfiguration | None = None
    model_provider: ModelProvider | None = None
    classifier: Classifier = classify_failure
    planner: Planner = plan_repair
    patch_generator: PatchGenerator = generate_patch
    policy_validator: PolicyValidator = validate_patch_policy
    evaluator: CandidateEvaluator = evaluate_candidate


def _strict_state(candidate: object) -> OrchestrationState:
    if type(candidate) is not OrchestrationState:
        raise ValueError("graph state has the wrong type")
    try:
        raw = candidate.model_dump(warnings="none")
        checked = OrchestrationState.model_validate(raw, strict=True)
        if checked.model_dump(warnings="none") != raw:
            raise ValueError
        return checked
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise ValueError("graph state is invalid") from None


def _transport(value: GraphTransport) -> OrchestrationState:
    if type(value) is not dict or set(value) != {"state"}:
        raise ValueError("graph transport is invalid")
    return _strict_state(value["state"])


def _replace(state: OrchestrationState, **updates: object) -> OrchestrationState:
    candidate = state.model_copy(update=updates, deep=True)
    return _strict_state(candidate)


def _finish(
    state: OrchestrationState,
    node: NodeName,
    updates: dict[str, object] | None = None,
) -> OrchestrationState:
    if node in state.completed_nodes or state.terminal_route is not None:
        raise ValueError("graph node cannot execute twice or after routing")
    values = {} if updates is None else dict(updates)
    values["completed_nodes"] = (*state.completed_nodes, node)
    return _replace(state, **values)


def _failure(
    state: OrchestrationState, node: NodeName, outcome: TerminalOutcome
) -> dict[str, object]:
    return {"state": _finish(state, node, {"failure_override": outcome})}


def _model_failure(error: ModelDiagnosticError) -> TerminalOutcome:
    if error.diagnostic.code is ModelDiagnosticCode.UNSUITABLE_AUTONOMOUS_REPAIR:
        return TerminalOutcome.CANDIDATE_FAILURE
    if error.diagnostic.code in _MODEL_MALFORMED_CODES:
        return TerminalOutcome.MALFORMED_MODEL_OUTPUT
    return TerminalOutcome.INFRASTRUCTURE_FAILURE


def _attempt_failure(
    state: OrchestrationState,
    dependencies: GraphDependencies,
    outcome: TerminalOutcome,
) -> dict[str, object]:
    legacy = state.legacy_state.model_copy(deep=True)
    legacy.attempt_number = 1
    configuration = dependencies.model_configuration
    if configuration is not None:
        legacy.model_provider = configuration.provider
        legacy.model_name = configuration.model_name
        legacy.prompt_version = configuration.prompt_version
    try:
        legacy.updated_at = _utc(dependencies.clock())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        outcome = TerminalOutcome.INFRASTRUCTURE_FAILURE
    return {
        "state": _finish(
            state,
            "plan_repair",
            {
                "attempt_number": 1,
                "legacy_state": legacy,
                "failure_override": outcome,
            },
        )
    }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clock returned a naive timestamp")
    return value.astimezone(UTC)


def new_orchestration_state(
    *,
    run_id: str,
    trusted_case: TrustedCase,
    workspaces: RunWorkspaces,
    started_at: datetime,
    model_configuration: ModelConfiguration | None = None,
    model_usage: CandidateModelUsage | None = None,
) -> OrchestrationState:
    case = TrustedCase.model_validate(trusted_case.model_dump(warnings="none"), strict=True)
    started = _utc(started_at)
    legacy = LocalRepairCaseState(
        case_id=case.manifest.case_id,
        created_at=started,
        updated_at=started,
    )
    return OrchestrationState(
        run_id=run_id,
        case_id=case.manifest.case_id,
        started_at=started,
        attempt_number=0,
        requested_revision=case.manifest.failing_revision,
        trusted_case=case,
        legacy_state=legacy,
        workspaces=workspaces,
        model_configuration=model_configuration,
        model_usage=model_usage,
    )


def route_state(state: OrchestrationState) -> TerminalRoute:
    return route_terminal(
        RoutingEvidence(
            baseline=state.baseline,
            policy=state.policy_result,
            static=state.static_result,
            verification=state.verification_result,
            evaluation=state.evaluation_result,
            failure_override=state.failure_override,
        )
    )


class _GraphNodes:
    def __init__(self, dependencies: GraphDependencies) -> None:
        self._dependencies = dependencies

    def prepare_repository(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            preparation = self._dependencies.preparer.prepare(
                state.case_id,
                state.requested_revision,
                state.workspaces.preparation,
                destination_name="repository",
            )
            preparation = RepositoryPreparation.model_validate(
                preparation.model_dump(warnings="none"), strict=True
            )
            if preparation.case_id != state.case_id:
                raise ValueError
            self._dependencies.workspace_integrity.validate_preparation(
                preparation,
                state.workspaces.preparation,
                state.requested_revision,
            )
            return {"state": _finish(state, "prepare_repository", {"preparation": preparation})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "prepare_repository", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def run_baseline(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if state.preparation is None:
                raise ValueError
            baseline = self._dependencies.baseline.run(state.preparation)
            baseline = BaselineResult.model_validate(
                baseline.model_dump(warnings="none"), strict=True
            )
            legacy = state.legacy_state.model_copy(deep=True)
            legacy.baseline_result = baseline
            legacy.status = RepairStatus.baseline_complete
            legacy.updated_at = _utc(self._dependencies.clock())
            return {
                "state": _finish(
                    state,
                    "run_baseline",
                    {"baseline": baseline, "legacy_state": legacy},
                )
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "run_baseline", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def classify_failure(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if state.baseline is None:
                raise ValueError
            analysis = self._dependencies.classifier(state.baseline)
            analysis = FailureAnalysis.model_validate(
                analysis.model_dump(warnings="none"), strict=True
            )
            legacy = apply_failure_analysis(state.legacy_state, analysis)
            legacy.updated_at = _utc(self._dependencies.clock())
            return {
                "state": _finish(
                    state,
                    "classify_failure",
                    {"failure_analysis": analysis, "legacy_state": legacy},
                )
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "classify_failure", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def select_relevant_context(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if state.baseline is None or state.failure_analysis is None:
                raise ValueError
            context = self._dependencies.context_selector.select(
                state.baseline, state.failure_analysis
            )
            context = ContextPackage.model_validate(
                context.model_dump(warnings="none"), strict=True
            )
            return {"state": _finish(state, "select_relevant_context", {"context": context})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(
                state,
                "select_relevant_context",
                TerminalOutcome.INFRASTRUCTURE_FAILURE,
            )

    def plan_repair(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            configuration = self._dependencies.model_configuration
            provider = self._dependencies.model_provider
            if state.context is None or configuration is None or provider is None:
                raise ValueError
            if state.model_configuration != configuration:
                raise ValueError
            plan = self._dependencies.planner(state.context, configuration, provider)
            plan = RepairPlan.model_validate(plan.model_dump(warnings="none"), strict=True)
            legacy = record_repair_plan(state.legacy_state, state.context, plan, configuration)
            legacy.updated_at = _utc(self._dependencies.clock())
            return {
                "state": _finish(
                    state,
                    "plan_repair",
                    {
                        "repair_plan": plan,
                        "legacy_state": legacy,
                        "attempt_number": 1,
                    },
                )
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except ModelDiagnosticError as error:
            return _attempt_failure(state, self._dependencies, _model_failure(error))
        except Exception:
            return _attempt_failure(
                state,
                self._dependencies,
                TerminalOutcome.INFRASTRUCTURE_FAILURE,
            )

    def generate_patch(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            configuration = self._dependencies.model_configuration
            provider = self._dependencies.model_provider
            if (
                state.context is None
                or state.repair_plan is None
                or configuration is None
                or provider is None
            ):
                raise ValueError
            proposal = self._dependencies.patch_generator(
                state.context, state.repair_plan, configuration, provider
            )
            proposal = PatchProposal.model_validate(
                proposal.model_dump(warnings="none"), strict=True
            )
            legacy = record_patch_proposal(
                state.legacy_state,
                state.context,
                state.repair_plan,
                proposal,
                configuration,
            )
            legacy.updated_at = _utc(self._dependencies.clock())
            return {
                "state": _finish(
                    state,
                    "generate_patch",
                    {
                        "proposal": proposal,
                        "proposal_digest": _proposal_digest(proposal),
                        "legacy_state": legacy,
                    },
                )
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except ModelDiagnosticError as error:
            return _failure(state, "generate_patch", _model_failure(error))
        except Exception:
            return _failure(state, "generate_patch", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def validate_patch_policy(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if state.proposal is None:
                raise ValueError
            forbidden = tuple(
                sorted(
                    (
                        PurePosixPath(value)
                        for value in state.trusted_case.manifest.forbidden_changed_files
                    ),
                    key=lambda item: (item.as_posix().casefold(), item.as_posix()),
                )
            )
            if state.context is None:
                raise ValueError
            policy = PatchPolicy(
                protected_path_policy=state.context.protected_path_policy,
                forbidden_changed_files=forbidden,
            )
            if policy.protected_path_policy.protected_paths != DEFAULT_PROTECTED_PATHS:
                raise ValueError
            result = self._dependencies.policy_validator(state.proposal, policy)
            result = PatchPolicyResult.model_validate(
                result.model_dump(warnings="none"), strict=True
            )
            if result != validate_patch_policy(state.proposal, policy):
                raise ValueError("policy result does not match the active proposal")
            return {"state": _finish(state, "validate_patch_policy", {"policy_result": result})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(
                state,
                "validate_patch_policy",
                TerminalOutcome.INFRASTRUCTURE_FAILURE,
            )

    def run_static_checks(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if (
                state.preparation is None
                or state.proposal is None
                or state.policy_result is None
                or not state.policy_result.approved
            ):
                raise ValueError
            result = self._dependencies.static_validation.validate(
                StaticValidationRequest(
                    preparation=state.preparation,
                    proposal=state.proposal,
                    policy_result=state.policy_result,
                    candidate_workspace_root=state.workspaces.candidate,
                )
            )
            result = StaticValidationResult.model_validate(
                result.model_dump(warnings="none"), strict=True
            )
            self._dependencies.workspace_integrity.validate_candidate(
                result, state.workspaces.candidate
            )
            return {"state": _finish(state, "run_static_checks", {"static_result": result})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "run_static_checks", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def verify_in_sandbox(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if (
                state.static_result is None
                or state.static_result.outcome is not StaticValidationOutcome.PASSED
            ):
                raise ValueError
            result = self._dependencies.verification.verify(
                CandidateVerificationRequest(
                    case_id=state.case_id,
                    static_result=state.static_result,
                    verification_workspace_root=state.workspaces.verification,
                )
            )
            result = CandidateVerificationResult.model_validate(
                result.model_dump(warnings="none"), strict=True
            )
            return {"state": _finish(state, "verify_in_sandbox", {"verification_result": result})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "verify_in_sandbox", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def evaluate_candidate(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        try:
            if state.proposal is None or state.verification_result is None:
                raise ValueError
            request = CandidateEvaluationRequest(
                case_id=state.case_id,
                proposal=state.proposal,
                verification=state.verification_result,
                model_usage=state.model_usage,
            )
            result = self._dependencies.evaluator(request)
            result = CandidateEvaluationResult.model_validate(
                result.model_dump(warnings="none"), strict=True
            )
            if result != evaluate_candidate(request):
                raise ValueError("evaluation result does not match the active inputs")
            return {"state": _finish(state, "evaluate_candidate", {"evaluation_result": result})}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _failure(state, "evaluate_candidate", TerminalOutcome.INFRASTRUCTURE_FAILURE)

    def route_result(self, transport: GraphTransport) -> dict[str, object]:
        state = _transport(transport)
        route = route_state(state)
        completed = _utc(self._dependencies.clock())
        return {
            "state": _finish(
                state,
                "route_result",
                {"terminal_route": route, "completed_at": completed},
            )
        }


def _after_prepare(transport: GraphTransport) -> Literal["run_baseline", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "run_baseline"


def _after_baseline(transport: GraphTransport) -> Literal["classify_failure", "route_result"]:
    state = _transport(transport)
    if state.failure_override is not None:
        return "route_result"
    if state.baseline is None or state.baseline.outcome is not BaselineOutcome.REPRODUCED_FAILURE:
        return "route_result"
    return "classify_failure"


def _after_classification(
    transport: GraphTransport,
) -> Literal["select_relevant_context", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "select_relevant_context"


def _after_context(transport: GraphTransport) -> Literal["plan_repair", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "plan_repair"


def _after_plan(transport: GraphTransport) -> Literal["generate_patch", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "generate_patch"


def _after_patch(
    transport: GraphTransport,
) -> Literal["validate_patch_policy", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "validate_patch_policy"


def _after_policy(transport: GraphTransport) -> Literal["run_static_checks", "route_result"]:
    state = _transport(transport)
    if state.failure_override is not None:
        return "route_result"
    if state.policy_result is None or not state.policy_result.approved:
        return "route_result"
    return "run_static_checks"


def _after_static(transport: GraphTransport) -> Literal["verify_in_sandbox", "route_result"]:
    state = _transport(transport)
    if state.failure_override is not None:
        return "route_result"
    if (
        state.static_result is None
        or state.static_result.outcome is not StaticValidationOutcome.PASSED
    ):
        return "route_result"
    return "verify_in_sandbox"


def _after_verification(
    transport: GraphTransport,
) -> Literal["evaluate_candidate", "route_result"]:
    state = _transport(transport)
    return "route_result" if state.failure_override is not None else "evaluate_candidate"


def build_repair_graph(
    dependencies: GraphDependencies,
) -> CompiledStateGraph[GraphTransport, None, GraphTransport, GraphTransport]:
    nodes = _GraphNodes(dependencies)
    graph = StateGraph(GraphTransport)
    actions: tuple[tuple[NodeName, Callable[[GraphTransport], object]], ...] = (
        ("prepare_repository", nodes.prepare_repository),
        ("run_baseline", nodes.run_baseline),
        ("classify_failure", nodes.classify_failure),
        ("select_relevant_context", nodes.select_relevant_context),
        ("plan_repair", nodes.plan_repair),
        ("generate_patch", nodes.generate_patch),
        ("validate_patch_policy", nodes.validate_patch_policy),
        ("run_static_checks", nodes.run_static_checks),
        ("verify_in_sandbox", nodes.verify_in_sandbox),
        ("evaluate_candidate", nodes.evaluate_candidate),
        ("route_result", nodes.route_result),
    )
    for name, action in actions:
        graph.add_node(name, cast("StateNode[GraphTransport, None]", action))
    graph.add_edge(START, "prepare_repository")
    graph.add_conditional_edges("prepare_repository", _after_prepare)
    graph.add_conditional_edges("run_baseline", _after_baseline)
    graph.add_conditional_edges("classify_failure", _after_classification)
    graph.add_conditional_edges("select_relevant_context", _after_context)
    graph.add_conditional_edges("plan_repair", _after_plan)
    graph.add_conditional_edges("generate_patch", _after_patch)
    graph.add_conditional_edges("validate_patch_policy", _after_policy)
    graph.add_conditional_edges("run_static_checks", _after_static)
    graph.add_conditional_edges("verify_in_sandbox", _after_verification)
    graph.add_edge("evaluate_candidate", "route_result")
    graph.add_edge("route_result", END)
    return graph.compile(name="tracefix-v0.1-single-attempt")


def invoke_repair_graph(
    graph: CompiledStateGraph[GraphTransport, None, GraphTransport, GraphTransport],
    initial_state: OrchestrationState,
) -> OrchestrationState:
    result: object = graph.invoke({"state": _strict_state(initial_state)})
    if type(result) is not dict:
        raise ValueError("graph returned an invalid transport")
    return _strict_state(result.get("state"))


def replace_with_infrastructure_failure(
    state: OrchestrationState, completed_at: datetime
) -> OrchestrationState:
    checked = _strict_state(state)
    if checked.terminal_route is None:
        raise ValueError("only a terminal state can be replaced")
    candidate = checked.model_copy(
        update={
            "failure_override": TerminalOutcome.INFRASTRUCTURE_FAILURE,
            "terminal_route": TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE),
            "completed_at": _utc(completed_at),
        },
        deep=True,
    )
    return _strict_state(candidate)


def terminalize_infrastructure_failure(
    state: OrchestrationState, completed_at: datetime
) -> OrchestrationState:
    checked = _strict_state(state)
    if checked.terminal_route is not None or "route_result" in checked.completed_nodes:
        raise ValueError("run is already terminal")
    candidate = checked.model_copy(
        update={
            "failure_override": TerminalOutcome.INFRASTRUCTURE_FAILURE,
            "terminal_route": TerminalRoute.for_outcome(TerminalOutcome.INFRASTRUCTURE_FAILURE),
            "completed_at": _utc(completed_at),
            "completed_nodes": (*checked.completed_nodes, "route_result"),
        },
        deep=True,
    )
    return _strict_state(candidate)
