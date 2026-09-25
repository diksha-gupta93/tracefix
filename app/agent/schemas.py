from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from app.sandbox.results import SandboxCompletion

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictInteger = Annotated[int, Field(strict=True)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]


def _require_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("value must be a float")
    return value


Confidence = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]
ModelTemperature = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(ge=0.0, le=2.0, allow_inf_nan=False),
]


class TestStatus(StrEnum):
    passed = "passed"
    failed = "failed"
    timed_out = "timed_out"
    error = "error"


class RepairStatus(StrEnum):
    pending = "pending"
    baseline_complete = "baseline_complete"
    analysis_complete = "analysis_complete"
    plan_complete = "plan_complete"
    patch_proposed = "patch_proposed"
    verified = "verified"
    failed = "failed"


class FailureCategory(StrEnum):
    syntax_failure = "syntax_failure"
    import_or_dependency_failure = "import_or_dependency_failure"
    assertion_failure = "assertion_failure"
    incorrect_exception_behaviour = "incorrect_exception_behaviour"
    type_related_failure = "type_related_failure"
    timeout = "timeout"
    test_environment_failure = "test_environment_failure"
    unsupported_or_ambiguous = "unsupported_or_ambiguous"


class EvaluationStatus(StrEnum):
    passed = "passed"
    failed = "failed"


class ModelOperation(StrEnum):
    REPAIR_PLANNING = "repair_planning"
    PATCH_GENERATION = "patch_generation"


class ModelDiagnosticStage(StrEnum):
    CONFIGURATION = "configuration"
    PLANNING = "planning"
    PATCH_GENERATION = "patch_generation"
    STATE_UPDATE = "state_update"


class ModelDiagnosticCode(StrEnum):
    MISSING_CONFIGURATION = "missing_configuration"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNSUPPORTED_PROMPT_VERSION = "unsupported_prompt_version"
    INVALID_INPUT_STATE = "invalid_input_state"
    PROVIDER_EXCEPTION = "provider_exception"
    OVERSIZED_RESPONSE = "oversized_response"
    MALFORMED_STRUCTURED_OUTPUT = "malformed_structured_output"
    UNSUITABLE_AUTONOMOUS_REPAIR = "unsuitable_autonomous_repair"
    INVALID_PATCH_ENVELOPE = "invalid_patch_envelope"


class RevisionKind(StrEnum):
    SNAPSHOT = "snapshot"


class BaselineOutcome(StrEnum):
    REPRODUCED_FAILURE = "reproduced_failure"
    NOT_REPRODUCIBLE = "baseline_not_reproducible"
    INFRASTRUCTURE_DIAGNOSTIC = "infrastructure_diagnostic"


class BaselineDiagnosticCode(StrEnum):
    BASELINE_NOT_REPRODUCIBLE = "baseline_not_reproducible"
    INVALID_VISIBLE_TEST = "invalid_visible_test"
    SANDBOX_TIMED_OUT = "sandbox_timed_out"
    SANDBOX_MEMORY_LIMIT = "sandbox_memory_limit"
    SANDBOX_VALIDATION_FAILED = "sandbox_validation_failed"
    SANDBOX_EXECUTION_FAILED = "sandbox_execution_failed"
    SANDBOX_CLEANUP_FAILED = "sandbox_cleanup_failed"
    MALFORMED_SANDBOX_RESULT = "malformed_sandbox_result"


class RepositoryFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    algorithm: Annotated[str, Field(pattern=r"^sha256$")] = "sha256"
    version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @property
    def identifier(self) -> str:
        return f"tracefix-repository-{self.algorithm}-v{self.version}:{self.digest}"


class RepositoryPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: NonBlankString
    requested_revision: NonBlankString
    revision_kind: RevisionKind
    prepared_repository: Path
    fingerprint: RepositoryFingerprint


class BaselineEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    completion: SandboxCompletion
    exit_code: StrictInteger | None
    stdout: str
    stderr: str
    stdout_truncated: Annotated[bool, Field(strict=True)]
    stderr_truncated: Annotated[bool, Field(strict=True)]
    duration_seconds: NonNegativeFiniteFloat


class BaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    preparation: RepositoryPreparation
    outcome: BaselineOutcome
    evidence: BaselineEvidence | None
    diagnostic_code: BaselineDiagnosticCode | None = None
    message: NonBlankString | None = None
    terminal: Literal[True] = True

    @model_validator(mode="after")
    def validate_outcome(self) -> BaselineResult:
        if self.outcome is BaselineOutcome.REPRODUCED_FAILURE:
            if (
                self.evidence is None
                or self.evidence.completion is not SandboxCompletion.NORMAL
                or self.evidence.exit_code in (None, 0)
                or self.diagnostic_code is not None
                or self.message is not None
            ):
                raise ValueError("reproduced failure requires evidence without a diagnostic")
        elif self.outcome is BaselineOutcome.NOT_REPRODUCIBLE:
            if (
                self.evidence is None
                or self.evidence.completion is not SandboxCompletion.NORMAL
                or self.evidence.exit_code != 0
                or self.diagnostic_code is not BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE
                or self.message is None
            ):
                raise ValueError("non-reproducible baseline requires complete pass evidence")
        elif (
            self.diagnostic_code is None
            or self.diagnostic_code is BaselineDiagnosticCode.BASELINE_NOT_REPRODUCIBLE
            or self.message is None
        ):
            raise ValueError("infrastructure diagnostic requires a matching code and message")
        elif self.diagnostic_code is BaselineDiagnosticCode.SANDBOX_TIMED_OUT:
            if (
                self.evidence is None
                or self.evidence.completion is not SandboxCompletion.TIMED_OUT
                or self.evidence.exit_code is not None
            ):
                raise ValueError("timeout diagnostic requires timed-out evidence")
        elif self.diagnostic_code is BaselineDiagnosticCode.SANDBOX_MEMORY_LIMIT:
            if (
                self.evidence is None
                or self.evidence.completion is not SandboxCompletion.MEMORY_LIMIT
            ):
                raise ValueError("memory-limit diagnostic requires memory-limit evidence")
        elif self.evidence is not None:
            raise ValueError("non-execution diagnostic cannot contain execution evidence")
        return self


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    status: TestStatus
    exit_code: StrictInteger
    stdout: str
    stderr: str
    duration_seconds: NonNegativeFiniteFloat


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    category: FailureCategory
    summary: NonBlankString


class ContextItemKind(StrEnum):
    ISSUE = "issue"
    CLASSIFICATION = "classification"
    PROTECTED_PATH_POLICY = "protected_path_policy"
    FAILURE_TRACE = "failure_trace"
    VISIBLE_TEST = "visible_test"
    SOURCE = "source"
    DEFINITION = "definition"
    IMPORT = "import"
    TYPE_DEFINITION = "type_definition"
    REPOSITORY_INSTRUCTION = "repository_instruction"


class ContextOmissionReason(StrEnum):
    BUDGET = "budget"
    AST_PARSE_FAILED = "ast_parse_failed"
    GENERATED = "generated"
    SECRET = "secret"
    UNSAFE_PATH = "unsafe_path"
    UNREADABLE = "unreadable"
    UNRELATED = "unrelated"


def _validate_context_path(path: PurePosixPath) -> None:
    value = path.as_posix()
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or not value
        or value == "."
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("context path must be a normalized relative POSIX path")


class ContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: PurePosixPath
    kind: ContextItemKind
    content: str
    priority: Annotated[int, Field(strict=True, ge=0, le=6)]
    source_line: Annotated[int, Field(strict=True, ge=0)] = 0
    truncated: Annotated[bool, Field(strict=True)] = False
    original_utf8_bytes: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def validate_relative_path_and_size(self) -> ContextItem:
        _validate_context_path(self.path)
        if self.original_utf8_bytes < len(self.content.encode("utf-8")):
            raise ValueError("original size cannot be smaller than included content")
        if self.truncated != (self.original_utf8_bytes > len(self.content.encode("utf-8"))):
            raise ValueError("truncation metadata is inconsistent")
        return self

    def downstream_text(self) -> str:
        return (
            f"[{self.kind.value}|{self.path.as_posix()}|p={self.priority}|l={self.source_line}|"
            f"t={int(self.truncated)}|b={self.original_utf8_bytes}]\n{self.content}\n"
        )


class ContextOmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: PurePosixPath
    kind: ContextItemKind
    reason: ContextOmissionReason

    @model_validator(mode="after")
    def validate_relative_path(self) -> ContextOmission:
        _validate_context_path(self.path)
        return self


class ProtectedPathPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protected_paths: tuple[PurePosixPath, ...]
    exceptions: tuple[PurePosixPath, ...] = ()

    @model_validator(mode="after")
    def validate_policy(self) -> ProtectedPathPolicy:
        if not self.protected_paths or self.exceptions:
            raise ValueError("protected-path policy must be non-empty and grant no exceptions")
        values = tuple(path.as_posix() for path in self.protected_paths)
        for path in self.protected_paths:
            _validate_context_path(path)
        if len({value.casefold() for value in values}) != len(values) or values != tuple(
            sorted(values, key=str.casefold)
        ):
            raise ValueError("protected paths must be unique and deterministically ordered")
        return self


class ContextPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: NonBlankString
    failure_analysis: FailureAnalysis
    protected_path_policy: ProtectedPathPolicy
    items: tuple[ContextItem, ...]
    omissions: tuple[ContextOmission, ...] = ()
    limit_utf8_bytes: Annotated[int, Field(strict=True, gt=0)]
    actual_utf8_bytes: Annotated[int, Field(strict=True, ge=0)]
    safe_material_omitted: Annotated[bool, Field(strict=True)]

    def downstream_text(self) -> str:
        return self.render_downstream(
            case_id=self.case_id,
            items=self.items,
            omissions=self.omissions,
            limit_utf8_bytes=self.limit_utf8_bytes,
            safe_material_omitted=self.safe_material_omitted,
        )

    @staticmethod
    def render_downstream(
        *,
        case_id: str,
        items: tuple[ContextItem, ...],
        omissions: tuple[ContextOmission, ...],
        limit_utf8_bytes: int,
        safe_material_omitted: bool,
    ) -> str:
        header = (
            f"[context|{case_id}|limit={limit_utf8_bytes}|omitted={int(safe_material_omitted)}]\n"
        )
        rendered_items = "".join(item.downstream_text() for item in items)
        rendered_omissions = "".join(
            f"[omission|{item.path.as_posix()}|{item.kind.value}|{item.reason.value}]\n"
            for item in omissions
        )
        return header + rendered_items + rendered_omissions

    @staticmethod
    def downstream_utf8_size(
        *,
        case_id: str,
        items: tuple[ContextItem, ...],
        omissions: tuple[ContextOmission, ...],
        limit_utf8_bytes: int,
        safe_material_omitted: bool,
    ) -> int:
        return len(
            ContextPackage.render_downstream(
                case_id=case_id,
                items=items,
                omissions=omissions,
                limit_utf8_bytes=limit_utf8_bytes,
                safe_material_omitted=safe_material_omitted,
            ).encode("utf-8")
        )

    @model_validator(mode="after")
    def validate_accounting_and_order(self) -> ContextPackage:
        actual = len(self.downstream_text().encode("utf-8"))
        if actual != self.actual_utf8_bytes or actual > self.limit_utf8_bytes:
            raise ValueError("context UTF-8 accounting is invalid")
        order = tuple(
            (item.priority, item.path.as_posix().casefold(), item.source_line, item.kind.value)
            for item in self.items
        )
        if order != tuple(sorted(order)):
            raise ValueError("context items are not deterministically ordered")
        safe_material_omitted = any(item.truncated for item in self.items) or any(
            omission.reason is ContextOmissionReason.BUDGET for omission in self.omissions
        )
        if self.safe_material_omitted != safe_material_omitted:
            raise ValueError("omission flag does not match omission metadata")
        item_keys = tuple(
            (item.path.as_posix().casefold(), item.kind.value, item.source_line)
            for item in self.items
        )
        if len(set(item_keys)) != len(item_keys):
            raise ValueError("context items contain duplicates")
        omission_keys = tuple(
            (item.path.as_posix().casefold(), item.kind.value, item.reason.value)
            for item in self.omissions
        )
        if omission_keys != tuple(sorted(omission_keys)):
            raise ValueError("context omissions are not deterministically ordered")
        if len(set(omission_keys)) != len(omission_keys):
            raise ValueError("context omissions contain duplicates")
        if not any(item.kind is ContextItemKind.ISSUE for item in self.items):
            raise ValueError("issue is not represented in downstream context")
        classification_content = (
            f"{self.failure_analysis.category.value}\n{self.failure_analysis.summary}"
        )
        if not any(
            item.kind is ContextItemKind.CLASSIFICATION and item.content == classification_content
            for item in self.items
        ):
            raise ValueError("failure analysis is not represented in downstream context")
        policy_content = "\n".join(
            path.as_posix() for path in self.protected_path_policy.protected_paths
        )
        if not any(
            item.kind is ContextItemKind.PROTECTED_PATH_POLICY and item.content == policy_content
            for item in self.items
        ):
            raise ValueError("protected-path policy is not represented in downstream context")
        return self


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: NonBlankString
    model_name: NonBlankString
    temperature: ModelTemperature
    max_output_tokens: Annotated[int, Field(strict=True, ge=1, le=32_768)]
    prompt_version: NonBlankString


class ModelProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    operation: ModelOperation
    model_name: NonBlankString
    temperature: ModelTemperature
    max_output_tokens: Annotated[int, Field(strict=True, ge=1, le=32_768)]
    prompt_version: NonBlankString
    attempt_number: Annotated[int, Field(strict=True, ge=1, le=1)]
    rendered_prompt: NonBlankString


class ModelProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    raw_json: str


class ModelDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage: ModelDiagnosticStage
    code: ModelDiagnosticCode
    message: NonBlankString


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    suspected_root_cause: NonBlankString
    files_expected_to_change: list[NonBlankString]
    intended_behavioural_correction: NonBlankString
    risks: list[NonBlankString]
    validation_strategy: NonBlankString
    autonomous_repair_suitable: Annotated[bool, Field(strict=True)]


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    summary: NonBlankString
    root_cause: NonBlankString
    files_changed: list[NonBlankString]
    unified_diff: NonBlankString
    expected_effect: NonBlankString
    risks: list[NonBlankString]
    confidence: Confidence


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    status: EvaluationStatus
    summary: NonBlankString


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LocalRepairCaseState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    case_id: NonBlankString
    status: RepairStatus = RepairStatus.pending
    attempt_number: Annotated[int, Field(strict=True, ge=0, le=1)] = 0
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    baseline_result: TestResult | BaselineResult | None = None
    failure_analysis: FailureAnalysis | None = None
    repair_plan: RepairPlan | None = None
    candidate_patch: PatchProposal | None = None
    verification_results: list[TestResult] = Field(default_factory=list)
    evaluation_results: list[EvaluationResult] = Field(default_factory=list)
    model_provider: NonBlankString | None = None
    model_name: NonBlankString | None = None
    prompt_version: NonBlankString | None = None
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)
