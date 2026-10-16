from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agent.model import ModelProvider
from app.agent.patches import normalize_repository_paths, parse_unified_diff
from app.agent.schemas import (
    ModelConfiguration,
    ModelOperation,
    ModelProviderRequest,
    ModelProviderResponse,
    PatchProposal,
    RepairPlan,
)

FAKE_PROVIDER_NAME = "tracefix-fake"
FAKE_MODEL_NAME = "tracefix-deterministic-v1"
FAKE_PROMPT_VERSION = "repair-v1.0"
FAKE_TEMPERATURE = 0.0
FAKE_MAX_OUTPUT_TOKENS = 4096

type FakeModelCatalogKey = tuple[str, str, int]


class FakeModelError(ValueError):
    """The deterministic provider rejected an invalid trusted or untrusted boundary value."""


class FakeModelResponseBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: Annotated[
        str,
        Field(strict=True, pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$", max_length=128),
    ]
    prompt_version: Annotated[str, Field(strict=True, min_length=1)]
    attempt_number: Annotated[int, Field(strict=True, ge=1, le=1)]
    planning_json: Annotated[str, Field(strict=True, min_length=1)]
    generation_json: Annotated[str, Field(strict=True, min_length=1)]

    @model_validator(mode="after")
    def validate_responses(self) -> Self:
        try:
            plan = RepairPlan.model_validate_json(self.planning_json, strict=True)
            proposal = PatchProposal.model_validate_json(self.generation_json, strict=True)
            plan_paths = normalize_repository_paths(plan.files_expected_to_change)
            proposal_paths = normalize_repository_paths(proposal.files_changed)
            parsed = parse_unified_diff(proposal.unified_diff)
        except (TypeError, ValueError, ValidationError):
            raise ValueError("fake model response bundle is invalid") from None
        if (
            plan_paths != proposal_paths
            or proposal_paths != parsed.changed_paths
            or parsed.normalized_text != proposal.unified_diff
            or plan.model_dump_json() != self.planning_json
            or proposal.model_dump_json() != self.generation_json
        ):
            raise ValueError("fake model response bundle is inconsistent")
        return self


def _bundle(
    *,
    case_id: str,
    plan: RepairPlan,
    proposal: PatchProposal,
) -> FakeModelResponseBundle:
    return FakeModelResponseBundle(
        case_id=case_id,
        prompt_version=FAKE_PROMPT_VERSION,
        attempt_number=1,
        planning_json=plan.model_dump_json(),
        generation_json=proposal.model_dump_json(),
    )


_RESPONSE_BUNDLES = (
    _bundle(
        case_id="boundary_condition",
        plan=RepairPlan(
            suspected_root_cause="Exact page multiples receive an unconditional extra page",
            files_expected_to_change=["src/pagination/pagination.py"],
            intended_behavioural_correction="Compute the ceiling division without an extra page",
            risks=["Non-positive page sizes remain outside this repair"],
            validation_strategy="Run targeted, visible, and hidden tests",
            autonomous_repair_suitable=True,
        ),
        proposal=PatchProposal(
            summary="Correct page-count ceiling division",
            root_cause="The calculation always adds one page",
            files_changed=["src/pagination/pagination.py"],
            unified_diff=(
                "--- a/src/pagination/pagination.py\n"
                "+++ b/src/pagination/pagination.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def page_count(items: int, size: int) -> int:\n"
                "-    return items // size + 1\n"
                "+    return (items + size - 1) // size\n"
            ),
            expected_effect="Exact and partial page counts are calculated correctly",
            risks=["Non-positive page sizes remain outside this repair"],
            confidence=1.0,
        ),
    ),
    _bundle(
        case_id="exception_handling",
        plan=RepairPlan(
            suspected_root_cause="Invalid numeric text raises ValueError without being handled",
            files_expected_to_change=["src/parser/parser.py"],
            intended_behavioural_correction="Return None for both invalid types and invalid text",
            risks=["Valid integer parsing behavior must remain unchanged"],
            validation_strategy="Run targeted, visible, and hidden tests",
            autonomous_repair_suitable=True,
        ),
        proposal=PatchProposal(
            summary="Handle invalid numeric text",
            root_cause="The exception handler omits ValueError",
            files_changed=["src/parser/parser.py"],
            unified_diff=(
                "--- a/src/parser/parser.py\n"
                "+++ b/src/parser/parser.py\n"
                "@@ -1,5 +1,5 @@\n"
                " def parse_port(value: str) -> int | None:\n"
                "     try:\n"
                "         return int(value)\n"
                "-    except TypeError:\n"
                "+    except (TypeError, ValueError):\n"
                "         return None\n"
            ),
            expected_effect="Invalid text returns None while valid ports still parse",
            risks=["Valid integer parsing behavior must remain unchanged"],
            confidence=1.0,
        ),
    ),
    _bundle(
        case_id="fixture_or_mocking",
        plan=RepairPlan(
            suspected_root_cause="Greeting calls a plural method absent from the client protocol",
            files_expected_to_change=["src/greeter/greeter.py"],
            intended_behavioural_correction="Call fetch_user exactly once",
            risks=["The client call must not be duplicated"],
            validation_strategy="Run targeted, visible, and hidden tests",
            autonomous_repair_suitable=True,
        ),
        proposal=PatchProposal(
            summary="Use the declared user-client method",
            root_cause="Greeting calls fetch_users instead of fetch_user",
            files_changed=["src/greeter/greeter.py"],
            unified_diff=(
                "--- a/src/greeter/greeter.py\n"
                "+++ b/src/greeter/greeter.py\n"
                "@@ -6,4 +6,4 @@ class UserClient(Protocol):\n"
                " \n"
                " \n"
                " def greeting(client: UserClient) -> str:\n"
                '-    return f"Hello, {client.fetch_users()}!"  # type: ignore[attr-defined]\n'
                '+    return f"Hello, {client.fetch_user()}!"\n'
            ),
            expected_effect="Greeting uses the protocol method once and formats its result",
            risks=["The client call must not be duplicated"],
            confidence=1.0,
        ),
    ),
    _bundle(
        case_id="incorrect_conditional",
        plan=RepairPlan(
            suspected_root_cause="The adult threshold excludes age eighteen",
            files_expected_to_change=["src/eligibility/eligibility.py"],
            intended_behavioural_correction="Include the minimum adult age",
            risks=["Younger ages must remain ineligible"],
            validation_strategy="Run targeted, visible, and hidden tests",
            autonomous_repair_suitable=True,
        ),
        proposal=PatchProposal(
            summary="Include the adult eligibility threshold",
            root_cause="The conditional uses a strict comparison",
            files_changed=["src/eligibility/eligibility.py"],
            unified_diff=(
                "--- a/src/eligibility/eligibility.py\n"
                "+++ b/src/eligibility/eligibility.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def is_eligible(age: int) -> bool:\n"
                "-    return age > 18\n"
                "+    return age >= 18\n"
            ),
            expected_effect="Age eighteen is eligible while minors remain ineligible",
            risks=["Younger ages must remain ineligible"],
            confidence=1.0,
        ),
    ),
    _bundle(
        case_id="incorrect_return_value",
        plan=RepairPlan(
            suspected_root_cause="Normalization uppercases input without trimming it",
            files_expected_to_change=["src/normalizer/normalizer.py"],
            intended_behavioural_correction="Trim surrounding whitespace and lowercase the name",
            risks=["Internal whitespace must remain unchanged"],
            validation_strategy="Run targeted, visible, and hidden tests",
            autonomous_repair_suitable=True,
        ),
        proposal=PatchProposal(
            summary="Return the normalized lowercase name",
            root_cause="The function returns uppercase untrimmed input",
            files_changed=["src/normalizer/normalizer.py"],
            unified_diff=(
                "--- a/src/normalizer/normalizer.py\n"
                "+++ b/src/normalizer/normalizer.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def normalize(name: str) -> str:\n"
                "-    return name.upper()\n"
                "+    return name.strip().lower()\n"
            ),
            expected_effect="Names are trimmed and lowercased without collapsing internal space",
            risks=["Internal whitespace must remain unchanged"],
            confidence=1.0,
        ),
    ),
)


def _build_catalog(
    bundles: tuple[FakeModelResponseBundle, ...],
) -> Mapping[FakeModelCatalogKey, FakeModelResponseBundle]:
    catalog: dict[FakeModelCatalogKey, FakeModelResponseBundle] = {}
    folded_keys: set[FakeModelCatalogKey] = set()
    try:
        for candidate in bundles:
            if type(candidate) is not FakeModelResponseBundle:
                raise ValueError("fake model catalog entry has the wrong type")
            bundle = FakeModelResponseBundle.model_validate(
                candidate.model_dump(warnings="none"), strict=True
            )
            key = (bundle.case_id, bundle.prompt_version, bundle.attempt_number)
            folded = (
                bundle.case_id.casefold(),
                bundle.prompt_version.casefold(),
                bundle.attempt_number,
            )
            if key in catalog or folded in folded_keys:
                raise ValueError("fake model catalog key is duplicated")
            catalog[key] = bundle
            folded_keys.add(folded)
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise FakeModelError("fake model catalog is invalid") from None
    return MappingProxyType(catalog)


FAKE_MODEL_CATALOG = _build_catalog(_RESPONSE_BUNDLES)
_EXPECTED_CATALOG_KEYS = frozenset(
    (case_id, FAKE_PROMPT_VERSION, 1)
    for case_id in (
        "boundary_condition",
        "exception_handling",
        "fixture_or_mocking",
        "incorrect_conditional",
        "incorrect_return_value",
    )
)
if frozenset(FAKE_MODEL_CATALOG) != _EXPECTED_CATALOG_KEYS:
    raise FakeModelError("fake model catalog is incomplete")


class DeterministicFakeModelProvider:
    def complete(self, request: ModelProviderRequest) -> ModelProviderResponse:
        try:
            if type(request) is not ModelProviderRequest:
                raise ValueError("fake model request has the wrong type")
            checked = ModelProviderRequest.model_validate(
                request.model_dump(warnings="none"), strict=True
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise FakeModelError("fake model request is invalid") from None
        if (
            checked.model_name != FAKE_MODEL_NAME
            or checked.temperature != FAKE_TEMPERATURE
            or checked.max_output_tokens != FAKE_MAX_OUTPUT_TOKENS
        ):
            raise FakeModelError("fake model request is unsupported")
        key = (checked.case_id, checked.prompt_version, checked.attempt_number)
        bundle = FAKE_MODEL_CATALOG.get(key)
        if bundle is None:
            raise FakeModelError("fake model request is unsupported")
        if checked.operation is ModelOperation.REPAIR_PLANNING:
            raw_json = bundle.planning_json
        elif checked.operation is ModelOperation.PATCH_GENERATION:
            raw_json = bundle.generation_json
        else:
            raise FakeModelError("fake model request is unsupported")
        return ModelProviderResponse(raw_json=raw_json)


_SUPPORTED_CONFIGURATION = ModelConfiguration(
    provider=FAKE_PROVIDER_NAME,
    model_name=FAKE_MODEL_NAME,
    temperature=FAKE_TEMPERATURE,
    max_output_tokens=FAKE_MAX_OUTPUT_TOKENS,
    prompt_version=FAKE_PROMPT_VERSION,
)


def create_fake_model_provider(configuration: ModelConfiguration) -> ModelProvider:
    try:
        if type(configuration) is not ModelConfiguration:
            raise ValueError("fake model configuration has the wrong type")
        checked = ModelConfiguration.model_validate(
            configuration.model_dump(warnings="none"), strict=True
        )
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise FakeModelError("fake model configuration is invalid") from None
    if checked != _SUPPORTED_CONFIGURATION:
        raise FakeModelError("fake model configuration is unsupported")
    return DeterministicFakeModelProvider()
