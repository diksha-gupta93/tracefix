from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from app.agent.model import ModelProvider
from app.agent.patches import normalize_repository_paths, parse_unified_diff
from app.agent.schemas import (
    ModelConfiguration,
    ModelOperation,
    ModelProviderRequest,
    PatchProposal,
    RepairPlan,
)
from app.evaluation.deterministic.fake_model import (
    FAKE_MODEL_CATALOG,
    DeterministicFakeModelProvider,
    FakeModelError,
    FakeModelResponseBundle,
    _build_catalog,
    create_fake_model_provider,
)

CASE_FILES = {
    "boundary_condition": "src/pagination/pagination.py",
    "exception_handling": "src/parser/parser.py",
    "fixture_or_mocking": "src/greeter/greeter.py",
    "incorrect_conditional": "src/eligibility/eligibility.py",
    "incorrect_return_value": "src/normalizer/normalizer.py",
}


def configuration() -> ModelConfiguration:
    return ModelConfiguration(
        provider="tracefix-fake",
        model_name="tracefix-deterministic-v1",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
    )


def request(
    *,
    case_id: str = "boundary_condition",
    operation: ModelOperation = ModelOperation.REPAIR_PLANNING,
) -> ModelProviderRequest:
    return ModelProviderRequest(
        case_id=case_id,
        operation=operation,
        model_name="tracefix-deterministic-v1",
        temperature=0.0,
        max_output_tokens=4096,
        prompt_version="repair-v1.0",
        attempt_number=1,
        rendered_prompt="untrusted prompt that must not be parsed",
    )


def test_catalog_has_exact_immutable_keys_and_valid_responses() -> None:
    assert tuple(FAKE_MODEL_CATALOG) == tuple(
        (case_id, "repair-v1.0", 1) for case_id in sorted(CASE_FILES)
    )

    with pytest.raises(TypeError):
        FAKE_MODEL_CATALOG[("other", "repair-v1.0", 1)] = next(iter(FAKE_MODEL_CATALOG.values()))

    for key, bundle in FAKE_MODEL_CATALOG.items():
        assert key == (bundle.case_id, bundle.prompt_version, bundle.attempt_number)
        restored = FakeModelResponseBundle.model_validate_json(
            bundle.model_dump_json(), strict=True
        )
        assert restored == bundle
        plan = RepairPlan.model_validate_json(bundle.planning_json, strict=True)
        proposal = PatchProposal.model_validate_json(bundle.generation_json, strict=True)
        parsed = parse_unified_diff(proposal.unified_diff)
        expected = (CASE_FILES[bundle.case_id],)
        assert tuple(normalize_repository_paths(plan.files_expected_to_change)) == expected
        assert tuple(normalize_repository_paths(proposal.files_changed)) == expected
        assert parsed.changed_paths == expected
        assert parsed.normalized_text == proposal.unified_diff
        assert "\r" not in bundle.planning_json
        assert "\r" not in bundle.generation_json


def test_provider_selects_operation_and_returns_byte_stable_fresh_envelopes() -> None:
    provider: ModelProvider = DeterministicFakeModelProvider()
    planning_request = request()
    generation_request = request(operation=ModelOperation.PATCH_GENERATION)

    first = provider.complete(planning_request)
    second = provider.complete(planning_request)
    generated = provider.complete(generation_request)

    assert first is not second
    assert first.raw_json == second.raw_json
    assert RepairPlan.model_validate_json(first.raw_json, strict=True)
    assert PatchProposal.model_validate_json(generated.raw_json, strict=True)
    assert first.raw_json != generated.raw_json


@pytest.mark.parametrize("case_id", ["unknown", "boundary-condition"])
def test_provider_rejects_unknown_case_without_parsing_prompt(case_id: str) -> None:
    with pytest.raises(FakeModelError, match="fake model request is unsupported"):
        DeterministicFakeModelProvider().complete(request(case_id=case_id))


def test_provider_rejects_subclassed_and_mutated_requests() -> None:
    class RequestSubclass(ModelProviderRequest):
        pass

    base = request()
    subclassed = RequestSubclass.model_validate(base.model_dump(), strict=True)
    mutated = request()
    object.__setattr__(mutated, "attempt_number", 2)

    provider = DeterministicFakeModelProvider()
    with pytest.raises(FakeModelError):
        provider.complete(subclassed)
    with pytest.raises(FakeModelError):
        provider.complete(mutated)
    with pytest.raises(FakeModelError):
        provider.complete(cast(ModelProviderRequest, object()))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other"),
        ("temperature", 0.5),
        ("max_output_tokens", 2048),
        ("prompt_version", "other"),
    ],
)
def test_provider_rejects_request_outside_supported_profile(field: str, value: object) -> None:
    unsupported = request().model_copy(update={field: value})

    with pytest.raises(FakeModelError, match="fake model request is unsupported"):
        DeterministicFakeModelProvider().complete(unsupported)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "other"),
        ("model_name", "other"),
        ("temperature", 0.5),
        ("max_output_tokens", 2048),
        ("prompt_version", "other"),
    ],
)
def test_factory_accepts_only_exact_supported_profile(field: str, value: object) -> None:
    mismatched = configuration().model_copy(update={field: value})

    with pytest.raises(FakeModelError, match="fake model configuration is unsupported"):
        create_fake_model_provider(mismatched)

    assert isinstance(create_fake_model_provider(configuration()), DeterministicFakeModelProvider)


def test_factory_and_provider_revalidate_mutated_base_models() -> None:
    mutated_configuration = configuration()
    object.__setattr__(mutated_configuration, "model_name", "other")
    mutated_request = request()
    object.__setattr__(mutated_request, "operation", "unknown")

    with pytest.raises(FakeModelError):
        create_fake_model_provider(mutated_configuration)
    with pytest.raises(FakeModelError):
        DeterministicFakeModelProvider().complete(mutated_request)


def test_catalog_builder_rejects_duplicate_casefold_collision_and_malformed_bundle() -> None:
    valid = next(iter(FAKE_MODEL_CATALOG.values()))
    duplicate = valid.model_copy()
    collision = valid.model_copy(update={"prompt_version": valid.prompt_version.upper()})
    malformed = FakeModelResponseBundle.model_construct(
        case_id="malformed",
        prompt_version="repair-v1.0",
        attempt_number=1,
        planning_json="{}",
        generation_json="{}",
    )

    with pytest.raises(FakeModelError):
        _build_catalog((valid, duplicate))
    with pytest.raises(FakeModelError):
        _build_catalog((valid, collision))
    with pytest.raises(FakeModelError):
        _build_catalog((malformed,))


def test_bundle_is_strict_frozen_and_rejects_missing_operation_response() -> None:
    bundle = next(iter(FAKE_MODEL_CATALOG.values()))
    with pytest.raises(ValidationError):
        bundle.case_id = "changed"
    with pytest.raises(ValidationError):
        FakeModelResponseBundle.model_validate(
            {
                "case_id": bundle.case_id,
                "prompt_version": bundle.prompt_version,
                "attempt_number": bundle.attempt_number,
                "planning_json": bundle.planning_json,
            },
            strict=True,
        )


def test_provider_construction_and_completion_use_no_external_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*values: object, **named: object) -> None:
        del values, named
        raise AssertionError("external capability used")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr("os.getenv", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr("benchmarks.loader.load_trusted_case", forbidden)

    provider = DeterministicFakeModelProvider()
    response = provider.complete(request())

    assert RepairPlan.model_validate_json(response.raw_json, strict=True)
    assert "hidden_tests" not in response.raw_json
    assert "reference.patch" not in response.raw_json
