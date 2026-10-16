from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from app.agent.routing import TerminalOutcome
from app.agent.schemas import BaselineOutcome, FailureCategory
from app.agent.static_checks import StaticCheckStatus, StaticValidationOutcome
from app.agent.verification import VerificationOutcome, VerificationStageStatus
from app.artifacts import RunArtifact
from app.evaluation.deterministic.candidate import CandidateFunctionalStatus
from app.sandbox.results import SandboxCompletion
from app.sandbox.runner import CONTAINER_NAME_PREFIX, SANDBOX_IMAGE
from app.sandbox.static_runner import STATIC_CHECK_IMAGE, STATIC_CONTAINER_NAME_PREFIX
from app.sandbox.verification_runner import (
    HIDDEN_BUNDLE_PREFIX,
    VERIFICATION_CONTAINER_NAME_PREFIX,
)

pytestmark = pytest.mark.integration

_EXPECTED = {
    "boundary_condition": (
        "src/pagination/pagination.py",
        FailureCategory.assertion_failure,
        ("test_partial_page", "items // size + 1"),
    ),
    "exception_handling": (
        "src/parser/parser.py",
        FailureCategory.unsupported_or_ambiguous,
        ("test_valid_port_is_parsed", "except TypeError:"),
    ),
    "fixture_or_mocking": (
        "src/greeter/greeter.py",
        FailureCategory.unsupported_or_ambiguous,
        ("test_client_is_called_once", "client.fetch_users()"),
    ),
    "incorrect_conditional": (
        "src/eligibility/eligibility.py",
        FailureCategory.assertion_failure,
        ("test_minor_is_not_eligible", "return age > 18"),
    ),
    "incorrect_return_value": (
        "src/normalizer/normalizer.py",
        FailureCategory.assertion_failure,
        ("test_preserves_internal_space", "return name.upper()"),
    ),
}
_CONTAINER_PREFIXES = (
    CONTAINER_NAME_PREFIX,
    STATIC_CONTAINER_NAME_PREFIX,
    VERIFICATION_CONTAINER_NAME_PREFIX,
)
_SUMMARY = re.compile(
    r"^case_id=(?P<case_id>[a-z0-9_-]+) outcome=success "
    r"summary=repair candidate passed all required checks exit_code=0 "
    r"artifact=(?P<artifact>\.tracefix/artifacts/[a-z0-9_-]+-[0-9a-f]{32}\.json)$"
)


def _docker(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("docker", *arguments),
        check=False,
        capture_output=True,
        shell=False,
        timeout=120.0,
    )


def _unavailable(message: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module", autouse=True)
def require_docker_and_images() -> None:
    try:
        version = _docker("version", "--format", "{{.Server.Os}}")
    except (OSError, subprocess.TimeoutExpired):
        _unavailable("Docker CLI is unavailable")
        return
    if version.returncode != 0 or version.stdout.strip() != b"linux":
        _unavailable("a usable Linux-container Docker daemon is unavailable")
        return
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.returncode != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
    for image in (SANDBOX_IMAGE, STATIC_CHECK_IMAGE):
        inspected = _docker("image", "inspect", image)
        if inspected.returncode != 0:
            pytest.fail(f"required integration image is absent: {image}")


def _tracefix_containers() -> set[str]:
    result = _docker("ps", "--all", "--format", "{{.Names}}")
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    names = result.stdout.decode("utf-8", errors="strict").splitlines()
    return {name for name in names if name.startswith(_CONTAINER_PREFIXES)}


def _sanitized_environment(runtime_temp: Path) -> tuple[dict[str, str], str]:
    environment = dict(os.environ)
    forbidden_fragments = (
        "api_key",
        "apikey",
        "access_token",
        "access_key",
        "auth_token",
        "credential",
        "password",
        "private_key",
        "proxy",
        "secret",
    )
    for name in tuple(environment):
        folded = name.casefold()
        if any(fragment in folded for fragment in forbidden_fragments):
            del environment[name]
    for name in (
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "MODEL_TEMPERATURE",
        "MODEL_MAX_TOKENS",
        "PROMPT_VERSION",
    ):
        environment.pop(name, None)
    sentinel = "tracefix-environment-value-must-not-leak"
    environment.update(
        {
            "MODEL_PROVIDER": "tracefix-fake",
            "MODEL_NAME": "tracefix-deterministic-v1",
            "MODEL_TEMPERATURE": "0",
            "MODEL_MAX_TOKENS": "4096",
            "PROMPT_VERSION": "repair-v1.0",
            "TRACEFIX_UNRELATED_VALUE": sentinel,
            "TEMP": str(runtime_temp),
            "TMP": str(runtime_temp),
            "TMPDIR": str(runtime_temp),
        }
    )
    return environment, sentinel


def _assert_no_disclosure(
    value: str,
    *,
    invocation: Path,
    runtime_temp: Path,
    case_tokens: tuple[str, str],
    sentinel: str,
) -> None:
    prohibited = (
        "rendered_prompt",
        "raw_json",
        "repair_plan",
        "proposal",
        "unified_diff",
        "reference.patch",
        "hidden_tests",
        "evaluator/",
        "evaluator\\",
        "/tracefix/",
        "Traceback",
        sentinel,
        *case_tokens,
    )
    for token in prohibited:
        assert token not in value
    for path in (Path.cwd().resolve(), invocation, runtime_temp):
        rendered = str(path)
        assert rendered not in value
        assert rendered.replace("\\", "/") not in value
        assert rendered.replace("\\", "\\\\") not in value


@pytest.mark.parametrize("case_id", tuple(_EXPECTED))
def test_installed_cli_repairs_every_fixture_through_real_pipeline(
    tmp_path: Path,
    case_id: str,
) -> None:
    expected_file, expected_category, case_tokens = _EXPECTED[case_id]
    invocation = (tmp_path / "invocation").resolve()
    runtime_temp = (tmp_path / "runtime-temp").resolve()
    invocation.mkdir()
    runtime_temp.mkdir()
    executable = Path(sys.executable).with_name(
        "tracefix.exe" if sys.platform == "win32" else "tracefix"
    )
    assert executable.is_file(), f"installed CLI not found: {executable}"
    environment, sentinel = _sanitized_environment(runtime_temp)
    containers_before = _tracefix_containers()

    completed = subprocess.run(
        (str(executable), "repair", case_id),
        cwd=invocation,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        shell=False,
        timeout=300.0,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    stdout_lines = completed.stdout.splitlines()
    assert len(stdout_lines) == 1
    matched = _SUMMARY.fullmatch(stdout_lines[0])
    assert matched is not None
    assert matched.group("case_id") == case_id
    relative_path = PurePosixPath(matched.group("artifact"))
    artifact_path = invocation.joinpath(*relative_path.parts)
    artifacts = tuple((invocation / ".tracefix" / "artifacts").iterdir())
    assert artifacts == (artifact_path,)
    raw_artifact = artifact_path.read_text(encoding="utf-8")
    artifact = RunArtifact.model_validate_json(raw_artifact, strict=True)
    assert RunArtifact.model_validate(artifact.model_dump(warnings="none"), strict=True) == artifact

    assert artifact.case_id == case_id
    assert artifact.outcome is TerminalOutcome.SUCCESS
    assert artifact.exit_code == 0
    assert artifact.attempt_number == 1
    assert artifact.max_attempts == 1
    assert artifact.provider_name == "tracefix-fake"
    assert artifact.model_name == "tracefix-deterministic-v1"
    assert artifact.prompt_version == "repair-v1.0"
    assert artifact.terminal_diagnostic is None
    assert artifact.completed_at >= artifact.started_at
    assert artifact.failure_category is expected_category

    assert artifact.baseline is not None
    assert artifact.baseline.outcome is BaselineOutcome.REPRODUCED_FAILURE
    assert artifact.baseline.completion is SandboxCompletion.NORMAL
    assert artifact.baseline.exit_code not in (None, 0)
    assert artifact.baseline.stdout_truncated is False
    assert artifact.baseline.stderr_truncated is False
    assert artifact.baseline.duration_seconds is not None

    assert artifact.policy is not None
    assert artifact.policy.approved
    assert artifact.policy.violations == ()
    assert artifact.static is not None
    assert artifact.static.outcome is StaticValidationOutcome.PASSED
    assert artifact.static.diagnostic is None
    assert all(result.status is StaticCheckStatus.PASSED for result in artifact.static.checks)

    assert artifact.verification is not None
    assert artifact.verification.outcome is VerificationOutcome.PASSED
    assert all(
        stage.status is VerificationStageStatus.PASSED
        and stage.completion is SandboxCompletion.NORMAL
        and stage.exit_code == 0
        and stage.diagnostic is None
        for stage in artifact.verification.stages
    )
    assert artifact.candidate_evaluation is not None
    assert artifact.candidate_evaluation.functional_status is CandidateFunctionalStatus.PASSED
    assert artifact.candidate_evaluation.files_changed == (expected_file,)
    assert artifact.candidate_evaluation.lines_added == 1
    assert artifact.candidate_evaluation.lines_removed == 1
    assert artifact.candidate_evaluation.sandbox_duration_seconds >= 0.0
    assert artifact.candidate_evaluation.model_latency_seconds is None
    assert artifact.candidate_evaluation.input_tokens is None
    assert artifact.candidate_evaluation.output_tokens is None
    assert artifact.candidate_evaluation.estimated_cost_usd is None

    _assert_no_disclosure(
        completed.stdout + completed.stderr + raw_artifact,
        invocation=invocation,
        runtime_temp=runtime_temp,
        case_tokens=case_tokens,
        sentinel=sentinel,
    )
    assert not tuple(runtime_temp.glob("tracefix-run-*"))
    assert not tuple(runtime_temp.rglob(f"{HIDDEN_BUNDLE_PREFIX}*"))
    assert _tracefix_containers() == containers_before
