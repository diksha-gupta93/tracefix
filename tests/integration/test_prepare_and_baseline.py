from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from app.agent.baseline import BaselineExecutor
from app.agent.preparation import RepositoryPreparer
from app.agent.schemas import BaselineOutcome
from app.sandbox import SandboxCompletion, SandboxRunner, SubprocessDockerCommandAdapter
from app.sandbox.runner import SANDBOX_IMAGE
from benchmarks.loader import TrustedCase, load_development_cases, load_trusted_case

pytestmark = pytest.mark.integration


def _docker(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(("docker", *arguments), check=False, capture_output=True, shell=False)


@pytest.fixture(scope="module", autouse=True)
def require_docker_and_image() -> None:
    try:
        version = _docker("version", "--format", "{{.Server.Os}}")
    except OSError:
        pytest.skip("Docker CLI is unavailable")
    if version.returncode != 0 or version.stdout.strip() != b"linux":
        pytest.skip("a usable Linux-container Docker daemon is unavailable")
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.returncode != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
    image = _docker("image", "inspect", SANDBOX_IMAGE)
    if image.returncode != 0:
        pytest.fail(
            f"sandbox image is absent; run: docker build --tag {SANDBOX_IMAGE} "
            "--file docker/sandbox/Dockerfile ."
        )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize("case", load_development_cases(), ids=lambda case: case.manifest.case_id)
def test_prepares_copy_and_reproduces_visible_failure(case: TrustedCase, tmp_path: Path) -> None:
    source_before = _tree_digest(case.case_path)
    preparation = RepositoryPreparer(load_trusted_case).prepare(
        case.manifest.case_id,
        case.manifest.failing_revision,
        tmp_path,
    )
    result = BaselineExecutor(
        load_trusted_case, SandboxRunner(SubprocessDockerCommandAdapter())
    ).run(preparation)

    assert preparation.prepared_repository != case.repository_path
    assert preparation.prepared_repository.is_relative_to(tmp_path.resolve())
    assert result.outcome is BaselineOutcome.REPRODUCED_FAILURE
    assert result.evidence is not None
    assert result.evidence.completion is SandboxCompletion.NORMAL
    assert result.evidence.exit_code is not None and result.evidence.exit_code != 0
    assert _tree_digest(case.case_path) == source_before
