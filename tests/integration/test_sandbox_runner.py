from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.sandbox.results import TRUNCATION_MARKER, SandboxCompletion, SandboxLimits
from app.sandbox.runner import (
    CONTAINER_NAME_PREFIX,
    SANDBOX_IMAGE,
    PytestCommand,
    SandboxRunner,
    SandboxValidationError,
    SubprocessDockerCommandAdapter,
)

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
    image = _docker("image", "inspect", SANDBOX_IMAGE)
    if image.returncode != 0:
        pytest.fail(
            "sandbox image is absent; run: docker build --tag "
            f"{SANDBOX_IMAGE} --file docker/sandbox/Dockerfile ."
        )


def _containers() -> set[str]:
    result = _docker(
        "ps", "--all", "--filter", f"name=^{CONTAINER_NAME_PREFIX}", "--format", "{{.Names}}"
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return set(result.stdout.decode("utf-8", errors="replace").splitlines())


def _write_repository(repository: Path, test_source: str) -> None:
    tests = repository / "tests"
    tests.mkdir(parents=True)
    test_file = tests / "test_sandbox.py"
    test_file.write_text(test_source, encoding="utf-8")

    # Native Linux preserves host ownership and modes for bind mounts. Pytest's
    # per-test temporary directory is private to the host runner by default, so
    # the sandbox's fixed non-root user otherwise cannot traverse the prepared
    # repository mounted at /tracefix/input. These permissions expose only the
    # synthetic repository; the bind mount remains read-only in the container.
    repository.chmod(0o755)
    tests.chmod(0o755)
    test_file.chmod(0o644)


def test_passing_pytest_observes_required_container_controls(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        """import os
import socket
from pathlib import Path


def test_controls_and_output():
    assert os.getuid() == 10001
    assert os.getgid() == 10001
    Path('workspace-write.txt').write_text('writable', encoding='utf-8')
    try:
        Path('/root-write.txt').write_text('forbidden', encoding='utf-8')
    except OSError:
        pass
    else:
        raise AssertionError('container root filesystem was writable')
    with socket.socket() as connection:
        connection.settimeout(1.0)
        assert connection.connect_ex(('1.1.1.1', 53)) != 0
    print('sandbox-stdout-marker')
""",
    )
    before = _containers()

    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(),
        (
            "python",
            "-m",
            "pytest",
            "tests/test_sandbox.py",
        ),
    )

    assert result.exit_code == 0
    assert result.completion is SandboxCompletion.NORMAL
    assert "1 passed" in result.stdout
    assert result.stderr == ""
    assert result.duration_seconds >= 0.0
    assert _containers() == before


def test_failing_pytest_returns_structured_failure_and_cleans_up(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        """import sys


def test_failure():
    print('failure-stdout-marker')
    print('failure-stderr-marker', file=sys.stderr)
    assert False, 'expected failure'
""",
    )
    before = _containers()

    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(), ("python", "-m", "pytest")
    )

    assert result.exit_code != 0
    assert result.completion is SandboxCompletion.NORMAL
    assert "failure-stdout-marker" in result.stdout
    assert "failure-stderr-marker" in result.stdout
    assert isinstance(result.stderr, str)
    assert result.duration_seconds >= 0.0
    assert _containers() == before


def test_invalid_command_is_rejected_without_container_creation(tmp_path: Path) -> None:
    before = _containers()
    runner = SandboxRunner(SubprocessDockerCommandAdapter())

    with pytest.raises(SandboxValidationError):
        runner.execute(tmp_path.resolve(), ("python", "-m", "pytest", "-q"))

    assert _containers() == before


def test_approved_command_is_immutable() -> None:
    command = PytestCommand(("python", "-m", "pytest"))
    with pytest.raises(AttributeError):
        command.tokens = ("sh",)


def test_infinite_workload_times_out_and_cleans_up(tmp_path: Path) -> None:
    _write_repository(tmp_path, "def test_forever():\n    while True:\n        pass\n")
    before = _containers()
    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), SandboxLimits(timeout_seconds=1.0)
    )
    assert result.completion is SandboxCompletion.TIMED_OUT
    assert result.exit_code is None
    assert result.duration_seconds < 5.0
    assert _containers() == before


def test_memory_exhaustion_is_classified_from_docker_state(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        "def test_memory():\n    blocks = []\n    while True:\n        blocks.append(bytearray(1024 * 1024))\n",
    )
    before = _containers()
    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(),
        ("python", "-m", "pytest"),
        SandboxLimits(memory_bytes=64 * 1024 * 1024, timeout_seconds=10.0),
    )
    assert result.completion is SandboxCompletion.MEMORY_LIMIT
    assert _containers() == before


def test_pid_limit_bounds_child_creation(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        """import subprocess
import sys

def test_process_bound():
    children = []
    try:
        for _ in range(64):
            children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']))
    except OSError:
        pass
    finally:
        for child in children:
            child.kill()
    assert len(children) < 64
""",
    )
    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), SandboxLimits(pids_limit=16)
    )
    assert result.completion is SandboxCompletion.NORMAL
    assert result.exit_code == 0


def test_stdout_and_stderr_are_independently_truncated(tmp_path: Path) -> None:
    _write_repository(tmp_path, "def test_output():\n    pass\n")
    (tmp_path / "conftest.py").write_text(
        "import os\ndef pytest_unconfigure(config):\n"
        "    os.write(1, b'x' * 10000)\n    os.write(2, b'y' * 10000)\n",
        encoding="utf-8",
    )
    limits = SandboxLimits(stdout_bytes=256, stderr_bytes=192)
    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), limits
    )
    assert result.stdout_truncated
    assert result.stderr_truncated
    assert result.stdout.endswith(TRUNCATION_MARKER.decode())
    assert result.stderr.endswith(TRUNCATION_MARKER.decode())
    assert len(result.stdout.encode()) <= limits.stdout_bytes
    assert len(result.stderr.encode()) <= limits.stderr_bytes
