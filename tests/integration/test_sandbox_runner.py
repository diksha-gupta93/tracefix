from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.sandbox.results import TRUNCATION_MARKER, SandboxCompletion, SandboxLimits
from app.sandbox.runner import (
    CONTAINER_NAME_PREFIX,
    SANDBOX_IMAGE,
    DockerCommandResult,
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
    security = _docker("info", "--format", "{{json .SecurityOptions}}")
    if security.returncode != 0 or b"name=seccomp" not in security.stdout:
        pytest.fail("the Linux Docker daemon must report seccomp support")
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
    if os.name != "nt":
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
    status = Path('/proc/self/status').read_text(encoding='utf-8')
    fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
    assert int(fields['CapEff'].strip(), 16) == 0
    assert fields['NoNewPrivs'].strip() == '1'
    try:
        os.setuid(0)
    except OSError:
        pass
    else:
        raise AssertionError('numeric non-root process gained root')
    Path('workspace-write.txt').write_text('writable', encoding='utf-8')
    Path('/tmp/temp-write.txt').write_text('writable', encoding='utf-8')
    for forbidden in (
        Path('/root-write.txt'),
        Path('/root/tracefix-write.txt'),
        Path('/etc/tracefix-write.txt'),
        Path('/tracefix/input/tracefix-write.txt'),
    ):
        try:
            forbidden.write_text('forbidden', encoding='utf-8')
        except OSError:
            pass
        else:
            raise AssertionError(f'{forbidden.parent} was writable')
    assert [name for _, name in socket.if_nameindex()] == ['lo']
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/run/docker.sock').exists()
    for credential in (
        Path('/root/.ssh'), Path('/root/.docker'), Path('/root/.aws'),
        Path('/root/.config/gcloud'), Path('/root/.azure'), Path('/root/.netrc'),
    ):
        assert not credential.exists()
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


def test_host_environment_is_not_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACEFIX_SYNTHETIC_SECRET", "not-for-container")
    monkeypatch.setenv("HTTPS_PROXY", "http://not-for-container")
    _write_repository(
        tmp_path,
        """import os


def test_environment_allowlist():
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert os.environ.get('PYTHONUNBUFFERED') == '1'
    assert 'TRACEFIX_SYNTHETIC_SECRET' not in os.environ
    assert 'HTTPS_PROXY' not in os.environ
""",
    )
    result = SandboxRunner(SubprocessDockerCommandAdapter()).execute(
        tmp_path.resolve(), ("python", "-m", "pytest")
    )
    assert result.exit_code == 0


def test_project_root_mount_is_rejected_without_container_creation() -> None:
    before = _containers()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(SubprocessDockerCommandAdapter()).execute(
            Path(__file__).resolve().parents[2], ("python", "-m", "pytest")
        )
    assert _containers() == before


class CrashAfterCreateAdapter:
    def __init__(self) -> None:
        self._delegate = SubprocessDockerCommandAdapter()

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        if arguments[:1] == ("start",):
            raise RuntimeError("injected runner crash")
        return self._delegate.run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def test_injected_runner_crash_removes_exact_container(tmp_path: Path) -> None:
    _write_repository(tmp_path, "def test_never_started():\n    assert True\n")
    before = _containers()
    with pytest.raises(RuntimeError, match="injected runner crash"):
        SandboxRunner(CrashAfterCreateAdapter()).execute(
            tmp_path.resolve(), ("python", "-m", "pytest")
        )
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
