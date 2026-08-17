from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.sandbox.results import TRUNCATION_MARKER, SandboxCompletion, SandboxLimits
from app.sandbox.runner import (
    DockerCommandError,
    DockerCommandResult,
    DockerCommandTimeout,
    PytestCommand,
    SandboxCleanupError,
    SandboxExecutionError,
    SandboxRunner,
    SandboxValidationError,
)


class FakeDockerAdapter:
    def __init__(self, responses: Sequence[DockerCommandResult | Exception] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, ...]] = []
        self.options: list[tuple[float | None, int, int]] = []

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        self.calls.append(tuple(arguments))
        self.options.append((timeout_seconds, stdout_bytes, stderr_bytes))
        if not self.responses:
            raise AssertionError("unexpected Docker call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def completed(exit_code: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> DockerCommandResult:
    return DockerCommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


def state(*, oom: bool = False, running: bool = False) -> DockerCommandResult:
    return completed(
        stdout=f'{{"OOMKilled":{str(oom).lower()},"Running":{str(running).lower()}}}\n'.encode()
    )


@pytest.mark.parametrize(
    "tokens",
    [
        ("python", "-m", "pytest"),
        ("python", "-m", "pytest", "tests/test_example.py"),
        (
            "python",
            "-m",
            "pytest",
            "tests/unit/test_example.py::TestCase::test_case",
            "tests/test_other.py",
        ),
    ],
)
def test_accepts_only_approved_pytest_commands(tokens: tuple[str, ...]) -> None:
    assert PytestCommand(tokens).tokens == tokens


@pytest.mark.parametrize(
    "tokens",
    [
        (),
        ("pytest",),
        ("python", "-m", "unittest"),
        ("sh", "-c", "pytest"),
        ("python", "-m", "pytest", "-q"),
        ("python", "-m", "pytest", "PYTHONPATH=x"),
        ("python", "-m", "pytest", "@args"),
        ("python", "-m", "pytest", "src/test_example.py"),
        ("python", "-m", "pytest", "tests"),
        ("python", "-m", "pytest", "tests/test_example.txt"),
        ("python", "-m", "pytest", "/tests/test_example.py"),
        ("python", "-m", "pytest", "C:/tests/test_example.py"),
        ("python", "-m", "pytest", "C:\\tests\\test_example.py"),
        ("python", "-m", "pytest", "//server/tests/test_example.py"),
        ("python", "-m", "pytest", "tests/../test_example.py"),
        ("python", "-m", "pytest", "tests/./test_example.py"),
        ("python", "-m", "pytest", "tests/test*.py"),
        ("python", "-m", "pytest", "https://host/tests/test.py"),
        ("python", "-m", "pytest", "tests/test example.py"),
        ("python", "-m", "pytest", "tests/test.py;id"),
        ("python", "-m", "pytest", "tests/test.py\nid"),
        ("python", "-m", "pytest", ""),
    ],
)
def test_rejects_prohibited_command_classes(tokens: tuple[str, ...]) -> None:
    with pytest.raises(SandboxValidationError):
        PytestCommand(tokens)


def test_invalid_command_causes_no_docker_interaction(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter()
    runner = SandboxRunner(adapter)

    with pytest.raises(SandboxValidationError):
        runner.execute(tmp_path, ("python", "-m", "pytest", "-q"))

    assert adapter.calls == []


def test_repository_must_be_absolute_existing_directory(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter()
    runner = SandboxRunner(adapter)

    for path in (Path("relative"), tmp_path / "missing"):
        with pytest.raises(SandboxValidationError):
            runner.execute(path, ("python", "-m", "pytest"))
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxValidationError):
        runner.execute(file_path, ("python", "-m", "pytest"))

    assert adapter.calls == []


def test_repository_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not permit test-created symlinks")

    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(link.absolute(), ("python", "-m", "pytest"))
    assert adapter.calls == []


def test_constructs_fixed_container_and_maps_result(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            completed(stdout=b"out\xff", stderr=b"err"),
            completed(stdout=b"1\n"),
            state(),
            completed(),
        ]
    )
    runner = SandboxRunner(adapter, clock=FakeClock([10.0, 10.5]))

    result = runner.execute(tmp_path.resolve(), ("python", "-m", "pytest", "tests/test_example.py"))

    assert result.exit_code == 1
    assert result.stdout == "out�"
    assert result.stderr == "err"
    assert result.duration_seconds == 0.5
    create = adapter.calls[0]
    name = create[create.index("--name") + 1]
    assert name.startswith("tracefix-sandbox-")
    assert "tracefix-sandbox:0.1.3a" in create
    assert create[create.index("--network") : create.index("--network") + 2] == (
        "--network",
        "none",
    )
    assert "--read-only" in create
    assert create[create.index("--user") : create.index("--user") + 2] == (
        "--user",
        "10001:10001",
    )
    mount = create[create.index("--mount") + 1]
    assert mount == f"type=bind,source={tmp_path.resolve()},target=/tracefix/input,readonly"
    assert "type=tmpfs,destination=/tracefix/workspace,tmpfs-mode=1777" in create
    assert "type=tmpfs,destination=/tmp,tmpfs-mode=1777" in create
    assert adapter.calls[1] == ("start", "--attach", name)
    assert adapter.calls[2] == ("wait", name)
    assert adapter.calls[3][:2] == ("inspect", "--format")
    assert adapter.calls[4] == ("rm", "--force", name)


def test_container_names_are_unique(tmp_path: Path) -> None:
    responses = [completed(), completed(), completed(stdout=b"0\n"), state(), completed()] * 2
    adapter = FakeDockerAdapter(responses)
    runner = SandboxRunner(adapter, clock=FakeClock([0.0, 0.0, 0.0, 0.0]))
    command = ("python", "-m", "pytest")
    runner.execute(tmp_path.resolve(), command)
    runner.execute(tmp_path.resolve(), command)

    names = [call[call.index("--name") + 1] for call in (adapter.calls[0], adapter.calls[5])]
    assert names[0] != names[1]


@pytest.mark.parametrize("wait_code", [0, 1, 5])
def test_cleanup_occurs_for_every_completed_pytest_exit(tmp_path: Path, wait_code: int) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            completed(),
            completed(stdout=f"{wait_code}\n".encode()),
            state(),
            completed(),
            completed(),
        ]
    )
    result = SandboxRunner(adapter, clock=FakeClock([0.0, 0.1])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest")
    )

    assert result.exit_code == wait_code
    assert adapter.calls[-1][:2] == ("rm", "--force")


@pytest.mark.parametrize(
    "responses",
    [
        [completed(), DockerCommandError("start failed"), completed()],
        [completed(), completed(), completed(stdout=b"not-an-integer"), completed()],
        [
            completed(),
            completed(),
            completed(stdout=b"0\n"),
            DockerCommandError("inspect failed"),
            completed(),
        ],
    ],
)
def test_cleanup_after_post_creation_runner_failure(
    tmp_path: Path, responses: list[DockerCommandResult | Exception]
) -> None:
    adapter = FakeDockerAdapter(responses)
    with pytest.raises(SandboxExecutionError):
        SandboxRunner(adapter, clock=FakeClock([0.0, 0.1])).execute(
            tmp_path.resolve(), ("python", "-m", "pytest")
        )
    assert adapter.calls[-1][:2] == ("rm", "--force")


def test_create_failure_still_attempts_exact_cleanup(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter([DockerCommandError("create failed"), completed()])
    with pytest.raises(SandboxExecutionError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.calls[-1][:2] == ("rm", "--force")


def test_cleanup_failure_is_not_hidden(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            completed(),
            completed(stdout=b"0\n"),
            state(),
            DockerCommandError("remove failed"),
        ]
    )
    with pytest.raises(SandboxCleanupError, match="tracefix-sandbox-"):
        SandboxRunner(adapter, clock=FakeClock([0.0, 0.1])).execute(
            tmp_path.resolve(), ("python", "-m", "pytest")
        )


def test_execution_and_cleanup_failures_preserve_both_facts(tmp_path: Path) -> None:
    execution_error = DockerCommandError("start failed")
    cleanup_error = DockerCommandError("remove failed")
    adapter = FakeDockerAdapter([completed(), execution_error, cleanup_error])

    with pytest.raises(SandboxCleanupError, match="execution also failed") as captured:
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))

    assert captured.value.__cause__ is execution_error
    assert captured.value.cleanup_cause is cleanup_error


def test_subprocess_adapter_contract_does_not_expose_completed_process() -> None:
    result = DockerCommandResult(exit_code=0, stdout=b"", stderr=b"")
    assert not isinstance(result, subprocess.CompletedProcess)


def test_custom_limits_map_to_create_and_wait_deadline(tmp_path: Path) -> None:
    limits = SandboxLimits(
        cpu_count=0.5,
        memory_bytes=64_000_000,
        pids_limit=12,
        timeout_seconds=2.5,
        stdout_bytes=100,
        stderr_bytes=200,
    )
    adapter = FakeDockerAdapter(
        [completed(), completed(), completed(stdout=b"0\n"), state(), completed()]
    )
    SandboxRunner(adapter, clock=FakeClock([0.0, 0.1])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), limits
    )
    create = adapter.calls[0]
    assert create[create.index("--cpus") + 1] == "0.5"
    assert create[create.index("--memory") + 1] == "64000000"
    assert create[create.index("--pids-limit") + 1] == "12"
    assert "max-size=300b" in create
    assert adapter.options[1] == (2.5, 100, 200)


def test_timeout_kills_exact_container_and_returns_typed_result(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            DockerCommandTimeout("deadline", result=completed(stdout=b"partial")),
            state(running=True),
            completed(),
            state(),
            completed(),
        ]
    )
    result = SandboxRunner(adapter, clock=FakeClock([1.0, 1.5])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), SandboxLimits(timeout_seconds=0.1)
    )
    name = adapter.calls[0][adapter.calls[0].index("--name") + 1]
    assert result.completion is SandboxCompletion.TIMED_OUT
    assert result.exit_code is None
    assert ("kill", "--signal", "KILL", name) in adapter.calls
    assert adapter.calls[-1] == ("rm", "--force", name)


def test_timeout_race_maps_natural_completion_without_kill(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            DockerCommandTimeout("deadline", result=completed()),
            state(running=False),
            completed(stdout=b"1\n"),
            state(),
            completed(),
        ]
    )
    result = SandboxRunner(adapter, clock=FakeClock([1.0, 1.1])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest"), SandboxLimits(timeout_seconds=0.1)
    )
    assert result.completion is SandboxCompletion.NORMAL
    assert result.exit_code == 1
    assert not any(call[0] == "kill" for call in adapter.calls)


def test_oom_requires_inspection_evidence(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter(
        [
            completed(),
            completed(),
            completed(stdout=b"137\n"),
            state(oom=True),
            completed(),
            completed(),
        ]
    )
    result = SandboxRunner(adapter, clock=FakeClock([0.0, 0.2])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest")
    )
    assert result.completion is SandboxCompletion.MEMORY_LIMIT
    assert result.exit_code == 137


@pytest.mark.parametrize(
    ("payload", "limit", "expected_truncated"),
    [
        (b"abc", 32, False),
        (b"a" * 32, 32, False),
        (b"a" * 33, 32, True),
        (("a€" + "z" * 40).encode(), len(TRUNCATION_MARKER) + 2, True),
        (b"a\xffb" * 20, 32, True),
    ],
)
def test_output_bounding_is_byte_safe(payload: bytes, limit: int, expected_truncated: bool) -> None:
    from app.sandbox.runner import _bounded_text

    text, truncated = _bounded_text(payload, limit)
    assert truncated is expected_truncated
    assert len(text.encode("utf-8")) <= limit
    assert text.endswith(TRUNCATION_MARKER.decode()) is expected_truncated
