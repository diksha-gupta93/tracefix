from __future__ import annotations

import os
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
    def __init__(
        self,
        responses: Sequence[DockerCommandResult | Exception] = (),
        *,
        docker_info: DockerCommandResult | Exception | None = None,
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, ...]] = []
        self.options: list[tuple[float | None, int, int]] = []
        self.docker_info = docker_info or completed(
            stdout=b'{"OSType":"linux","SecurityOptions":["name=seccomp,profile=builtin"]}'
        )

    @property
    def container_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[:1] != ("info",)]

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
        if arguments[:1] == ("info",):
            if isinstance(self.docker_info, Exception):
                raise self.docker_info
            return self.docker_info
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


def test_nested_repository_symlink_is_rejected_before_docker(tmp_path: Path) -> None:
    target = tmp_path.parent / "outside"
    target.mkdir(exist_ok=True)
    link = tmp_path / "nested-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not permit test-created symlinks")

    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.calls == []


@pytest.mark.parametrize(
    "protected",
    [
        Path(__file__).resolve().parents[2],
        Path.home(),
    ],
)
def test_protected_locations_are_rejected_without_docker(protected: Path) -> None:
    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(protected.absolute(), ("python", "-m", "pytest"))
    assert adapter.calls == []


def test_ancestor_of_home_is_rejected_without_docker() -> None:
    ancestor = Path(Path.home().anchor)
    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(ancestor, ("python", "-m", "pytest"))
    assert adapter.calls == []


def test_protected_policy_contains_required_credential_and_socket_locations() -> None:
    import app.sandbox.runner as runner_module

    protected = {os.path.normcase(os.fspath(path)) for path in runner_module._protected_locations()}
    home = Path.home().resolve(strict=False)
    required = {
        home / ".ssh",
        home / ".docker",
        home / ".aws",
        home / ".config" / "gcloud",
        home / ".azure",
        home / ".netrc",
        home / "_netrc",
        home / ".npmrc",
        home / ".pypirc",
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
    }
    assert {os.path.normcase(os.fspath(path)) for path in required} <= protected


def test_windows_docker_named_pipe_form_is_protected() -> None:
    import app.sandbox.runner as runner_module

    assert runner_module._is_windows_docker_pipe(Path(r"\\.\pipe\docker_engine"))


def test_incomplete_repository_inspection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_scandir = os.scandir

    def denied(path: os.PathLike[str] | str) -> os.ScandirIterator[str]:
        if Path(path) == tmp_path.resolve():
            raise PermissionError("synthetic denial")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", denied)
    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.calls == []


def test_repository_identity_change_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.sandbox.runner as runner_module

    original_snapshot = runner_module._tree_snapshot
    snapshots = iter((original_snapshot(tmp_path), ()))
    monkeypatch.setattr(runner_module, "_tree_snapshot", lambda repository: next(snapshots))
    (tmp_path / "entry").write_text("content", encoding="utf-8")
    snapshots = iter((original_snapshot(tmp_path), ()))
    adapter = FakeDockerAdapter()
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.calls == []


@pytest.mark.parametrize(
    "docker_info",
    [
        completed(stdout=b'{"OSType":"windows","SecurityOptions":["name=seccomp"]}'),
        completed(stdout=b'{"OSType":"linux","SecurityOptions":[]}'),
        completed(stdout=b"not-json"),
        completed(stdout=b'{"OSType":"linux","SecurityOptions":["seccomp=unconfined"]}'),
        DockerCommandError("daemon unavailable"),
    ],
)
def test_invalid_daemon_security_posture_prevents_container_creation(
    tmp_path: Path, docker_info: DockerCommandResult | Exception
) -> None:
    adapter = FakeDockerAdapter(docker_info=docker_info)
    with pytest.raises(SandboxValidationError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.calls[0][0] == "info"
    assert not any(call[0] == "create" for call in adapter.calls)


def test_create_environment_and_mounts_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACEFIX_SYNTHETIC_SECRET", "must-not-cross-boundary")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-cross-boundary")
    adapter = FakeDockerAdapter(
        [completed(), completed(), completed(stdout=b"0\n"), state(), completed()]
    )
    SandboxRunner(adapter, clock=FakeClock([0.0, 0.1])).execute(
        tmp_path.resolve(), ("python", "-m", "pytest")
    )
    create = adapter.container_calls[0]
    environments = tuple(
        create[index + 1] for index, value in enumerate(create) if value == "--env"
    )
    mounts = tuple(create[index + 1] for index, value in enumerate(create) if value == "--mount")
    assert environments == ("PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1")
    assert len(mounts) == 3
    assert sum(mount.startswith("type=bind,") for mount in mounts) == 1
    assert "must-not-cross-boundary" not in "\0".join(create)


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
    calls = adapter.container_calls
    create = calls[0]
    name = create[create.index("--name") + 1]
    assert name.startswith("tracefix-sandbox-")
    assert "tracefix-sandbox:0.1.3a" in create
    assert create[create.index("--network") : create.index("--network") + 2] == (
        "--network",
        "none",
    )
    assert "--read-only" in create
    assert create[create.index("--cap-drop") : create.index("--cap-drop") + 2] == (
        "--cap-drop",
        "ALL",
    )
    assert create[create.index("--security-opt") : create.index("--security-opt") + 2] == (
        "--security-opt",
        "no-new-privileges=true",
    )
    assert not any("seccomp=unconfined" in argument for argument in create)
    assert create[create.index("--user") : create.index("--user") + 2] == (
        "--user",
        "10001:10001",
    )
    mount = create[create.index("--mount") + 1]
    assert mount == f"type=bind,source={tmp_path.resolve()},target=/tracefix/input,readonly"
    assert "type=tmpfs,destination=/tracefix/workspace,tmpfs-mode=1777" in create
    assert "type=tmpfs,destination=/tmp,tmpfs-mode=1777" in create
    assert calls[1] == ("start", "--attach", name)
    assert calls[2] == ("wait", name)
    assert calls[3][:2] == ("inspect", "--format")
    assert calls[4] == ("rm", "--force", name)


def test_container_names_are_unique(tmp_path: Path) -> None:
    responses = [completed(), completed(), completed(stdout=b"0\n"), state(), completed()] * 2
    adapter = FakeDockerAdapter(responses)
    runner = SandboxRunner(adapter, clock=FakeClock([0.0, 0.0, 0.0, 0.0]))
    command = ("python", "-m", "pytest")
    runner.execute(tmp_path.resolve(), command)
    runner.execute(tmp_path.resolve(), command)

    calls = adapter.container_calls
    names = [call[call.index("--name") + 1] for call in (calls[0], calls[5])]
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
    assert adapter.container_calls[-1][:2] == ("rm", "--force")


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
    assert adapter.container_calls[-1][:2] == ("rm", "--force")


def test_unexpected_runner_exception_after_creation_still_cleans_up(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter([completed(), RuntimeError("injected runner crash"), completed()])
    with pytest.raises(RuntimeError, match="injected runner crash"):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    create = adapter.container_calls[0]
    name = create[create.index("--name") + 1]
    assert adapter.container_calls[-1] == ("rm", "--force", name)


def test_create_failure_still_attempts_exact_cleanup(tmp_path: Path) -> None:
    adapter = FakeDockerAdapter([DockerCommandError("create failed"), completed()])
    with pytest.raises(SandboxExecutionError):
        SandboxRunner(adapter).execute(tmp_path.resolve(), ("python", "-m", "pytest"))
    assert adapter.container_calls[-1][:2] == ("rm", "--force")


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
    create = adapter.container_calls[0]
    assert create[create.index("--cpus") + 1] == "0.5"
    assert create[create.index("--memory") + 1] == "64000000"
    assert create[create.index("--pids-limit") + 1] == "12"
    assert "max-size=300b" in create
    assert adapter.options[2] == (2.5, 100, 200)


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
    calls = adapter.container_calls
    name = calls[0][calls[0].index("--name") + 1]
    assert result.completion is SandboxCompletion.TIMED_OUT
    assert result.exit_code is None
    assert ("kill", "--signal", "KILL", name) in calls
    assert calls[-1] == ("rm", "--force", name)


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
    assert not any(call[0] == "kill" for call in adapter.container_calls)


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
