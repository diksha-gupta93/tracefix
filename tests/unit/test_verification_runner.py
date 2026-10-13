from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from app.sandbox.results import SandboxCompletion
from app.sandbox.runner import DockerCommandError, DockerCommandResult, DockerCommandTimeout
from app.sandbox.verification_runner import (
    HiddenTestBundle,
    VerificationRunnerError,
    VerificationRunnerErrorCode,
    VerificationRunnerValidationError,
    VerificationSandboxEvidence,
    VerificationSandboxRunner,
)


def _completed(stdout: bytes = b"", stderr: bytes = b"") -> DockerCommandResult:
    return DockerCommandResult(exit_code=0, stdout=stdout, stderr=stderr)


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        call = tuple(arguments)
        self.calls.append(call)
        if call[0] == "info":
            return _completed(
                b'{"OSType":"linux","SecurityOptions":["name=seccomp,profile=builtin"]}'
            )
        if call[0] == "wait":
            return _completed(b"0\n")
        if call[0] == "inspect":
            return _completed(b'{"OOMKilled":false,"Running":false}')
        if call[0] == "start":
            return _completed(b"secret output", b"hidden error")
        return _completed()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "candidate"
    repository.mkdir()
    (repository / "tests").mkdir()
    (repository / "tests" / "test_behavior.py").write_text("def test_x(): pass\n")
    (repository / "src").mkdir()
    return repository.resolve()


def test_visible_runner_uses_fixed_hardened_arguments_and_discards_output(tmp_path: Path) -> None:
    docker = FakeDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 1.5)).__next__,
        identifier_factory=lambda: "a" * 32,
    )
    repository = _repository(tmp_path)

    result = runner.execute_targeted(repository, ("tests/test_behavior.py",))

    assert result.completion is SandboxCompletion.NORMAL
    assert result.duration_seconds == 0.5
    assert not hasattr(result, "stdout")
    create = next(call for call in docker.calls if call[0] == "create")
    assert create[create.index("--network") : create.index("--network") + 2] == (
        "--network",
        "none",
    )
    assert "--read-only" in create
    assert create[create.index("--cap-drop") : create.index("--cap-drop") + 2] == (
        "--cap-drop",
        "ALL",
    )
    assert "no-new-privileges=true" in create
    assert create[create.index("--user") : create.index("--user") + 2] == ("--user", "10001:10001")
    assert "tracefix-sandbox:0.1.3a" in create
    assert create[-1] == "tests/test_behavior.py"
    assert create[create.index("--cpus") + 1] == "1"
    assert create[create.index("--memory") + 1] == "536870912"
    assert create[create.index("--pids-limit") + 1] == "128"
    assert create[create.index("--log-driver") + 1] == "local"
    assert sum("type=tmpfs" in value for value in create) == 2
    assert "--entrypoint" not in create
    assert all("secret output" not in value for value in create)
    assert docker.calls[-1][:2] == ("rm", "--force")


def test_hidden_runner_mounts_only_deterministic_read_only_bundle(tmp_path: Path) -> None:
    docker = FakeDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 1.1)).__next__,
        identifier_factory=lambda: "a" * 32,
    )
    repository = _repository(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    root = workspace / (".tracefix-verification-" + "b" * 32)
    root.mkdir()
    (root / "test_0000.py").write_text("def test_hidden(): pass\n")
    bundle = HiddenTestBundle(
        workspace_root=workspace,
        root=root,
        test_files=("test_0000.py",),
    )

    runner.execute_hidden(repository, bundle)

    create = next(call for call in docker.calls if call[0] == "create")
    mounts = tuple(create[index + 1] for index, value in enumerate(create) if value == "--mount")
    assert len(mounts) == 4
    assert f"type=bind,source={root},target=/tracefix/hidden,readonly" in mounts
    assert create[-1] == "/tracefix/hidden/test_0000.py"
    assert not any("evaluator" in argument or "reference.patch" in argument for argument in create)


@pytest.mark.parametrize("name", ["hidden.py", "test_1.py", "test_0000.py/escape"])
def test_hidden_runner_rejects_non_generated_names(tmp_path: Path, name: str) -> None:
    repository = _repository(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    if os.name != "nt":
        workspace.chmod(0o700)
    root = workspace / (".tracefix-verification-" + "b" * 32)
    root.mkdir()
    docker = FakeDocker()
    runner = VerificationSandboxRunner(docker)
    bundle = HiddenTestBundle(workspace_root=workspace, root=root, test_files=(name,))

    with pytest.raises(VerificationRunnerValidationError):
        runner.execute_hidden(repository, bundle)
    assert docker.calls == []


@pytest.mark.parametrize(
    "values",
    [
        {"completion": SandboxCompletion.NORMAL, "exit_code": None},
        {"completion": SandboxCompletion.TIMED_OUT, "exit_code": 1},
        {"completion": SandboxCompletion.NORMAL, "exit_code": True},
        {"completion": SandboxCompletion.NORMAL, "exit_code": -1},
        {
            "completion": SandboxCompletion.NORMAL,
            "exit_code": 0,
            "duration_seconds": float("inf"),
        },
    ],
)
def test_evidence_rejects_impossible_or_coerced_values(values: dict[str, object]) -> None:
    complete: dict[str, object] = {
        "completion": SandboxCompletion.NORMAL,
        "exit_code": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "duration_seconds": 0.0,
    }
    complete.update(values)
    with pytest.raises(ValidationError):
        VerificationSandboxEvidence.model_validate(complete)


def test_visible_suite_has_no_caller_controlled_pytest_arguments(tmp_path: Path) -> None:
    docker = FakeDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 1.1)).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    runner.execute_visible_suite(_repository(tmp_path))

    create = next(call for call in docker.calls if call[0] == "create")
    image_index = create.index("tracefix-sandbox:0.1.3a")
    assert image_index == len(create) - 1
    environments = tuple(
        create[index + 1] for index, value in enumerate(create) if value == "--env"
    )
    assert environments == (
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONPATH=/tracefix/workspace/repository/src",
    )
    assert not any("proxy" in value.casefold() for value in create)


@pytest.mark.parametrize(
    "path",
    [
        "-q",
        "tests/test_behavior.py::test_x",
        "tests/../test_behavior.py",
        "tests\\test_behavior.py",
        "/tests/test_behavior.py",
        "evaluator/test_hidden.py",
        "tests/test_behavior.txt",
        "tests/test_behavior.py;id",
    ],
)
def test_targeted_paths_cannot_inject_options_nodes_shells_or_other_roots(
    tmp_path: Path, path: str
) -> None:
    docker = FakeDocker()
    runner = VerificationSandboxRunner(docker)

    with pytest.raises(VerificationRunnerValidationError):
        runner.execute_targeted(_repository(tmp_path), (path,))

    assert docker.calls == []


def test_nested_repository_symlink_is_rejected_before_docker(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    external = (tmp_path / "external").resolve()
    external.mkdir()
    link = repository / "src" / "alias"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this filesystem")
    docker = FakeDocker()

    with pytest.raises(VerificationRunnerValidationError):
        VerificationSandboxRunner(docker).execute_visible_suite(repository)

    assert docker.calls == []


def test_repository_reparse_root_is_rejected_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    original_lstat = Path.lstat

    def reparse_lstat(path: Path) -> os.stat_result:
        if path == repository:
            return cast(
                os.stat_result,
                SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400),
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    docker = FakeDocker()

    with pytest.raises(VerificationRunnerValidationError):
        VerificationSandboxRunner(docker).execute_visible_suite(repository)

    assert docker.calls == []


class InsecureDocker(FakeDocker):
    def __init__(self, info: bytes) -> None:
        super().__init__()
        self.info = info

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        del timeout_seconds, stdout_bytes, stderr_bytes, check
        call = tuple(arguments)
        self.calls.append(call)
        if call[0] == "info":
            return _completed(self.info)
        return _completed()


@pytest.mark.parametrize(
    "info",
    [
        b'{"OSType":"windows","SecurityOptions":["name=seccomp"]}',
        b'{"OSType":"linux","SecurityOptions":[]}',
        b'{"OSType":"linux","SecurityOptions":["seccomp=unconfined"]}',
        b"not-json",
    ],
)
def test_insecure_or_malformed_daemon_blocks_before_container_creation(
    tmp_path: Path, info: bytes
) -> None:
    docker = InsecureDocker(info)
    runner = VerificationSandboxRunner(docker)

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DAEMON_SECURITY
    assert not any(call[0] == "create" for call in docker.calls)


class CreateFailingDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        del timeout_seconds, stdout_bytes, stderr_bytes, check
        call = tuple(arguments)
        self.calls.append(call)
        if call[0] == "info":
            return _completed(
                b'{"OSType":"linux","SecurityOptions":["name=seccomp,profile=builtin"]}'
            )
        if call[0] == "create":
            raise DockerCommandError("container name already exists")
        return _completed()


def test_create_collision_does_not_remove_preexisting_container(tmp_path: Path) -> None:
    docker = CreateFailingDocker()
    runner = VerificationSandboxRunner(docker, identifier_factory=lambda: "a" * 32)

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DOCKER_FAILED
    assert any(call[0] == "create" for call in docker.calls)
    assert not any(call[:2] == ("rm", "--force") for call in docker.calls)


class RuntimeFailingDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        if tuple(arguments)[0] == "start":
            self.calls.append(tuple(arguments))
            raise RuntimeError("secret adapter crash")
        return super().run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def test_unexpected_docker_adapter_failure_is_typed_and_cleans_created_container(
    tmp_path: Path,
) -> None:
    docker = RuntimeFailingDocker()
    runner = VerificationSandboxRunner(docker, identifier_factory=lambda: "a" * 32)

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DOCKER_FAILED
    assert "secret" not in str(caught.value)
    assert docker.calls[-1][:2] == ("rm", "--force")


@pytest.mark.parametrize("failure_command", ["start", "wait", "inspect"])
def test_docker_stage_failures_are_typed_and_cleanup_runs(
    tmp_path: Path, failure_command: str
) -> None:
    class StageFailingDocker(FakeDocker):
        def run(
            self,
            arguments: Sequence[str],
            *,
            timeout_seconds: float | None = None,
            stdout_bytes: int = 65_536,
            stderr_bytes: int = 65_536,
            check: bool = True,
        ) -> DockerCommandResult:
            if tuple(arguments)[0] == failure_command:
                self.calls.append(tuple(arguments))
                raise DockerCommandError("secret stage failure")
            return super().run(
                arguments,
                timeout_seconds=timeout_seconds,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                check=check,
            )

    docker = StageFailingDocker()
    runner = VerificationSandboxRunner(docker, identifier_factory=lambda: "a" * 32)

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DOCKER_FAILED
    assert "secret" not in str(caught.value)
    assert docker.calls[-1][:2] == ("rm", "--force")


class MalformedInfoDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        del timeout_seconds, stdout_bytes, stderr_bytes, check
        self.calls.append(tuple(arguments))
        return cast(DockerCommandResult, None)


def test_malformed_docker_info_return_is_typed_daemon_failure(tmp_path: Path) -> None:
    docker = MalformedInfoDocker()
    runner = VerificationSandboxRunner(docker)

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DAEMON_SECURITY
    assert docker.calls == [("info", "--format", "{{json .}}")]


class TimeoutDocker(FakeDocker):
    def __init__(self) -> None:
        super().__init__()
        self.killed = False

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        call = tuple(arguments)
        self.calls.append(call)
        if call[0] == "info":
            return _completed(
                b'{"OSType":"linux","SecurityOptions":["name=seccomp,profile=builtin"]}'
            )
        if call[0] == "start":
            raise DockerCommandTimeout(
                "deadline",
                result=DockerCommandResult(
                    exit_code=-9,
                    stdout=b"secret",
                    stderr=b"hidden",
                    stdout_truncated=True,
                    stderr_truncated=False,
                ),
            )
        if call[0] == "inspect":
            return _completed(
                b'{"OOMKilled":false,"Running":true}'
                if not self.killed
                else b'{"OOMKilled":false,"Running":false}'
            )
        if call[0] == "kill":
            self.killed = True
        return _completed()


def test_timeout_kills_and_returns_only_safe_bounded_metadata(tmp_path: Path) -> None:
    docker = TimeoutDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 121.0)).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    result = runner.execute_visible_suite(_repository(tmp_path))

    assert result.completion is SandboxCompletion.TIMED_OUT
    assert result.exit_code is None
    assert result.stdout_truncated
    assert not hasattr(result, "stdout")
    assert any(call[:3] == ("kill", "--signal", "KILL") for call in docker.calls)
    assert docker.calls[-1][:2] == ("rm", "--force")


class OomDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        if tuple(arguments)[0] == "inspect":
            self.calls.append(tuple(arguments))
            return _completed(b'{"OOMKilled":true,"Running":false}')
        return super().run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def test_oom_is_distinct_completion_and_container_is_removed(tmp_path: Path) -> None:
    docker = OomDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 1.2)).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    result = runner.execute_visible_suite(_repository(tmp_path))

    assert result.completion is SandboxCompletion.MEMORY_LIMIT
    assert result.exit_code == 0
    assert docker.calls[-1][:2] == ("rm", "--force")


def test_nonfinite_clock_duration_is_malformed_and_container_is_removed(tmp_path: Path) -> None:
    docker = FakeDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, float("nan"))).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.MALFORMED_RESULT
    assert docker.calls[-1][:2] == ("rm", "--force")


class KillFailingDocker(TimeoutDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        if tuple(arguments)[0] == "kill":
            self.calls.append(tuple(arguments))
            raise DockerCommandError("secret kill failure")
        return super().run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def test_timeout_kill_failure_is_typed_and_forced_removal_still_runs(tmp_path: Path) -> None:
    docker = KillFailingDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 121.0)).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.DOCKER_FAILED
    assert "secret" not in str(caught.value)
    assert docker.calls[-1][:2] == ("rm", "--force")


class CleanupFailingDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        if tuple(arguments)[:2] == ("rm", "--force"):
            self.calls.append(tuple(arguments))
            raise OSError("secret cleanup failure")
        return super().run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def test_cleanup_failure_is_visible_without_exposing_docker_detail(tmp_path: Path) -> None:
    docker = CleanupFailingDocker()
    runner = VerificationSandboxRunner(
        docker,
        clock=iter((1.0, 1.1)).__next__,
        identifier_factory=lambda: "a" * 32,
    )

    with pytest.raises(VerificationRunnerError) as caught:
        runner.execute_visible_suite(_repository(tmp_path))

    assert caught.value.code is VerificationRunnerErrorCode.CLEANUP_FAILED
    assert "secret" not in str(caught.value)
