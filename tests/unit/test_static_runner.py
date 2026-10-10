from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from app.sandbox.runner import DockerCommandError, DockerCommandResult, DockerCommandTimeout
from app.sandbox.static_runner import (
    STATIC_CHECK_IMAGE,
    STATIC_CONTAINER_NAME_PREFIX,
    DockerStaticRunner,
    StaticContainerResult,
    StaticRunnerError,
    StaticRunnerErrorCode,
    StaticToolCheck,
    StaticToolFinding,
    StaticToolResult,
    StaticToolStatus,
)


class FakeDocker:
    def __init__(self, document: bytes | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []
        self.options: list[tuple[tuple[str, ...], float | None, int, int, bool]] = []
        self.document = document or _passing_document()

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        args = tuple(arguments)
        self.calls.append((args, check))
        self.options.append((args, timeout_seconds, stdout_bytes, stderr_bytes, check))
        if args[:2] == ("info", "--format"):
            return DockerCommandResult(
                0, b'{"OSType":"linux","SecurityOptions":["name=seccomp"]}', b""
            )
        if args[0] == "start":
            return DockerCommandResult(0, self.document, b"")
        if args[0] == "wait":
            return DockerCommandResult(0, b"0\n", b"")
        if args[0] == "inspect":
            return DockerCommandResult(0, b'{"OOMKilled":false,"Running":false}', b"")
        return DockerCommandResult(0, b"", b"")


class TimeoutDocker(FakeDocker):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        args = tuple(arguments)
        if args[0] == "start":
            self.calls.append((args, check))
            raise DockerCommandTimeout(
                "synthetic timeout", result=DockerCommandResult(-9, b"", b"")
            )
        if args[0] == "inspect":
            self.calls.append((args, check))
            return DockerCommandResult(0, b'{"OOMKilled":false,"Running":true}', b"")
        return super().run(
            args,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


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
        args = tuple(arguments)
        if args[0] == "inspect":
            self.calls.append((args, check))
            return DockerCommandResult(0, b'{"OOMKilled":true,"Running":false}', b"")
        return super().run(
            args,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


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
        args = tuple(arguments)
        if args[:2] == ("rm", "--force"):
            self.calls.append((args, check))
            raise DockerCommandError("synthetic cleanup failure")
        return super().run(
            args,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


class LifecycleEvidenceDocker(FakeDocker):
    def __init__(
        self,
        operation: str,
        *,
        exit_code: int = 0,
        stdout_truncated: bool = False,
        stderr: bytes = b"",
    ) -> None:
        super().__init__()
        self.operation = operation
        self.exit_code = exit_code
        self.stdout_truncated = stdout_truncated
        self.injected_stderr = stderr

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        result = super().run(
            arguments,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )
        if tuple(arguments)[0] != self.operation:
            return result
        return DockerCommandResult(
            exit_code=self.exit_code,
            stdout=result.stdout,
            stderr=self.injected_stderr,
            stdout_truncated=self.stdout_truncated,
        )


class InterruptingCleanupFailingDocker(CleanupFailingDocker):
    def __init__(self, interruption: KeyboardInterrupt | SystemExit) -> None:
        super().__init__()
        self.interruption = interruption

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        stdout_bytes: int = 65_536,
        stderr_bytes: int = 65_536,
        check: bool = True,
    ) -> DockerCommandResult:
        args = tuple(arguments)
        if args[0] == "start":
            self.calls.append((args, check))
            raise self.interruption
        return super().run(
            args,
            timeout_seconds=timeout_seconds,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            check=check,
        )


def _passing_document() -> bytes:
    return json.dumps(
        {
            "version": 1,
            "results": [
                {"check": check.value, "status": "passed", "findings": [], "error_code": None}
                for check in StaticToolCheck
            ],
        },
        separators=(",", ":"),
    ).encode()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "candidate"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "example.py").write_text("value = 1\n", encoding="utf-8")
    return repository.resolve()


def _image_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "docker" / "static-checks" / "runner.py"
    specification = importlib.util.spec_from_file_location("tracefix_static_image_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_static_protocol_models_reject_inconsistent_unsafe_or_unsorted_evidence() -> None:
    finding = StaticToolFinding(path="src/example.py", line=1, column=1, rule_code="F401")
    with pytest.raises(ValidationError):
        StaticToolResult(
            check=StaticToolCheck.RUFF,
            status=StaticToolStatus.PASSED,
            findings=(finding,),
        )
    with pytest.raises(ValidationError):
        StaticToolFinding(path="/etc/passwd", line=1, column=1, rule_code="F401")
    with pytest.raises(ValidationError):
        StaticContainerResult(
            version=1,
            results=tuple(
                StaticToolResult(check=check, status=StaticToolStatus.PASSED, findings=())
                for check in reversed(tuple(StaticToolCheck))
            ),
        )


def test_fixed_docker_arguments_and_protocol_mapping(tmp_path: Path) -> None:
    docker = FakeDocker()
    runner = DockerStaticRunner(
        docker,
        clock=lambda: 1.0,
        identifier_factory=lambda: "0123456789abcdef0123456789abcdef",
    )

    result = runner.execute(_repository(tmp_path), ("src/example.py",))

    assert all(item.status is StaticToolStatus.PASSED for item in result.results)
    create = next(call[0] for call in docker.calls if call[0][0] == "create")
    assert create[1:3] == (
        "--name",
        f"{STATIC_CONTAINER_NAME_PREFIX}0123456789abcdef0123456789abcdef",
    )
    assert "--network" in create and create[create.index("--network") + 1] == "none"
    assert "--read-only" in create
    assert create[create.index("--user") + 1] == "10001:10001"
    assert create[create.index("--cap-drop") + 1] == "ALL"
    assert create[create.index("--security-opt") + 1] == "no-new-privileges=true"
    assert create[-2:] == (STATIC_CHECK_IMAGE, "src/example.py")
    mount_values = tuple(
        create[index + 1] for index, value in enumerate(create) if value == "--mount"
    )
    assert any("readonly" in value and "target=/tracefix/input" in value for value in mount_values)
    assert sum("type=tmpfs" in value for value in mount_values) == 2
    environment = tuple(create[index + 1] for index, value in enumerate(create) if value == "--env")
    assert environment == (
        "HOME=/nonexistent",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "PYTHONNOUSERSITE=1",
        "PYTHONUNBUFFERED=1",
    )
    assert not any("PROXY" in value or "DOCKER" in value or "TOKEN" in value for value in create)
    assert docker.calls[-1][0][:2] == ("rm", "--force")
    assert all(timeout is not None and timeout > 0 for _, timeout, _, _, _ in docker.options)
    assert all(stdout > 0 and stderr > 0 for _, _, stdout, stderr, _ in docker.options)


@pytest.mark.parametrize(
    ("paths", "code"),
    [
        (("../escape.py",), StaticRunnerErrorCode.INVALID_INPUT),
        (("src\\example.py",), StaticRunnerErrorCode.INVALID_INPUT),
        (("src/example.txt",), StaticRunnerErrorCode.INVALID_INPUT),
        (("--config-file=attacker.py",), StaticRunnerErrorCode.INVALID_INPUT),
        (("@attacker.py",), StaticRunnerErrorCode.INVALID_INPUT),
        (("SRC/example.py", "src/EXAMPLE.py"), StaticRunnerErrorCode.INVALID_INPUT),
    ],
)
def test_changed_paths_are_data_only_and_strictly_validated(
    tmp_path: Path, paths: tuple[str, ...], code: StaticRunnerErrorCode
) -> None:
    docker = FakeDocker()
    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(docker).execute(_repository(tmp_path), paths)
    assert raised.value.code is code
    assert docker.calls == []


@pytest.mark.parametrize(
    "document",
    [b"not-json", b"{}", b"{}" * 40_000],
    ids=("invalid-json", "missing-fields", "oversized"),
)
def test_malformed_or_oversized_protocol_is_infrastructure_failure(
    tmp_path: Path, document: bytes
) -> None:
    docker = FakeDocker(document)
    runner = DockerStaticRunner(docker)

    with pytest.raises(StaticRunnerError) as raised:
        runner.execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value.code is StaticRunnerErrorCode.MALFORMED_RESULT
    assert docker.calls[-1][0][:2] == ("rm", "--force")


@pytest.mark.parametrize(
    ("docker", "expected"),
    [
        (TimeoutDocker(), StaticRunnerErrorCode.TIMED_OUT),
        (OomDocker(), StaticRunnerErrorCode.MEMORY_LIMIT),
        (CleanupFailingDocker(), StaticRunnerErrorCode.CLEANUP_FAILED),
    ],
)
def test_timeout_oom_and_cleanup_fail_closed_and_attempt_removal(
    tmp_path: Path, docker: FakeDocker, expected: StaticRunnerErrorCode
) -> None:
    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(docker).execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value.code is expected
    assert any(call[0][:2] == ("rm", "--force") for call in docker.calls)
    if expected is StaticRunnerErrorCode.TIMED_OUT:
        assert any(call[0][:3] == ("kill", "--signal", "KILL") for call in docker.calls)


@pytest.mark.parametrize(
    ("docker", "expected"),
    [
        (
            LifecycleEvidenceDocker("info", stdout_truncated=True),
            StaticRunnerErrorCode.DAEMON_SECURITY,
        ),
        (
            LifecycleEvidenceDocker("wait", stdout_truncated=True),
            StaticRunnerErrorCode.DOCKER_FAILED,
        ),
        (
            LifecycleEvidenceDocker("inspect", stderr=b"unexpected"),
            StaticRunnerErrorCode.DOCKER_FAILED,
        ),
        (
            LifecycleEvidenceDocker("start", exit_code=9),
            StaticRunnerErrorCode.UNEXPECTED_EXIT,
        ),
    ],
    ids=("truncated-info", "truncated-wait", "inspect-stderr", "attach-nonzero"),
)
def test_lifecycle_evidence_is_validated_before_a_pass(
    tmp_path: Path, docker: FakeDocker, expected: StaticRunnerErrorCode
) -> None:
    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(docker).execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value.code is expected
    if expected is not StaticRunnerErrorCode.DAEMON_SECURITY:
        assert any(call[0][:2] == ("rm", "--force") for call in docker.calls)


def test_single_host_deadline_covers_the_full_docker_lifecycle(tmp_path: Path) -> None:
    docker = FakeDocker()
    clock = iter((0.0, 0.0, 0.0, 0.0, 121.0)).__next__

    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(docker, clock=clock).execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value.code is StaticRunnerErrorCode.TIMED_OUT
    assert any(call[0][:2] == ("rm", "--force") for call in docker.calls)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_process_control_is_preserved_when_container_cleanup_fails(
    tmp_path: Path, interruption: KeyboardInterrupt | SystemExit
) -> None:
    docker = InterruptingCleanupFailingDocker(interruption)

    with pytest.raises(type(interruption)) as raised:
        DockerStaticRunner(docker).execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value is interruption
    assert any(call[0][:2] == ("rm", "--force") for call in docker.calls)


def test_cleanup_failure_remains_visible_when_primary_protocol_also_fails(tmp_path: Path) -> None:
    docker = CleanupFailingDocker(b"not-json")

    with pytest.raises(StaticRunnerError) as raised:
        DockerStaticRunner(docker).execute(_repository(tmp_path), ("src/example.py",))

    assert raised.value.code is StaticRunnerErrorCode.CLEANUP_FAILED
    assert isinstance(raised.value.__cause__, StaticRunnerError)
    assert raised.value.__cause__.code is StaticRunnerErrorCode.MALFORMED_RESULT


def test_image_runner_accepts_only_normalized_python_paths_as_data() -> None:
    runner = _image_runner()

    assert runner.validated_paths(["src/a.py", "src/b.pyi"]) == ("src/a.py", "src/b.pyi")
    for invalid in (
        ["--config-file=attacker.py"],
        ["@attacker.py"],
        ["../escape.py"],
        ["src\\escape.py"],
        ["src/a.txt"],
        ["src/A.py", "src/a.py"],
    ):
        with pytest.raises(ValueError):
            runner.validated_paths(invalid)


def test_image_runner_no_python_paths_marks_every_tool_not_applicable() -> None:
    runner = _image_runner()

    results = runner.execute(())

    assert [item["check"] for item in results] == list(runner.CHECKS)
    assert {item["status"] for item in results} == {"not_applicable"}


def test_image_runner_compile_and_ast_are_separate_and_do_not_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _image_runner()
    repository = tmp_path / "repository"
    path = repository / "src" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_text("return 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    compile_result = runner.syntax_result("python_compile", ("src/example.py",), parse_ast=False)
    ast_result = runner.syntax_result("ast_parse", ("src/example.py",), parse_ast=True)

    assert compile_result["status"] == "failed"
    assert compile_result["findings"][0]["rule_code"] == "PY_COMPILE"
    assert ast_result["status"] == "passed"


def test_image_runner_maps_an_ast_component_failure_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _image_runner()
    repository = tmp_path / "repository"
    path = repository / "src" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    def fail_ast(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise SyntaxError("synthetic", ("src/example.py", 1, 1, "value = 1"))

    monkeypatch.setattr(runner.ast, "parse", fail_ast)

    compile_result = runner.syntax_result("python_compile", ("src/example.py",), parse_ast=False)
    ast_result = runner.syntax_result("ast_parse", ("src/example.py",), parse_ast=True)

    assert compile_result["status"] == "passed"
    assert ast_result["status"] == "failed"
    assert ast_result["findings"][0]["rule_code"] == "AST_PARSE"


def test_image_runner_uses_fixed_tool_arguments_and_discards_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _image_runner()
    calls: list[tuple[str, ...]] = []

    def fake_tool(arguments: tuple[str, ...]) -> tuple[int, bytes, bytes, bool]:
        calls.append(arguments)
        if arguments[0] == "/usr/local/bin/ruff":
            return (
                1,
                json.dumps(
                    [
                        {
                            "filename": "src/example.py",
                            "location": {"row": 2, "column": 3},
                            "code": "F401",
                            "message": "candidate secret must not survive",
                        }
                    ]
                ).encode(),
                b"",
                False,
            )
        if arguments[0] == "/usr/local/bin/mypy":
            return 1, b"src/example.py:2:3: error: candidate secret [return-value]\n", b"", False
        return (
            1,
            json.dumps(
                {
                    "errors": [],
                    "results": [
                        {
                            "filename": "src/example.py",
                            "line_number": 2,
                            "test_id": "B404",
                            "issue_text": "candidate secret",
                        }
                    ],
                }
            ).encode(),
            b"",
            False,
        )

    monkeypatch.setattr(runner, "run_tool", fake_tool)
    monkeypatch.setattr(runner, "mypy_directive_findings", lambda paths: [])

    results = [
        runner.ruff_result(("src/example.py",)),
        runner.mypy_result(("src/example.py",)),
        runner.bandit_result(("src/example.py",)),
    ]

    serialized = json.dumps(results)
    assert "candidate secret" not in serialized
    assert all(item["status"] == "failed" for item in results)
    assert "--isolated" in calls[0] and "--no-cache" in calls[0] and "--ignore-noqa" in calls[0]
    assert calls[0][calls[0].index("--select") + 1] == "E4,E7,E9,F,I,UP,B,SIM"
    assert "/dev/null" in calls[1] and "--strict" in calls[1]
    assert calls[1][-2:] == ("--", "src/example.py")
    assert calls[0][0] == "/usr/local/bin/ruff"
    assert calls[1][0] == "/usr/local/bin/mypy"
    assert calls[2][:4] == ("/usr/local/bin/bandit", "--format", "json", "--quiet")
    assert "--ignore-nosec" in calls[2]
    assert all("attacker" not in token for call in calls for token in call)


def test_image_runner_rejects_candidate_mypy_suppression_directives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _image_runner()
    repository = tmp_path / "repository"
    path = repository / "src" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_text("value: int = 'hidden'  # type: ignore[assignment]\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    findings = runner.mypy_directive_findings(("src/example.py",))

    assert findings == [
        {
            "path": "src/example.py",
            "line": 1,
            "column": 24,
            "rule_code": "MYPY_DIRECTIVE",
        }
    ]


@pytest.mark.parametrize(
    "comment",
    ["#\ttype\t:\tignore[assignment]", "#  mypy \t: disable-error-code=assignment"],
)
def test_image_runner_rejects_whitespace_variant_mypy_directives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, comment: str
) -> None:
    runner = _image_runner()
    repository = tmp_path / "repository"
    path = repository / "example.py"
    repository.mkdir()
    path.write_text(f"value = 1  {comment}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    findings = runner.mypy_directive_findings(("example.py",))

    assert len(findings) == 1
    assert findings[0]["rule_code"] == "MYPY_DIRECTIVE"


def test_image_runner_does_not_reject_near_miss_mypy_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _image_runner()
    repository = tmp_path / "repository"
    path = repository / "example.py"
    repository.mkdir()
    path.write_text("value = 1  # type: ignored by prose\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    assert runner.mypy_directive_findings(("example.py",)) == []
