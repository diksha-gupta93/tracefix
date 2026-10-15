from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from app.cli import (
    ApplicationDependencies,
    LocalWorkspaceManager,
    ProviderRegistry,
    RepairApplication,
    WorkspaceError,
    main,
)
from benchmarks.loader import BenchmarkLoadError, load_trusted_case


class NeverCalled:
    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected side effect: {name}")

    def __call__(self, *args: object, **kwargs: object):
        raise AssertionError("unexpected side effect")


def _unknown_application() -> RepairApplication:
    def missing(case_id: str):
        raise BenchmarkLoadError(case_id)

    stream = StringIO()
    return RepairApplication(
        ApplicationDependencies(
            case_loader=missing,
            configuration_loader=NeverCalled(),
            provider_registry=ProviderRegistry(),
            workspace_manager=NeverCalled(),
            artifact_writer=NeverCalled(),
            graph_dependencies_factory=NeverCalled(),
            clock=NeverCalled(),
            run_id_factory=NeverCalled(),
            invocation_directory=NeverCalled(),
            environment={},
            stdout=stream,
            stderr=stream,
        )
    )


def test_cli_help_and_argument_surface(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"], application_factory=_unknown_application)
    assert help_exit.value.code == 0
    assert "{repair}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing_exit:
        main([], application_factory=_unknown_application)
    assert missing_exit.value.code == 2
    with pytest.raises(SystemExit) as extra_exit:
        main(["repair", "boundary_condition", "extra"], application_factory=_unknown_application)
    assert extra_exit.value.code == 2


@pytest.mark.parametrize("case_id", ["../case", "C:\\case", "/case", "UPPER", "a" * 129])
def test_invalid_or_unknown_case_has_no_preflight_side_effects(
    case_id: str, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["repair", case_id], application_factory=_unknown_application)
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "invalid or unknown case_id" in captured.err
    assert "Traceback" not in captured.err


def test_workspace_manager_creates_separate_owned_areas_and_cleans(tmp_path: Path) -> None:
    manager = LocalWorkspaceManager(tmp_path.resolve())
    workspaces = manager.create("a" * 32)
    assert workspaces.root.exists()
    assert len({workspaces.preparation, workspaces.candidate, workspaces.verification}) == 3
    assert all(
        path.parent == workspaces.root
        for path in (
            workspaces.preparation,
            workspaces.candidate,
            workspaces.verification,
        )
    )
    manager.cleanup(workspaces)
    assert not workspaces.root.exists()
    with pytest.raises(WorkspaceError):
        manager.cleanup(workspaces)


def test_workspace_manager_rejects_aliased_base(tmp_path: Path) -> None:
    target = (tmp_path / "workspace-target").resolve()
    target.mkdir()
    alias = tmp_path / "workspace-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    with pytest.raises(WorkspaceError, match="creation failed"):
        LocalWorkspaceManager(alias).create("f" * 32)


def test_workspace_manager_normalizes_base_validation_failure(tmp_path: Path) -> None:
    missing_base = tmp_path / "missing-workspace-base"
    with pytest.raises(WorkspaceError, match="creation failed"):
        LocalWorkspaceManager(missing_base).create("f" * 32)


def test_empty_provider_registry_has_no_fallback() -> None:
    from app.agent.schemas import ModelConfiguration

    configuration = ModelConfiguration(
        provider="not-registered",
        model_name="model",
        temperature=0.0,
        max_output_tokens=100,
        prompt_version="repair-v1.0",
    )
    assert ProviderRegistry().resolve(configuration) is None


def test_keyboard_interrupt_maps_to_130() -> None:
    class InterruptingApplication:
        def repair(self, case_id: str) -> int:
            del case_id
            raise KeyboardInterrupt

    assert main(["repair", "boundary_condition"], lambda: InterruptingApplication()) == 130


def test_unexpected_application_factory_failure_is_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_factory() -> RepairApplication:
        raise RuntimeError("untrusted initialization detail")

    assert main(["repair", "boundary_condition"], fail_factory) == 6
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "tracefix: repair run failed safely\n"
    assert "untrusted" not in captured.err


def test_loader_operational_failure_is_not_reported_as_unknown() -> None:
    stdout = StringIO()
    stderr = StringIO()

    def fail_loader(case_id: str):
        del case_id
        raise RuntimeError("untrusted loader detail")

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=fail_loader,
            configuration_loader=NeverCalled(),
            provider_registry=ProviderRegistry(),
            workspace_manager=NeverCalled(),
            artifact_writer=NeverCalled(),
            graph_dependencies_factory=NeverCalled(),
            clock=NeverCalled(),
            run_id_factory=NeverCalled(),
            invocation_directory=NeverCalled(),
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )
    assert application.repair("boundary_condition") == 6
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "tracefix: run initialization failed\n"


def test_run_identifier_failure_is_contained_before_configuration() -> None:
    stdout = StringIO()
    stderr = StringIO()

    def fail_identifier() -> str:
        raise RuntimeError("untrusted identifier detail")

    application = RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=NeverCalled(),
            provider_registry=ProviderRegistry(),
            workspace_manager=NeverCalled(),
            artifact_writer=NeverCalled(),
            graph_dependencies_factory=NeverCalled(),
            clock=NeverCalled(),
            run_id_factory=fail_identifier,
            invocation_directory=NeverCalled(),
            environment={},
            stdout=stdout,
            stderr=stderr,
        )
    )
    assert application.repair("boundary_condition") == 6
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "tracefix: run initialization failed\n"


def test_installed_entry_point_help_smoke() -> None:
    executable = Path(sys.executable).with_name(
        "tracefix.exe" if sys.platform == "win32" else "tracefix"
    )
    result = subprocess.run(
        (str(executable), "--help"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "{repair}" in result.stdout
