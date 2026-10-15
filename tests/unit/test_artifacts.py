from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

import app.artifacts as artifacts_module
from app.agent.graph import (
    RunWorkspaces,
    new_orchestration_state,
    terminalize_infrastructure_failure,
)
from app.agent.routing import TerminalOutcome
from app.artifacts import (
    ArtifactSummary,
    ArtifactWriteError,
    LocalArtifactWriter,
    RunArtifact,
    artifact_bytes,
    project_artifact,
)
from benchmarks.loader import load_trusted_case


def _terminal_state(tmp_path: Path):
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    preparation = root / "preparation"
    candidate = root / "candidate"
    verification = root / "verification"
    for child in (preparation, candidate, verification):
        child.mkdir()
    state = new_orchestration_state(
        run_id="a" * 32,
        trusted_case=load_trusted_case("boundary_condition"),
        workspaces=RunWorkspaces(
            root=root,
            preparation=preparation,
            candidate=candidate,
            verification=verification,
            ownership_token="a" * 32,
        ),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return terminalize_infrastructure_failure(state, datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC))


def test_safe_projection_is_deterministic_and_excludes_graph_secrets(tmp_path: Path) -> None:
    state = _terminal_state(tmp_path)
    artifact = project_artifact(state)
    first = artifact_bytes(artifact)
    second = artifact_bytes(RunArtifact.model_validate_json(first, strict=True))
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first)["schema_version"] == 1
    text = first.decode()
    assert str(state.workspaces.root) not in text
    for forbidden in (
        "unified_diff",
        "rendered_prompt",
        "raw_json",
        "stdout",
        "stderr",
        "reference_patch",
        "hidden_tests",
    ):
        assert forbidden not in text


def test_writer_publishes_once_without_overwrite(tmp_path: Path) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    writer = LocalArtifactWriter(lambda: "c" * 32)
    publication = writer.write(artifact, invocation)
    destination = invocation / Path(publication.relative_path.as_posix())
    assert destination.read_bytes() == artifact_bytes(artifact)
    assert not tuple(destination.parent.glob("*.tmp"))
    with pytest.raises(ArtifactWriteError):
        writer.write(artifact, invocation)
    assert destination.read_bytes() == artifact_bytes(artifact)


def test_writer_rejects_aliased_output_component(tmp_path: Path) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation").resolve()
    invocation.mkdir()
    target = invocation / "target"
    target.mkdir()
    alias = invocation / ".tracefix"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    with pytest.raises(ArtifactWriteError):
        LocalArtifactWriter(lambda: "d" * 32).write(artifact, invocation)


def test_artifact_and_summary_contracts_reject_inconsistency(tmp_path: Path) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    with pytest.raises(ValidationError):
        RunArtifact.model_validate({**artifact.model_dump(), "exit_code": 0}, strict=True)
    summary = ArtifactSummary(
        case_id=artifact.case_id,
        outcome=TerminalOutcome.INFRASTRUCTURE_FAILURE,
        summary="repair run encountered an infrastructure failure",
        exit_code=6,
        artifact_path=PurePosixPath(".tracefix/artifacts/file.json"),
    )
    assert "outcome=infrastructure_failure" in summary.render()


def test_artifact_rejects_non_utc_and_impossible_success(tmp_path: Path) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    offset = timezone(timedelta(hours=5))
    with pytest.raises(ValidationError, match="timestamps must be UTC"):
        RunArtifact.model_validate(
            {
                **artifact.model_dump(),
                "started_at": artifact.started_at.astimezone(offset),
                "completed_at": artifact.completed_at.astimezone(offset),
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="terminal outcome"):
        RunArtifact.model_validate(
            {
                **artifact.model_dump(),
                "outcome": TerminalOutcome.SUCCESS,
                "exit_code": 0,
                "terminal_diagnostic": None,
            },
            strict=True,
        )


def test_writer_removes_partial_file_when_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-failure").resolve()
    invocation.mkdir()

    def fail_flush(descriptor: int) -> None:
        del descriptor
        raise OSError("untrusted flush detail")

    monkeypatch.setattr("app.artifacts.os.fsync", fail_flush)
    with pytest.raises(ArtifactWriteError, match="publication failed"):
        LocalArtifactWriter(lambda: "e" * 32).write(artifact, invocation)
    output = invocation / ".tracefix" / "artifacts"
    assert output.is_dir()
    assert not tuple(output.iterdir())


def test_writer_removes_partial_file_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-write-failure").resolve()
    invocation.mkdir()

    def fail_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise OSError("untrusted write detail")

    monkeypatch.setattr("app.artifacts.os.write", fail_write)
    with pytest.raises(ArtifactWriteError, match="publication failed"):
        LocalArtifactWriter(lambda: "1" * 32).write(artifact, invocation)
    assert not tuple((invocation / ".tracefix" / "artifacts").iterdir())


def test_writer_removes_partial_file_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-close-failure").resolve()
    invocation.mkdir()
    real_close = os.close
    failed = False

    def fail_first_close(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            real_close(descriptor)
            raise OSError("untrusted close detail")
        real_close(descriptor)

    monkeypatch.setattr("app.artifacts.os.close", fail_first_close)
    with pytest.raises(ArtifactWriteError, match="publication failed"):
        LocalArtifactWriter(lambda: "2" * 32).write(artifact, invocation)
    assert not tuple((invocation / ".tracefix" / "artifacts").iterdir())


def test_writer_removes_partial_file_when_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-rename-failure").resolve()
    invocation.mkdir()

    def fail_rename(temporary: Path, destination: Path) -> None:
        del temporary, destination
        raise OSError("untrusted rename detail")

    monkeypatch.setattr("app.artifacts._publish_no_replace", fail_rename)
    with pytest.raises(ArtifactWriteError, match="publication failed"):
        LocalArtifactWriter(lambda: "3" * 32).write(artifact, invocation)
    assert not tuple((invocation / ".tracefix" / "artifacts").iterdir())


def test_writer_preserves_racing_destination_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-rename-collision").resolve()
    invocation.mkdir()
    existing = b"pre-existing artifact"

    def collide(temporary: Path, destination: Path) -> None:
        del temporary
        destination.write_bytes(existing)
        raise FileExistsError("destination appeared before rename")

    monkeypatch.setattr("app.artifacts._publish_no_replace", collide)
    with pytest.raises(ArtifactWriteError, match="publication failed"):
        LocalArtifactWriter(lambda: "4" * 32).write(artifact, invocation)
    output = invocation / ".tracefix" / "artifacts"
    destination = output / f"{artifact.case_id}-{artifact.run_id}.json"
    assert destination.read_bytes() == existing
    assert tuple(output.iterdir()) == (destination,)


def test_writer_cleans_temp_and_preserves_post_creation_artifact_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / "invocation-artifact-error").resolve()
    invocation.mkdir()
    original = artifacts_module._require_directory_identity
    failure = ArtifactWriteError("directory identity changed")
    calls = 0

    def fail_after_creation(path: Path, identity: tuple[int, int, int, int]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        original(path, identity)

    monkeypatch.setattr("app.artifacts._require_directory_identity", fail_after_creation)
    with pytest.raises(ArtifactWriteError) as raised:
        LocalArtifactWriter(lambda: "5" * 32).write(artifact, invocation)
    assert raised.value is failure
    assert not tuple((invocation / ".tracefix" / "artifacts").iterdir())


@pytest.mark.parametrize("interruption", [KeyboardInterrupt("stop"), SystemExit(19)])
def test_writer_cleans_temp_and_preserves_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: KeyboardInterrupt | SystemExit,
) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    invocation = (tmp_path / f"invocation-{type(interruption).__name__}").resolve()
    invocation.mkdir()

    def interrupt_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise interruption

    monkeypatch.setattr("app.artifacts.os.write", interrupt_write)
    with pytest.raises(type(interruption)) as raised:
        LocalArtifactWriter(lambda: "6" * 32).write(artifact, invocation)
    assert raised.value is interruption
    assert not tuple((invocation / ".tracefix" / "artifacts").iterdir())


def test_writer_rejects_aliased_invocation_root(tmp_path: Path) -> None:
    artifact = project_artifact(_terminal_state(tmp_path))
    target = (tmp_path / "invocation-target").resolve()
    target.mkdir()
    alias = tmp_path / "invocation-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    with pytest.raises(ArtifactWriteError, match="output root is invalid"):
        LocalArtifactWriter(lambda: "f" * 32).write(artifact, alias.absolute())
