from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts import check


class RecordingRunner:
    def __init__(self, failing_call: int | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.failing_call = failing_call

    def __call__(self, command: Sequence[str], *, cwd: Path) -> None:
        self.calls.append((tuple(command), cwd))
        if self.failing_call == len(self.calls):
            raise subprocess.CalledProcessError(returncode=7, cmd=command)


def test_run_checks_uses_required_commands_in_order_from_repository_root() -> None:
    runner = RecordingRunner()

    check.run_checks(runner)

    repository_root = Path(check.__file__).resolve().parents[1]
    assert runner.calls == [
        ((sys.executable, "-m", "ruff", "format", "--check", "."), repository_root),
        ((sys.executable, "-m", "ruff", "check", "."), repository_root),
        (
            (sys.executable, "-m", "mypy", "app", "benchmarks", "scripts"),
            repository_root,
        ),
        ((sys.executable, "-m", "pytest"), repository_root),
    ]


def test_run_checks_stops_at_first_failure() -> None:
    runner = RecordingRunner(failing_call=2)

    try:
        check.run_checks(runner)
    except subprocess.CalledProcessError as error:
        assert error.returncode == 7
    else:
        raise AssertionError("run_checks did not propagate the command failure")

    assert len(runner.calls) == 2
