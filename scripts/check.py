from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    def __call__(self, command: Sequence[str], *, cwd: Path) -> None: ...


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_checks(runner: CommandRunner | None = None) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    commands = (
        (sys.executable, "-m", "ruff", "format", "--check", "."),
        (sys.executable, "-m", "ruff", "check", "."),
        (sys.executable, "-m", "mypy", "app", "benchmarks", "scripts"),
        (sys.executable, "-m", "pytest"),
    )
    command_runner = runner or _run_command

    for command in commands:
        command_runner(command, cwd=repository_root)


def main() -> None:
    run_checks()


if __name__ == "__main__":
    main()
