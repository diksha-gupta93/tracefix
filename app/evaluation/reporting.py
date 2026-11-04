from __future__ import annotations

import errno
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from app.evaluation.benchmark import BenchmarkExecutionStatus, BenchmarkReport

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_TEMP_ID = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_REPARSE_POINT = 0x400


class BenchmarkReportWriteError(RuntimeError):
    """The benchmark report pair could not be published with certainty."""


def benchmark_json_bytes(report: BenchmarkReport) -> bytes:
    if type(report) is not BenchmarkReport:
        raise ValueError("benchmark report has the wrong type")
    checked = BenchmarkReport.model_validate(report.model_dump(warnings="none"), strict=True)
    serialized = json.dumps(
        checked.model_dump(mode="json", warnings="none"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{serialized}\n".encode()


def _display(value: object | None) -> str:
    return "—" if value is None else str(value)


def benchmark_markdown_bytes(report: BenchmarkReport) -> bytes:
    checked = BenchmarkReport.model_validate(report.model_dump(warnings="none"), strict=True)
    lines = [
        "# TraceFix Live Benchmark",
        "",
        f"- Schema: `{checked.schema_version}`",
        f"- Run ID: `{checked.benchmark_run_id}`",
        f"- Split: `{checked.split}`",
        f"- Provider: `{checked.provider}`",
        f"- Model: `{checked.model}`",
        f"- Prompt version: `{checked.prompt_version}`",
        f"- Attempts per case: `{checked.max_attempts}`",
        f"- Started (UTC): `{checked.started_at.isoformat()}`",
        f"- Completed (UTC): `{checked.completed_at.isoformat()}`",
        f"- Execution status: `{checked.execution_status.value}`",
        f"- Request timeout seconds: `{checked.request_timeout_seconds}`",
        f"- Input price USD per million tokens: `{checked.input_price_usd_per_million_tokens}`",
        f"- Output price USD per million tokens: `{checked.output_price_usd_per_million_tokens}`",
        "",
        "## Cases",
        "",
        "| Case | Classification | Outcome | Latency (s) | Input tokens | "
        "Output tokens | Estimated USD | Artifact |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in checked.cases:
        lines.append(
            "| "
            + " | ".join(
                (
                    item.case_id,
                    item.classification.value,
                    _display(
                        None if item.terminal_outcome is None else item.terminal_outcome.value
                    ),
                    _display(item.model_latency_seconds),
                    _display(item.input_tokens),
                    _display(item.output_tokens),
                    _display(item.estimated_cost_usd),
                    _display(None if item.artifact_path is None else item.artifact_path.as_posix()),
                )
            )
            + " |"
        )
    counts = checked.counts
    usage = checked.aggregate_usage
    lines.extend(
        (
            "",
            "## Summary",
            "",
            f"- Attempted: `{counts.attempted}`",
            f"- Not run: `{counts.not_run}`",
            f"- Repair success: `{counts.repair_success}`",
            f"- Repair failure: `{counts.repair_failure}`",
            f"- Execution failure: `{counts.execution_failure}`",
            f"- Success rate over completed repairs: `{_display(checked.success_rate)}`",
            f"- Aggregate model latency seconds: `{usage.model_latency_seconds}`",
            f"- Aggregate input tokens: `{usage.input_tokens}`",
            f"- Aggregate output tokens: `{usage.output_tokens}`",
            f"- Aggregate estimated USD cost: `{usage.estimated_cost_usd}`",
            "",
            "Estimated cost uses the recorded operator-supplied rates and is not an invoice.",
            "",
        )
    )
    return "\n".join(lines).encode()


def _validate_report_path(value: PurePosixPath, filename: str) -> None:
    if (
        value.parts[:2] != (".tracefix", "benchmarks")
        or len(value.parts) != 4
        or value.parts[3] != filename
        or _RUN_ID.fullmatch(value.parts[2]) is None
    ):
        raise ValueError("benchmark report path is unsafe")


class BenchmarkReportPublication(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    json_path: PurePosixPath
    markdown_path: PurePosixPath

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        _validate_report_path(self.json_path, "report.json")
        _validate_report_path(self.markdown_path, "report.md")
        if self.json_path.parent != self.markdown_path.parent:
            raise ValueError("benchmark reports must share a directory")
        return self


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    report: BenchmarkReport
    exit_code: int
    publication: BenchmarkReportPublication

    @model_validator(mode="after")
    def validate_exit(self) -> Self:
        expected = 0 if self.report.execution_status is BenchmarkExecutionStatus.COMPLETED else 6
        if self.exit_code != expected:
            raise ValueError("benchmark exit code is inconsistent")
        if self.publication.json_path.parts[2] != self.report.benchmark_run_id:
            raise ValueError("benchmark publication identity is inconsistent")
        return self

    def render(self) -> str:
        counts = self.report.counts
        return (
            f"split={self.report.split} provider={self.report.provider} "
            f"model={self.report.model} attempted={counts.attempted} "
            f"not_run={counts.not_run} repair_success={counts.repair_success} "
            f"repair_failure={counts.repair_failure} "
            f"execution_failure={counts.execution_failure} "
            f"status={self.report.execution_status.value} exit_code={self.exit_code} "
            f"json={self.publication.json_path.as_posix()} "
            f"markdown={self.publication.markdown_path.as_posix()}"
        )


def _ordinary_directory(path: Path) -> tuple[int, int, int, int]:
    if not path.is_absolute():
        raise BenchmarkReportWriteError("benchmark output directory is invalid")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise BenchmarkReportWriteError("benchmark output directory is invalid") from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        path != resolved
        or not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or attributes & _WINDOWS_REPARSE_POINT
    ):
        raise BenchmarkReportWriteError("benchmark output directory is invalid")
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, attributes


def _private_directory(parent: Path, name: str) -> Path:
    parent_identity = _ordinary_directory(parent)
    child = parent / name
    try:
        child.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError:
        raise BenchmarkReportWriteError("benchmark output directory creation failed") from None
    if child.parent != parent or _ordinary_directory(parent) != parent_identity:
        raise BenchmarkReportWriteError("benchmark output directory identity changed")
    _ordinary_directory(child)
    return child


def _write_file(path: Path, payload: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError(errno.EIO, "benchmark report write made no progress")
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    except BaseException:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        with suppress(OSError, RuntimeError):
            _regular_file_identity(path, identity)
            path.unlink()
        raise
    finally:
        os.close(descriptor)


def _regular_file_identity(path: Path, expected: tuple[int, int] | None = None) -> tuple[int, int]:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or attributes & _WINDOWS_REPARSE_POINT
        or (expected is not None and identity != expected)
    ):
        raise OSError(errno.EIO, "benchmark report identity changed")
    return identity


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(temporary: Path, destination: Path, identity: tuple[int, int]) -> tuple[int, int]:
    _regular_file_identity(temporary, identity)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError
    if os.name == "nt":
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination, follow_symlinks=False)
        _regular_file_identity(destination, identity)
        _regular_file_identity(temporary, identity)
        temporary.unlink()
    return _regular_file_identity(destination, identity)


@dataclass(frozen=True, slots=True)
class _OwnedReportPair:
    publication: BenchmarkReportPublication
    output: Path
    output_identity: tuple[int, int, int, int]
    files: tuple[tuple[Path, tuple[int, int]], ...]


class LocalBenchmarkReportWriter:
    def __init__(self, identifier_factory: Callable[[], str] = lambda: uuid.uuid4().hex) -> None:
        self._identifier_factory = identifier_factory
        self._publications: dict[tuple[Path, str], _OwnedReportPair] = {}

    def write(
        self, report: BenchmarkReport, invocation_directory: Path
    ) -> BenchmarkReportPublication:
        checked = BenchmarkReport.model_validate(report.model_dump(warnings="none"), strict=True)
        if _RUN_ID.fullmatch(checked.benchmark_run_id) is None:
            raise BenchmarkReportWriteError("benchmark run identity is invalid")
        invocation = invocation_directory.absolute()
        _ordinary_directory(invocation)
        tracefix = _private_directory(invocation, ".tracefix")
        benchmarks = _private_directory(tracefix, "benchmarks")
        if any(
            child.name.casefold() == checked.benchmark_run_id.casefold()
            for child in benchmarks.iterdir()
        ):
            raise BenchmarkReportWriteError("benchmark destination already exists")
        output = benchmarks / checked.benchmark_run_id
        try:
            output.mkdir(mode=0o700)
            output_identity = _ordinary_directory(output)
            _sync_directory(benchmarks)
        except (FileExistsError, OSError):
            raise BenchmarkReportWriteError("benchmark destination creation failed") from None
        identifier = self._identifier_factory()
        if type(identifier) is not str or _TEMP_ID.fullmatch(identifier) is None:
            with suppress(OSError):
                output.rmdir()
            raise BenchmarkReportWriteError("benchmark temporary identity is invalid")
        json_temp = output / f".report.json.{identifier}.tmp"
        markdown_temp = output / f".report.md.{identifier}.tmp"
        json_path = output / "report.json"
        markdown_path = output / "report.md"
        owned: dict[Path, tuple[int, int]] = {}
        try:
            owned[json_temp] = _write_file(json_temp, benchmark_json_bytes(checked))
            owned[markdown_temp] = _write_file(markdown_temp, benchmark_markdown_bytes(checked))
            if _ordinary_directory(output) != output_identity:
                raise OSError(errno.EIO, "benchmark output identity changed")
            json_identity = owned[json_temp]
            published_json_identity = _publish(json_temp, json_path, json_identity)
            del owned[json_temp]
            owned[json_path] = published_json_identity
            markdown_identity = owned[markdown_temp]
            published_markdown_identity = _publish(markdown_temp, markdown_path, markdown_identity)
            del owned[markdown_temp]
            owned[markdown_path] = published_markdown_identity
            _sync_directory(output)
            if _ordinary_directory(output) != output_identity or set(output.iterdir()) != set(
                owned
            ):
                raise OSError(errno.EIO, "benchmark output contents changed")
            for path, identity in owned.items():
                _regular_file_identity(path, identity)
            publication = BenchmarkReportPublication(
                json_path=PurePosixPath(
                    ".tracefix", "benchmarks", checked.benchmark_run_id, "report.json"
                ),
                markdown_path=PurePosixPath(
                    ".tracefix", "benchmarks", checked.benchmark_run_id, "report.md"
                ),
            )
            self._publications[(invocation, checked.benchmark_run_id)] = _OwnedReportPair(
                publication=publication,
                output=output,
                output_identity=output_identity,
                files=tuple(owned.items()),
            )
            return publication
        except (KeyboardInterrupt, SystemExit):
            self._cleanup(output, output_identity, owned)
            raise
        except Exception:
            cleaned = self._cleanup(output, output_identity, owned)
            message = (
                "benchmark publication failed"
                if cleaned
                else "benchmark publication state is uncertain"
            )
            raise BenchmarkReportWriteError(message) from None

    def discard(self, publication: BenchmarkReportPublication, invocation_directory: Path) -> None:
        checked = BenchmarkReportPublication.model_validate(
            publication.model_dump(warnings="none"), strict=True
        )
        invocation = invocation_directory.absolute()
        run_id = checked.json_path.parts[2]
        key = (invocation, run_id)
        owned = self._publications.get(key)
        if owned is None or owned.publication != checked:
            raise BenchmarkReportWriteError("benchmark publication is not owned")
        if not self._cleanup(owned.output, owned.output_identity, dict(owned.files)):
            raise BenchmarkReportWriteError("benchmark publication state is uncertain")
        del self._publications[key]

    @staticmethod
    def _cleanup(
        output: Path,
        identity: tuple[int, int, int, int],
        candidates: dict[Path, tuple[int, int]],
    ) -> bool:
        try:
            if _ordinary_directory(output) != identity:
                return False
            existing = set(output.iterdir())
            if not existing.issubset(candidates.keys()):
                return False
            for path in existing:
                _regular_file_identity(path, candidates[path])
            for path in existing:
                path.unlink()
            _sync_directory(output)
            output.rmdir()
            _sync_directory(output.parent)
            return True
        except (OSError, RuntimeError, BenchmarkReportWriteError):
            return False
