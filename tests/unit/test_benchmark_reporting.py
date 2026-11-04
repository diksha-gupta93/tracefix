from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath

import pytest

import app.evaluation.reporting as reporting
from app.agent.routing import TerminalOutcome
from app.evaluation.benchmark import (
    BenchmarkCaseExecution,
    build_benchmark_report,
    run_benchmark_cases,
)
from app.evaluation.deterministic.candidate import CandidateModelUsage
from app.evaluation.live_model import LiveBenchmarkConfiguration
from app.evaluation.reporting import (
    BenchmarkReportWriteError,
    BenchmarkSummary,
    LocalBenchmarkReportWriter,
    benchmark_json_bytes,
    benchmark_markdown_bytes,
)
from benchmarks.loader import load_development_cases


def report() -> object:
    cases = run_benchmark_cases(
        load_development_cases(),
        lambda case: BenchmarkCaseExecution(
            case_id=case.manifest.case_id,
            terminal_outcome=TerminalOutcome.SUCCESS,
            model_usage=CandidateModelUsage(
                model_latency_seconds=0.5,
                input_tokens=10,
                output_tokens=5,
                estimated_cost_usd=Decimal("0.0000045"),
            ),
            artifact_path=PurePosixPath(
                ".tracefix", "artifacts", f"{case.manifest.case_id}-{'a' * 32}.json"
            ),
        ),
    )
    start = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)
    return build_benchmark_report(
        benchmark_run_id="e" * 32,
        model="openai/gpt-oss-120b",
        prompt_version="repair-v1.0",
        started_at=start,
        completed_at=start + timedelta(seconds=1),
        live_configuration=LiveBenchmarkConfiguration(
            request_timeout_seconds=120,
            input_price_usd_per_million_tokens=Decimal("0.15"),
            output_price_usd_per_million_tokens=Decimal("0.60"),
        ),
        cases=cases,
    )


def test_json_and_markdown_are_deterministic_and_semantically_equivalent() -> None:
    candidate = report()
    json_payload = benchmark_json_bytes(candidate)  # type: ignore[arg-type]
    markdown_payload = benchmark_markdown_bytes(candidate)  # type: ignore[arg-type]
    assert json_payload.endswith(b"\n")
    assert markdown_payload.endswith(b"\n")
    decoded = json.loads(json_payload)
    markdown = markdown_payload.decode()
    assert decoded["benchmark_run_id"] in markdown
    assert decoded["model"] in markdown
    assert decoded["execution_status"] in markdown
    for item in decoded["cases"]:
        assert item["case_id"] in markdown
        assert item["classification"] in markdown
    assert decoded["aggregate_usage"]["estimated_cost_usd"] == "0.0000225"


def test_writer_publishes_pair_without_overwrite(tmp_path: Path) -> None:
    candidate = report()
    writer = LocalBenchmarkReportWriter(identifier_factory=lambda: "f" * 32)
    publication = writer.write(candidate, tmp_path)  # type: ignore[arg-type]
    json_path = tmp_path.joinpath(*publication.json_path.parts)
    markdown_path = tmp_path.joinpath(*publication.markdown_path.parts)
    assert json_path.is_file()
    assert markdown_path.is_file()
    summary = BenchmarkSummary(
        report=candidate,
        exit_code=0,
        publication=publication,  # type: ignore[arg-type]
    )
    rendered = summary.render()
    assert str(tmp_path) not in rendered
    assert publication.json_path.as_posix() in rendered

    with pytest.raises(BenchmarkReportWriteError):
        writer.write(candidate, tmp_path)  # type: ignore[arg-type]
    assert json_path.is_file() and markdown_path.is_file()


def test_writer_rejects_aliased_tracefix_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / ".tracefix"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    with pytest.raises(BenchmarkReportWriteError):
        LocalBenchmarkReportWriter().write(report(), tmp_path)  # type: ignore[arg-type]


def test_writer_cleans_owned_pair_after_partial_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = reporting._publish

    def fail_second_publish(
        temporary: Path, destination: Path, identity: tuple[int, int]
    ) -> tuple[int, int]:
        if destination.name == "report.md":
            raise OSError("injected publication failure")
        return real_publish(temporary, destination, identity)

    monkeypatch.setattr(reporting, "_publish", fail_second_publish)
    with pytest.raises(BenchmarkReportWriteError, match="publication failed"):
        LocalBenchmarkReportWriter(identifier_factory=lambda: "f" * 32).write(
            report(),
            tmp_path,  # type: ignore[arg-type]
        )
    assert not (tmp_path / ".tracefix" / "benchmarks" / ("e" * 32)).exists()


def test_writer_detects_temporary_replacement_without_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = reporting._publish

    def replace_before_publish(
        temporary: Path, destination: Path, identity: tuple[int, int]
    ) -> tuple[int, int]:
        if destination.name == "report.json":
            temporary.unlink()
            temporary.write_bytes(b"replacement")
        return real_publish(temporary, destination, identity)

    monkeypatch.setattr(reporting, "_publish", replace_before_publish)
    with pytest.raises(BenchmarkReportWriteError, match="state is uncertain"):
        LocalBenchmarkReportWriter(identifier_factory=lambda: "f" * 32).write(
            report(),
            tmp_path,  # type: ignore[arg-type]
        )
    output = tmp_path / ".tracefix" / "benchmarks" / ("e" * 32)
    replacement = output / f".report.json.{'f' * 32}.tmp"
    assert replacement.read_bytes() == b"replacement"


def test_writer_discards_only_its_owned_publication(tmp_path: Path) -> None:
    writer = LocalBenchmarkReportWriter(identifier_factory=lambda: "f" * 32)
    publication = writer.write(report(), tmp_path)  # type: ignore[arg-type]
    output = tmp_path.joinpath(*publication.json_path.parts).parent
    writer.discard(publication, tmp_path)
    assert not output.exists()
    with pytest.raises(BenchmarkReportWriteError, match="not owned"):
        writer.discard(publication, tmp_path)
