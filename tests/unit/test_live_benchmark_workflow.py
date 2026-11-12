from __future__ import annotations

import re
from pathlib import Path


def test_live_benchmark_workflow_is_manual_pinned_and_scopes_credential() -> None:
    workflow = Path(".github/workflows/live-benchmark.yml").read_text(encoding="utf-8")
    assert "pacing" not in workflow.lower()
    assert "rate_limit" not in workflow.lower()
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    paid_step = workflow.index("- name: Run paid live benchmark")
    assert "GROQ_API_KEY" not in workflow[:paid_step]
    paid_block = workflow[paid_step : workflow.index("- name: Upload safe benchmark evidence")]
    assert "GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}" in paid_block
    assert "LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}" in paid_block
    assert 'TRACEFIX_LANGSMITH_DIAGNOSTICS: "true"' in paid_block
    assert 'LANGSMITH_TRACING: "false"' in paid_block
    assert 'LANGSMITH_TRACING: "true"' not in workflow
    assert "LANGSMITH_ENDPOINT" not in workflow
    assert "LANGCHAIN_ENDPOINT" not in workflow
    assert "LANGSMITH_TRACING_MODE" not in workflow
    assert "LANGCHAIN_TRACING_MODE" not in workflow
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in workflow
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in workflow
    assert "LANGSMITH_PROJECT: tracefix-live-diagnostics" in paid_block
    assert 'LANGSMITH_HIDE_INPUTS: "true"' in paid_block
    assert 'LANGSMITH_HIDE_OUTPUTS: "true"' in paid_block
    assert "LANGSMITH_API_KEY" not in workflow[:paid_step]
    assert "OPENAI_API_KEY" not in workflow
    assert '--provider "${BENCHMARK_PROVIDER}"' in paid_block
    assert "- groq" in workflow
    assert "- openai/gpt-oss-120b" in workflow
    assert "reasoning_effort:" in workflow
    assert "- low" in workflow
    model_block = workflow[
        workflow.index("      model:") : workflow.index("      reasoning_effort:")
    ]
    reasoning_block = workflow[
        workflow.index("      reasoning_effort:") : workflow.index("      request_timeout_seconds:")
    ]
    assert "type: choice" in model_block
    assert "type: choice" in reasoning_block
    assert 'MODEL_MAX_TOKENS: "4096"' in workflow
    job_block = workflow[workflow.index("jobs:") : paid_step]
    assert "GROQ_API_KEY" not in job_block
    uses = re.findall(r"^\s*uses:\s*(\S+)$", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    assert ".tracefix/benchmarks/**/report.json" in workflow
    assert ".tracefix/benchmarks/**/report.md" in workflow
    assert ".tracefix/artifacts/*.json" in workflow
