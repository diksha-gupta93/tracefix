from __future__ import annotations

import re
from pathlib import Path


def test_live_benchmark_workflow_is_manual_pinned_and_scopes_credential() -> None:
    workflow = Path(".github/workflows/live-benchmark.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    paid_step = workflow.index("- name: Run paid live benchmark")
    assert "OPENAI_API_KEY" not in workflow[:paid_step]
    paid_block = workflow[paid_step : workflow.index("- name: Upload safe benchmark evidence")]
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in paid_block
    assert '--provider "${BENCHMARK_PROVIDER}"' in paid_block
    assert "- openai" in workflow
    uses = re.findall(r"^\s*uses:\s*(\S+)$", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    assert ".tracefix/benchmarks/**/report.json" in workflow
    assert ".tracefix/benchmarks/**/report.md" in workflow
    assert ".tracefix/artifacts/*.json" in workflow
