# TraceFix — Execution Plan (v0.1 → v0.2 → v1.0)

This plan turns the TraceFix Product Development Architecture into a sequence of **single-issue, bounded tasks** that can be implemented with Codex one task at a time.

The central rule is simple:

> One task → one specification → one implementation branch → one reviewable change.

Do not batch multiple tasks into one Codex request. Each task must be small enough for Codex to inspect, implement, test, and report without making undocumented assumptions.

## Governing Documents and Precedence

TraceFix development is governed by four levels of documentation:

1. `AGENTS.md`
   Repository-wide engineering, security, quality, and Codex rules.

2. `docs/architecture/tracefix-system-architecture.md`
   Canonical system architecture, domain concepts, core contracts,
   trust boundaries, and long-term design intent.

3. `docs/plan/tracefix-execution-plan.md`
   Version scope, implementation sequence, and bounded task definitions.

4. `docs/specs/<task>.md`
   Exact implementation contract for the active task.

An active task specification may intentionally implement only a subset
of the architecture appropriate to the current version.

A specification may narrow scope, but it must not silently contradict
the architecture or execution plan.

If a conflict is discovered between these documents, implementation
must stop and the conflict must be resolved explicitly before coding.

---

# 1. How to Use This Plan

1. Create `AGENTS.md` at the repository root before implementation begins.
2. Store this file at `docs/plan/tracefix-execution-plan.md`.
3. For every task:
   - create one corresponding specification under `docs/specs/`;
   - review and approve the specification;
   - commit the approved specification;
   - start a fresh Codex session;
   - ask Codex to implement only that specification;
   - independently run the required quality gates;
   - review the diff;
   - commit the implementation.
4. Keep specifications and implementation tasks 1:1.
5. Do not implement version `N+1` until version `N` satisfies its full Definition of Done.
6. Do not allow Codex to silently change a specification during implementation.
7. Do not combine deterministic platform testing with live-model benchmark evaluation.
8. Every task must finish with a standard completion report.

---

# 2. Repository Documentation Structure

```text
docs/
├── plan/
│   └── tracefix-execution-plan.md
├── specs/
│   ├── 0.1-000-bootstrap.md
│   ├── 0.1-001-core-schemas.md
│   └── ...
├── architecture/
│   ├── sandbox-design.md
│   ├── retry-policy.md
│   ├── checkpoint-recovery.md
│   └── ...
└── progress.md
```

- `docs/plan/`: the complete roadmap and task order.
- `docs/specs/`: one detailed implementation contract per task.
- `docs/architecture/`: architecture decisions, diagrams, retry rules, security design, ADRs, and operational behavior.
- `docs/progress.md`: branch, commit, status, validation evidence, and remaining risks for each task.

---

# 3. Specification Naming Convention

```text
docs/specs/0.1-000-bootstrap.md
docs/specs/0.1-001-core-schemas.md
docs/specs/0.1-003a-basic-docker-execution.md
docs/specs/0.2-005b-worker-restart-recovery.md
docs/specs/1.0-009c-slo-validation.md
```

Format:

```text
<version>-<sequence><optional-suffix>-<short-slug>.md
```

Suffixes `a`, `b`, `c`, and `d` are used when a task is deliberately split into smaller bounded units.

---

# 4. `AGENTS.md` Baseline

Create this file at the repository root before implementation begins.

```markdown
# Engineering Rules — TraceFix

## Architecture
- FastAPI, once introduced in v0.2, handles ingress only.
- Long-running work must never execute inside HTTP request handlers.
- Long-running work executes through the worker queue introduced in v0.2.
- PostgreSQL, once introduced in v0.2, is the source of truth for persistent job state.
- LangGraph state must be checkpointed after persistence is introduced.
- Untrusted code must never execute on the host.
- Untrusted code must execute only inside the Docker sandbox from v0.1 onward.
- External systems must be accessed only through typed adapters.
- Business logic must not call external systems ad hoc.

## Quality Gates
Before completing any task:
1. Run formatting checks.
2. Run linting.
3. Run type checking.
4. Run unit tests.
5. Run applicable integration tests.
6. Add tests for every new behavior.
7. Write failing tests first where practical.
8. Update architecture documentation when a decision changes.
9. Report remaining risks explicitly.
10. Do not silently absorb ambiguity.

## Security
- Never mount the Docker socket into execution containers.
- Never expose application, model-provider, or GitHub credentials to execution containers.
- Never mount a developer home directory into execution containers.
- Use non-root execution for all untrusted code.
- Disable network access during untrusted execution.
- Use a read-only root filesystem.
- Use explicit writable temporary directories only.
- Reject protected-path modifications before sandbox execution.
- Do not weaken, skip, delete, or rewrite tests to make a patch pass.
- Do not disable security controls to make integration tests easier.

## Implementation Style
- Python 3.12.
- Pydantic v2 for structured schemas.
- No implicit untyped dictionary passing across module boundaries.
- No `Any` unless explicitly justified in the active specification.
- Use typed interfaces between modules.
- Use dependency injection for external systems.
- Use structured JSON logs.
- In v0.1, logs must include `case_id`, and `job_id` when available.
- In v0.2+, logs must include `job_id` and `trace_id`.
- Every task must be reviewable in one sitting.
- If a task requires more than one specification file, split it.

## Codex Task Discipline
- Implement exactly one specification per task.
- Read the active specification before changing files.
- Inspect the current architecture before implementation.
- Do not implement adjacent or future tasks.
- Do not modify the active specification during implementation.
- Do not add unrelated abstractions or dependencies.
- Do not commit or push changes.
- The human reviewer performs Git operations.
- Run all required quality gates before claiming completion.
- Report all assumptions and remaining risks.

## Required Completion Report
Every implementation task must finish with:
- Files changed
- Commands executed
- Test results
- Acceptance criteria evidence
- Architecture decisions
- Security considerations
- Remaining risks or deferred work
```

## v0.2 Overlay

Append when v0.2 begins:

```markdown
## Architecture — v0.2 Additions
- Every significant job-state transition must produce an audit record.
- Every persisted job must be keyed by a delivery or idempotency identifier.
- Duplicate idempotency identifiers must return the existing job.
- Duplicate submissions must never create duplicate jobs.
- Retries must distinguish repair attempts from infrastructure retries.
- External operations must be designed for at-least-once execution with idempotent recovery.
```

## v1.0 Overlay

Append when v1.0 begins:

```markdown
## Architecture — v1.0 Additions
- Always operate on immutable commit SHAs.
- Never operate on mutable branch names after ingestion.
- Never merge a generated patch automatically into a protected branch.
- Human approval is mandatory before creating a repair branch.
- Every delivery ID, job ID, trace ID, attempt ID, and patch hash must be correlatable.
- SLOs must be enforced by automated tests.
- Revalidate the current pull-request head SHA before publishing or creating a branch.
- Cancel or supersede stale jobs when the pull-request head commit changes.
```

---

# 5. Task Specification Template

```text
# Task <ID> — <Task Name>

We are preparing Task <ID> only.

Read:
- AGENTS.md
- docs/architecture/tracefix-system-architecture.md
- docs/plan/tracefix-execution-plan.md
- all existing architecture documents relevant to this task
- existing code relevant to this task

Do not implement application code.

Create:
docs/specs/<spec-name>.md

The specification must include:
 1. Goal
 2. Context
 3. In Scope
 4. Out of Scope
 5. Expected Repository Structure
 6. Python and Packaging Requirements
 7. Development Tool Requirements
 8. Cross-Platform Requirements
 9. GitHub Actions CI Requirements
 10. Security Considerations
 11. Failure Cases
 12. Acceptance Criteria
 13. Tests and Validation Commands
 14. Documentation Updates
 15. Definition of Done

Do not expand the task beyond the execution plan.
After creating the spec, report the assumptions that require human review.

```

---

# 6. Standard Codex Implementation Instruction

```text
Implement <task name> exactly as described in:

docs/specs/<spec-file>.md

Before editing:
1. Read AGENTS.md in full.
2. Read the active specification in full.
3. Inspect the current repository and related modules.
4. Present the exact files you propose creating or modifying.
5. Explain the implementation sequence.
6. Identify security and failure cases.
7. Identify which tests will be created first.
8. Stop and wait for approval before editing.

Constraints:
- Do not implement adjacent or future tasks.
- Do not modify the active specification.
- Do not weaken, skip, delete, or rewrite tests.
- Do not add unrelated dependencies.
- Do not commit or push changes.
- Keep public interfaces typed.
- Use dependency injection for external services.

After implementation, report:
- Files changed
- Commands executed
- Test results
- Acceptance criteria evidence
- Architecture decisions
- Security controls verified
- Remaining risks or deferred work
```

---

# VERSION 0.1 — Local Working Repair Agent

## Scope

Build a local, single-attempt repair pipeline operating on seeded fixture repositories.

v0.1 includes:
- five seeded defect cases;
- a Docker execution sandbox;
- a single-attempt LangGraph workflow;
- deterministic fake-model integration testing;
- optional live-model evaluation;
- a CLI entrypoint;
- JSON evaluation artifacts.

v0.1 excludes FastAPI, PostgreSQL, Redis, worker queues, GitHub integration, persistent checkpoints, distributed tracing, Prometheus, and Grafana.

## Definition of Done — v0.1

- `tracefix repair <case_id>` runs the complete local pipeline.
- All untrusted code executes only inside the Docker sandbox.
- The sandbox enforces non-root execution, no network, read-only root filesystem, bounded writable workspace, CPU/memory/PID limits, timeout, bounded output, dropped capabilities, `no-new-privileges`, seccomp, and automatic cleanup.
- Each run produces a structured JSON artifact.
- Deterministic fake-model integration tests run in ordinary CI.
- Live-model benchmarks are separate and do not block ordinary CI.
- Ruff, mypy, unit tests, integration tests, and security tests pass.

## Task 0.1.0 — Repository Bootstrap and `AGENTS.md`

Spec: `docs/specs/0.1-000-bootstrap.md`

Requirements:
- Create the initial v0.1 repository structure:
  - `app/agent`
  - `app/sandbox`
  - `app/evaluation/deterministic`
  - `benchmarks/development`
  - `tests/unit`
  - `tests/integration`
  - `docs/plan`
  - `docs/specs`
  - `docs/architecture`
  - `scripts`
- Add root `AGENTS.md`.
- Add `pyproject.toml` requiring Python 3.12.
- Configure Ruff, mypy, and pytest.
- Add `.gitignore`.
- Add `.gitattributes` with LF line-ending rules.
- Add a cross-platform validation command, preferably `python scripts/check.py`.
- Add a basic GitHub Actions CI workflow.
- Add a bootstrap README.
- Do not add application behavior.

Report:
- Files created.
- Tool versions selected.
- Packaging decision.
- Local development commands.
- Windows/Linux compatibility decisions.

## Task 0.1.1 — Core Pydantic Schemas

Spec: `docs/specs/0.1-001-core-schemas.md`

Requirements:
- Define `TestResult`, `FailureAnalysis`, `RepairPlan`, `PatchProposal`, `EvaluationResult`, and `LocalRepairCaseState`.
- `LocalRepairCaseState` includes:
  - `case_id`
  - `status`
  - `attempt_number`
  - `max_attempts=1`
  - `baseline_result`
  - `failure_analysis`
  - `repair_plan`
  - `candidate_patch`
  - `verification_results`
  - `evaluation_results`
  - `model_provider`
  - `model_name`
  - `prompt_version`
  - `created_at`
  - `updated_at`
- Use Pydantic v2.
- Fully type all fields.
- Avoid `Any`.
- Test construction, malformed input rejection, serialization round trip, enum validation, and timestamps.

Report:
- Schema decisions.
- Fields intentionally deferred to v0.2.

## Task 0.1.2 — Fixture Repository with Five Seeded Defects

Spec: `docs/specs/0.1-002-fixture-defects.md`

Requirements:
- Create five small pytest repositories:
  1. Incorrect conditional.
  2. Boundary-condition defect.
  3. Incorrect return value.
  4. Exception-handling defect.
  5. Fixture or mocking defect.
- Each case contains:
  - repository snapshot;
  - base commit;
  - failing commit;
  - visible tests;
  - hidden tests;
  - issue description;
  - expected behavior;
  - reference patch;
  - expected changed files;
  - forbidden changed files;
  - bug category;
  - difficulty;
  - risk level.
- Add `benchmarks/loader.py`.
- The loader must not expose hidden tests or reference patches to model-context selection.

Tests:
- Valid case loading.
- Missing case.
- Malformed manifest.
- Hidden-test isolation.
- Reference-patch isolation.
- Invalid path rejection.

## Task 0.1.3a — Basic Docker Execution

Spec: `docs/specs/0.1-003a-basic-docker-execution.md`

Requirements:
- Create `app/sandbox/runner.py`, `app/sandbox/results.py`, a sandbox Dockerfile, and basic integration tests.
- Accept a prepared repository path and a strictly validated pytest command.
- Launch a fresh Linux container.
- Execute as a non-root user.
- Disable networking.
- Use a read-only root filesystem.
- Provide a temporary writable workspace.
- Capture stdout, stderr, exit code, and duration.
- Always remove the container.
- Reject arbitrary shell execution such as `bash -c`, `sh -c`, PowerShell, or `cmd.exe`.

Tests:
- Passing pytest command.
- Failing pytest command.
- Invalid command rejection.
- Structured result.
- Container removed after success and failure.

## Task 0.1.3b — Resource, Timeout, and Output Enforcement

Spec: `docs/specs/0.1-003b-resource-timeout-enforcement.md`

Requirements:
- Add typed `SandboxLimits`.
- Enforce:
  - CPU quota;
  - memory limit;
  - PID limit;
  - wall-clock timeout;
  - stdout limit;
  - stderr limit;
  - truncation markers;
  - timeout classification;
  - resource-failure classification.
- Suggested defaults:
  - 1 CPU;
  - 512 MB memory;
  - 128 PIDs;
  - 120-second timeout;
  - 1 MB stdout;
  - 1 MB stderr.

Tests:
- Infinite loop timeout.
- Memory exhaustion bounded.
- Excessive processes bounded.
- stdout and stderr truncation.
- Normal execution unaffected.
- Cleanup still occurs.

## Task 0.1.3c — Security Hardening and Adversarial Tests

Spec: `docs/specs/0.1-003c-security-hardening.md`

Requirements:
- Apply `--network none`, read-only root filesystem, `--cap-drop ALL`, `no-new-privileges`, non-root execution, resource limits, seccomp, no Docker socket, no home-directory mount, no credential mount, validated mount sources, unique container names, and cleanup in `finally`.
- Require Docker's default seccomp profile.
- Reject `seccomp=unconfined`.
- Optionally support a project-owned profile.
- Do not begin with an overly restrictive Python syscall allowlist.

Adversarial tests:
- Network blocked.
- Host credentials inaccessible.
- Docker socket inaccessible.
- Write outside workspace blocked.
- Privilege attempt blocked.
- Process explosion blocked.
- Container removed after runner crash.
- No secrets in container environment.
- Invalid mount rejected.

## Task 0.1.4 — Prepare Repository and Run Baseline Nodes

Spec: `docs/specs/0.1-004-prepare-and-baseline.md`

Requirements:
- Prepare node:
  - load benchmark case;
  - checkout required commit into an isolated workspace;
  - prevent path traversal;
  - compute repository fingerprint;
  - record commit identity.
- Baseline node:
  - run the visible failing test in the sandbox;
  - capture baseline evidence;
  - stop with a diagnostic artifact if the baseline passes;
  - do not invoke the model when the failure is not reproducible.

Tests:
- Known failure reproduces for all five cases.
- Unexpected-pass path.
- Missing commit.
- Invalid workspace.
- Fingerprint determinism.

## Task 0.1.5 — Classify Failure and Select Relevant Context

Spec: `docs/specs/0.1-005-classify-and-context.md`

Requirements:
- Classify failures into supported categories.
- Build a bounded context package with:
  - issue description;
  - failing test;
  - stack trace;
  - referenced source;
  - relevant definitions;
  - imports;
  - types;
  - repository instructions;
  - protected-path policy.
- Exclude hidden tests, reference patches, unrelated files, secrets, and generated artifacts.

Tests:
- Classification correctness.
- Hidden-test exclusion.
- Reference-patch exclusion.
- Unrelated-file exclusion.
- Context-size enforcement.
- Path safety.

## Task 0.1.6 — Plan Repair and Generate Patch

Spec: `docs/specs/0.1-006-plan-and-generate.md`

Requirements:
- Create a typed model-provider interface.
- Configure through:
  - `MODEL_PROVIDER`
  - `MODEL_NAME`
  - `MODEL_TEMPERATURE`
  - `MODEL_MAX_TOKENS`
  - `PROMPT_VERSION`
- v0.1 supports one repair attempt.
- Repair plan returns root cause, expected files, risks, validation strategy, and autonomy suitability.
- Patch generation returns `PatchProposal`, never free text.
- Malformed model output becomes a structured diagnostic failure.

Tests:
- Fake provider success.
- Malformed structured output.
- Provider exception.
- Missing configuration.
- Prompt version recorded.
- Invalid patch rejected.

## Task 0.1.7 — Validate Patch Policy

Spec: `docs/specs/0.1-007-patch-policy.md`

Reject patches that:
- modify protected files;
- modify too many files;
- exceed line-change thresholds;
- contain binary data;
- contain secrets;
- disable tests;
- remove assertions;
- change CI or security configuration;
- introduce unrestricted subprocess execution;
- add unapproved dependencies;
- modify hidden tests;
- modify benchmark metadata;
- modify reference patches.

Policy rejection occurs before sandbox execution.

Tests:
- One test per rejection rule.
- Valid patch pass-through.
- Combined violation reporting.
- Deterministic rule ordering.

## Task 0.1.8 — Static Checks

Spec: `docs/specs/0.1-008-static-checks.md`

Requirements:
- Patch-apply validation.
- Python compilation.
- AST parsing.
- Ruff.
- mypy where applicable.
- Bandit.
- Changed-file validation.
- All failures block progression.

Tests:
- Every check individually triggerable.
- Every failure blocks.
- Clean patch passes.
- Tool errors become structured failures.

## Task 0.1.9 — Sandbox Verification and Candidate Evaluation

Spec: `docs/specs/0.1-009-verify-and-evaluate.md`

Run:
1. Original targeted failing test.
2. Full visible suite.
3. Hidden tests.

Evaluate:
- functional result;
- files changed;
- lines added/removed;
- execution duration;
- model latency;
- input/output tokens;
- estimated cost where available.

Tests:
- Full pass path for at least two cases.
- Full fail path for at least two cases.
- Hidden-test failure after visible success.
- Regression failure.
- Sandbox infrastructure failure mapping.

## Task 0.1.10a — Graph Routing and CLI Wiring

Spec: `docs/specs/0.1-010a-routing-and-cli.md`

Requirements:
- Wire all v0.1 nodes into a single-attempt LangGraph workflow.
- Support outcomes:
  - success;
  - candidate failure;
  - policy violation;
  - static-check failure;
  - baseline not reproducible;
  - infrastructure failure;
  - malformed model output.
- Add `tracefix repair <case_id>`.
- Print a summary, write an artifact, and return meaningful exit codes.

Tests:
- Routing branch coverage.
- CLI argument validation.
- Artifact creation.
- Exit-code validation.
- Unknown case.

## Task 0.1.10b — Deterministic Fake-Model CI Integration

Spec: `docs/specs/0.1-010b-fake-model-ci.md`

Requirements:
- Create deterministic fake model provider.
- Return predetermined plans and patches by `case_id`, `prompt_version`, and `attempt_number`.
- Run full platform pipeline in ordinary CI.
- Validate graph wiring, schemas, patch application, policy, static checks, sandbox execution, artifacts, and CLI behavior.
- Require no model credentials or external network.

Tests:
- All five fixtures through the CLI.
- Expected artifacts.
- Stable deterministic results.

## Task 0.1.11a — Live-Model Benchmark Command and Report

Spec: `docs/specs/0.1-011a-live-model-benchmark.md`

Requirements:
- Add:
  `tracefix benchmark --split development --provider <provider>`
- Use a real model provider.
- Write JSON and human-readable results.
- Record latency, tokens, estimated cost, and repair outcome.
- Treat model repair failure as a valid benchmark result.
- Treat platform or provider infrastructure failure as an execution failure.
- Do not include live evaluation in required ordinary CI.
- Add a manual `workflow_dispatch` workflow.

## Task 0.1.11b — README, Quick Start, and v0.1 Release Gate

Spec: `docs/specs/0.1-011b-docs-and-release-gate.md`

Requirements:
- Document architecture, Python/Docker requirements, installation, local checks, fake-model CI, live benchmark, CLI usage, artifacts, and limitations.
- Limitations:
  - one repair attempt;
  - no persistence;
  - no queue;
  - no GitHub integration;
  - no checkpoint recovery;
  - no production observability.
- Create v0.1 release checklist.

Definition of Done:
- All fake-model cases run end to end.
- Sandbox security tests pass.
- Ruff, mypy, unit, and integration tests pass.
- Live benchmark triggers independently.
- Documentation matches actual behavior.

---

# VERSION 0.2 — Persistence, Queue, Checkpointing, Retries, and 15 Benchmarks

## Scope

v0.2 introduces PostgreSQL, SQLAlchemy, Alembic, transactional job persistence, Redis, a worker queue, FastAPI manual ingress, LangGraph checkpoint persistence, worker restart recovery, external-operation recovery, bounded repair retries, 15 benchmark cases, and confidence-interval reporting.

## Definition of Done — v0.2

- Job state survives worker restart.
- The graph resumes from the latest valid checkpoint.
- Duplicate idempotency keys do not create duplicate jobs.
- Repair attempts and infrastructure retries are tracked separately.
- Repair attempts are bounded at three.
- Fifteen cases cover at least six categories.
- Reports include confidence intervals.
- Failure-injection tests assert final job state.

## Task 0.2.1 — PostgreSQL Schema and Migrations

Spec: `docs/specs/0.2-001-postgres-schema.md`

Requirements:
- Full `RepairJobState`.
- SQLAlchemy models.
- Alembic migrations.
- Job statuses.
- Attempt, evaluation, and audit records.
- Idempotency identifier.
- Timestamps and schema versioning.

Tests:
- Upgrade/downgrade.
- Enum constraints.
- Unique idempotency key.
- Foreign keys.
- Required fields.

## Task 0.2.2 — Transactional Job Persistence

Spec: `docs/specs/0.2-002-job-repository.md`

Requirements:
- Create/retrieve job.
- Update status.
- Append attempt/evaluation/audit event.
- Idempotent create.
- Existing idempotency key returns existing job.
- Every significant transition produces exactly one audit event.

Tests:
- Concurrent duplicate creation.
- Rollback.
- Audit uniqueness.
- Invalid transitions.

## Task 0.2.3 — FastAPI Manual Ingress and Status API

Spec: `docs/specs/0.2-003-fastapi-manual-trigger.md`

Endpoints:
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/attempts`
- `GET /api/v1/jobs/{job_id}/evaluations`

Requirements:
- Persist and enqueue only.
- No repair work in request handlers.
- Idempotency.
- Prompt accepted response.
- 404 for unknown jobs.

## Task 0.2.4 — Redis Queue and Worker Wiring

Spec: `docs/specs/0.2-004-queue-worker.md`

Requirements:
- Dramatiq or Celery.
- Redis-backed queue.
- Worker process.
- Concurrency limit.
- Graceful shutdown.
- Infrastructure backoff.
- Structured logs.

Tests:
- Dispatch.
- Processing.
- Shutdown.
- Duplicate message.
- Retry scheduling.

## Task 0.2.5a — Persist and Restore LangGraph Checkpoints

Spec: `docs/specs/0.2-005a-checkpoint-persistence.md`

Requirements:
- PostgreSQL-backed checkpointer.
- Stable `thread_id = job_id`.
- Checkpoint after every graph node.
- Checkpoint schema version.
- Restore latest valid checkpoint.
- Reject incompatible versions.
- Do not rerun completed nodes after normal resume.

## Task 0.2.5b — Worker Restart Recovery

Spec: `docs/specs/0.2-005b-worker-restart-recovery.md`

Requirements:
- Start a job.
- Complete one or more nodes.
- Kill worker.
- Start fresh worker.
- Recover in-progress job.
- Resume from latest checkpoint.
- Reach exactly one terminal state.
- Preserve audit trail.
- Begin with termination between nodes, not mid-operation.

## Task 0.2.5c — Failure Injection During External Operations

Spec: `docs/specs/0.2-005c-external-operation-recovery.md`

Requirements:
- Add external-operation ledger with:
  - `operation_id`
  - `job_id`
  - `attempt_number`
  - `operation_type`
  - `request_fingerprint`
  - `status`
  - `started_at`
  - `completed_at`
  - `result_reference`
  - `error_category`
- Use stable operation IDs.
- Model request:
  - persist `STARTED`;
  - persist `COMPLETED`;
  - reuse completed result;
  - classify unresolved `STARTED` as uncertain;
  - record possible duplicate cost;
  - do not claim exactly-once behavior.
- Sandbox:
  - disposable container/workspace;
  - cleanup after crash;
  - safe retry on uncertainty.

Tests:
- Kill during model request.
- Kill during sandbox run.
- Restart and assert final state.
- Assert ledger state.
- Assert no orphan container.
- Assert duplicate-cost marker where relevant.

## Task 0.2.6 — Bounded Repair Retry Loop

Spec: `docs/specs/0.2-006-retry-routing.md`

Requirements:
- Maximum three repair attempts.
- Attach prior failure evidence to next attempt.
- Repair failures increment attempt count.
- Infrastructure retries do not.
- Policy violation, baseline not reproducible, and unsupported project are non-retriable.
- Provider outage, Docker daemon interruption, DB interruption, and worker termination are infrastructure retries.

Tests:
- Attempt 2 success.
- Non-retriable stop.
- Attempt 3 terminal failure.
- Infrastructure retry not counted.
- Prior evidence propagated.

## Task 0.2.7 — Static Validation Gate Hardening

Spec: `docs/specs/0.2-007-static-gate-hardening.md`

Add:
- secret detection;
- dependency-diff inspection;
- complexity comparison;
- suspicious generated-file detection;
- dependency-policy validation.

All failures remain blocking unless explicitly documented otherwise.

## Task 0.2.8 — Expand Benchmark to 15 Cases

Spec: `docs/specs/0.2-008-benchmark-expansion.md`

Requirements:
- Add ten cases.
- Cover at least six categories.
- Include ambiguous/unfixable, protected-file trap, dependency trap, multi-file defect, and regression-prone defect.
- Split into development, validation, and locked test.
- Never expose hidden tests or reference patches.

## Task 0.2.9 — Evaluation Reporting with Confidence Intervals

Spec: `docs/specs/0.2-009-eval-reporting.md`

Metrics:
- success@1/@2/@3;
- hidden-test pass rate;
- regression failure rate;
- median and p95 latency;
- median attempts per success;
- infrastructure retries;
- cost per successful repair;
- confidence intervals.

## Task 0.2.10 — Queue, Database, and Duplicate-Submission Resilience Suite

Spec: `docs/specs/0.2-010-resilience-suite.md`

Scenarios:
- Restart DB.
- Restart Redis.
- Duplicate submission during processing.
- Duplicate worker delivery.
- Worker termination during checkpoint-safe phase.
- Stale claim recovery.

Assert:
- job status;
- attempt count;
- infrastructure retries;
- audit history;
- terminal-state uniqueness;
- no duplicate jobs.

---

# VERSION 1.0 — GitHub Integration, Observability, Reliability, Security, and Demo

## Definition of Done — v1.0

- A failing PR triggers the full pipeline.
- HMAC verification and delivery deduplication are enforced.
- Immutable SHAs are used.
- Results publish to GitHub.
- Human approval is mandatory before branch creation.
- No protected branch is automatically modified.
- Observability receives real run data.
- SLO and security tests pass.
- Documentation and demo are complete.

## Task 1.0.1 — GitHub App Adapter

Spec: `docs/specs/1.0-001-github-app-adapter.md`

Requirements:
- Typed GitHub client.
- Installation-token auth and refresh.
- Read repo/PR/checks.
- Create check runs.
- Publish comments.
- Create draft branch after approval.
- Minimal permissions.

## Task 1.0.2 — Webhook Ingestion

Spec: `docs/specs/1.0-002-webhook-ingestion.md`

Requirements:
- Validate HMAC.
- Deduplicate with `X-GitHub-Delivery`.
- Persist transactionally.
- Store installation/repository/delivery IDs, base/head SHA, and PR number.
- Enqueue only after persistence.
- Reject invalid signature, unauthorized repo, and unsupported event.

## Task 1.0.3 — Check Run and PR Comment Publishing

Spec: `docs/specs/1.0-003-github-publish.md`

Publish:
- root cause;
- proposed patch;
- files changed;
- tests;
- static/security/regression results;
- model/prompt versions;
- attempts;
- tokens;
- cost;
- latency;
- confidence/risk.

Publishing failure is infrastructure failure, not repair failure.

## Task 1.0.4 — Human Approval and Draft Branch Creation

Spec: `docs/specs/1.0-004-approval-and-branch.md`

Endpoints:
- `POST /jobs/{id}/approve`
- `POST /jobs/{id}/reject`

Requirements:
- Approval revalidates head SHA.
- Approval creates draft repair branch only.
- Never write to protected branch.
- Store decisions as evaluation labels.
- Rejection creates no branch.

## Task 1.0.5 — LangSmith Tracing

Spec: `docs/specs/1.0-005-langsmith.md`

Instrument prompts, model calls, tool calls, transitions, branch decisions, tokens, latency, retries, and errors. Tracing failure must not block the pipeline. Redact sensitive values.

## Task 1.0.6 — OpenTelemetry Distributed Tracing

Spec: `docs/specs/1.0-006-opentelemetry.md`

Create connected spans across webhook, persistence, enqueue, queue wait, worker, graph, model, policy, static checks, sandbox, evaluation, GitHub publish, approval, and branch creation. Correlate delivery/job/trace/attempt/operation IDs.

## Task 1.0.7 — Prometheus Metrics

Spec: `docs/specs/1.0-007-prometheus.md`

Metrics:
- webhook rate;
- invalid signatures;
- duplicates;
- queue depth/wait;
- active workers;
- job success/failure;
- sandbox time;
- model latency/tokens/cost;
- policy violations;
- repair attempts;
- infrastructure retries;
- human acceptance;
- DLQ count.

## Task 1.0.8 — Grafana Dashboards and Docker Compose Stack

Spec: `docs/specs/1.0-008-grafana-and-compose.md`

Services:
- API;
- worker;
- PostgreSQL;
- Redis;
- Prometheus;
- Grafana;
- OpenTelemetry Collector.

Provision dashboards and data sources as code. Include health checks and local startup docs.

## Task 1.0.9a — Dead-Letter Queue

Spec: `docs/specs/1.0-009a-dead-letter-queue.md`

Requirements:
- DLQ schema.
- Retry-exhaustion threshold.
- `DEAD_LETTERED` state.
- Preserve failure evidence.
- Inspection endpoint.
- Manual requeue.
- Audit and metrics.

## Task 1.0.9b — Webhook Rate Limiting

Spec: `docs/specs/1.0-009b-rate-limiting.md`

Requirements:
- Per-installation or repository limits.
- Burst and sustained limits.
- `429` with retry guidance.
- Metrics.
- Do not trust spoofable identity where verified identity exists.

## Task 1.0.9c — Automated SLO Validation

Spec: `docs/specs/1.0-009c-slo-validation.md`

Validate:
- webhook acknowledgement latency;
- processing-start latency;
- terminal-state rate;
- credential isolation;
- deduplication;
- trace coverage;
- audit coverage;
- zero auto-merges;
- orphan-container rate;
- stale-job handling.

## Task 1.0.9d — GitHub Lifecycle and Failure Handling

Spec: `docs/specs/1.0-009d-github-lifecycle.md`

Scenarios:
- Publish denied.
- New commit during repair.
- PR closed/reopened.
- Head SHA changed before approval.
- Installation access revoked.
- Repository unauthorized.

Requirements:
- Revalidate SHA.
- Cancel/supersede stale jobs.
- Never publish stale patches.
- Never create branch from outdated commit.
- Record lifecycle events.

## Task 1.0.10 — Security Documentation, ADRs, and Security Tests

Spec: `docs/specs/1.0-010-security-and-adrs.md`

Create:
- threat model;
- trust-boundary diagram;
- sandbox design;
- credential isolation policy;
- dependency policy;
- protected-path policy;
- incident-response notes;
- ADRs.

Tests:
- sandbox escape;
- secret detection;
- forbidden dependency;
- protected-path modification;
- CI/security config modification;
- hidden-test modification;
- Docker socket access;
- host-path access;
- unauthorized webhook.

## Task 1.0.11 — Documentation Pass

Spec: `docs/specs/1.0-011-documentation.md`

Complete:
- README;
- architecture and sequence diagrams;
- state machine;
- benchmark methodology and locked-test report;
- API docs;
- runbooks;
- troubleshooting;
- local development;
- security and observability guides;
- limitations.

## Task 1.0.12 — Demonstration Scenario and Recording

Spec: `docs/specs/1.0-012-demo.md`

Demonstrate:
1. Failing PR.
2. Webhook.
3. Persist/enqueue.
4. Attempt 1 fails.
5. Evidence checkpointed.
6. Worker resumes.
7. Attempt 2 succeeds.
8. Hidden tests pass.
9. GitHub check/comment published.
10. Human approval.
11. Draft branch created.
12. No auto-merge.
13. Metrics, traces, logs, and audit visible.

Record a video or terminal capture. Update CV-positioning claims using only measured, implemented results.

---

# 7. Revised Task Count

- v0.1: 16 tasks.
- v0.2: 12 tasks.
- v1.0: 15 tasks.
- Total: 43 bounded tasks.

The increase is intentional. Large security, recovery, and reliability tasks were split so they are easier to implement, test, and review.

---

# 8. Deterministic CI vs Live Evaluation

## Deterministic CI Integration

Uses a fake model provider.

Purpose:
- verify platform behavior;
- verify graph wiring;
- verify schemas;
- verify policy enforcement;
- verify sandbox execution;
- verify artifact generation.

Characteristics:
- deterministic;
- stable;
- no model credentials;
- no network dependency;
- runs on every commit;
- blocks merge on failure.

## Live-Model Evaluation

Uses a real model provider.

Purpose:
- measure model repair capability;
- measure success, latency, tokens, and cost.

Characteristics:
- non-deterministic;
- may cost money;
- may face provider/network failures;
- runs manually or on schedule;
- does not block ordinary CI because repair failure is a legitimate benchmark result.

The platform test answers:

> Does TraceFix behave correctly?

The benchmark answers:

> How well does a particular model repair the benchmark defects?

---

# 9. Git Workflow

For every task:

```text
main
└── task/<task-id>-<slug>
```

Examples:
- `task/0.1.0-bootstrap`
- `task/0.1.3a-basic-docker`
- `task/0.2.5c-external-recovery`
- `task/1.0.9c-slo-validation`

Recommended commits:
- `docs: specify task <task-id>`
- `feat: implement task <task-id>`
- `docs: record completion of task <task-id>`

Codex must not commit or push. The human reviewer handles Git.

---

# 10. Progress Tracking

Use `docs/progress.md`.

```markdown
# TraceFix Implementation Progress

| Task | Spec | Branch | Commit | Status | Validation Evidence | Remaining Risks |
|---|---|---|---|---|---|---|
| 0.1.0 | 0.1-000-bootstrap.md | task/0.1.0-bootstrap | — | In progress | — | — |
```

Statuses:
- Not started
- Spec drafted
- Spec approved
- In progress
- Review required
- Complete
- Blocked

---

# 11. Quality-Gate Commands

Provide one cross-platform command:

```text
python scripts/check.py
```

It should run:

```text
python -m ruff format --check .
python -m ruff check .
python -m mypy app benchmarks
python -m pytest
```

Task-specific integration and security tests must also be documented.

Codex's completion report is not sufficient. The human reviewer independently runs the checks.

---

# 12. Suggested Delivery Cadence

Planning ranges:

- v0.1: approximately 1.5–3 weeks.
- v0.2: approximately 2–4 weeks.
- v1.0: approximately 3–5 weeks.

These are estimates, not promises. Docker security, checkpoint recovery, GitHub App configuration, and observability integration can require additional troubleshooting.

---

# 13. Final Working Rule

Do not optimize for the smallest number of tasks.

Optimize for:
- reviewability;
- testability;
- security;
- traceability;
- clear acceptance criteria;
- reproducible evidence;
- honest benchmark reporting.

A task is correctly sized when:
- one specification fully describes it;
- one Codex session can implement it;
- one reviewer can understand the diff;
- one branch contains the complete change;
- the acceptance criteria can be independently verified.
