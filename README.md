# TraceFix

TraceFix is an in-progress autonomous code-repair and evaluation platform for failing Python
repositories. Its current foundation prepares deterministic benchmark snapshots, fingerprints their
contents, reproduces known failures inside a hardened Docker sandbox, classifies captured
pytest/Python evidence deterministically, and selects bounded model-safe context through static
text and AST inspection. It can now pass that context through a typed, dependency-injected model
boundary to produce a strictly validated repair plan and structurally validated inert patch
proposal for the single v0.1 attempt.

The broader platform will enforce patch policy, verify candidates, and present results with
reproducible evidence. No generated patch is applied or executed at the current stage.

## Current Status

The repository has completed the v0.1 foundation through Task 0.1.6:

- Python 3.14 packaging, CI, and deterministic quality tooling
- strict Pydantic v2 schemas for repair state, results, plans, patches, and evaluations
- five deterministic seeded benchmark cases with trusted evaluator-only material
- hardened Docker execution with CPU, memory, PID, time, and output limits
- network isolation, non-root execution, a read-only root filesystem, capability dropping, and
  `no-new-privileges`
- repository preparation with path validation, isolated workspaces, and deterministic content
  fingerprinting
- baseline reproduction of each seeded visible failure in the Docker sandbox
- typed handling of reproduced failures, unexpected passes, timeouts, OOM termination, sandbox
  failures, malformed results, and cleanup failures
- deterministic runtime failure classification with ambiguity and truncation handled fail-closed
- immutable, exact-byte-bounded context packages containing relevant visible tests, tracebacks,
  traceback-referenced source and static definitions/imports/types
- evaluator, secret-bearing, generated, binary, invalid UTF-8, aliased, stale, and unrelated
  context exclusion with immutable protected-path metadata
- strict frozen configuration for the five model and prompt environment variables, with no
  defaults or credential handling
- a typed dependency-injected model-provider protocol with no production fake, live adapter, SDK,
  network call, streaming, tool call, or retry behavior
- deterministic versioned planning and generation prompts that delimit the exact bounded context
  as untrusted data and request schema-only JSON
- strict `RepairPlan` and `PatchProposal` parsing, bounded provider output, safe typed diagnostics,
  normalized path identities, and minimal unified-diff envelope validation
- copy-on-update single-attempt state transitions through `plan_complete` and `patch_proposed`, with
  provider/model/prompt identity recording
- adversarial filesystem and sandbox security tests, including Linux TOCTOU regression coverage

[`docs/progress.md`](docs/progress.md) is the source of truth for task completion and recorded
validation evidence. It currently records no task as in progress. The next planned task in the
execution plan is Task 0.1.7: deterministic patch-policy validation.

## Why TraceFix

Generating a diff is only one part of automated repair. A credible repair system also needs to
reproduce the original failure, control untrusted execution, preserve evidence, distinguish repair
failures from infrastructure failures, and evaluate changes without leaking hidden answers.

TraceFix is being built around those constraints:

- reproducible snapshots and versioned content fingerprints;
- typed boundaries instead of implicit dictionaries between components;
- deterministic fixtures separated from future stochastic model evaluation;
- fail-closed path validation and isolated execution;
- explicit completion and failure classification;
- adversarial tests for security-sensitive filesystem behavior; and
- specification-driven tasks with formatting, linting, typing, and test gates.

## Architecture

The intended local repair flow is:

```text
benchmark case
  -> repository preparation and fingerprinting       [implemented]
  -> baseline failure reproduction                    [implemented]
  -> failure classification and relevant context      [implemented]
  -> repair planning and patch generation             [implemented]
  -> patch policy and static validation               [roadmap]
  -> sandbox verification                             [roadmap]
  -> evaluation and routing                           [roadmap]
```

The current implementation deliberately stops after producing a structurally valid inert patch
proposal. Model access is available only through an injected protocol; there is no production fake
or live provider. The patch is not applied, policy-approved, statically checked, compiled,
imported, executed, or treated as evidence of correctness. Later v0.1 tasks add those deterministic
gates; v0.2 adds persistence, queue-backed workers, API ingress, and checkpoint recovery; v1.0 adds
GitHub integration, human approval, and production observability.
External systems are designed to sit behind typed adapters, and long-running work will run in
workers rather than HTTP handlers once those layers exist.

See the [system architecture](docs/architecture/tracefix-system-architecture.md) for the target
component model, trust boundaries, and versioned evolution.

## Current Capabilities

TraceFix can currently:

- load trusted and model-safe views of five validated benchmark manifests;
- keep hidden tests and reference patches outside prepared repositories and baseline commands;
- materialize a declared failing snapshot into a fresh caller-owned workspace;
- reject unsafe paths, links/reparse points, unsupported entries, source mutation, destination
  escape, and protected or aliased locations;
- calculate a location-independent, versioned SHA-256 repository fingerprint;
- execute only validated `python -m pytest` commands for declared visible test paths;
- retain bounded stdout, stderr, truncation flags, exit status, duration, and completion evidence;
  classify recognized syntax, import, assertion, exception, type, timeout, and environment
  evidence without using benchmark answers;
- produce deterministic context under an exact 32,768-byte default limit while revalidating the
  prepared fingerprint and evaluator boundary;
- load strict model configuration from exactly five named environment variables and construct
  deterministic `repair-v1.0` requests through an injected typed provider; and
- produce a canonical repair plan and inert patch proposal with exact plan/proposal/diff path-set
  agreement for attempt `1`.

Current benchmark revisions are symbolic snapshot identities, not Git commit SHAs. Successful
prepared workspaces are caller-owned and must be cleaned up after downstream work completes.

TraceFix does not yet provide a production model adapter, apply patches, enforce full patch policy,
run static or sandbox candidate verification, execute a LangGraph workflow, expose a CLI or API,
persist jobs, run a queue, integrate with GitHub, or report repair success rates.

## Quick Start

Prerequisites are CPython 3.14, `pip`, and Docker with a usable Linux-container daemon and built-in
seccomp support. Docker Desktop with Linux containers is supported for Windows development, subject
to the host limitation described under [Security Model](#security-model).

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --editable ".[dev]"
docker build --tag tracefix-sandbox:0.1.3a --file docker/sandbox/Dockerfile .
python scripts/check.py
```

On POSIX systems, activate the environment with `. .venv/bin/activate`; the remaining commands are
the same. After setup, use the existing `.venv` rather than creating another runtime environment.

## Validation

The canonical local and CI validation entry point is:

```text
python scripts/check.py
```

It runs Ruff formatting checks, Ruff linting, strict mypy checks, and pytest. The pytest suite
includes the applicable Docker integration and adversarial security tests. Local Docker tests skip
only when the Docker CLI or a usable Linux-container daemon is unavailable; CI builds the sandbox
image first and executes the integration suite. A reachable daemon without seccomp support, or a
missing required image when Docker is usable, is a failure.

Recorded milestone results belong in [`docs/progress.md`](docs/progress.md), not in manually
duplicated README counters.

## Security Model

Untrusted benchmark code is never executed directly on the host. Each run uses a fresh Linux
container with the prepared repository mounted read-only. The container has no network, runs as
numeric UID/GID `10001:10001`, drops all capabilities, enables `no-new-privileges`, retains Docker's
default seccomp profile, and uses a read-only root filesystem with only explicit temporary writable
mounts. Host secrets, credentials, proxy configuration, Docker configuration, the Docker socket, and
developer home directories are not exposed. Resource and returned-output limits are typed and
bounded, and containers are forcibly terminated on timeout and removed after every outcome.

Repository context, repair plans, provider responses, and generated diffs are untrusted data. Model
responses are byte-bounded before strict JSON parsing, diagnostics do not expose prompts, context,
raw responses, exception text, credentials, or host paths, and generated diffs remain inert. The
model boundary receives no filesystem, command, Docker, credential, or network capability.

Linux is the production-supported host for adversarial repository preparation and execution.
Windows is development-supported, but native Windows does not claim complete host-side TOCTOU
protection during repository preparation. Running Linux containers through Docker Desktop does not
remove that Windows host preparation limitation.

The Docker daemon, project-owned sandbox image, host administrator, and container engine remain
trusted. The detailed assumptions, protected-path behavior, and future hardening boundaries are
documented in the [system architecture](docs/architecture/tracefix-system-architecture.md) and the
[sandbox hardening specification](docs/specs/0.1-003c-security-hardening.md).

## Benchmark / Evaluation Approach

The development benchmark contains five small, deterministic Python defects:

- `boundary_condition`
- `exception_handling`
- `fixture_or_mocking`
- `incorrect_conditional`
- `incorrect_return_value`

Each case contains a failing repository snapshot, visible test metadata, hidden regression tests,
and a reference patch. Trusted loaders may access evaluator material; model-safe views and prepared
repositories may not. The current integration suite verifies that all five declared visible
failures reproduce inside the real sandbox while fixture sources remain unchanged.

This is deterministic platform and integration testing: it answers whether preparation, isolation,
execution, and evidence handling behave correctly. Future live-model evaluation will separately
measure repair capability, latency, token use, and cost. No repair success rate is reported until
real-model benchmark execution exists.

## Repository Structure

- [`app/`](app/) — typed agent state, model boundary, planning, repository preparation, baseline
  execution, and Docker adapter
- [`benchmarks/`](benchmarks/) — deterministic development fixtures and trusted/model-safe loaders
- [`tests/`](tests/) — unit, Docker integration, filesystem, and adversarial security tests
- [`docs/architecture/`](docs/architecture/) — target architecture and security boundaries
- [`docs/specs/`](docs/specs/) — one implementation contract per completed task
- [`docs/plan/`](docs/plan/) — versioned execution roadmap
- [`scripts/check.py`](scripts/check.py) — canonical validation entry point

## Roadmap

- **Remaining v0.1:** enforce patch policy, verify candidates, route outcomes, add a local CLI, and
  separate deterministic fake-model integration from live-model evaluation.
- **v0.2:** add PostgreSQL-backed state, queue workers, FastAPI ingress, LangGraph checkpointing,
  recovery, bounded retries, and expanded evaluation.
- **v1.0:** add GitHub App workflows, human approval, observability, resilience hardening, and
  operational documentation.

These are roadmap items, not current capabilities. The authoritative sequence and scope are in the
[execution plan](docs/plan/tracefix-execution-plan.md).

## Documentation

- [Implementation progress](docs/progress.md)
- [System architecture](docs/architecture/tracefix-system-architecture.md)
- [Execution plan](docs/plan/tracefix-execution-plan.md)
- [Task specifications](docs/specs/)
- [Engineering and security rules](AGENTS.md)
