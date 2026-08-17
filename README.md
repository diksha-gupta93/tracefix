# TraceFix

TraceFix is planned as a repair agent for diagnosing failures, proposing constrained patches,
and evaluating them safely. The current repository contains its typed core schemas, a
deterministic five-case development benchmark, and a resource-bounded typed Docker execution boundary for
strictly validated pytest commands. Repair, command-line, model, and evaluation behavior remain
deferred to later tasks.

## Prerequisites

- CPython 3.14
- `pip`
- Docker with a usable Linux-container daemon (Docker Desktop is supported on Windows)

## Development setup

Create a virtual environment from the repository root:

```text
python -m venv .venv
```

Activate it on Windows PowerShell:

```text
.venv\Scripts\Activate.ps1
```

Or activate it on a POSIX shell:

```text
. .venv/bin/activate
```

Install the project and development tools in editable mode:

```text
python -m pip install --editable ".[dev]"
```

Activation is optional if the environment's Python executable is invoked directly.

## Validation

Build the fixed project-owned sandbox image explicitly before validation:

```text
docker build --tag tracefix-sandbox:0.1.3a --file docker/sandbox/Dockerfile .
```

Run the same cross-platform validation entry point used by continuous integration:

```text
python scripts/check.py
```

It checks formatting, linting, strict typing, unit tests, and Docker integration tests in that
order. On a developer machine, Docker integration tests skip only when the Docker CLI or a usable
Linux-container daemon is unavailable. A missing prebuilt image is a failure with the build command
shown above. CI builds the image first, so integration tests must execute there.

## Repository layout

- `app/` contains the importable package skeleton.
- `app/sandbox/` contains the typed Docker adapter and basic pytest runner.
- `benchmarks/development/` contains five seeded defects with typed trusted and model-safe loader
  views. Hidden tests and reference patches remain evaluator-only; execution is deferred to Docker.
- `tests/unit/` and `tests/integration/` contain the test suites.
- `docs/` contains the execution plan, task specifications, and future architecture notes.
- `scripts/check.py` is the canonical local and CI validation entry point.

## Sandbox scope

Each execution uses a fresh, uniquely named Linux container. The prepared repository is the only
host bind mount and is mounted read-only; the runner copies it into a fresh writable container
workspace. The container runs as numeric non-root UID/GID `10001:10001`, with networking disabled
and a read-only root filesystem. The runner applies typed immutable defaults of 1 CPU, 512 MiB
memory, 128 processes, a 120-second host deadline, and independent 1 MiB returned stdout and
stderr limits. Callers customize them by passing a strict `SandboxLimits` instance. Docker uses a
bounded aggregate local log; the adapter retains each returned stream independently within its
configured in-memory cap.

Results classify normal completion, host wall-clock timeout, and Docker-confirmed OOM termination.
An ordinary nonzero pytest exit remains normal completion. A truncated stream ends with the fixed
ASCII marker `[tracefix output truncated]`, and its complete UTF-8 encoding including that marker
never exceeds its byte limit. The exact generated container is forcibly terminated on timeout and
removed after every outcome.

This is the Task 0.1.3b resource-bounded sandbox, not final hardening. Capability dropping,
`no-new-privileges`, explicit
seccomp selection, sensitive-host-path policy, and the full adversarial suite remain deferred to
Task 0.1.3c. TraceFix still contains no graph, model provider, API, database, queue, GitHub
integration, observability service, or production repair workflow.
