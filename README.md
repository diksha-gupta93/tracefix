# TraceFix

TraceFix is planned as a repair agent for diagnosing failures, proposing constrained patches,
and evaluating them safely. The current repository contains its typed core schemas, a
deterministic five-case development benchmark, and a resource-bounded typed Docker execution boundary for
strictly validated pytest commands. Repair, command-line, model, and evaluation behavior remain
deferred to later tasks.

## Prerequisites

- CPython 3.14
- `pip`
- Docker with a usable Linux-container daemon and built-in seccomp support (Docker Desktop is
  supported on Windows)

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
shown above. A daemon that is available but does not report seccomp support is a failure. CI builds
the image first, so the complete adversarial integration suite must execute there.

## Repository layout

- `app/` contains the importable package skeleton.
- `app/sandbox/` contains the typed Docker adapter and basic pytest runner.
- `benchmarks/development/` contains five seeded defects with typed trusted and model-safe loader
  views. Hidden tests and reference patches remain evaluator-only; execution is deferred to Docker.
- `tests/unit/` and `tests/integration/` contain the test suites.
- `docs/` contains the execution plan, task specifications, and future architecture notes.
- `scripts/check.py` is the canonical local and CI validation entry point.

## Sandbox scope

Each execution uses a fresh, uniquely named Linux container after validating that the daemon serves
Linux containers with seccomp enabled. Docker's built-in default seccomp profile remains active;
the runner exposes no security-profile override. The prepared repository is the only host bind
mount and is mounted read-only after a complete, stable filesystem inspection. Protected host
locations, protected-path ancestors, symlinks, junctions, reparse points, unsupported entries, and
inspection failures are rejected before container creation. The runner does not inspect credential
file contents.

The container runs as numeric non-root UID/GID `10001:10001`, with all capabilities dropped,
`no-new-privileges` enabled, networking disabled, and a read-only root filesystem. Only the fixed
workspace and temporary directory are writable tmpfs mounts. The environment is a fixed allowlist
containing deterministic Python settings; host secrets, credentials, proxy settings, and Docker
configuration are not inherited. The runner applies typed immutable defaults of 1 CPU, 512 MiB
memory, 128 processes, a 120-second host deadline, and independent 1 MiB returned stdout and
stderr limits. Callers customize them by passing a strict `SandboxLimits` instance. Docker uses a
bounded aggregate local log; the adapter retains each returned stream independently within its
configured in-memory cap.

Results classify normal completion, host wall-clock timeout, and Docker-confirmed OOM termination.
An ordinary nonzero pytest exit remains normal completion. A truncated stream ends with the fixed
ASCII marker `[tracefix output truncated]`, and its complete UTF-8 encoding including that marker
never exceeds its byte limit. The exact generated container is forcibly terminated on timeout and
removed after every outcome.

The sandbox assumes the Docker daemon, fixed project-owned image, host administrator, and Docker
engine are trusted. It does not protect against a malicious daemon or image, host compromise,
Docker engine vulnerabilities, or abrupt host-process/operating-system/daemon loss. Repeated
filesystem identity checks reduce path-confusion risk but cannot eliminate races caused by a
hostile local process mutating a prepared workspace concurrently. Rootless Docker, custom seccomp,
user namespaces, AppArmor/SELinux policy, alternate runtimes, dependency installation, and repair
patch policy remain outside Task 0.1.3c. TraceFix still contains no graph, model provider, API,
database, queue, GitHub integration, observability service, or production repair workflow.
