# TraceFix

TraceFix is planned as a repair agent for diagnosing failures, proposing constrained patches,
and evaluating them safely. Task 0.1.0 provides only the repository foundation: there is no
application, repair, command-line, sandbox, model, or evaluation behavior yet.

## Prerequisites

- CPython 3.12 (versions before 3.12 and Python 3.13 or newer are unsupported)
- `pip`

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

Run the same cross-platform validation entry point used by continuous integration:

```text
python scripts/check.py
```

It checks formatting, linting, strict typing, and tests in that order.

## Repository layout

- `app/` contains the importable package skeleton.
- `benchmarks/development/` reserves the future development benchmark location.
- `tests/unit/` and `tests/integration/` contain the test suites.
- `docs/` contains the execution plan, task specifications, and future architecture notes.
- `scripts/check.py` is the canonical local and CI validation entry point.

## Current scope

This bootstrap intentionally contains no schemas, benchmark defects, Docker sandbox, graph,
model provider, API, database, queue, GitHub integration, observability service, production
behavior, or runtime dependency. Those belong to later, separately specified tasks.

`.dockerignore` is deliberately deferred to Task 0.1.3a, when the Docker build context and its
exclusion policy will be defined together.
