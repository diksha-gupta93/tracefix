# TraceFix

TraceFix is planned as a repair agent for diagnosing failures, proposing constrained patches,
and evaluating them safely. The current repository contains its typed core schemas and a
deterministic five-case development benchmark. Repair, command-line, sandbox, model, and
evaluation behavior remain deferred to later tasks.

## Prerequisites

- CPython 3.14
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
- `benchmarks/development/` contains five seeded defects with typed trusted and model-safe loader
  views. Hidden tests and reference patches remain evaluator-only; execution is deferred to Docker.
- `tests/unit/` and `tests/integration/` contain the test suites.
- `docs/` contains the execution plan, task specifications, and future architecture notes.
- `scripts/check.py` is the canonical local and CI validation entry point.

## Current scope

The current scope is repository-local schemas and benchmark data/loading only. It contains no
Docker sandbox, graph, model provider, API, database, queue, GitHub integration, observability
service, or production repair behavior. Those belong to later, separately specified tasks.

`.dockerignore` is deliberately deferred to Task 0.1.3a, when the Docker build context and its
exclusion policy will be defined together.
