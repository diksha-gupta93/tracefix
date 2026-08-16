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
- External systems include Docker, databases, queues, model providers, GitHub, tracing systems, and metrics systems.
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
- Never expose application secrets to execution containers.
- Never expose model-provider credentials to execution containers.
- Never expose GitHub credentials to execution containers.
- Never mount a developer home directory into execution containers.
- Use non-root execution for all untrusted code.
- Disable network access during untrusted execution.
- Use a read-only root filesystem.
- Use explicit writable temporary directories only.
- Reject protected-path modifications before sandbox execution.
- Do not weaken, skip, delete, or rewrite tests to make a patch pass.
- Do not disable security controls to make integration tests easier.

## Implementation Style

- Python 3.14.
- Pydantic v2 for structured schemas.
- No implicit untyped dictionary passing across module boundaries.
- No `Any` unless explicitly justified in the active specification.
- Use typed interfaces between modules.
- Use dependency injection for external systems.
- Use structured JSON logs.
- In v0.1, logs must include `case_id`, and `job_id` when available.
- In v0.2+, logs must include `job_id` and `trace_id`.
- Every task must be reviewable in one sitting.
- If a task requires more than one specification file, split the task.

## Codex Task Discipline

- Implement exactly one specification per task.
- Read the active specification before changing files.
- Inspect the current architecture before implementation.
- Do not implement adjacent or future tasks.
- Do not modify the active specification during implementation.
- Do not add unrelated abstractions.
- Do not add unrelated dependencies.
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

## Local Development Environment

- Use the existing project virtual environment.
- The standard virtual-environment directory is `.venv`.
- On Windows, use `.venv\Scripts\python.exe`.
- Invoke Python tools through `python -m`.
- Do not download or install an additional Python runtime without explicit human approval.
- Do not create alternative runtime directories such as `.python314`.
- If the required Python version is unavailable, stop and report the detected versions and interpreter paths.