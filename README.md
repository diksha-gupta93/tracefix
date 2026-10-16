# TraceFix

TraceFix is an in-progress autonomous code-repair and evaluation platform for failing Python
repositories. Its current foundation prepares deterministic benchmark snapshots, fingerprints their
contents, reproduces known failures inside a hardened Docker sandbox, classifies captured
pytest/Python evidence deterministically, and selects bounded model-safe context through static
text and AST inspection. It can now pass that context through a typed, dependency-injected model
boundary to produce a strictly validated repair plan and structurally validated inert patch
proposal for the single v0.1 attempt. A deterministic typed patch-policy gate now rejects
out-of-scope or unsafe proposals before a deterministic static-validation gate applies an approved
patch to a fresh candidate and runs compilation, AST parsing, Ruff, strict mypy, and Bandit in a
dedicated hardened container. Passing candidates are now verified in fixed increasing scope inside
the hardened pytest sandbox and evaluated with deterministic functional, patch-size, execution-
duration, and optional supplied model-usage metrics. A synchronous eleven-node LangGraph now
routes one repair attempt to a deterministic local outcome, and the installed CLI publishes a
safe metadata-only run artifact.

A static pass remains blocking prerequisite evidence only; functional success requires targeted,
full-visible, and hidden verification to pass.

## Current Status

The repository has completed the v0.1 foundation through Task 0.1.10a:

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
- pure, deterministic patch-policy validation with fixed limits of three changed files and 200
  added-plus-removed hunk lines
- aggregated, deduplicated, stably ordered typed violations for protected paths, binary or secret
  content, test/assertion manipulation, CI/security changes, direct process execution,
  dependencies, and evaluator/benchmark/reference material
- distinct fail-closed typed diagnostics for malformed policy, proposal, path, and diff input
- exact, atomic UTF-8 text patch application for modifications, additions, and deletions, with no
  offset search, fuzz, path stripping, Git invocation, or partial publication
- byte-level changed-file validation before and after static tooling, preserving the prepared
  snapshot and publishing only a fully passing candidate
- fixed-order compilation, AST, Ruff, strict-mypy, and Bandit checks in the dedicated
  `tracefix-static-checks:0.1.8` image, with candidate configuration and suppressions disabled or
  rejected
- strict safe results that distinguish deterministic candidate findings from Docker, tool,
  protocol, filesystem-integrity, and cleanup diagnostics
- adversarial filesystem and sandbox security tests, including Linux TOCTOU regression coverage
- fixed-order candidate verification against targeted, complete-visible, and trusted hidden tests
- private hidden-only evaluator bundles mounted read-only and removed after every outcome
- safe verification evidence that discards raw test and Docker output and distinguishes functional
  failure from timeout, OOM, integrity, Docker, protocol, evaluator, and cleanup diagnostics
- deterministic candidate evaluation with ordered changed files, added/removed line counts,
  executed sandbox duration, and nullable exact model latency/token/Decimal-USD values
- an in-memory, single-attempt LangGraph with eleven named nodes, explicit terminal short circuits,
  no retry edge, and seven fixed terminal outcomes
- `tracefix repair <case_id>` with preflight case validation, stable exit codes, graph-owned private
  workspaces, deterministic safe artifacts, and an empty production provider registry

[`docs/progress.md`](docs/progress.md) is the source of truth for task completion and recorded
validation evidence. It currently records no task as in progress. The next planned task in the
execution plan is Task 0.1.10b: deterministic fake-provider integration.

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
  -> patch policy                                      [implemented]
  -> patch application and static validation           [implemented]
  -> sandbox verification                              [implemented]
  -> deterministic candidate evaluation                [implemented]
  -> routing and local artifacts                        [implemented]
```

The current implementation stops after one routed local attempt. Model access remains behind an
injected protocol, with one production-composed deterministic local fake provider and no live
provider. Policy approval and static validation are necessary but not sufficient: a candidate is
successful only when targeted, full-visible, and hidden verification all pass. The graph has no
checkpointer, persistence, retry, or resume behavior. Task 0.1.11a adds the live-provider benchmark
boundary. v0.2 adds persistence, queue-backed workers, API ingress, and checkpoint recovery; v1.0
adds GitHub integration, human approval, and production observability. External systems are
designed to sit behind typed adapters, and long-running work will run in workers rather than HTTP
handlers once those layers exist.

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
- resolve the explicitly configured `tracefix-fake` provider to immutable, predetermined plan and
  patch responses for all five development cases without credentials or model-provider network
  access;
- produce a canonical repair plan and inert patch proposal with exact plan/proposal/diff path-set
  agreement for attempt `1`; and
- inspect a proposal entirely in memory against immutable protected and forbidden paths, the fixed
  v0.1 size limits, high-confidence local content signatures, an empty-by-default dependency
  allowlist, and evaluator isolation rules, returning only safe typed metadata;
- apply an approved modification/addition/deletion diff with exact coordinates, contexts, removed
  lines, UTF-8 and LF/CRLF semantics in a fresh private candidate copy;
- prove the actual added/modified/deleted byte set equals both the proposal and parsed diff before
  and after analysis; and
- run all surviving changed `.py` and `.pyi` files through Python 3.14 compilation and AST parsing,
  isolated Ruff, strict mypy, and Bandit in a fixed-command hardened container;
- verify candidates in targeted, full-visible, then hidden order with deterministic
  short-circuiting and identity/fingerprint checks before, between, and after execution;
- copy only declared hidden-test bytes into deterministic private names, mount the bundle
  read-only, discard raw output, and clean up the bundle while leaving the candidate untouched; and
- purely evaluate functional outcome, changed paths, added/removed lines, executed sandbox
  duration, and optional supplied model latency, token counts, and exact USD cost. Missing
  measurements remain `null`, while zero remains a valid observation.

Current benchmark revisions are symbolic snapshot identities, not Git commit SHAs. Successful
prepared workspaces are caller-owned and must be cleaned up after downstream work completes.

TraceFix does not yet provide a live model adapter, API, persistent jobs, queue, GitHub integration,
benchmark aggregation, retries, or live-model repair success rates.

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
docker build --tag tracefix-static-checks:0.1.8 --file docker/static-checks/Dockerfile .
python scripts/check.py
```

On POSIX systems, activate the environment with `. .venv/bin/activate`; the remaining commands are
the same. After setup, use the existing `.venv` rather than creating another runtime environment.

## Local Repair Command

The installed public command is exactly:

```text
tracefix repair <case_id>
```

The command validates malformed and unknown case IDs before creating a workspace, loading model
configuration, invoking Docker, or writing an artifact. To run any of the five development cases
through the local deterministic provider and the complete eleven-node pipeline, set exactly:

```text
MODEL_PROVIDER=tracefix-fake
MODEL_NAME=tracefix-deterministic-v1
MODEL_TEMPERATURE=0
MODEL_MAX_TOKENS=4096
PROMPT_VERSION=repair-v1.0
```

The fake provider accepts no alternate profile. It uses explicit project-owned responses keyed by
case ID, prompt version, and attempt number; it does not parse prompts, read benchmark/evaluator
files at runtime, require model credentials, or make provider network calls. An unknown provider or
unsupported fake profile follows the existing safe `infrastructure_failure` route. Live provider
and benchmark support remains deferred to Task 0.1.11a.

For identical valid requests, fake response bytes are stable. Across complete runs, the terminal
route, exit code, changed-file and line-count evidence, stage outcomes, and provider/model/prompt
identity are stable. Run IDs, timestamps, measured durations, temporary paths, and artifact
filenames are intentionally volatile and are validated for shape and internal consistency instead
of byte equality.

Completed known-case runs print one summary and write
`.tracefix/artifacts/<case_id>-<run_id>.json` relative to the invocation directory. The versioned
artifact contains only safe structured metadata: identities and UTC timestamps, the terminal route,
baseline completion metadata, failure category, policy/static metadata, safe verification evidence,
and candidate evaluation metrics when available. It explicitly excludes source/context content,
model prompts and raw responses, proposal prose and diff, stdout/stderr, raw Docker/tool output,
hidden-test names or content, reference material, exception text, environment values, credentials,
secrets, and absolute paths. Publication is exclusive and atomic; an existing destination is never
overwritten.

Each run owns one private temporary root with separate preparation, candidate, and verification
areas. The trees remain available through routing and artifact projection, then the complete owned
root is removed on success, every terminal failure, and interruption. Cleanup uncertainty changes
the result to `infrastructure_failure`.

| Exit code | Outcome |
| ---: | --- |
| 0 | `success` |
| 1 | `candidate_failure` |
| 2 | CLI syntax/input error, including an unknown case |
| 3 | `policy_violation` |
| 4 | `static_check_failure` |
| 5 | `baseline_not_reproducible` |
| 6 | `infrastructure_failure`, cleanup failure, or artifact-publication failure |
| 7 | `malformed_model_output` |

`KeyboardInterrupt` retains the conventional process exit code `130` after best-effort cleanup.

## Validation

The canonical local and CI validation entry point is:

```text
python scripts/check.py
```

It runs Ruff formatting checks, Ruff linting, strict mypy checks, and pytest. The pytest suite
includes the applicable Docker integration and adversarial security tests. Local Docker tests skip
only when the Docker CLI or a usable Linux-container daemon is unavailable; CI builds both fixed
images first and executes the integration suite. A reachable daemon without seccomp support, or a
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

Candidate verification likewise never executes candidate or test code on the host. Visible stages
mount only the validated candidate. The hidden stage creates a fresh bundle containing only the
loader-declared hidden-test bytes under TraceFix-owned names and mounts it read-only at a fixed
path; it never mounts the evaluator directory, manifest, reference patch, benchmark root, project
root, home directory, credentials, or Docker socket. Candidate identity and content fingerprints
are checked around execution. Raw visible/hidden output, Docker output, exception text, and paths
do not cross the verification adapter. Timeout, OOM, integrity, Docker, protocol, evaluator, and
cleanup failures remain infrastructure diagnostics rather than functional repair failures.

Static validation never compiles or analyzes candidate Python on the host. Host code only copies,
hashes, compares, and transforms inert bytes in a fresh contained staging tree. The dedicated
static image receives a read-only candidate mount and normalized changed Python paths only, copies
the candidate into tmpfs, and invokes fixed tool argument sequences with no shell. Candidate Ruff,
mypy, and Bandit configuration, plugins, caches, baselines, inline Ruff/Bandit suppressions, and
mypy suppression directives cannot change the gate. Raw tool output and source text are discarded;
cross-module findings retain only bounded normalized path, location, and rule metadata.

Repository context, repair plans, provider responses, and generated diffs are untrusted data. Model
responses are byte-bounded before strict JSON parsing, diagnostics do not expose prompts, context,
raw responses, exception text, credentials, or host paths, and generated diffs remain inert. Patch
policy revalidates the proposal and policy, normalizes paths and line endings, reports every safely
detectable violation without retaining matched content, and uses no filesystem, environment,
network, process, Docker, or candidate-execution capability. The model boundary receives no
filesystem, command, Docker, credential, or network capability.

Graph routing revalidates every typed handoff and binds policy, static validation, verification, and
evaluation to the generated proposal identity. Baseline non-reproduction bypasses the model and all
candidate nodes; policy rejection bypasses patch application; static failure bypasses verification;
all well-formed verification outcomes are evaluated before routing. Artifacts serialize an explicit
allowlist rather than the graph state. Workspace cleanup validates the exact recorded owner identity
before recursively removing only the run-owned temporary root.

Linux is the production-supported host for adversarial repository preparation and execution.
Windows is development-supported, but native Windows does not claim complete host-side TOCTOU
protection during repository preparation. Running Linux containers through Docker Desktop does not
remove that Windows host preparation limitation.

The Docker daemon, project-owned sandbox image, host administrator, and container engine remain
trusted. The detailed assumptions, protected-path behavior, and future hardening boundaries are
documented in the [system architecture](docs/architecture/tracefix-system-architecture.md) and the
[sandbox hardening specification](docs/specs/0.1-003c-security-hardening.md).

## Benchmark / Evaluation Approach

The development benchmark contains five small, deterministic Python defects, all of which can now
run end to end through `tracefix repair <case_id>` with the configured fake provider:

- `boundary_condition`
- `exception_handling`
- `fixture_or_mocking`
- `incorrect_conditional`
- `incorrect_return_value`

Each case contains a failing repository snapshot, visible test metadata, hidden regression tests,
and a reference patch. Trusted loaders may access evaluator material; model-safe views and prepared
repositories may not. The current integration suite verifies that all five declared visible
failures reproduce inside the real sandbox while fixture sources remain unchanged.

This is deterministic platform and integration testing: it answers whether preparation,
classification, model-boundary handling, patch enforcement, isolation, verification, evaluation,
artifact publication, and cleanup behave correctly. It does not measure or claim live-model repair
capability. Candidate evaluation reparses the inert diff for deterministic files and added/removed
counts, sums only executed verification-stage duration, and preserves optional trusted upstream
model measurements without pricing or provider access. Future live-model evaluation will separately
measure repair capability. No live-model repair success rate is reported until that boundary exists.

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

- **Remaining v0.1:** add the Task 0.1.11a live-provider benchmark boundary while keeping it
  separate from deterministic fake-provider platform validation.
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
