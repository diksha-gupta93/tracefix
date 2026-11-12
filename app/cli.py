from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TextIO

from pydantic import ValidationError

from app.observability import NOOP_TRACER, DiagnosticTracer, safe_metadata

if TYPE_CHECKING:
    from app.agent.graph import Clock, GraphDependencies, OrchestrationState, RunWorkspaces
    from app.agent.model import ModelProvider
    from app.agent.schemas import ModelConfiguration
    from app.artifacts import ArtifactWriter
    from app.evaluation.benchmark import BenchmarkReport
    from app.evaluation.live_model import LiveBenchmarkConfiguration
    from app.evaluation.reporting import BenchmarkReportPublication
    from benchmarks.loader import TrustedCase

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_REPARSE_POINT = 0x400


class CliInputError(Exception):
    """Command input was invalid before repair side effects began."""


class WorkspaceError(Exception):
    """The graph-owned workspace lifecycle could not be proven safe."""


class RunInitializationError(Exception):
    """A known-case run could not be initialized safely."""


class RepairExecutionError(RuntimeError):
    """A repair case failed without a trustworthy terminal artifact."""


class RepairArtifactPublicationError(RepairExecutionError):
    """A repair completed but its safe artifact could not be published."""


class TrustedCaseLoader(Protocol):
    def __call__(self, case_id: str) -> TrustedCase: ...


class ConfigurationLoader(Protocol):
    def __call__(self, environment: Mapping[str, str] | None = None) -> ModelConfiguration: ...


class ProviderFactory(Protocol):
    def __call__(self, configuration: ModelConfiguration) -> ModelProvider: ...


class DevelopmentCaseLoader(Protocol):
    def __call__(self) -> tuple[TrustedCase, ...]: ...


class LiveConfigurationLoader(Protocol):
    def __call__(self, environment: Mapping[str, str]) -> LiveBenchmarkConfiguration: ...


class LiveProviderFactory(Protocol):
    def __call__(
        self,
        configuration: ModelConfiguration,
        live_configuration: LiveBenchmarkConfiguration,
        environment: Mapping[str, str],
    ) -> ModelProvider: ...


class BenchmarkReportWriter(Protocol):
    def write(
        self, report: BenchmarkReport, invocation_directory: Path
    ) -> BenchmarkReportPublication: ...

    def discard(
        self, publication: BenchmarkReportPublication, invocation_directory: Path
    ) -> None: ...


class ProviderRegistry:
    def __init__(self, factories: Mapping[str, ProviderFactory] | None = None) -> None:
        source = {} if factories is None else dict(factories)
        if any(type(name) is not str or not name for name in source):
            raise ValueError("provider registry key is invalid")
        self._factories: Mapping[str, ProviderFactory] = MappingProxyType(source)

    def resolve(self, configuration: ModelConfiguration) -> ModelProvider | None:
        from app.agent.schemas import ModelConfiguration

        if type(configuration) is not ModelConfiguration:
            return None
        factory = self._factories.get(configuration.provider)
        if factory is None:
            return None
        try:
            provider = factory(configuration)
        except Exception:
            return None
        return provider


class WorkspaceManager(Protocol):
    def create(self, run_id: str) -> RunWorkspaces: ...

    def cleanup(self, workspaces: RunWorkspaces) -> None: ...


@dataclass(frozen=True, slots=True)
class _OwnedWorkspace:
    workspaces: RunWorkspaces
    device: int
    inode: int
    mode: int


def _workspace_metadata(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceError("workspace identity validation failed") from None
    if (
        path != resolved
        or not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    ):
        raise WorkspaceError("workspace identity validation failed")
    return metadata


class LocalWorkspaceManager:
    def __init__(self, base_directory: Path | None = None) -> None:
        self._base_directory = base_directory
        self._owned: dict[str, _OwnedWorkspace] = {}

    def create(self, run_id: str) -> RunWorkspaces:
        from app.agent.graph import RunWorkspaces

        if _RUN_ID.fullmatch(run_id) is None or run_id in self._owned:
            raise WorkspaceError("workspace run identity is invalid")
        base: Path | None = None
        root_value: str | None = None
        root_identity: tuple[int, int, int] | None = None
        try:
            if self._base_directory is not None:
                base = self._base_directory.absolute()
                _workspace_metadata(base)
            root_value = tempfile.mkdtemp(
                prefix=f"tracefix-run-{run_id}-",
                dir=None if base is None else os.fspath(base),
            )
            root = Path(root_value).absolute()
            metadata = _workspace_metadata(root)
            root_identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
            with suppress(OSError):
                root.chmod(0o700)
            metadata = _workspace_metadata(root)
            root_identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
            preparation = root / "preparation"
            candidate = root / "candidate"
            verification = root / "verification"
            for child in (preparation, candidate, verification):
                child.mkdir(mode=0o700)
                _workspace_metadata(child)
            workspaces = RunWorkspaces(
                root=root,
                preparation=preparation,
                candidate=candidate,
                verification=verification,
                ownership_token=run_id,
            )
            self._owned[run_id] = _OwnedWorkspace(
                workspaces=workspaces,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                mode=metadata.st_mode,
            )
            return workspaces
        except (OSError, RuntimeError, ValueError, ValidationError, WorkspaceError):
            if root_value is not None and root_identity is not None:
                root = Path(root_value).absolute()
                with suppress(OSError, RuntimeError, WorkspaceError):
                    current = _workspace_metadata(root)
                    if (
                        current.st_dev,
                        current.st_ino,
                        current.st_mode,
                    ) == root_identity and root.name.startswith(f"tracefix-run-{run_id}-"):
                        shutil.rmtree(root)
            raise WorkspaceError("workspace creation failed") from None

    def cleanup(self, workspaces: RunWorkspaces) -> None:
        from app.agent.graph import RunWorkspaces

        if type(workspaces) is not RunWorkspaces:
            raise WorkspaceError("workspace cleanup target is invalid")
        owned = self._owned.get(workspaces.ownership_token)
        if owned is None or owned.workspaces != workspaces:
            raise WorkspaceError("workspace cleanup target is not owned")
        root = workspaces.root
        metadata = _workspace_metadata(root)
        if (
            metadata.st_dev != owned.device
            or metadata.st_ino != owned.inode
            or metadata.st_mode != owned.mode
            or not root.name.startswith(f"tracefix-run-{workspaces.ownership_token}-")
            or any(
                child.parent != root
                for child in (
                    workspaces.preparation,
                    workspaces.candidate,
                    workspaces.verification,
                )
            )
        ):
            raise WorkspaceError("workspace identity changed before cleanup")
        try:
            for child in (
                workspaces.preparation,
                workspaces.candidate,
                workspaces.verification,
            ):
                child_metadata = _workspace_metadata(child)
                if child.parent != root or child_metadata.st_dev != metadata.st_dev:
                    raise WorkspaceError("workspace child identity changed before cleanup")
            metadata = _workspace_metadata(root)
            if (
                metadata.st_dev != owned.device
                or metadata.st_ino != owned.inode
                or metadata.st_mode != owned.mode
            ):
                raise WorkspaceError("workspace identity changed before cleanup")
            shutil.rmtree(root)
            if root.exists() or root.is_symlink():
                raise OSError("workspace still exists")
        except (OSError, RuntimeError, WorkspaceError):
            raise WorkspaceError("workspace cleanup failed") from None
        del self._owned[workspaces.ownership_token]


class GraphDependenciesFactory(Protocol):
    def __call__(
        self,
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock: Clock,
    ) -> GraphDependencies: ...


class DefaultGraphDependenciesFactory:
    def __call__(
        self,
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        clock: Clock,
    ) -> GraphDependencies:
        from app.agent.baseline import BaselineExecutor
        from app.agent.context import ContextSelector
        from app.agent.graph import (
            GraphDependencies,
            LocalWorkspaceIntegrityBoundary,
            ModelUsageReader,
        )
        from app.agent.preparation import LocalPreparationFilesystem, RepositoryPreparer
        from app.agent.static_checks import (
            LocalStaticFilesystem,
            Sha256RepositoryFingerprinter,
            StaticValidationService,
        )
        from app.agent.verification import (
            LocalCandidateFilesystem,
            LocalEvaluatorBundleFilesystem,
            VerificationService,
        )
        from app.sandbox.runner import SandboxRunner, SubprocessDockerCommandAdapter
        from app.sandbox.static_runner import DockerStaticRunner
        from app.sandbox.verification_runner import VerificationSandboxRunner
        from benchmarks.loader import load_trusted_case

        docker = SubprocessDockerCommandAdapter()
        preparation_filesystem = LocalPreparationFilesystem()
        preparer = RepositoryPreparer(load_trusted_case, filesystem=preparation_filesystem)
        baseline = BaselineExecutor(load_trusted_case, SandboxRunner(docker))
        context_selector = ContextSelector(load_trusted_case)
        static_filesystem = LocalStaticFilesystem()
        fingerprinter = Sha256RepositoryFingerprinter(static_filesystem)
        static_validation = StaticValidationService(
            filesystem=static_filesystem,
            fingerprinter=fingerprinter,
            runner=DockerStaticRunner(docker),
        )
        verification = VerificationService(
            loader=load_trusted_case,
            fingerprinter=fingerprinter,
            candidate_filesystem=LocalCandidateFilesystem(),
            bundle_filesystem=LocalEvaluatorBundleFilesystem(),
            sandbox=VerificationSandboxRunner(docker),
            identifier_factory=lambda: uuid.uuid4().hex,
        )
        return GraphDependencies(
            preparer=preparer,
            baseline=baseline,
            context_selector=context_selector,
            static_validation=static_validation,
            verification=verification,
            workspace_integrity=LocalWorkspaceIntegrityBoundary(fingerprinter),
            clock=clock,
            model_configuration=configuration,
            model_provider=provider,
            model_usage_reader=(provider if isinstance(provider, ModelUsageReader) else None),
        )


@dataclass(frozen=True, slots=True)
class ApplicationDependencies:
    case_loader: TrustedCaseLoader
    configuration_loader: ConfigurationLoader
    provider_registry: ProviderRegistry
    workspace_manager: WorkspaceManager
    artifact_writer: ArtifactWriter
    graph_dependencies_factory: GraphDependenciesFactory
    clock: Clock
    run_id_factory: Callable[[], str]
    invocation_directory: Callable[[], Path]
    environment: Mapping[str, str]
    stdout: TextIO
    stderr: TextIO


@dataclass(frozen=True, slots=True)
class RepairRunExecution:
    state: OrchestrationState
    artifact_path: PurePosixPath


class RepairApplication:
    def __init__(self, dependencies: ApplicationDependencies) -> None:
        self._dependencies = dependencies

    @staticmethod
    def _case(case_id: str, loader: TrustedCaseLoader) -> TrustedCase:
        from benchmarks.loader import BenchmarkLoadError, TrustedCase

        if type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None:
            raise CliInputError("case_id is invalid")
        try:
            raw = loader(case_id)
        except BenchmarkLoadError:
            raise CliInputError("case_id is unknown") from None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RunInitializationError("trusted case loading failed") from None
        try:
            if type(raw) is not TrustedCase:
                raise ValueError
            case = TrustedCase.model_validate(raw.model_dump(warnings="none"), strict=True)
            if case.manifest.case_id != case_id:
                raise ValueError
            return case
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise RunInitializationError("trusted case validation failed") from None

    def repair(self, case_id: str) -> int:
        from app.agent.schemas import ModelConfiguration
        from app.artifacts import ArtifactSummary

        dependencies = self._dependencies
        try:
            case = self._case(case_id, dependencies.case_loader)
        except CliInputError:
            raise
        except RunInitializationError:
            self._safe_error("tracefix: run initialization failed\n")
            return 6
        configuration: ModelConfiguration | None = None
        provider: ModelProvider | None = None
        try:
            configuration = dependencies.configuration_loader(dependencies.environment)
            configuration = ModelConfiguration.model_validate(
                configuration.model_dump(warnings="none"), strict=True
            )
            provider = dependencies.provider_registry.resolve(configuration)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            configuration = None
            provider = None
        try:
            execution = self.execute_trusted_case(case, configuration, provider)
        except (KeyboardInterrupt, SystemExit):
            raise
        except RunInitializationError:
            self._safe_error("tracefix: run initialization failed\n")
            return 6
        except RepairArtifactPublicationError:
            self._safe_error("tracefix: artifact publication failed\n")
            return 6
        except RepairExecutionError:
            self._safe_error("tracefix: repair run failed safely\n")
            return 6
        try:
            route = execution.state.terminal_route
            if route is None:
                raise ValueError
            summary = ArtifactSummary(
                case_id=execution.state.case_id,
                outcome=route.outcome,
                summary=(
                    route.diagnostic.message
                    if route.diagnostic is not None
                    else "repair candidate passed all required checks"
                ),
                exit_code=route.exit_code,
                artifact_path=execution.artifact_path,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: artifact publication failed\n")
            return 6
        try:
            dependencies.stdout.write(f"{summary.render()}\n")
            dependencies.stdout.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: console output failed\n")
            return 6
        return route.exit_code

    def execute_trusted_case(
        self,
        case: TrustedCase,
        configuration: ModelConfiguration | None,
        provider: ModelProvider | None,
        *,
        tracer: DiagnosticTracer = NOOP_TRACER,
    ) -> RepairRunExecution:
        from app.agent.graph import (
            build_repair_graph,
            invoke_repair_graph,
            new_orchestration_state,
            replace_with_infrastructure_failure,
            terminalize_infrastructure_failure,
        )
        from app.artifacts import ArtifactPublication, project_artifact
        from benchmarks.loader import TrustedCase

        dependencies = self._dependencies
        try:
            if type(case) is not TrustedCase:
                raise ValueError
            checked_case = TrustedCase.model_validate(case.model_dump(warnings="none"), strict=True)
            run_id = dependencies.run_id_factory()
            started = dependencies.clock()
            if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
                raise ValueError
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RunInitializationError("repair run initialization failed") from None

        workspaces: RunWorkspaces | None = None
        state: OrchestrationState | None = None
        projected = None
        try:
            workspaces = dependencies.workspace_manager.create(run_id)
            initial = new_orchestration_state(
                run_id=run_id,
                trusted_case=checked_case,
                workspaces=workspaces,
                started_at=started,
                model_configuration=configuration,
            )
            state = initial
            try:
                graph_dependencies = dependencies.graph_dependencies_factory(
                    configuration, provider, dependencies.clock
                )
                graph_dependencies = replace(graph_dependencies, tracer=tracer)
                state = invoke_repair_graph(build_repair_graph(graph_dependencies), initial)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                state = terminalize_infrastructure_failure(initial, dependencies.clock())
            projected = project_artifact(state)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            projected = None
        finally:
            if workspaces is not None:
                try:
                    dependencies.workspace_manager.cleanup(workspaces)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    if state is not None and state.terminal_route is not None:
                        try:
                            state = replace_with_infrastructure_failure(state, dependencies.clock())
                            projected = project_artifact(state)
                        except Exception:
                            projected = None

        if state is None or state.terminal_route is None or projected is None:
            raise RepairExecutionError("repair run failed safely")
        try:
            invocation_value = dependencies.invocation_directory()
            if not isinstance(invocation_value, Path):
                raise ValueError
            publication = dependencies.artifact_writer.write(projected, invocation_value.absolute())
            if type(publication) is not ArtifactPublication:
                raise ValueError
            checked_publication = ArtifactPublication.model_validate(
                publication.model_dump(warnings="none"), strict=True
            )
            return RepairRunExecution(
                state=state,
                artifact_path=checked_publication.relative_path,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RepairArtifactPublicationError("repair artifact publication failed") from None

    def _safe_error(self, message: str) -> None:
        with suppress(Exception):
            self._dependencies.stderr.write(message)
            self._dependencies.stderr.flush()


@dataclass(frozen=True, slots=True)
class BenchmarkApplicationDependencies:
    repair_application: RepairApplication
    development_case_loader: DevelopmentCaseLoader
    configuration_loader: ConfigurationLoader
    live_configuration_loader: LiveConfigurationLoader
    live_provider_factory: LiveProviderFactory
    report_writer: BenchmarkReportWriter
    clock: Clock
    run_id_factory: Callable[[], str]
    invocation_directory: Callable[[], Path]
    environment: Mapping[str, str]
    stdout: TextIO
    stderr: TextIO
    tracer: DiagnosticTracer = NOOP_TRACER


class BenchmarkApplication:
    def __init__(self, dependencies: BenchmarkApplicationDependencies) -> None:
        self._dependencies = dependencies

    def benchmark(self, split: str, provider_argument: str) -> int:
        from app.agent.schemas import ModelConfiguration
        from app.evaluation.benchmark import (
            BenchmarkCaseExecution,
            BenchmarkExecutionStatus,
            build_benchmark_report,
            run_benchmark_cases,
        )
        from app.evaluation.live_model import (
            GROQ_PROVIDER_NAME,
            apply_live_model_configuration,
        )
        from app.evaluation.reporting import (
            BenchmarkReportPublication,
            BenchmarkSummary,
        )

        dependencies = self._dependencies
        if split != "development" or provider_argument != GROQ_PROVIDER_NAME:
            raise CliInputError("benchmark arguments are invalid")
        try:
            benchmark_run_id = dependencies.run_id_factory()
            started_at = dependencies.clock()
            if type(benchmark_run_id) is not str or _RUN_ID.fullmatch(benchmark_run_id) is None:
                raise ValueError
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: benchmark preflight failed: benchmark_run_id\n")
            return 6
        try:
            base_configuration = dependencies.configuration_loader(dependencies.environment)
            base_configuration = ModelConfiguration.model_validate(
                base_configuration.model_dump(warnings="none"), strict=True
            )
            configuration = apply_live_model_configuration(
                base_configuration, provider_argument, dependencies.environment
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: benchmark preflight failed: model_configuration\n")
            return 6
        try:
            live_configuration = dependencies.live_configuration_loader(dependencies.environment)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: benchmark preflight failed: live_configuration\n")
            return 6
        try:
            first_provider = dependencies.live_provider_factory(
                configuration, live_configuration, dependencies.environment
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: benchmark preflight failed: provider_initialization\n")
            return 6
        try:
            cases = dependencies.development_case_loader()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: benchmark preflight failed: benchmark_cases\n")
            return 6

        providers: list[ModelProvider] = [first_provider]
        with dependencies.tracer.span(
            "tracefix.live_benchmark",
            safe_metadata(
                benchmark_run_id=benchmark_run_id,
                provider_name=configuration.provider,
                model_name=configuration.model_name,
                prompt_version=configuration.prompt_version,
            ),
        ) as benchmark_span:

            def execute_case(case: TrustedCase) -> BenchmarkCaseExecution:
                with dependencies.tracer.span(
                    f"case.{case.manifest.case_id}",
                    safe_metadata(
                        benchmark_run_id=benchmark_run_id,
                        case_id=case.manifest.case_id,
                        provider_name=configuration.provider,
                        model_name=configuration.model_name,
                        prompt_version=configuration.prompt_version,
                        attempt_number=1,
                    ),
                ) as case_span:
                    try:
                        model_provider = (
                            providers.pop()
                            if providers
                            else dependencies.live_provider_factory(
                                configuration, live_configuration, dependencies.environment
                            )
                        )
                        execution = dependencies.repair_application.execute_trusted_case(
                            case,
                            configuration,
                            model_provider,
                            tracer=dependencies.tracer,
                        )
                        route = execution.state.terminal_route
                        if route is None:
                            raise RepairExecutionError("repair run did not terminate")
                        classification = (
                            "repair_success"
                            if route.outcome.value == "success"
                            else (
                                "execution_failure"
                                if route.outcome.value
                                in {
                                    "infrastructure_failure",
                                    "baseline_not_reproducible",
                                }
                                else "repair_failure"
                            )
                        )
                        case_span.add_metadata(
                            safe_metadata(
                                terminal_repair_status=route.outcome.value,
                                terminal_diagnostic_code=(
                                    route.outcome.value
                                    if route.diagnostic is None
                                    else route.diagnostic.code.value
                                ),
                                benchmark_classification=classification,
                            )
                        )
                        return BenchmarkCaseExecution(
                            case_id=execution.state.case_id,
                            terminal_outcome=route.outcome,
                            model_usage=execution.state.model_usage,
                            artifact_path=execution.artifact_path,
                        )
                    except (KeyboardInterrupt, SystemExit):
                        case_span.add_metadata(
                            safe_metadata(terminal_diagnostic_code="interrupted")
                        )
                        raise
                    except Exception:
                        case_span.add_metadata(
                            safe_metadata(
                                terminal_diagnostic_code="execution_failure",
                                benchmark_classification="execution_failure",
                            )
                        )
                        raise

            try:
                case_results = run_benchmark_cases(cases, execute_case)
                completed_at = dependencies.clock()
                report = build_benchmark_report(
                    benchmark_run_id=benchmark_run_id,
                    model=configuration.model_name,
                    prompt_version=configuration.prompt_version,
                    started_at=started_at,
                    completed_at=completed_at,
                    live_configuration=live_configuration,
                    cases=case_results,
                )
                invocation_value = dependencies.invocation_directory()
                if not isinstance(invocation_value, Path):
                    raise ValueError
                publication = dependencies.report_writer.write(report, invocation_value.absolute())
                if type(publication) is not BenchmarkReportPublication:
                    raise ValueError
                checked_publication = BenchmarkReportPublication.model_validate(
                    publication.model_dump(warnings="none"), strict=True
                )
                exit_code = (
                    0 if report.execution_status is BenchmarkExecutionStatus.COMPLETED else 6
                )
                summary = BenchmarkSummary(
                    report=report,
                    exit_code=exit_code,
                    publication=checked_publication,
                )
                benchmark_span.add_metadata(
                    safe_metadata(execution_status=report.execution_status.value)
                )
            except (KeyboardInterrupt, SystemExit):
                benchmark_span.add_metadata(safe_metadata(terminal_diagnostic_code="interrupted"))
                raise
            except Exception:
                benchmark_span.add_metadata(
                    safe_metadata(
                        execution_status="failed",
                        terminal_diagnostic_code="benchmark_execution_failure",
                    )
                )
                self._safe_error("tracefix: benchmark execution failed safely\n")
                return 6
            try:
                dependencies.stdout.write(f"{summary.render()}\n")
                dependencies.stdout.flush()
            except (KeyboardInterrupt, SystemExit):
                with suppress(Exception):
                    dependencies.report_writer.discard(
                        checked_publication, invocation_value.absolute()
                    )
                raise
            except Exception:
                with suppress(Exception):
                    dependencies.report_writer.discard(
                        checked_publication, invocation_value.absolute()
                    )
                self._safe_error("tracefix: benchmark console output failed\n")
                return 6
            return exit_code

    def _safe_error(self, message: str) -> None:
        with suppress(Exception):
            self._dependencies.stderr.write(message)
            self._dependencies.stderr.flush()


def _clock() -> datetime:
    return datetime.now(UTC)


def default_application_factory() -> RepairApplication:
    from app.agent.model import load_model_configuration
    from app.artifacts import LocalArtifactWriter
    from app.evaluation.deterministic.fake_model import (
        FAKE_PROVIDER_NAME,
        create_fake_model_provider,
    )
    from benchmarks.loader import load_trusted_case

    return RepairApplication(
        ApplicationDependencies(
            case_loader=load_trusted_case,
            configuration_loader=load_model_configuration,
            provider_registry=ProviderRegistry({FAKE_PROVIDER_NAME: create_fake_model_provider}),
            workspace_manager=LocalWorkspaceManager(),
            artifact_writer=LocalArtifactWriter(),
            graph_dependencies_factory=DefaultGraphDependenciesFactory(),
            clock=_clock,
            run_id_factory=lambda: uuid.uuid4().hex,
            invocation_directory=Path.cwd,
            environment=os.environ,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    )


def default_benchmark_application_factory() -> BenchmarkApplication:
    from app.agent.model import load_model_configuration
    from app.evaluation.live_model import (
        GroqRequestPacer,
        create_groq_model_provider,
        load_live_benchmark_configuration,
    )
    from app.evaluation.reporting import LocalBenchmarkReportWriter
    from app.observability import create_diagnostic_tracer
    from benchmarks.loader import load_development_cases

    tracer = create_diagnostic_tracer(os.environ)
    request_pacer = GroqRequestPacer()

    def live_provider_factory(
        configuration: ModelConfiguration,
        live_configuration: LiveBenchmarkConfiguration,
        environment: Mapping[str, str],
    ) -> ModelProvider:
        return create_groq_model_provider(
            configuration,
            live_configuration,
            environment,
            request_pacer=request_pacer,
            tracer=tracer,
        )

    return BenchmarkApplication(
        BenchmarkApplicationDependencies(
            repair_application=default_application_factory(),
            development_case_loader=load_development_cases,
            configuration_loader=load_model_configuration,
            live_configuration_loader=load_live_benchmark_configuration,
            live_provider_factory=live_provider_factory,
            report_writer=LocalBenchmarkReportWriter(),
            clock=_clock,
            run_id_factory=lambda: uuid.uuid4().hex,
            invocation_directory=Path.cwd,
            environment=os.environ,
            stdout=sys.stdout,
            stderr=sys.stderr,
            tracer=tracer,
        )
    )


ApplicationFactory = Callable[[], RepairApplication]
BenchmarkApplicationFactory = Callable[[], BenchmarkApplication]


def _enable_editable_project_imports() -> None:
    source_root = Path(__file__).resolve().parents[1]
    benchmark_package = source_root / "benchmarks" / "__init__.py"
    rendered = os.fspath(source_root)
    if benchmark_package.is_file() and rendered not in sys.path:
        sys.path.insert(0, rendered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracefix")
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser("repair")
    repair.add_argument("case_id")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--split", required=True, choices=("development",))
    benchmark.add_argument("--provider", required=True, choices=("groq",))
    return parser


def main(
    argv: Sequence[str] | None = None,
    application_factory: ApplicationFactory = default_application_factory,
    benchmark_application_factory: BenchmarkApplicationFactory = (
        default_benchmark_application_factory
    ),
) -> int:
    arguments = _parser().parse_args(None if argv is None else list(argv))
    try:
        _enable_editable_project_imports()
        if arguments.command == "repair" and type(arguments.case_id) is str:
            return application_factory().repair(arguments.case_id)
        if (
            arguments.command == "benchmark"
            and type(arguments.split) is str
            and type(arguments.provider) is str
        ):
            return benchmark_application_factory().benchmark(arguments.split, arguments.provider)
        raise CliInputError("command is invalid")
    except CliInputError:
        with suppress(Exception):
            sys.stderr.write("tracefix: error: invalid or unknown case_id\n")
            sys.stderr.flush()
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception:
        with suppress(Exception):
            sys.stderr.write("tracefix: repair run failed safely\n")
            sys.stderr.flush()
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
