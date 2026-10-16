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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TextIO

from pydantic import ValidationError

if TYPE_CHECKING:
    from app.agent.graph import Clock, GraphDependencies, OrchestrationState, RunWorkspaces
    from app.agent.model import ModelProvider
    from app.agent.schemas import ModelConfiguration
    from app.artifacts import ArtifactWriter
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


class TrustedCaseLoader(Protocol):
    def __call__(self, case_id: str) -> TrustedCase: ...


class ConfigurationLoader(Protocol):
    def __call__(self, environment: Mapping[str, str] | None = None) -> ModelConfiguration: ...


class ProviderFactory(Protocol):
    def __call__(self, configuration: ModelConfiguration) -> ModelProvider: ...


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
        from app.agent.graph import GraphDependencies, LocalWorkspaceIntegrityBoundary
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
        from app.agent.graph import (
            build_repair_graph,
            invoke_repair_graph,
            new_orchestration_state,
            replace_with_infrastructure_failure,
            terminalize_infrastructure_failure,
        )
        from app.agent.schemas import ModelConfiguration
        from app.artifacts import (
            ArtifactSummary,
            project_artifact,
        )

        dependencies = self._dependencies
        try:
            case = self._case(case_id, dependencies.case_loader)
        except CliInputError:
            raise
        except RunInitializationError:
            self._safe_error("tracefix: run initialization failed\n")
            return 6
        try:
            run_id = dependencies.run_id_factory()
            started = dependencies.clock()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self._safe_error("tracefix: run initialization failed\n")
            return 6
        if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
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

        workspaces: RunWorkspaces | None = None
        state: OrchestrationState | None = None
        projected = None
        try:
            workspaces = dependencies.workspace_manager.create(run_id)
            initial = new_orchestration_state(
                run_id=run_id,
                trusted_case=case,
                workspaces=workspaces,
                started_at=started,
                model_configuration=configuration,
            )
            state = initial
            try:
                graph_dependencies = dependencies.graph_dependencies_factory(
                    configuration, provider, dependencies.clock
                )
                state = invoke_repair_graph(build_repair_graph(graph_dependencies), initial)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                state = terminalize_infrastructure_failure(initial, dependencies.clock())
            projected = project_artifact(state)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if state is not None and state.terminal_route is not None:
                projected = None
            else:
                self._safe_error("tracefix: run initialization failed\n")
                return 6
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
            self._safe_error("tracefix: repair run failed safely\n")
            return 6
        try:
            invocation_value = dependencies.invocation_directory()
            if not isinstance(invocation_value, Path):
                raise ValueError("invocation directory has the wrong type")
            invocation = invocation_value.absolute()
            publication = dependencies.artifact_writer.write(projected, invocation)
            from app.artifacts import ArtifactPublication

            if type(publication) is not ArtifactPublication:
                raise ValueError("artifact writer returned the wrong type")
            publication = ArtifactPublication.model_validate(
                publication.model_dump(warnings="none"), strict=True
            )
            summary = ArtifactSummary(
                case_id=state.case_id,
                outcome=state.terminal_route.outcome,
                summary=(
                    state.terminal_route.diagnostic.message
                    if state.terminal_route.diagnostic is not None
                    else "repair candidate passed all required checks"
                ),
                exit_code=state.terminal_route.exit_code,
                artifact_path=publication.relative_path,
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
        return state.terminal_route.exit_code

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


ApplicationFactory = Callable[[], RepairApplication]


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
    return parser


def main(
    argv: Sequence[str] | None = None,
    application_factory: ApplicationFactory = default_application_factory,
) -> int:
    arguments = _parser().parse_args(None if argv is None else list(argv))
    try:
        _enable_editable_project_imports()
        application = application_factory()
        if arguments.command != "repair" or type(arguments.case_id) is not str:
            raise CliInputError("command is invalid")
        return application.repair(arguments.case_id)
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
