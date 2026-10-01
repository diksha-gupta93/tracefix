from __future__ import annotations

import ast
import re
import tokenize
from enum import StrEnum
from io import StringIO
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agent.patches import (
    DiffLineKind,
    ParsedDiffFile,
    ParsedDiffLine,
    ParsedUnifiedDiff,
    UnifiedDiffError,
    normalize_repository_paths,
    parse_binary_diff_paths,
    parse_unified_diff,
)
from app.agent.schemas import DEFAULT_PROTECTED_PATHS, PatchProposal, ProtectedPathPolicy

MAX_CHANGED_FILES = 3
MAX_CHANGED_LINES = 200


class PatchPolicyViolationCode(StrEnum):
    PROTECTED_PATH_MODIFIED = "protected_path_modified"
    TOO_MANY_FILES = "too_many_files"
    LINE_CHANGE_LIMIT_EXCEEDED = "line_change_limit_exceeded"
    BINARY_CONTENT = "binary_content"
    SECRET_DETECTED = "secret_detected"
    TEST_DISABLED = "test_disabled"
    ASSERTION_REMOVED = "assertion_removed"
    CI_OR_SECURITY_CONFIG_MODIFIED = "ci_or_security_config_modified"
    UNRESTRICTED_SUBPROCESS_INTRODUCED = "unrestricted_subprocess_introduced"
    UNAPPROVED_DEPENDENCY_ADDED = "unapproved_dependency_added"
    HIDDEN_TEST_MODIFIED = "hidden_test_modified"
    BENCHMARK_METADATA_MODIFIED = "benchmark_metadata_modified"
    REFERENCE_PATCH_MODIFIED = "reference_patch_modified"


_VIOLATION_MESSAGES: dict[PatchPolicyViolationCode, str] = {
    PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED: "protected path modification is not allowed",
    PatchPolicyViolationCode.TOO_MANY_FILES: "patch changes too many files",
    PatchPolicyViolationCode.LINE_CHANGE_LIMIT_EXCEEDED: "patch exceeds the changed-line limit",
    PatchPolicyViolationCode.BINARY_CONTENT: "binary patch content is not allowed",
    PatchPolicyViolationCode.SECRET_DETECTED: "credential or private-key material is not allowed",
    PatchPolicyViolationCode.TEST_DISABLED: "test deletion or disabling is not allowed",
    PatchPolicyViolationCode.ASSERTION_REMOVED: "assertion removal is not allowed",
    PatchPolicyViolationCode.CI_OR_SECURITY_CONFIG_MODIFIED: (
        "CI or security configuration modification is not allowed"
    ),
    PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED: (
        "direct process or shell execution is not allowed"
    ),
    PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED: (
        "dependency addition is not allowlisted"
    ),
    PatchPolicyViolationCode.HIDDEN_TEST_MODIFIED: "hidden-test material modification is not allowed",
    PatchPolicyViolationCode.BENCHMARK_METADATA_MODIFIED: (
        "benchmark metadata modification is not allowed"
    ),
    PatchPolicyViolationCode.REFERENCE_PATCH_MODIFIED: (
        "reference patch modification is not allowed"
    ),
}
_RULE_ORDER = {code: index for index, code in enumerate(PatchPolicyViolationCode, start=1)}


def _validate_relative_path(path: PurePosixPath) -> None:
    value = path.as_posix()
    windows = PureWindowsPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or windows.drive
        or windows.root
        or "\\" in value
        or value != value.strip()
        or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("path must be a normalized relative POSIX path")


def _normalized_package_name(value: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", value) is None
    ):
        raise ValueError("dependency allowlist name is invalid")
    return re.sub(r"[-_.]+", "-", value).casefold()


class PatchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protected_path_policy: ProtectedPathPolicy
    forbidden_changed_files: tuple[PurePosixPath, ...] = ()
    max_changed_files: Literal[3] = 3
    max_changed_lines: Literal[200] = 200
    dependency_allowlist: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_policy(self) -> PatchPolicy:
        paths = tuple(path.as_posix() for path in self.forbidden_changed_files)
        for path in self.forbidden_changed_files:
            _validate_relative_path(path)
        if paths != tuple(sorted(paths, key=str.casefold)) or len(
            {path.casefold() for path in paths}
        ) != len(paths):
            raise ValueError("forbidden paths must be unique and deterministically ordered")
        normalized = tuple(_normalized_package_name(name) for name in self.dependency_allowlist)
        if normalized != self.dependency_allowlist or normalized != tuple(sorted(normalized)):
            raise ValueError(
                "dependency allowlist must be normalized and deterministically ordered"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("dependency allowlist must not contain duplicates")
        if self.protected_path_policy.protected_paths != DEFAULT_PROTECTED_PATHS:
            raise ValueError("protected-path policy must contain the architecture defaults")
        return self


class PatchPolicyViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: PatchPolicyViolationCode
    message: str
    path: PurePosixPath | None = None
    line: Annotated[int, Field(strict=True, ge=1)] | None = None

    @model_validator(mode="after")
    def validate_safe_constant_content(self) -> PatchPolicyViolation:
        if self.message != _VIOLATION_MESSAGES[self.code]:
            raise ValueError("violation message must be the safe constant for its code")
        if self.path is not None:
            _validate_relative_path(self.path)
        if self.line is not None and self.path is None:
            raise ValueError("violation line metadata requires a path")
        return self


def _violation_sort_key(violation: PatchPolicyViolation) -> tuple[int, str, str, int, str]:
    path = "" if violation.path is None else violation.path.as_posix()
    return (
        _RULE_ORDER[violation.code],
        path.casefold(),
        path,
        0 if violation.line is None else violation.line,
        violation.message,
    )


class PatchPolicyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    approved: Annotated[bool, Field(strict=True)]
    violations: tuple[PatchPolicyViolation, ...]

    @model_validator(mode="after")
    def validate_result(self) -> PatchPolicyResult:
        if self.approved != (not self.violations):
            raise ValueError("approval must agree with violations")
        if self.violations != tuple(sorted(self.violations, key=_violation_sort_key)):
            raise ValueError("violations must be deterministically ordered")
        keys = tuple(_violation_sort_key(item) for item in self.violations)
        if len(set(keys)) != len(keys):
            raise ValueError("violations must not contain duplicates")
        return self


class PatchPolicyInputDiagnosticCode(StrEnum):
    INVALID_PROPOSAL = "invalid_proposal"
    INVALID_POLICY = "invalid_policy"
    INVALID_PATH = "invalid_path"
    INVALID_DIFF = "invalid_diff"
    INCONSISTENT_PATHS = "inconsistent_paths"


_INPUT_MESSAGES: dict[PatchPolicyInputDiagnosticCode, str] = {
    PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL: "patch proposal input is invalid",
    PatchPolicyInputDiagnosticCode.INVALID_POLICY: "patch policy input is invalid",
    PatchPolicyInputDiagnosticCode.INVALID_PATH: "patch path input is invalid",
    PatchPolicyInputDiagnosticCode.INVALID_DIFF: "patch diff input is invalid",
    PatchPolicyInputDiagnosticCode.INCONSISTENT_PATHS: "patch paths are inconsistent",
}


class PatchPolicyInputDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    code: PatchPolicyInputDiagnosticCode
    message: str

    @model_validator(mode="after")
    def validate_safe_constant_message(self) -> PatchPolicyInputDiagnostic:
        if self.message != _INPUT_MESSAGES[self.code]:
            raise ValueError("input diagnostic message must be the safe constant for its code")
        return self


class PatchPolicyInputError(Exception):
    def __init__(self, diagnostic: PatchPolicyInputDiagnostic) -> None:
        super().__init__(f"patch_policy_input:{diagnostic.code.value}")
        self.diagnostic = diagnostic


def _raise_input(code: PatchPolicyInputDiagnosticCode) -> NoReturn:
    raise PatchPolicyInputError(
        PatchPolicyInputDiagnostic(code=code, message=_INPUT_MESSAGES[code])
    ) from None


def _make_violation(
    code: PatchPolicyViolationCode,
    *,
    path: str | None = None,
    line: int | None = None,
) -> PatchPolicyViolation:
    return PatchPolicyViolation(
        code=code,
        message=_VIOLATION_MESSAGES[code],
        path=None if path is None else PurePosixPath(path),
        line=line,
    )


def _path_is_equal_or_below(path: str, protected: PurePosixPath) -> bool:
    path_parts = tuple(part.casefold() for part in PurePosixPath(path).parts)
    protected_parts = tuple(part.casefold() for part in protected.parts)
    return path_parts[: len(protected_parts)] == protected_parts


def _has_prefix(path: str, *parts: str) -> bool:
    actual = tuple(part.casefold() for part in PurePosixPath(path).parts)
    expected = tuple(part.casefold() for part in parts)
    return actual[: len(expected)] == expected


def _is_ci_or_security_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    parts = tuple(part.casefold() for part in candidate.parts)
    name = candidate.name.casefold()
    if any(
        _has_prefix(path, *prefix)
        for prefix in (
            (".github", "workflows"),
            (".github", "actions"),
            (".circleci",),
            ("docker",),
            ("security",),
        )
    ):
        return True
    if name in {
        ".gitlab-ci.yml",
        "azure-pipelines.yml",
        "jenkinsfile",
        "dockerfile",
        ".bandit",
        "bandit.yml",
        "bandit.yaml",
        "bandit.toml",
        "security.md",
    }:
        return True
    return (
        name in {"seccomp.json", "seccomp.yml", "seccomp.yaml", "seccomp.profile"}
        or (name.startswith("seccomp-") or name.startswith("seccomp."))
        or "security" in parts
    )


def _is_hidden_test_path(path: str) -> bool:
    parts = tuple(part.casefold().replace("-", "_") for part in PurePosixPath(path).parts)
    return any(part in {"evaluator", "hidden_tests", "hiddentests"} for part in parts) or any(
        left == "hidden" and right == "tests" for left, right in zip(parts, parts[1:], strict=False)
    )


def _is_benchmark_metadata_path(path: str) -> bool:
    parts = tuple(part.casefold() for part in PurePosixPath(path).parts)
    if "benchmarks" not in parts:
        return False
    benchmark_index = parts.index("benchmarks")
    if PurePosixPath(path).name.casefold() == "manifest.json":
        return True
    return "repository" not in parts[benchmark_index + 1 :]


def _is_reference_patch_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    name = candidate.name.casefold()
    if name == "reference.patch":
        return True
    stem = candidate.stem.casefold().replace("-", "_")
    parts = {part.casefold() for part in candidate.parts}
    return (
        "evaluator" in parts
        and stem.startswith("reference_patch")
        and candidate.suffix.casefold()
        in {
            ".patch",
            ".diff",
        }
    )


_BINARY_MARKER = re.compile(r"(?m)^(?:GIT binary patch|Binary files [^\r\n]+ differ)$")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
    flags=re.IGNORECASE,
)
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_GITHUB_TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{60,255})\b")
_ASSIGNED_SECRET = re.compile(
    r"(?im)(?<![A-Za-z0-9_])(?P<key_quote>[\"']?)"
    r"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|private[_-]?key)"
    r"(?P=key_quote)(?![A-Za-z0-9_])\s*[:=]\s*(?:\(\s*)?"
    r"(?P<value>[\"'][^\"'\r\n]+[\"']|[^\s,;#)\r\n]+)"
)
_PLACEHOLDER_SECRET = re.compile(
    r"(?i)^(?:[\"']?)(?:placeholder|changeme|change_me|example|dummy|test|redacted|none|null|"
    r"x+|your[_-].*|<[^>]+>|\$\{[^}]+\})(?:[\"']?)$"
)


def _contains_non_text_control(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in "\t\n\r")
        or ord(character) == 127
        or 128 <= ord(character) <= 159
        for character in value
    )


def _proposal_strings(proposal: PatchProposal) -> tuple[str, ...]:
    return (
        proposal.summary,
        proposal.root_cause,
        *proposal.files_changed,
        proposal.unified_diff,
        proposal.expected_effect,
        *proposal.risks,
    )


def _contains_secret(value: str) -> bool:
    if _PRIVATE_KEY.search(value) or _AWS_KEY.search(value) or _GITHUB_TOKEN.search(value):
        return True
    for match in _ASSIGNED_SECRET.finditer(value):
        assigned = match.group("value").strip("\"'")
        if len(assigned) >= 8 and _PLACEHOLDER_SECRET.fullmatch(match.group("value")) is None:
            return True
    return False


_TEST_DEFINITION = re.compile(r"^\s*(?:async\s+)?def\s+test_[A-Za-z0-9_]*\s*\(")


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    values: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        values.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        values.append(current.id)
    return tuple(reversed(values))


class _PolicyInspectionError(Exception):
    pass


_MAX_PYTHON_TOKENS_PER_LINE = 4096
_IGNORED_TOKEN_TYPES = frozenset(
    {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.STRING,
    }
)


def _python_token_values(content: str) -> tuple[str, ...]:
    values: list[str] = []
    count = 0
    try:
        for token in tokenize.generate_tokens(StringIO(content).readline):
            count += 1
            if count > _MAX_PYTHON_TOKENS_PER_LINE:
                raise _PolicyInspectionError
            if token.type not in _IGNORED_TOKEN_TYPES and not token.string.isspace():
                values.append(token.string)
    except (IndentationError, tokenize.TokenError):
        pass
    except (MemoryError, RecursionError):
        raise _PolicyInspectionError from None
    return tuple(values)


def _has_token_sequence(values: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    width = len(expected)
    return any(
        values[index : index + width] == expected for index in range(len(values) - width + 1)
    )


def _has_rooted_token_sequence(values: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    width = len(expected)
    return any(
        values[index : index + width] == expected and (index == 0 or values[index - 1] != ".")
        for index in range(len(values) - width + 1)
    )


def _parsed_python_line(
    content: str, *, decorator: bool = False, suite: bool = False
) -> ast.AST | None:
    stripped = content.lstrip()
    if not stripped or stripped.startswith("#"):
        return None
    source = stripped
    if decorator:
        source = f"{stripped}\ndef _tracefix_policy_target():\n    pass"
    elif suite:
        source = f"{stripped}\n    pass"
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    except (MemoryError, RecursionError):
        raise _PolicyInspectionError from None


def _introduces_test_disable(content: str) -> bool:
    stripped = content.lstrip()
    tokens = _python_token_values(content)
    if any(
        _has_rooted_token_sequence(tokens, ("pytest", ".", "mark", ".", name))
        for name in ("skip", "skipif", "xfail")
    ):
        return True
    if _has_rooted_token_sequence(tokens, ("pytest", ".", "skip")):
        return True
    if any(
        tokens[index : index + 2] == ("unittest", ".")
        and (index == 0 or tokens[index - 1] != ".")
        and tokens[index + 2].startswith("skip")
        for index in range(max(0, len(tokens) - 2))
    ):
        return True
    tree = _parsed_python_line(content, decorator=stripped.startswith("@"))
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain[:2] == ("pytest", "mark") and chain[2:] in {
                ("skip",),
                ("skipif",),
                ("xfail",),
            }:
                return True
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain == ("pytest", "skip"):
                return True
            if chain[:2] == ("pytest", "mark") and chain[2:] in {
                ("skip",),
                ("skipif",),
                ("xfail",),
            }:
                return True
            if len(chain) == 2 and chain[0] == "unittest" and chain[1].startswith("skip"):
                return True
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
            ):
                value = node.value
                if value is None:
                    continue
                for child in ast.walk(value):
                    chain = _attribute_chain(child)
                    if chain[:2] == ("pytest", "mark") and chain[2:] in {
                        ("skip",),
                        ("skipif",),
                        ("xfail",),
                    }:
                        return True
    return False


_UNITTEST_ASSERTIONS = frozenset(
    {
        "assertAlmostEqual",
        "assertCountEqual",
        "assertDictEqual",
        "assertEndsWith",
        "assertEqual",
        "assertFalse",
        "assertGreater",
        "assertGreaterEqual",
        "assertHasAttr",
        "assertIn",
        "assertIs",
        "assertIsInstance",
        "assertIsNone",
        "assertIsNot",
        "assertIsNotNone",
        "assertIsSubclass",
        "assertLess",
        "assertLessEqual",
        "assertListEqual",
        "assertLogs",
        "assertMultiLineEqual",
        "assertNoLogs",
        "assertNotAlmostEqual",
        "assertNotEndsWith",
        "assertNotEqual",
        "assertNotHasAttr",
        "assertNotIn",
        "assertNotIsInstance",
        "assertNotIsSubclass",
        "assertNotRegex",
        "assertNotStartsWith",
        "assertRaises",
        "assertRaisesRegex",
        "assertRegex",
        "assertSequenceEqual",
        "assertSetEqual",
        "assertStartsWith",
        "assertTrue",
        "assertTupleEqual",
        "assertWarns",
        "assertWarnsRegex",
    }
)


def _removes_assertion(content: str) -> bool:
    stripped = content.lstrip()
    tokens = _python_token_values(content)
    if tokens and tokens[0] == "assert":
        return True
    if any(
        _has_rooted_token_sequence(tokens, (owner, ".", assertion))
        for owner in ("self", "cls")
        for assertion in _UNITTEST_ASSERTIONS
    ):
        return True
    if any(
        _has_rooted_token_sequence(tokens, ("unittest", ".", "TestCase", ".", assertion))
        for assertion in _UNITTEST_ASSERTIONS
    ):
        return True
    if any(
        _has_rooted_token_sequence(tokens, ("pytest", ".", context))
        for context in ("raises", "warns")
    ):
        return True
    suite = stripped.startswith("with ") and stripped.endswith(":")
    tree = _parsed_python_line(content, suite=suite)
    if tree is None:
        return False
    if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                expression = item.context_expr
                if isinstance(expression, ast.Call):
                    chain = _attribute_chain(expression.func)
                    if chain in {("pytest", "raises"), ("pytest", "warns")}:
                        return True
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if len(chain) == 2 and chain[0] in {"self", "cls"} and chain[1] in _UNITTEST_ASSERTIONS:
                return True
            if (
                len(chain) == 3
                and chain[:2] == ("unittest", "TestCase")
                and chain[2] in _UNITTEST_ASSERTIONS
            ):
                return True
    return False


def _introduces_subprocess(content: str) -> bool:
    tokens = _python_token_values(content)
    if tokens[:2] == ("import", "subprocess") or tokens[:3] == (
        "from",
        "subprocess",
        "import",
    ):
        return True
    if tokens[:3] == ("from", "os", "import") and any(
        value in {"system", "popen"} or value.startswith("spawn") for value in tokens[3:]
    ):
        return True
    if _has_rooted_token_sequence(tokens, ("subprocess", ".")):
        return True
    if any(
        _has_rooted_token_sequence(tokens, ("os", ".", name)) for name in ("system", "popen")
    ) or any(
        tokens[index : index + 2] == ("os", ".")
        and (index == 0 or tokens[index - 1] != ".")
        and tokens[index + 2].startswith("spawn")
        for index in range(max(0, len(tokens) - 2))
    ):
        return True
    if _has_token_sequence(tokens, ("shell", "=", "True")):
        return True
    tree = _parsed_python_line(content)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "subprocess" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                return True
            if node.module == "os" and any(
                alias.name in {"system", "popen"} or alias.name.startswith("spawn")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain and chain[0] == "subprocess":
                return True
            if (
                len(chain) == 2
                and chain[0] == "os"
                and (chain[1] in {"system", "popen"} or chain[1].startswith("spawn"))
            ):
                return True
            if any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                return True
    return False


def _is_test_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    parts = tuple(part.casefold() for part in candidate.parts)
    name = candidate.name.casefold()
    return (
        "tests" in parts
        or (name.startswith("test_") and name.endswith(".py"))
        or (name.endswith("_test.py"))
    )


def _is_dependency_path(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return name in {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "uv.lock",
    } or (name.startswith("requirements") and name.endswith(".txt"))


_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_DEPENDENCY_KEYS = frozenset(
    {"dependencies", "dependency", "requires", "install_requires", "extras_require"}
)
_NON_DEPENDENCY_KEYS = frozenset(
    {
        "name",
        "version",
        "description",
        "python",
        "requires-python",
        "readme",
        "license",
        "authors",
        "classifiers",
        "build-backend",
    }
)
_INLINE_DEPENDENCY_TABLE = re.compile(r"(?i)(?:^|[{,]\s*)(?:git|url|path|version)\s*=")


def _requirement_name(value: str) -> str | None:
    stripped = value.strip().strip("\"'").strip().rstrip(",")
    match = _REQUIREMENT_NAME.match(stripped)
    if match is None:
        return None
    tail = stripped[match.end() :]
    if tail and not tail.startswith(("[", "=", "!", "~", ">", "<", "@", ";", " ")):
        return None
    try:
        return _normalized_package_name(match.group(1))
    except ValueError:
        return None


def _dependency_names(content: str, path: str) -> tuple[str, ...] | None:
    stripped = content.strip()
    if not stripped or stripped.startswith("#"):
        return None
    name = PurePosixPath(path).name.casefold()
    without_comment = stripped.split("#", maxsplit=1)[0].strip()
    if not without_comment:
        return None
    if name.startswith("requirements") and name.endswith(".txt"):
        if without_comment.startswith(("-", "git+", "http://", "https://")):
            return ()
        package = _requirement_name(without_comment)
        return () if package is None else (package,)
    if without_comment.startswith("[") and without_comment.endswith("]"):
        return None
    if without_comment in {"[", "]", "{", "}", "},", "],"}:
        return None
    assignment = re.match(
        r"^[\"']?(?P<key>[A-Za-z0-9_.-]+)[\"']?\s*=\s*(?P<value>.+)$", without_comment
    )
    quoted = tuple(re.findall(r"[\"']([^\"']+)[\"']", without_comment))
    if assignment is not None:
        key = assignment.group("key")
        folded_key = key.casefold()
        if name in {"poetry.lock", "uv.lock"} and folded_key == "name" and quoted:
            package = _requirement_name(quoted[0])
            return () if package is None else (package,)
        if folded_key in _NON_DEPENDENCY_KEYS or folded_key.startswith("_"):
            return None
        if folded_key in _DEPENDENCY_KEYS:
            packages = tuple(
                package
                for candidate in quoted
                if (package := _requirement_name(candidate)) is not None
            )
            return packages if packages else ()
        if name in {"pipfile", "pipfile.lock", "poetry.lock", "uv.lock"}:
            try:
                return (_normalized_package_name(key),)
            except ValueError:
                return ()
        if name == "pyproject.toml" and _INLINE_DEPENDENCY_TABLE.search(assignment.group("value")):
            try:
                return (_normalized_package_name(key),)
            except ValueError:
                return ()
        if quoted and re.fullmatch(r"[~^<>=!0-9.*+,-]+", quoted[-1]) is not None:
            try:
                return (_normalized_package_name(key),)
            except ValueError:
                return ()
    packages = tuple(
        package for candidate in quoted if (package := _requirement_name(candidate)) is not None
    )
    if packages:
        return packages
    if name in {"setup.cfg", "pyproject.toml", "setup.py"} and any(
        key in without_comment.casefold() for key in _DEPENDENCY_KEYS
    ):
        return ()
    if name == "setup.cfg":
        package = _requirement_name(without_comment)
        if package is not None:
            return (package,)
    return None


def _safe_violation(
    code: PatchPolicyViolationCode,
    path: str,
    secret_paths: frozenset[str],
    *,
    line: int | None = None,
) -> PatchPolicyViolation:
    if path in secret_paths:
        return _make_violation(code)
    return _make_violation(code, path=path, line=line)


def _path_violations(
    paths: tuple[str, ...],
    policy: PatchPolicy,
    secret_paths: frozenset[str],
) -> list[PatchPolicyViolation]:
    violations: list[PatchPolicyViolation] = []
    protected = policy.protected_path_policy.protected_paths + policy.forbidden_changed_files
    for path in paths:
        if any(_path_is_equal_or_below(path, value) for value in protected):
            violations.append(
                _safe_violation(
                    PatchPolicyViolationCode.PROTECTED_PATH_MODIFIED, path, secret_paths
                )
            )
        if _is_ci_or_security_path(path):
            violations.append(
                _safe_violation(
                    PatchPolicyViolationCode.CI_OR_SECURITY_CONFIG_MODIFIED,
                    path,
                    secret_paths,
                )
            )
        if _is_hidden_test_path(path):
            violations.append(
                _safe_violation(PatchPolicyViolationCode.HIDDEN_TEST_MODIFIED, path, secret_paths)
            )
        if _is_benchmark_metadata_path(path):
            violations.append(
                _safe_violation(
                    PatchPolicyViolationCode.BENCHMARK_METADATA_MODIFIED,
                    path,
                    secret_paths,
                )
            )
        if _is_reference_patch_path(path):
            violations.append(
                _safe_violation(
                    PatchPolicyViolationCode.REFERENCE_PATCH_MODIFIED, path, secret_paths
                )
            )
    return violations


def _line_violation(
    code: PatchPolicyViolationCode,
    file: ParsedDiffFile,
    line: ParsedDiffLine,
    secret_paths: frozenset[str],
) -> PatchPolicyViolation:
    source_line = line.new_line if line.kind is DiffLineKind.ADDED else line.old_line
    return _safe_violation(code, file.path, secret_paths, line=source_line)


def _content_violations(
    parsed: ParsedUnifiedDiff,
    policy: PatchPolicy,
    secret_paths: frozenset[str],
) -> list[PatchPolicyViolation]:
    violations: list[PatchPolicyViolation] = []
    allowlist = frozenset(policy.dependency_allowlist)
    for file in parsed.files:
        if file.is_deleted and _is_test_path(file.path):
            violations.append(
                _safe_violation(PatchPolicyViolationCode.TEST_DISABLED, file.path, secret_paths)
            )
        for line in file.lines:
            if line.kind is DiffLineKind.REMOVED:
                if _TEST_DEFINITION.match(line.content):
                    violations.append(
                        _line_violation(
                            PatchPolicyViolationCode.TEST_DISABLED,
                            file,
                            line,
                            secret_paths,
                        )
                    )
                if _removes_assertion(line.content):
                    violations.append(
                        _line_violation(
                            PatchPolicyViolationCode.ASSERTION_REMOVED,
                            file,
                            line,
                            secret_paths,
                        )
                    )
            elif line.kind is DiffLineKind.ADDED:
                if _introduces_test_disable(line.content):
                    violations.append(
                        _line_violation(
                            PatchPolicyViolationCode.TEST_DISABLED,
                            file,
                            line,
                            secret_paths,
                        )
                    )
                if _introduces_subprocess(line.content):
                    violations.append(
                        _line_violation(
                            PatchPolicyViolationCode.UNRESTRICTED_SUBPROCESS_INTRODUCED,
                            file,
                            line,
                            secret_paths,
                        )
                    )
                if _is_dependency_path(file.path):
                    dependencies = _dependency_names(line.content, file.path)
                    if dependencies is not None and (
                        not dependencies or any(name not in allowlist for name in dependencies)
                    ):
                        violations.append(
                            _line_violation(
                                PatchPolicyViolationCode.UNAPPROVED_DEPENDENCY_ADDED,
                                file,
                                line,
                                secret_paths,
                            )
                        )
    return violations


def _result(violations: list[PatchPolicyViolation]) -> PatchPolicyResult:
    unique = {
        (
            item.code,
            None if item.path is None else item.path.as_posix(),
            item.line,
            item.message,
        ): item
        for item in violations
    }
    ordered = tuple(sorted(unique.values(), key=_violation_sort_key))
    return PatchPolicyResult(approved=not ordered, violations=ordered)


def validate_patch_policy(proposal: PatchProposal, policy: PatchPolicy) -> PatchPolicyResult:
    if type(proposal) is not PatchProposal:
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL)
    try:
        proposal_paths = normalize_repository_paths(proposal.files_changed)
    except UnifiedDiffError:
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_PATH)
    except (AttributeError, TypeError):
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL)

    validated_proposal: PatchProposal | None = None
    try:
        raw_proposal = proposal.model_dump(warnings="none")
        validated_proposal = PatchProposal.model_validate(raw_proposal, strict=True)
        if validated_proposal.model_dump(warnings="none") != raw_proposal:
            validated_proposal = None
    except (AttributeError, TypeError, ValueError, ValidationError):
        pass
    if validated_proposal is None:
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_PROPOSAL)

    if (
        type(policy) is not PatchPolicy
        or type(policy.protected_path_policy) is not ProtectedPathPolicy
    ):
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_POLICY)

    validated_policy: PatchPolicy | None = None
    try:
        raw_policy = policy.model_dump(warnings="none")
        validated_policy = PatchPolicy.model_validate(raw_policy, strict=True)
        if validated_policy.model_dump(warnings="none") != raw_policy:
            validated_policy = None
    except (AttributeError, TypeError, ValueError, ValidationError):
        pass
    if validated_policy is None:
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_POLICY)

    secret_paths = frozenset(path for path in proposal_paths if _contains_secret(path))
    violations = _path_violations(proposal_paths, validated_policy, secret_paths)
    if len(proposal_paths) > validated_policy.max_changed_files:
        violations.append(_make_violation(PatchPolicyViolationCode.TOO_MANY_FILES))
    if any(_contains_secret(value) for value in _proposal_strings(validated_proposal)):
        violations.append(_make_violation(PatchPolicyViolationCode.SECRET_DETECTED))

    has_binary_marker = bool(_BINARY_MARKER.search(validated_proposal.unified_diff))
    binary = _contains_non_text_control(validated_proposal.unified_diff) or has_binary_marker
    if binary:
        violations.append(_make_violation(PatchPolicyViolationCode.BINARY_CONTENT))

    try:
        parsed = parse_unified_diff(validated_proposal.unified_diff.replace("\x00", " "))
    except UnifiedDiffError:
        if not has_binary_marker:
            _raise_input(PatchPolicyInputDiagnosticCode.INVALID_DIFF)
        try:
            binary_paths = parse_binary_diff_paths(validated_proposal.unified_diff)
        except UnifiedDiffError:
            _raise_input(PatchPolicyInputDiagnosticCode.INVALID_DIFF)
        if set(proposal_paths) != set(binary_paths):
            _raise_input(PatchPolicyInputDiagnosticCode.INCONSISTENT_PATHS)
        return _result(violations)
    if set(proposal_paths) != set(parsed.changed_paths):
        _raise_input(PatchPolicyInputDiagnosticCode.INCONSISTENT_PATHS)
    if parsed.changed_line_count > validated_policy.max_changed_lines:
        violations.append(_make_violation(PatchPolicyViolationCode.LINE_CHANGE_LIMIT_EXCEEDED))
    try:
        violations.extend(_content_violations(parsed, validated_policy, secret_paths))
    except _PolicyInspectionError:
        _raise_input(PatchPolicyInputDiagnosticCode.INVALID_DIFF)
    return _result(violations)
