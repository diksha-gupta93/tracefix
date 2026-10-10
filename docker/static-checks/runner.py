from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tokenize
from io import StringIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

CHECKS = ("python_compile", "ast_parse", "ruff", "mypy", "bandit")
INPUT_ROOT = Path("/tracefix/input")
WORKSPACE_ROOT = Path("/tracefix/workspace")
REPOSITORY = WORKSPACE_ROOT / "repository"
MYPY_CONFIG = WORKSPACE_ROOT / "tracefix-mypy.ini"
MAX_PATHS = 3
MAX_PATH_BYTES = 1_024
MAX_TOOL_OUTPUT = 262_144
MAX_FINDINGS = 1_000
MAX_SOURCE_BYTES = 67_108_864
RULE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RUFF_EXECUTABLE = "/usr/local/bin/ruff"
MYPY_EXECUTABLE = "/usr/local/bin/mypy"
BANDIT_EXECUTABLE = "/usr/local/bin/bandit"
RUFF_PREVIEW_WARNING = (
    b"warning: Support for Python 3.14 is in preview and may undergo breaking changes. "
    b"Enable `preview` to remove this warning.\n"
)
MYPY_ERROR = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>[1-9][0-9]*):(?P<column>[1-9][0-9]*): "
    r"error: .+ \[(?P<code>[A-Za-z0-9_.-]+)\]$"
)


def result(
    check: str,
    status: str,
    findings: list[dict[str, object]] | None = None,
    error_code: str | None = None,
) -> dict[str, object]:
    return {
        "check": check,
        "status": status,
        "findings": [] if findings is None else findings,
        "error_code": error_code,
    }


def infrastructure(check: str, code: str) -> dict[str, object]:
    return result(check, "infrastructure_error", error_code=code)


def validate_path(value: str) -> str:
    windows = PureWindowsPath(value)
    path = PurePosixPath(value)
    if (
        not value
        or len(value.encode("utf-8")) > MAX_PATH_BYTES
        or value != value.strip()
        or "\\" in value
        or windows.drive
        or windows.root
        or path.is_absolute()
        or path.as_posix() != value
        or value.startswith(("-", "@"))
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        or path.suffix.casefold() not in {".py", ".pyi"}
    ):
        raise ValueError
    return value


def validated_paths(values: list[str]) -> tuple[str, ...]:
    if len(values) > MAX_PATHS:
        raise ValueError
    paths = tuple(validate_path(value) for value in values)
    if paths != tuple(sorted(paths, key=lambda item: (item.casefold(), item))):
        raise ValueError
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError
    return paths


def safe_path(value: object) -> str:
    if type(value) is not str:
        raise ValueError
    repository_prefix = f"{REPOSITORY.as_posix()}/"
    normalized = value[len(repository_prefix) :] if value.startswith(repository_prefix) else value
    normalized = normalized.removeprefix("./")
    return validate_path(normalized)


def safe_location(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1 or value > 10_000_000:
        raise ValueError
    return value


def finding(path: object, line: object, column: object, rule: object) -> dict[str, object]:
    normalized_path = safe_path(path)
    normalized_line = safe_location(line)
    normalized_column = safe_location(column)
    if normalized_column is not None and normalized_line is None:
        raise ValueError
    if type(rule) is not str or RULE_CODE.fullmatch(rule) is None:
        raise ValueError
    return {
        "path": normalized_path,
        "line": normalized_line,
        "column": normalized_column,
        "rule_code": rule,
    }


def finding_key(item: dict[str, object]) -> tuple[str, str, int, int, str]:
    path = str(item["path"])
    line = item["line"]
    column = item["column"]
    return (
        path.casefold(),
        path,
        0 if line is None else int(line),
        0 if column is None else int(column),
        str(item["rule_code"]),
    )


def ordered(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    findings.sort(key=finding_key)
    if len(findings) > MAX_FINDINGS:
        raise OverflowError
    keys = [finding_key(item) for item in findings]
    if len(set(keys)) != len(keys):
        raise ValueError
    return findings


def read_source(relative: str) -> str:
    path = REPOSITORY.joinpath(*PurePosixPath(relative).parts)
    resolved = path.resolve(strict=True)
    resolved.relative_to(REPOSITORY.resolve(strict=True))
    data = resolved.read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise OSError
    return data.decode("utf-8")


def syntax_result(check: str, paths: tuple[str, ...], *, parse_ast: bool) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    try:
        for relative in paths:
            source = read_source(relative)
            try:
                if parse_ast:
                    ast.parse(source, filename=relative, mode="exec", feature_version=(3, 14))
                else:
                    compile(source, relative, "exec", dont_inherit=True, optimize=0)
            except (SyntaxError, ValueError) as error:
                findings.append(
                    finding(
                        relative,
                        getattr(error, "lineno", None) or 1,
                        getattr(error, "offset", None),
                        "AST_PARSE" if parse_ast else "PY_COMPILE",
                    )
                )
    except (OSError, UnicodeError, ValueError):
        return infrastructure(check, "source_read_failed")
    if findings:
        return result(check, "failed", ordered(findings))
    return result(check, "passed")


def bounded_stream(stream: BinaryIO, limit: int) -> tuple[bytearray, list[bool], threading.Thread]:
    retained = bytearray()
    truncated = [False]

    def drain() -> None:
        while chunk := stream.read(65_536):
            remaining = limit - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return retained, truncated, thread


def run_tool(arguments: tuple[str, ...]) -> tuple[int, bytes, bytes, bool]:
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MYPY_CACHE_DIR": "/dev/null",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    process = subprocess.Popen(
        arguments,
        cwd=REPOSITORY,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise OSError
    stdout, stdout_cut, stdout_thread = bounded_stream(process.stdout, MAX_TOOL_OUTPUT)
    stderr, stderr_cut, stderr_thread = bounded_stream(process.stderr, MAX_TOOL_OUTPUT)
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return return_code, bytes(stdout), bytes(stderr), stdout_cut[0] or stderr_cut[0]


def version_tuple(distribution: str) -> tuple[int, ...]:
    text = importlib.metadata.version(distribution)
    match = re.match(r"^(\d+(?:\.\d+)+)", text)
    if match is None:
        raise ValueError
    return tuple(int(part) for part in match.group(1).split("."))


def versions_valid() -> bool:
    ruff = version_tuple("ruff")
    mypy = version_tuple("mypy")
    bandit = version_tuple("bandit")
    return (0, 12) <= ruff < (0, 13) and (1, 17) <= mypy < (2,) and (1, 9) <= bandit < (2,)


def ruff_result(paths: tuple[str, ...]) -> dict[str, object]:
    arguments = (
        RUFF_EXECUTABLE,
        "check",
        "--isolated",
        "--no-cache",
        "--target-version",
        "py314",
        "--select",
        "E4,E7,E9,F,I,UP,B,SIM",
        "--ignore-noqa",
        "--output-format",
        "json",
        "--",
        *paths,
    )
    try:
        code, stdout, stderr, truncated = run_tool(arguments)
    except OSError:
        return infrastructure("ruff", "executable_missing")
    if truncated:
        return infrastructure("ruff", "output_limit")
    if stderr not in {b"", RUFF_PREVIEW_WARNING} or code not in {0, 1}:
        return infrastructure("ruff", "invocation_failed")
    try:
        document = json.loads(stdout)
        if type(document) is not list:
            raise ValueError
        findings = []
        for item in document:
            if type(item) is not dict:
                raise ValueError
            location = item.get("location")
            if type(location) is not dict:
                raise ValueError
            findings.append(
                finding(
                    item.get("filename"),
                    location.get("row"),
                    location.get("column"),
                    item.get("code"),
                )
            )
        findings = ordered(findings)
    except (UnicodeError, ValueError, OverflowError, json.JSONDecodeError):
        return infrastructure("ruff", "output_invalid")
    if bool(findings) != (code == 1):
        return infrastructure("ruff", "output_invalid")
    return result("ruff", "failed", findings) if findings else result("ruff", "passed")


def mypy_result(paths: tuple[str, ...]) -> dict[str, object]:
    arguments = (
        MYPY_EXECUTABLE,
        "--config-file",
        str(MYPY_CONFIG),
        "--strict",
        "--no-incremental",
        "--cache-dir",
        "/dev/null",
        "--no-site-packages",
        "--show-error-codes",
        "--show-column-numbers",
        "--no-error-summary",
        "--hide-error-context",
        "--no-pretty",
        "--no-color-output",
        "--",
        *paths,
    )
    try:
        code, stdout, stderr, truncated = run_tool(arguments)
    except OSError:
        return infrastructure("mypy", "executable_missing")
    if truncated:
        return infrastructure("mypy", "output_limit")
    if stderr or code not in {0, 1}:
        return infrastructure("mypy", "invocation_failed")
    try:
        text = stdout.decode("utf-8")
        tool_findings = []
        for line in text.splitlines():
            if ": note:" in line:
                continue
            match = MYPY_ERROR.fullmatch(line)
            if match is None:
                raise ValueError
            tool_findings.append(
                finding(
                    match.group("path"),
                    int(match.group("line")),
                    int(match.group("column")),
                    match.group("code"),
                )
            )
        tool_findings = ordered(tool_findings)
        findings = ordered([*tool_findings, *mypy_directive_findings(paths)])
    except (OSError, UnicodeError, ValueError, OverflowError):
        return infrastructure("mypy", "output_invalid")
    if bool(tool_findings) != (code == 1):
        return infrastructure("mypy", "output_invalid")
    return result("mypy", "failed", findings) if findings else result("mypy", "passed")


def mypy_directive_findings(paths: tuple[str, ...]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in paths:
        source = read_source(relative)
        try:
            tokens = tokenize.generate_tokens(StringIO(source).readline)
            for token in tokens:
                if token.type != tokenize.COMMENT:
                    continue
                normalized = token.string.casefold()
                if re.match(r"^#\s*type\s*:\s*ignore(?:\s|\[|$)", normalized) or re.match(
                    r"^#\s*mypy\s*:", normalized
                ):
                    findings.append(
                        finding(relative, token.start[0], token.start[1] + 1, "MYPY_DIRECTIVE")
                    )
        except (IndentationError, tokenize.TokenError):
            continue
    return ordered(findings)


def bandit_result(paths: tuple[str, ...]) -> dict[str, object]:
    arguments = (
        BANDIT_EXECUTABLE,
        "--format",
        "json",
        "--quiet",
        "--ignore-nosec",
        "--",
        *paths,
    )
    try:
        code, stdout, stderr, truncated = run_tool(arguments)
    except OSError:
        return infrastructure("bandit", "executable_missing")
    if truncated:
        return infrastructure("bandit", "output_limit")
    if stderr or code not in {0, 1}:
        return infrastructure("bandit", "invocation_failed")
    try:
        document = json.loads(stdout)
        if (
            type(document) is not dict
            or type(document.get("results")) is not list
            or document.get("errors") != []
        ):
            raise ValueError
        findings = []
        for item in document["results"]:
            if type(item) is not dict:
                raise ValueError
            findings.append(
                finding(item.get("filename"), item.get("line_number"), None, item.get("test_id"))
            )
        findings = ordered(findings)
    except (UnicodeError, ValueError, OverflowError, json.JSONDecodeError):
        return infrastructure("bandit", "output_invalid")
    if bool(findings) != (code == 1):
        return infrastructure("bandit", "output_invalid")
    return result("bandit", "failed", findings) if findings else result("bandit", "passed")


def execute(paths: tuple[str, ...]) -> list[dict[str, object]]:
    if not paths:
        return [result(check, "not_applicable") for check in CHECKS]
    compile_check = syntax_result("python_compile", paths, parse_ast=False)
    ast_check = syntax_result("ast_parse", paths, parse_ast=True)
    try:
        compatible = versions_valid()
    except (importlib.metadata.PackageNotFoundError, ValueError):
        compatible = False
    if not compatible:
        tools = [infrastructure(check, "version_mismatch") for check in CHECKS[2:]]
    else:
        tools = [ruff_result(paths), mypy_result(paths), bandit_result(paths)]
    return [compile_check, ast_check, *tools]


def emit(results: list[dict[str, object]]) -> None:
    payload = json.dumps({"version": 1, "results": results}, separators=(",", ":"), sort_keys=True)
    if len(payload.encode("utf-8")) > 65_536:
        payload = json.dumps(
            {
                "version": 1,
                "results": [infrastructure(check, "output_limit") for check in CHECKS],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    sys.stdout.write(payload)


def main() -> int:
    try:
        paths = validated_paths(sys.argv[1:])
        if REPOSITORY.exists():
            raise OSError
        MYPY_CONFIG.write_text("[mypy]\nplugins =\n", encoding="utf-8")
        shutil.copytree(INPUT_ROOT, REPOSITORY)
        os.chdir(REPOSITORY)
        emit(execute(paths))
        return 0
    except (OSError, RuntimeError, UnicodeError, ValueError):
        emit([infrastructure(check, "invocation_failed") for check in CHECKS])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
