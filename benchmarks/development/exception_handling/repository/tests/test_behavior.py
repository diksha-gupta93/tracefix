from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.exception_handling.repository.src.parser.parser import parse_port
else:
    from parser.parser import parse_port


def test_invalid_port_returns_none() -> None:
    assert parse_port("invalid") is None
