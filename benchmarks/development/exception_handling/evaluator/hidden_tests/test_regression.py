from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.exception_handling.repository.src.parser.parser import parse_port
else:
    from parser.parser import parse_port


def test_valid_port_is_parsed() -> None:
    assert parse_port("8080") == 8080
