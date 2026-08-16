from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.incorrect_return_value.repository.src.normalizer.normalizer import (
        normalize,
    )
else:
    from normalizer.normalizer import normalize


def test_preserves_internal_space() -> None:
    assert normalize("Ada Lovelace") == "ada lovelace"
