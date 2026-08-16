from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.incorrect_conditional.repository.src.eligibility.eligibility import (
        is_eligible,
    )
else:
    from eligibility.eligibility import is_eligible


def test_adult_is_eligible() -> None:
    assert is_eligible(18)
