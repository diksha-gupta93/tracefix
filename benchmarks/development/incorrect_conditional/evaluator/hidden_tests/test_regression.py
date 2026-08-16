from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.incorrect_conditional.repository.src.eligibility.eligibility import (
        is_eligible,
    )
else:
    from eligibility.eligibility import is_eligible


def test_minor_is_not_eligible() -> None:
    assert not is_eligible(17)
