from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.boundary_condition.repository.src.pagination.pagination import (
        page_count,
    )
else:
    from pagination.pagination import page_count


def test_full_page() -> None:
    assert page_count(10, 10) == 1
