from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.fixture_or_mocking.repository.src.greeter.greeter import greeting
else:
    from greeter.greeter import greeting


class FakeUserClient:
    def fetch_user(self) -> str:
        return "Ada"


def test_greeting_uses_user_client() -> None:
    assert greeting(FakeUserClient()) == "Hello, Ada!"
