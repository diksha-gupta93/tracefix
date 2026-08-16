from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.development.fixture_or_mocking.repository.src.greeter.greeter import greeting
else:
    from greeter.greeter import greeting


class RecordingUserClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_user(self) -> str:
        self.calls += 1
        return "Grace"


def test_client_is_called_once() -> None:
    client = RecordingUserClient()

    assert greeting(client) == "Hello, Grace!"
    assert client.calls == 1
