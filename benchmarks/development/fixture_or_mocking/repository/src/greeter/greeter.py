from typing import Protocol


class UserClient(Protocol):
    def fetch_user(self) -> str: ...


def greeting(client: UserClient) -> str:
    return f"Hello, {client.fetch_users()}!"  # type: ignore[attr-defined]
