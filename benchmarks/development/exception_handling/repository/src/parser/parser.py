def parse_port(value: str) -> int | None:
    try:
        return int(value)
    except TypeError:
        return None
