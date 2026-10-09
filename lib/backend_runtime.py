"""Shared local server launch checks."""
import socket


def require_free_port(port: int) -> None:
    # Do not use SO_REUSEADDR: even a bound, not-yet-listening socket conflicts.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"requested port {port} is occupied or unavailable") from exc
