# ./src/layer_manager/wire.py
"""Length-prefixed JSON messaging over TCP, shared by the socket nodes.

The three node programs (``tx``, ``ch``, ``rx``) share only
``src/layer_manager`` across their containers, so the socket framing and the
config loading live here rather than being copied into each node.

A message is a 10-digit zero-padded ASCII byte-length header followed by that
many UTF-8 JSON bytes -- matching the ``utf-8`` / 1024-byte buffer convention in
``config.toml``. Length-prefixing lets the receiver read an exact, complete
message off a stream socket (which has no notion of message boundaries).
"""

import json
import socket
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

_HEADER_BYTES = 10  # zero-padded decimal byte count, e.g. b"0000004096"


@dataclass
class WireConfig:
    """Socket settings read from ``config.toml``."""

    address: str  # bind address for the listening nodes (e.g. 0.0.0.0)
    buffer_size: int  # recv chunk size
    encoding: str  # payload text encoding
    ports: dict[str, int]  # role -> TCP port (dev/tx/ch/rx)


def load_config(path: str | Path = "config.toml") -> WireConfig:
    """Read the ``[socket]`` and ``[container.*]`` tables from ``config.toml``.

    Args:
        path: Location of the TOML config (defaults to the repo-root copy that
            every container mounts).

    Returns:
        The parsed :class:`WireConfig`.
    """
    raw = tomllib.loads(Path(path).read_text())
    sock = raw["socket"]
    ports = {
        role: int(raw["container"][role]["port"]) for role in raw["container"]
    }
    return WireConfig(
        address=str(sock["address"]),
        buffer_size=int(sock["buffer_size"]),
        encoding=str(sock["encoding"]),
        ports=ports,
    )


def send_message(sock: socket.socket, payload: dict[str, object]) -> None:
    """Send one JSON ``payload`` with a length header.

    Args:
        sock: A connected stream socket.
        payload: A JSON-serialisable mapping.
    """
    data = json.dumps(payload).encode("utf-8")
    header = f"{len(data):0{_HEADER_BYTES}d}".encode("ascii")
    sock.sendall(header + data)


def recv_message(
    sock: socket.socket, buffer_size: int = 1024
) -> dict[str, object] | None:
    """Read one length-prefixed JSON message from ``sock``.

    Args:
        sock: A connected stream socket.
        buffer_size: Maximum bytes per ``recv`` call.

    Returns:
        The decoded mapping, or ``None`` if the peer closed the connection
        before a full message arrived.
    """
    header = _recv_exactly(sock, _HEADER_BYTES, buffer_size)
    if header is None:
        return None
    body = _recv_exactly(sock, int(header.decode("ascii")), buffer_size)
    if body is None:
        return None
    message: dict[str, object] = json.loads(body.decode("utf-8"))
    return message


def connect(
    host: str, port: int, retries: int = 30, delay: float = 1.0
) -> socket.socket:
    """Open a connection to ``host:port``, retrying while it is not yet up.

    The nodes start together under docker-compose, so a downstream server may
    not be listening yet; this retries instead of failing the race.

    Args:
        host: Target hostname (a container name on the bridge network).
        port: Target TCP port.
        retries: How many attempts before giving up.
        delay: Seconds to wait between attempts.

    Returns:
        The connected socket.

    Raises:
        OSError: If every attempt fails.
    """
    last: OSError | None = None
    for _ in range(retries):
        try:
            return socket.create_connection((host, port))
        except OSError as error:
            last = error
            time.sleep(delay)
    raise OSError(f"could not connect to {host}:{port}: {last}")


def _recv_exactly(
    sock: socket.socket, count: int, buffer_size: int
) -> bytes | None:
    """Read exactly ``count`` bytes, or ``None`` if the peer closes early."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(min(buffer_size, remaining))
        if not chunk:  # peer closed the connection
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
