"""Length-prefixed JSON message framing for the geometry worker socket protocol.

The GPU worker pool (parent process) and each GPU worker communicate over a
Unix-domain socket. Sockets are byte streams and do not preserve message
boundaries, so every message is framed as:

    [4-byte big-endian payload length][UTF-8 JSON payload]

NOTE: This protocol MUST NOT run over stdout/stderr. Third-party libraries
(``torch``, ``warp``, SAM3D, Hydra, ...) emit arbitrary text to those streams,
which would corrupt a text-based wire protocol. stdout/stderr are reserved for
logging only.

Message types (dict ``"type"`` field):
    "hello"     worker -> parent: who am I (gpu_id/pid/generation/token)
    "init"      parent -> worker: backend/use_mini/sam3d_config/preload_pipeline
    "ready"     worker -> parent: imports done (+ pipeline preloaded if requested)
    "request"   parent -> worker: a geometry generation job
    "result"    worker -> parent: outcome of a job (success/error)
    "shutdown"  parent -> worker: graceful stop signal
"""

from __future__ import annotations

import json
import socket
import struct

from typing import Any


# Message type identifiers.
MSG_HELLO = "hello"
MSG_INIT = "init"
MSG_READY = "ready"
MSG_REQUEST = "request"
MSG_RESULT = "result"
MSG_SHUTDOWN = "shutdown"

# Length of the framing header in bytes.
_HEADER_SIZE = struct.calcsize("!I")

MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # 8 MiB safety cap on a single message.


def encode_message(message: dict[str, Any]) -> bytes:
    """Serialize a message dict into a framed byte sequence."""
    payload = json.dumps(message).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Message payload too large: {len(payload)} bytes "
            f"(max {MAX_MESSAGE_BYTES})"
        )
    return struct.pack("!I", len(payload)) + payload


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send one framed message over ``sock``. Raises on failure."""
    send_encoded_message(sock, encode_message(message))


def send_encoded_message(sock: socket.socket, frame: bytes) -> None:
    """Send a frame previously returned by :func:`encode_message`."""
    sock.sendall(frame)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes, or raise EOFError if the peer closes early."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket closed by peer")
        buf += chunk
    return buf


def recv_message(sock: socket.socket) -> dict[str, Any]:
    """Read one framed message from ``sock``.

    Raises:
        EOFError: peer closed the connection cleanly (no more bytes).
        json.JSONDecodeError: malformed payload (should not happen for a trusted
            peer, but is surfaced rather than silently ignored).
    """
    header = _recv_exact(sock, _HEADER_SIZE)
    (length,) = struct.unpack("!I", header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message length exceeds cap: {length} bytes")
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))
