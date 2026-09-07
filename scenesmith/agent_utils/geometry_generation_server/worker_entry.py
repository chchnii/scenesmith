"""Standalone entry point for a GPU geometry worker.

This script is launched by the worker pool (parent process) via ``subprocess``
as a **fresh interpreter**, instead of ``fork`` from the CUDA-initialized parent.
That is the whole point of the restart fix: a new interpreter inherits none of
the parent's CUDA/PyTorch/Warp state, so ``wp.init()`` runs cleanly even after
the parent has already initialized CUDA.

Critical rules:
- This module MUST NOT import any CUDA-dependent code at module level.
  ``scenesmith.agent_utils.geometry_generation_server.geometry_generation``
  (which calls ``ensure_cuda_env()`` -> ``wp.init()``) is imported only inside
  ``main()``, after ``CUDA_VISIBLE_DEVICES`` has been verified.
- It MUST NOT import the package ``__main__`` (e.g. ``main.py``). This is why
  launching via this entry point avoids the ``spawn`` re-import incompatibility.
  (bpy itself IS imported inside ``main()`` — the geometry pipeline needs the
  ``mathutils`` module that the bpy wheel registers — but only after the
  handshake, never at module level.)

Launch forms (both supported by ``_build_command`` in the worker pool):
- ``python -m scenesmith.agent_utils.geometry_generation_server.worker_entry ...``
- ``blender -b --python <abs path to this file> -- ...``  (Blender exposes
  everything after ``--`` in ``sys.argv``.)
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
    MSG_HELLO,
    MSG_INIT,
    send_message,
)

console_logger = logging.getLogger(__name__)


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse worker arguments, tolerating both ``-m`` and Blender ``--python`` forms.

    With ``python -m <module>``, ``sys.argv`` is ``[<module>, *args]``.
    With Blender ``--python file.py -- a b c``, Blender passes ``--`` through to
    the script's ``sys.argv``; strip everything up to and including a bare ``--``
    so argparse sees only the intended worker arguments.
    """
    argv = list(args)
    # Blender form: ["...blender", "--", "--socket-path", ...] or similar where a
    # bare "--" separates blender's own args from ours.
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="GPU geometry worker")
    parser.add_argument("--socket-path", required=True, help="Path to the AF_UNIX socket.")
    parser.add_argument("--gpu-id", required=True, type=int, help="GPU index this worker owns.")
    parser.add_argument(
        "--token-file",
        required=True,
        help="Path to the session token file (0600, written by the parent).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to append worker logs to (also mirrored to stderr).",
    )
    return parser.parse_args(argv)


def _read_token(token_file: str) -> str:
    """Read the session token from the parent-owned 0600 file."""
    return Path(token_file).read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    """Worker entry point."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Verify CUDA_VISIBLE_DEVICES was set by the parent before start.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        console_logger.error("CUDA_VISIBLE_DEVICES is not configured; aborting.")
        return 1
    if cuda_visible != str(args.gpu_id):
        console_logger.error(
            f"CUDA_VISIBLE_DEVICES={cuda_visible!r} does not match --gpu-id {args.gpu_id}; aborting."
        )
        return 1

    # Read the session token before connecting (avoids accepting a stale socket).
    try:
        token = _read_token(args.token_file)
    except OSError as e:
        console_logger.error(f"Cannot read token file {args.token_file}: {e}")
        return 1

    # Connect to the parent's listener. retry briefly in case the worker process
    # was spawned slightly before the listener is ready.
    sock: socket.socket | None = None
    last_error: Exception | None = None
    for attempt in range(100):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(args.socket_path)
            break
        except OSError as e:
            last_error = e
            if sock is not None:
                sock.close()
            time.sleep(0.1)
    if sock is None:
        console_logger.error(f"Could not connect to {args.socket_path}: {last_error}")
        return 1

    try:
        # Handshake: tell the parent who we are. The parent must acknowledge with
        # an 'init' message, otherwise the connection is not trusted.
        send_message(
            sock,
            {
                "type": MSG_HELLO,
                "gpu_id": args.gpu_id,
                "pid": os.getpid(),
                "generation": os.environ.get("SCENESMITH_WORKER_GENERATION"),
                "token": token,
            },
        )

        from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
            recv_message,
        )

        init_message = recv_message(sock)
        if init_message.get("type") != MSG_INIT:
            console_logger.error(f"Expected init but got: {init_message.get('type')!r}")
            return 1

        init_config: dict[str, Any] = init_message.get("config", {})
        use_mini = bool(init_config.get("use_mini", False))
        backend = str(init_config.get("backend", "hunyuan3d"))
        sam3d_config: dict | None = init_config.get("sam3d_config")
        preload_pipeline = bool(init_config.get("preload_pipeline", True))

        # bpy must be imported before the geometry pipeline chain: mesh_utils
        # does `from mathutils import Vector`, and mathutils is not a standalone
        # package — the bpy pip wheel registers it in sys.modules on import
        # (same reason main.py imports bpy first). Inside a Blender-launched
        # interpreter bpy is already present, so this is a no-op there.
        import bpy  # noqa: F401

        # CUDA-imports happen here, after the env var is confirmed to be correct.
        from scenesmith.agent_utils.geometry_generation_server.gpu_worker import (
            socket_worker_run,
        )

        socket_worker_run(
            connection=sock,
            gpu_id=args.gpu_id,
            use_mini=use_mini,
            backend=backend,
            sam3d_config=sam3d_config,
            preload_pipeline=preload_pipeline,
            log_file=args.log_file,
        )
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [worker] %(levelname)s: %(message)s",
    )
    raise SystemExit(main())
