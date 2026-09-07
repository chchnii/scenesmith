"""GPU worker pool for multi-GPU geometry generation.

This module manages a pool of GPU worker *processes*, distributing geometry
generation requests across all available GPUs for parallel processing.

Design (post fork-after-CUDA fix):
- Workers are launched with ``subprocess.Popen`` running a fresh interpreter
  (``worker_entry.py``), NOT by ``fork`` from this (CUDA-initialized) parent.
  This is what makes worker auto-restart safe: a restarted worker inherits none
  of the parent's CUDA/PyTorch/Warp state, so ``wp.init()`` runs cleanly.
- The parent owns an AF_UNIX listener; each worker gets its own bidirectional
  connection and speaks a length-prefixed JSON protocol (``worker_protocol``).
  ``multiprocessing.Queue`` is used nowhere for worker IPC, because its file
  descriptors are close-on-exec (PEP 446) and would not survive an exec.
- Worker identity is pinned by ``(gpu_id, pid, generation, token)`` so stale
  connections, PID reuse and leftover sockets are rejected.

This module intentionally does NOT import torch/CUDA at module level.
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
    MSG_INIT,
    MSG_READY,
    MSG_REQUEST,
    MSG_RESULT,
    MSG_SHUTDOWN,
    encode_message,
    recv_message,
    send_encoded_message,
)

console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerToken:
    """An availability-token for a worker, tied to its identity.

    Carried through the parent-internal availability queue so that stale tokens
    (from a dead/restarted worker) can be detected and discarded at consume
    time rather than requiring precise deletion from a concurrent queue.
    """

    gpu_id: int
    pid: int
    generation: int


@dataclass
class WorkerInfo:
    """State for a single worker process."""

    gpu_id: int
    generation: int
    process: subprocess.Popen
    token: str
    ready_event: threading.Event = field(default_factory=threading.Event)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    connection: socket.socket | None = None
    current_request_id: str | None = None
    current_request_started_at: float | None = None
    ready: bool = False
    reader_thread: threading.Thread | None = None
    log_handle: object | None = None


@dataclass
class PoolStats:
    """Statistics from the worker pool for health reporting."""

    num_workers: int
    total_requests: int
    completed_requests: int
    failed_requests: int
    avg_processing_time_s: float | None
    avg_end_to_end_latency_s: float | None
    avg_queue_wait_s: float | None
    max_queue_wait_s: float | None
    worker_details: list[dict]


class GPUWorkerPool:
    """Manages pool of GPU worker processes with on-demand dispatch.

    Workers signal availability after completing each request.
    ``submit_request()`` blocks until a worker is free, ensuring:
    - Fair scheduler ordering is preserved (requests dispatched in order)
    - Natural load balancing (faster GPUs process more requests)
    - Works identically with 1 GPU or N GPUs (single code path)

    Example:
        >>> pool = GPUWorkerPool(use_mini=False, backend="hunyuan3d")
        >>> pool.start()
        >>> print(f"Pool has {pool.num_workers} workers")
        >>>
        >>> def callback(index, result):
        ...     print(f"Request {index}: {result}")
        >>>
        >>> pool.submit_request(request, callback, request_index=0)
        >>> pool.stop()
    """

    def __init__(
        self,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        preload_pipeline: bool = True,
        log_file: Path | None = None,
        worker_start_timeout_s: float = 900.0,
        worker_acquire_timeout_s: float = 3600.0,
        max_restarts_per_window: int = 3,
        restart_window_s: float = 300.0,
        request_timeout_s: float | None = None,
    ) -> None:
        """Initialize the GPU worker pool.

        Args:
            use_mini: Whether to use mini model variant (Hunyuan3D only).
            backend: Generation backend ("hunyuan3d" or "sam3d").
            sam3d_config: Configuration for SAM3D backend.
            preload_pipeline: Whether to preload pipeline in workers on start.
            log_file: Path for worker-side logging (mirrored to stderr as well).
            worker_start_timeout_s: Max seconds to wait for each worker to become
                ready after launch (pipeline preload can take minutes).
            worker_acquire_timeout_s: Max seconds for ``submit_request`` to block
                waiting for a free worker before raising.
            max_restarts_per_window: Max worker restarts allowed per gpu within
                ``restart_window_s`` before auto-restart is disabled (crash-loop
                guard).
            restart_window_s: Time window for the crash-loop guard.
            request_timeout_s: Optional max seconds a single request may run on a
                worker before it is treated as hung. None disables this check
                (default; long generations exceed minutes).
        """
        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._preload_pipeline = preload_pipeline
        self._log_file = str(log_file) if log_file else None
        self._worker_start_timeout_s = worker_start_timeout_s
        self._worker_acquire_timeout_s = worker_acquire_timeout_s
        self._max_restarts_per_window = max_restarts_per_window
        self._restart_window_s = restart_window_s
        self._request_timeout_s = request_timeout_s

        # Detect available GPUs (respects CUDA_VISIBLE_DEVICES if set).
        self._gpu_ids = self._detect_gpu_ids()
        self._num_gpus = len(self._gpu_ids)
        console_logger.info(
            f"Detected {self._num_gpus} GPU(s) for worker pool: {self._gpu_ids}"
        )

        # Worker tracking. Plain thread-safe structures — no multiprocessing.
        self._workers: dict[int, WorkerInfo] = {}
        self._worker_generations: dict[int, int] = {}
        # Holds WorkerToken items; a None sentinel is pushed by stop() to wake
        # any thread blocked in _acquire_available_worker.
        self._available_workers: queue.Queue[WorkerToken | None] = queue.Queue()
        self._pending_callbacks: dict[str, tuple[Callable, int]] = {}
        self._pending_callbacks_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._worker_lock = threading.RLock()

        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._processing_times: list[float] = []
        self._end_to_end_latencies: list[float] = []
        self._max_queue_wait: float | None = None
        self._per_worker_completed: dict[int, int] = {}
        self._per_worker_failed: dict[int, int] = {}

        self._running = False
        self._listener: socket.socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._health_monitor_thread: threading.Thread | None = None
        self._socket_dir: str | None = None
        self._socket_path: str | None = None
        self._restart_history: dict[int, deque[float]] = {}
        self._exhausted_gpus: set[int] = set()

    @property
    def num_workers(self) -> int:
        """Get the number of workers in the pool."""
        return self._num_gpus

    def start(self) -> None:
        """Start all GPU worker processes.

        Workers are launched serially and each is awaited until it reports ready,
        which serializes pipeline preloading across GPUs (a single 15 GB SAM3D
        checkpoint load can starve disk I/O/CPU and GPU memory if done
        concurrently).
        """
        if self._running:
            raise RuntimeError("Worker pool is already running")

        console_logger.info(
            f"Starting GPU worker pool with {self._num_gpus} workers..."
        )
        self._running = True

        try:
            self._setup_listener()
            self._listener_thread = threading.Thread(
                target=self._accept_loop, daemon=True, name="WorkerPoolAccept"
            )
            self._listener_thread.start()

            # Launch workers serially, waiting for each to become ready.
            for gpu_id in self._gpu_ids:
                if not self._running:
                    break
                self._start_single_worker(gpu_id)
                self._wait_for_gpu_ready(gpu_id)

            # Start health monitor thread.
            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True,
                name="WorkerHealthMonitor",
            )
            self._health_monitor_thread.start()

        except Exception:
            self._running = False
            self._cleanup_listener()
            # Do not leave half-started workers around.
            for worker in list(self._workers.values()):
                self._terminate_process(worker.process)
                self._close_worker(worker)
            self._workers.clear()
            raise

        console_logger.info("GPU worker pool started successfully")

    def stop(self) -> None:
        """Stop all worker processes gracefully."""
        if not self._running:
            console_logger.warning("Worker pool is not running")
            return

        # 1. Stop restarts first so nothing resurrects a worker during teardown.
        console_logger.info("Stopping GPU worker pool...")
        self._running = False

        # Join the health monitor (it exits on _running == False).
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=2.0)

        # 2. Send shutdown to every worker that still has a live connection.
        for worker in list(self._workers.values()):
            try:
                self._send_worker_message(worker, {"type": MSG_SHUTDOWN})
            except OSError:
                pass

        # Wake up any coordinator thread blocked in _acquire_available_worker.
        self._available_workers.put(None)

        # 3. Wait gracefully, then terminate, then kill.
        for worker in list(self._workers.values()):
            self._terminate_process(worker.process)

        # 4. Close connections and reader threads.
        for worker in list(self._workers.values()):
            self._close_worker(worker)

        # 5. Teardown listener + socket dir.
        self._cleanup_listener()

        self._workers.clear()
        console_logger.info("GPU worker pool stopped")

    def submit_request(
        self,
        request: object,
        callback: Callable[[int, tuple[str, dict | str]], None],
        request_index: int,
        received_timestamp: float,
    ) -> None:
        """Submit a request to an available worker.

        This method blocks until a worker is available, preserving the fair
        ordering from the StrictRoundRobinScheduler.

        Args:
            request: A GeometryGenerationServerRequest (or anything with
                ``to_dict()`` producing a JSON-safe dict).
            callback: Function to call with (index, result) when complete.
            request_index: Index of this request in the batch.
            received_timestamp: Time when request was received by server.
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running")

        request_id = str(uuid.uuid4())

        # Register the callback BEFORE acquiring a worker so that every failure
        # path below (and the worker-exit path) can resolve it idempotently.
        with self._pending_callbacks_lock:
            self._pending_callbacks[request_id] = (callback, request_index)

        # Build and encode the complete frame BEFORE reserving a worker. Request
        # conversion, JSON serialization and the protocol size check can all
        # fail; none of those failures indicates a bad worker, and they must not
        # consume an availability token or leave a worker marked busy.
        try:
            request_frame = encode_message(
                {
                    "type": MSG_REQUEST,
                    "request_id": request_id,
                    "received_timestamp": received_timestamp,
                    "request": request.to_dict(),
                }
            )
        except Exception as e:
            console_logger.error(
                f"Failed to serialize geometry request {request_id}: {e}"
            )
            self._complete_request_with_error(
                request_id, f"Invalid geometry request: {e}"
            )
            return

        # Block until a valid (not stale) worker token is available.
        worker = self._acquire_available_worker()
        if worker is None:
            # Workers exhausted, acquire timeout, or pool stopping. Resolve the
            # callback so the HTTP streaming endpoint doesn't hang. Do NOT
            # raise: this runs on the coordinator thread and an exception would
            # kill the dispatch loop for all subsequent requests.
            console_logger.error(
                "No GPU worker available for request; failing it (workers "
                "exhausted, acquire timeout, or pool stopping)"
            )
            self._complete_request_with_error(
                request_id, "No geometry worker available"
            )
            return

        # Re-validate identity under the lock before dispatching: the worker may
        # have died (and been replaced by a new generation) between token
        # acquisition and now. Operating on a stale WorkerInfo would send into a
        # closed socket, and its exit would already have been handled — leaving
        # this request's callback dangling forever.
        with self._worker_lock:
            current = self._workers.get(worker.gpu_id)
            valid = (
                self._running
                and current is worker
                and worker.ready
                and worker.connection is not None
                and worker.process is not None
                and worker.process.poll() is None
            )
            if not valid:
                dispatch_conn = None
            else:
                worker.current_request_id = request_id
                worker.current_request_started_at = time.monotonic()
                dispatch_conn = worker.connection
            generation = worker.generation

        if dispatch_conn is None:
            # Guarantee resolution regardless of whether the exit handler
            # already ran for this worker (it may have handled an older
            # generation and skipped this request entirely).
            self._complete_request_with_error(
                request_id,
                f"Geometry worker GPU {worker.gpu_id} became unavailable "
                "before dispatch",
            )
            self._handle_worker_exit(worker.gpu_id, generation)
            return

        with self._stats_lock:
            self._total_requests += 1

        try:
            self._send_worker_frame(worker, request_frame)
        except OSError as e:
            # Always resolve this request first (idempotent), THEN let the exit
            # handler clean up/restart. The exit handler alone is not enough: if
            # the worker was already replaced by a newer generation, it no-ops
            # and would leave this callback pending forever.
            self._complete_request_with_error(
                request_id,
                f"Geometry worker GPU {worker.gpu_id} connection failed "
                f"during dispatch: {e}",
            )
            with self._worker_lock:
                if worker.current_request_id == request_id:
                    worker.current_request_id = None
                    worker.current_request_started_at = None
            self._handle_worker_exit(worker.gpu_id, generation)

    def get_stats(self) -> PoolStats:
        """Get aggregate statistics from the pool."""
        with self._stats_lock:
            avg_time = (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times
                else None
            )
            avg_latency = (
                sum(self._end_to_end_latencies) / len(self._end_to_end_latencies)
                if self._end_to_end_latencies
                else None
            )
            avg_queue_wait = (
                (avg_latency - avg_time)
                if (avg_latency is not None and avg_time is not None)
                else None
            )

        with self._worker_lock:
            total_processed = self._completed_requests + self._failed_requests
            worker_details = []
            for gpu_id, worker in self._workers.items():
                completed = self._per_worker_completed.get(gpu_id, 0)
                failed = self._per_worker_failed.get(gpu_id, 0)
                worker_total = completed + failed
                proportion = (
                    worker_total / total_processed if total_processed > 0 else 0
                )
                worker_details.append(
                    {
                        "gpu_id": gpu_id,
                        "pid": worker.process.pid,
                        "generation": worker.generation,
                        "alive": worker.process.poll() is None,
                        "ready": worker.ready,
                        "completed_requests": completed,
                        "failed_requests": failed,
                        "total_requests": worker_total,
                        "proportion": round(proportion, 4),
                    }
                )

        return PoolStats(
            num_workers=self._num_gpus,
            total_requests=self._total_requests,
            completed_requests=self._completed_requests,
            failed_requests=self._failed_requests,
            avg_processing_time_s=avg_time,
            avg_end_to_end_latency_s=avg_latency,
            avg_queue_wait_s=avg_queue_wait,
            max_queue_wait_s=self._max_queue_wait,
            worker_details=worker_details,
        )

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _wait_for_gpu_ready(self, gpu_id: int) -> None:
        """Wait until the CURRENT worker for ``gpu_id`` reports ready.

        Not bound to the WorkerInfo returned by launch: if the initial worker
        dies mid-preload, the reader thread restarts it as a new generation,
        and this wait tracks whichever generation is current. Fails fast when
        the process exits before connecting (no reader thread exists yet, so
        nothing would restart it during initial startup).

        Raises:
            RuntimeError: if no worker becomes ready within
                ``worker_start_timeout_s``, or if the current worker process
                exits before ever connecting to the pool socket.
        """
        deadline = time.monotonic() + self._worker_start_timeout_s
        while time.monotonic() < deadline:
            with self._worker_lock:
                worker = self._workers.get(gpu_id)
                if gpu_id in self._exhausted_gpus:
                    raise RuntimeError(
                        f"Worker for GPU {gpu_id} crash-looped during startup"
                    )
            if worker is None:
                # Popped by an exit handler and not (yet) restarted.
                time.sleep(0.2)
                continue
            if worker.ready_event.wait(timeout=0.5):
                console_logger.info(
                    f"Worker for GPU {gpu_id} ready (PID {worker.process.pid})"
                )
                return
            # Fail fast if the process died before establishing a connection:
            # with no connection there is no reader thread to restart it.
            if (
                worker.process is not None
                and worker.process.poll() is not None
                and worker.connection is None
            ):
                raise RuntimeError(
                    f"Worker for GPU {gpu_id} (PID {worker.process.pid}) exited "
                    f"with code {worker.process.poll()} before connecting; see "
                    f"worker log in {self._socket_dir}"
                )
        raise RuntimeError(
            f"Worker for GPU {gpu_id} did not become ready within "
            f"{self._worker_start_timeout_s:.0f}s"
        )

    def _send_worker_message(self, worker: WorkerInfo, message: dict) -> None:
        """Send one framed message to a worker, serialized by its send lock.

        ``socket.sendall`` does not guarantee frame atomicity across threads;
        stop() (shutdown) and the coordinator (request) may target the same
        socket concurrently, so every parent->worker send must hold the lock.

        Raises:
            TypeError: if the message is not JSON serializable.
            ValueError: if the encoded message exceeds the protocol size cap.
            OSError: if the connection is gone or the send fails.
        """
        self._send_worker_frame(worker, encode_message(message))

    def _send_worker_frame(self, worker: WorkerInfo, frame: bytes) -> None:
        """Send a pre-encoded frame under the worker's per-connection lock."""
        with worker.send_lock:
            conn = worker.connection
            if conn is None:
                raise OSError("worker connection is closed")
            send_encoded_message(conn, frame)

    def _build_worker_command(
        self, socket_path: str, gpu_id: int, token_file: str, log_file: str | None
    ) -> list[str]:
        """Build the argv to launch one worker.

        Two forms are supported so the pool works whether the parent interpreter
        is a normal Python (``python -m``) or Blender (``--python --``):
        - ``python -m scenesmith...worker_entry --socket-path ...``
        - ``blender -b --python <abs worker_entry.py> -- --socket-path ...``
        """
        worker_args = [
            "--socket-path",
            socket_path,
            "--gpu-id",
            str(gpu_id),
            "--token-file",
            token_file,
        ]
        if log_file:
            worker_args += ["--log-file", log_file]

        exe = sys.executable
        if Path(exe).name.startswith("blender"):
            entry = Path(__file__).resolve().parent / "worker_entry.py"
            return [exe, "-b", "--python", str(entry), "--"] + worker_args
        return [
            exe,
            "-m",
            "scenesmith.agent_utils.geometry_generation_server.worker_entry",
        ] + worker_args

    def _start_single_worker(self, gpu_id: int) -> WorkerInfo:
        """Launch a single worker process as a fresh interpreter."""
        with self._worker_lock:
            generation = self._worker_generations.get(gpu_id, 0) + 1
            self._worker_generations[gpu_id] = generation
            self._exhausted_gpus.discard(gpu_id)

            token = secrets.token_hex(16)
            token_file = Path(self._socket_dir) / f"token_{gpu_id}.txt"
            token_file.write_text(token, encoding="utf-8")
            os.chmod(token_file, 0o600)

            # Per-generation stdout/stderr capture. Placed in a persistent
            # directory (next to the pool log file if given), NOT the temp
            # socket dir: these files are the only postmortem evidence when a
            # worker is OOM-killed/SIGKILLed, and must survive pool teardown.
            log_dir = (
                Path(self._log_file).parent if self._log_file else Path.cwd()
            )
            worker_log = log_dir / f"geometry_worker_gpu{gpu_id}_gen{generation}.log"
            log_handle = open(worker_log, "a", buffering=1)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["SCENESMITH_WORKER_GENERATION"] = str(generation)

            cmd = self._build_worker_command(
                self._socket_path, gpu_id, str(token_file), self._log_file
            )

            # Register the worker BEFORE Popen so that the accept thread, which
            # runs concurrently once the child connects, can already find it.
            # The child may connect as soon as it starts; if the entry is absent
            # its hello is rejected as stale.
            worker = WorkerInfo(
                gpu_id=gpu_id,
                generation=generation,
                process=None,  # type: ignore[arg-type]  # filled in below
                token=token,
                log_handle=log_handle,
            )
            self._workers[gpu_id] = worker

            process = subprocess.Popen(
                cmd,
                env=env,
                close_fds=True,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            worker.process = process

            console_logger.info(
                f"Started worker for GPU {gpu_id} (PID: {process.pid}, "
                f"generation {generation})"
            )

            return worker

    def _restart_worker(self, gpu_id: int) -> None:
        """Restart a dead worker, unless the crash-loop guard trips.

        Does NOT block waiting for ready: the new worker becomes ready
        asynchronously via the accept thread -> ``_on_worker_ready``, which
        publishes its availability token. A synchronous ready-wait here would
        deadlock, because it would hold the worker lock while the accept thread
        needs that same lock to process the new worker's hello.
        """
        if not self._running:
            # Shutting down: never spawn during teardown (avoids orphan workers).
            return

        if not self._should_restart(gpu_id):
            with self._worker_lock:
                console_logger.error(
                    f"Worker GPU {gpu_id} has restarted too many times within "
                    f"{self._restart_window_s:.0f}s; disabling auto-restart. "
                    "Geometry requests on this GPU will fail until the pool is "
                    "restarted."
                )
                self._exhausted_gpus.add(gpu_id)
                self._workers.pop(gpu_id, None)
            return

        try:
            worker = self._start_single_worker(gpu_id)
        except Exception as e:
            with self._worker_lock:
                console_logger.error(
                    f"Failed to start worker for GPU {gpu_id}: {e}; "
                    "disabling auto-restart."
                )
                self._exhausted_gpus.add(gpu_id)
                self._workers.pop(gpu_id, None)
            return

        # If the restarted worker dies again before becoming ready, the health
        # monitor will call _handle_worker_exit again and the crash-loop guard
        # will eventually trip.
        console_logger.info(
            f"Restarted worker for GPU {gpu_id} (PID {worker.process.pid})"
        )

    def _should_restart(self, gpu_id: int) -> bool:
        """Crash-loop guard: only allow a bounded number of restarts per window."""
        now = time.time()
        history = self._restart_history.setdefault(gpu_id, deque())
        while history and history[0] < now - self._restart_window_s:
            history.popleft()
        if len(history) >= self._max_restarts_per_window:
            return False
        history.append(now)
        return True

    def _handle_worker_exit(
        self, gpu_id: int, generation: int, restart: bool = True
    ) -> None:
        """Idempotently handle a dead worker: resolve in-flight, then restart.

        Called from both the socket reader thread (on EOF) and the health monitor
        (on ``poll()``). Identity is verified by ``generation`` so only one of the
        two callers wins, preventing duplicate restarts.

        CRITICAL: the worker is detached (popped) from ``self._workers`` and its
        connection closed while holding the lock, but ``_restart_worker`` is
        called OUTSIDE the lock. Otherwise the restarted worker's hello could
        never be accepted: the accept thread needs the same lock that the restart
        path would hold while waiting for ready.
        """
        with self._worker_lock:
            worker = self._workers.get(gpu_id)
            if worker is None or worker.generation != generation:
                return  # Already handled by another thread.
            exit_code = worker.process.poll()

            console_logger.error(
                f"Worker GPU {gpu_id} PID {worker.process.pid} generation "
                f"{generation} exited with code {exit_code}"
            )

            # Resolve any in-flight request so the HTTP endpoint doesn't hang.
            if worker.current_request_id is not None:
                req_id = worker.current_request_id
                worker.current_request_id = None
                worker.current_request_started_at = None
                self._complete_request_with_error(
                    req_id,
                    f"Geometry worker GPU {gpu_id} exited with code {exit_code} "
                    f"while processing request",
                )

            worker.ready = False
            if worker.connection is not None:
                try:
                    worker.connection.close()
                except OSError:
                    pass
                worker.connection = None

            # Detach so a fresh WorkerInfo is registered by the restart. A stale
            # reader thread from the old worker will no-op via generation check.
            self._workers.pop(gpu_id, None)

        # Reclaim resources OUTSIDE the lock: reap the child (avoid zombies) and
        # release the log file handle explicitly rather than relying on GC.
        if worker.process is not None and worker.process.poll() is None:
            self._terminate_process(worker.process)
        if worker.process is not None:
            try:
                worker.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if worker.log_handle is not None:
            try:
                worker.log_handle.close()
            except OSError:
                pass
            worker.log_handle = None

        if not restart:
            with self._worker_lock:
                self._exhausted_gpus.add(gpu_id)
            return

        if not self._running:
            # Shutting down; do not resurrect.
            return

        self._restart_worker(gpu_id)

    def _terminate_process(self, process: subprocess.Popen) -> None:
        """Graceful -> SIGTERM -> SIGKILL escalation for a worker process."""
        if process is None or process.poll() is not None:
            return
        try:
            process.wait(timeout=10)
        except (subprocess.TimeoutExpired, ValueError):
            pass
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        if process.poll() is not None:
            return
        try:
            process.kill()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass

    def _close_worker(self, worker: WorkerInfo) -> None:
        """Close a worker's connection, reader thread and log handle."""
        if worker.connection is not None:
            try:
                worker.connection.close()
            except OSError:
                pass
            worker.connection = None
        if worker.log_handle is not None:
            try:
                worker.log_handle.close()
            except OSError:
                pass
            worker.log_handle = None
        worker.ready = False

    # ------------------------------------------------------------------
    # Socket listener / accept
    # ------------------------------------------------------------------

    def _setup_listener(self) -> None:
        """Create a short-path AF_UNIX listener under /tmp."""
        self._socket_dir = tempfile.mkdtemp(prefix="ss-geom-", dir="/tmp")
        self._socket_path = str(Path(self._socket_dir) / "control.sock")

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self._socket_path)
        os.chmod(self._socket_path, 0o600)
        listener.listen(self._num_gpus if self._num_gpus else 1)
        self._listener = listener
        console_logger.info(
            f"Worker socket listener at {self._socket_path} (mode 0600)"
        )

    def _cleanup_listener(self) -> None:
        """Close the listener and remove the socket directory."""
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._socket_dir is not None:
            try:
                for p in Path(self._socket_dir).iterdir():
                    try:
                        p.unlink()
                    except OSError:
                        pass
                Path(self._socket_dir).rmdir()
            except OSError:
                pass
            self._socket_dir = None
            self._socket_path = None

    def _accept_loop(self) -> None:
        """Accept connections, one handler thread per worker."""
        while self._running:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                break
            handler = threading.Thread(
                target=self._handle_connection, args=(conn,), daemon=True
            )
            handler.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Validate a worker's hello, send init, and start its reader thread."""
        conn.settimeout(10.0)
        try:
            hello = recv_message(conn)
            if not isinstance(hello, dict):
                raise ValueError("worker hello must be a JSON object")
        except (OSError, EOFError, ValueError):
            conn.close()
            return

        gpu_id = hello.get("gpu_id")
        token = hello.get("token")

        with self._worker_lock:
            worker = self._workers.get(gpu_id)
            valid = (
                self._running
                and worker is not None
                and worker.process is not None
                and worker.process.poll() is None
                and hello.get("pid") == worker.process.pid
                and str(hello.get("generation")) == str(worker.generation)
                and token == worker.token
            )
            if not valid:
                console_logger.warning(
                    f"Rejected stale/invalid worker hello for GPU {gpu_id!r}"
                )
                conn.close()
                return

            worker.connection = conn
            generation = worker.generation

        # Encode outside the worker-state lock. A deterministic config encoding
        # error cannot be fixed by restarting the same worker, so mark this GPU
        # exhausted and let startup fail immediately instead of crash-looping.
        try:
            init_frame = encode_message(
                {
                    "type": MSG_INIT,
                    "config": {
                        "use_mini": self._use_mini,
                        "backend": self._backend,
                        "sam3d_config": self._sam3d_config,
                        "preload_pipeline": self._preload_pipeline,
                    },
                }
            )
        except Exception as e:
            console_logger.error(
                f"Invalid init config for worker GPU {gpu_id} generation "
                f"{generation}: {e}"
            )
            try:
                conn.close()
            except OSError:
                pass
            self._handle_worker_exit(gpu_id, generation, restart=False)
            return

        try:
            # Serialize pipeline preload across workers by waiting for ready in
            # the start() loop; here we only hand the worker its config. INIT uses
            # the same send lock as requests and shutdown, so frames cannot
            # interleave if stop() races with a connecting worker.
            self._send_worker_frame(worker, init_frame)
            conn.settimeout(None)  # blocking reads for result frames

            reader = threading.Thread(
                target=self._read_loop,
                args=(gpu_id, generation, conn),
                daemon=True,
                name=f"WorkerReader-{gpu_id}",
            )

            # The worker may have died, been replaced, or the pool may have
            # started stopping while INIT was being sent. Never attach a reader
            # to a stale WorkerInfo/connection.
            with self._worker_lock:
                current = self._workers.get(gpu_id)
                valid = (
                    self._running
                    and current is worker
                    and worker.generation == generation
                    and worker.connection is conn
                    and worker.process is not None
                    and worker.process.poll() is None
                )
                if not valid:
                    raise OSError("worker became stale while sending init")
                worker.reader_thread = reader
                reader.start()
        except Exception as e:
            console_logger.error(
                f"Failed to initialize worker GPU {gpu_id} generation "
                f"{generation}: {e}"
            )
            try:
                conn.close()
            except OSError:
                pass
            # This is a connection/process failure, so normal bounded restart is
            # appropriate. _handle_worker_exit is idempotent if another monitor
            # already replaced this generation.
            self._handle_worker_exit(gpu_id, generation)

    def _read_loop(self, gpu_id: int, generation: int, conn: socket.socket) -> None:
        """Read framed messages from one worker until it disconnects."""
        try:
            while True:
                message = recv_message(conn)
                msg_type = message.get("type")
                if msg_type == MSG_READY:
                    self._on_worker_ready(gpu_id, generation)
                elif msg_type == MSG_RESULT:
                    self._handle_result(gpu_id, generation, conn, message)
                else:
                    console_logger.warning(
                        f"Worker GPU {gpu_id} sent unexpected message type "
                        f"{msg_type!r}"
                    )
        except (EOFError, OSError, ValueError) as e:
            console_logger.debug(
                f"Worker GPU {gpu_id} reader ending: {e!r}"
            )
        finally:
            self._handle_worker_exit(gpu_id, generation)

    def _on_worker_ready(self, gpu_id: int, generation: int) -> None:
        """Mark a worker ready and publish its availability token."""
        with self._worker_lock:
            worker = self._workers.get(gpu_id)
            if worker is None or worker.generation != generation:
                return
            worker.ready = True
            worker.ready_event.set()
            if not self._running:
                # Shutting down: do not re-submit this worker to the available
                # pool, otherwise submit_request could block forever waiting for
                # a worker that stop() is about to kill.
                return
            self._available_workers.put(
                WorkerToken(
                    gpu_id=gpu_id, pid=worker.process.pid, generation=worker.generation
                )
            )
        console_logger.info(
            f"Worker GPU {gpu_id} initialized and ready for requests"
        )

    # ------------------------------------------------------------------
    # Dispatch / results
    # ------------------------------------------------------------------

    def _acquire_available_worker(self) -> WorkerInfo | None:
        """Block for a valid worker token, discarding stale ones.

        Returns the WorkerInfo for a live, ready worker, or None when the pool
        is stopping, the acquire timeout elapses, or every GPU has been marked
        exhausted by the crash-loop guard.
        """
        deadline = time.monotonic() + self._worker_acquire_timeout_s
        while True:
            if not self._running:
                return None
            if time.monotonic() >= deadline:
                console_logger.error(
                    f"Timed out after {self._worker_acquire_timeout_s:.0f}s "
                    "waiting for an available worker"
                )
                return None
            with self._worker_lock:
                if set(self._gpu_ids).issubset(self._exhausted_gpus):
                    return None

            try:
                token = self._available_workers.get(timeout=0.5)
            except queue.Empty:
                continue

            if token is None:
                # Sentinel from stop(): re-queue for any other waiting thread
                # and bail out.
                self._available_workers.put(None)
                return None

            with self._worker_lock:
                worker = self._workers.get(token.gpu_id)
                if (
                    worker is not None
                    and worker.ready
                    and worker.process is not None
                    and worker.process.poll() is None
                    and worker.process.pid == token.pid
                    and worker.generation == token.generation
                ):
                    return worker
            # Stale token; drop it and keep waiting.
            console_logger.warning(f"Discarding stale worker token: {token}")

    def _handle_result(
        self, gpu_id: int, generation: int, conn: socket.socket, message: dict
    ) -> None:
        """Process a result message from a worker.

        Validates that the result comes from the CURRENT worker (matching
        generation and connection) and matches its in-flight request. A late
        result from a dead/replaced worker must be discarded: acting on it would
        clear the new worker's request state, complete the wrong callback, and
        push a duplicate availability token (double dispatch).
        """
        request_id = message.get("request_id")

        with self._worker_lock:
            worker = self._workers.get(gpu_id)
            stale = (
                worker is None
                or worker.generation != generation
                or worker.connection is not conn
                or worker.current_request_id != request_id
            )
            if stale:
                console_logger.warning(
                    f"Discarding stale result for request {request_id} from "
                    f"GPU {gpu_id} generation {generation}"
                )
                return
            worker.current_request_id = None
            worker.current_request_started_at = None

        with self._pending_callbacks_lock:
            callback_info = self._pending_callbacks.pop(request_id, None)

        if callback_info is None:
            console_logger.warning(f"No callback found for request {request_id}")
            self._return_worker_to_available(gpu_id)
            return

        callback, request_index = callback_info
        status = message.get("status")
        processing_time = message.get("processing_time_seconds")
        end_to_end = message.get("end_to_end_latency_seconds")

        with self._stats_lock:
            if status == "success":
                self._completed_requests += 1
                self._per_worker_completed[gpu_id] = (
                    self._per_worker_completed.get(gpu_id, 0) + 1
                )
            else:
                self._failed_requests += 1
                self._per_worker_failed[gpu_id] = (
                    self._per_worker_failed.get(gpu_id, 0) + 1
                )
            if processing_time is not None:
                self._processing_times.append(processing_time)
                if len(self._processing_times) > 10000:
                    self._processing_times.pop(0)
            if end_to_end is not None:
                self._end_to_end_latencies.append(end_to_end)
                if len(self._end_to_end_latencies) > 10000:
                    self._end_to_end_latencies.pop(0)
            if processing_time is not None and end_to_end is not None:
                queue_wait = end_to_end - processing_time
                if self._max_queue_wait is None or queue_wait > self._max_queue_wait:
                    self._max_queue_wait = queue_wait

        try:
            if status == "success":
                callback(request_index, ("success", message.get("data")))
            else:
                callback(request_index, ("error", message.get("error")))
        except Exception as e:
            console_logger.error(f"Callback failed for request {request_id}: {e}")

        self._return_worker_to_available(gpu_id)

    def _complete_request_with_error(
        self, request_id: str, error: str, fail_only: bool = False
    ) -> None:
        """Complete a pending request with an error (idempotent per request_id)."""
        with self._pending_callbacks_lock:
            callback_info = self._pending_callbacks.pop(request_id, None)
            if callback_info is None:
                if not fail_only:
                    console_logger.warning(
                        f"No pending callback for request {request_id}"
                    )
                return

        callback, request_index = callback_info
        with self._stats_lock:
            self._failed_requests += 1
        try:
            callback(request_index, ("error", error))
        except Exception as e:
            console_logger.error(f"Callback for {request_id} raised: {e}")

    def _return_worker_to_available(self, gpu_id: int) -> None:
        """Return a live, ready worker to the availability queue."""
        with self._worker_lock:
            worker = self._workers.get(gpu_id)
            if (
                worker is None
                or not worker.ready
                or worker.process is None
                or worker.process.poll() is not None
            ):
                return
            self._available_workers.put(
                WorkerToken(
                    gpu_id=gpu_id, pid=worker.process.pid, generation=worker.generation
                )
            )

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def _health_monitor_loop(self) -> None:
        """Monitor worker health; restart dead workers, kill hung ones."""
        while self._running:
            time.sleep(5.0)
            if not self._running:
                break

            with self._worker_lock:
                workers = list(self._workers.values())

            for worker in workers:
                if not self._running:
                    break
                if worker.process is not None and worker.process.poll() is not None:
                    self._handle_worker_exit(worker.gpu_id, worker.generation)
                    continue

                # Hung-request detection: if a request has been running longer
                # than request_timeout_s, kill the worker. Its exit is then
                # handled by the normal exit path (resolve callback, restart).
                if self._request_timeout_s is not None:
                    with self._worker_lock:
                        started = worker.current_request_started_at
                        hung = (
                            started is not None
                            and time.monotonic() - started > self._request_timeout_s
                        )
                    if hung and worker.process is not None:
                        console_logger.error(
                            f"Worker GPU {worker.gpu_id} exceeded request timeout "
                            f"({self._request_timeout_s:.0f}s); killing it"
                        )
                        try:
                            worker.process.kill()
                        except OSError:
                            pass
                        # Exit handling (callback resolution + restart) happens
                        # via the reader-EOF / poll() paths.

    # ------------------------------------------------------------------
    # GPU detection (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu_ids() -> list[int]:
        """Detect available GPU IDs, respecting CUDA_VISIBLE_DEVICES if set.

        Uses nvidia-smi to avoid importing torch (which would initialize CUDA
        in the parent process).
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None and cuda_visible.strip():
            try:
                gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                if gpu_ids:
                    console_logger.info(
                        f"Using GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}"
                    )
                    return gpu_ids
            except ValueError:
                console_logger.warning(
                    f"Failed to parse CUDA_VISIBLE_DEVICES='{cuda_visible}', "
                    "falling back to nvidia-smi detection"
                )

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_ids = [int(line.strip()) for line in lines if line.strip()]
                if gpu_ids:
                    return gpu_ids
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            console_logger.warning(f"nvidia-smi detection failed: {e}")

        console_logger.warning("Could not detect GPUs, defaulting to GPU 0")
        return [0]
