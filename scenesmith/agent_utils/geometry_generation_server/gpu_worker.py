"""GPU worker process for geometry generation with CUDA isolation.

This module implements the worker process that runs on each GPU. The key design
principle is that CUDA_VISIBLE_DEVICES must be set BEFORE any CUDA-dependent
imports occur.

CRITICAL: This module must NOT import any CUDA/torch/warp code at module level.
All CUDA imports must be deferred to inside gpu_worker_main() after
CUDA_VISIBLE_DEVICES is set.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
    MSG_READY,
    MSG_REQUEST,
    MSG_RESULT,
    MSG_SHUTDOWN,
    recv_message,
    send_message,
)

from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
)

console_logger = logging.getLogger(__name__)


@dataclass
class WorkRequest:
    """Request to process geometry generation on a worker."""

    request_id: str
    """Unique identifier for this request."""

    request: GeometryGenerationServerRequest
    """The actual generation request with all parameters."""

    received_timestamp: float
    """Time when request was received by server (time.time())."""


@dataclass
class WorkResult:
    """Result from a worker after processing a request."""

    request_id: str
    """Unique identifier matching the original request."""

    worker_id: int
    """ID of the worker that processed this request (for availability tracking)."""

    status: str
    """Result status: "success" or "error"."""

    data: dict | None
    """Response data for successful requests (contains geometry_path)."""

    error: str | None
    """Error message for failed requests."""

    processing_time_seconds: float | None = None
    """Time taken to process the request (GPU time only), in seconds."""

    end_to_end_latency_seconds: float | None = None
    """Total time from request received by server to result ready, in seconds."""


class ShutdownRequest:
    """Sentinel class to signal worker shutdown."""


@dataclass
class WorkerReady:
    """Signal that a worker has finished initialization and is ready for requests."""

    worker_id: int
    """ID of the worker that is now ready."""


def socket_worker_run(
    connection: socket.socket,
    gpu_id: int,
    use_mini: bool,
    backend: str,
    sam3d_config: dict | None,
    preload_pipeline: bool,
    log_file: str | None = None,
) -> None:
    """Socket-based main loop for a GPU worker subprocess.

    This replaces the old ``multiprocessing.Queue``-based ``gpu_worker_main``.
    The worker connects to the parent's Unix-domain socket and exchanges framed
    JSON messages (see ``worker_protocol``). Running over a fresh socket means a
    restarted worker never inherits the parent's CUDA state, which is what makes
    worker auto-restart safe after CUDA has been initialized in the parent.

    Args:
        connection: Connected AF_UNIX socket to the worker pool (parent).
        gpu_id: The GPU index this worker is assigned to.
        use_mini: Whether to use mini model variant (Hunyuan3D only).
        backend: Generation backend ("hunyuan3d" or "sam3d").
        sam3d_config: Configuration for SAM3D backend.
        preload_pipeline: Whether to preload pipeline on startup.
        log_file: Optional path to a log file for this worker's stderr copy.
    """
    # CUDA_VISIBLE_DEVICES is set by the parent via the subprocess environment
    # (before this interpreter starts), then verified here. It must be set before
    # ANY CUDA-dependent import.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible != str(gpu_id):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES mismatch: expected {gpu_id}, got {cuda_visible!r}"
        )

    # Configure logging for this worker process. basicConfig(force=True)
    # installs a stderr handler on the root logger; module loggers propagate to
    # it, so no per-logger stderr handler is added (it would double every line).
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[GPU-{gpu_id}] %(levelname)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Log to file if path provided (e.g., experiment.log).
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(f"%(asctime)s - [GPU-{gpu_id}] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

    logger.info(f"Worker starting on GPU {gpu_id}, PID={os.getpid()}")

    # Now safe to import CUDA-dependent code.
    logger.info("Importing geometry_generation module...")
    import_start = time.time()
    from scenesmith.agent_utils.geometry_generation_server.geometry_generation import (
        generate_geometry_from_image,
    )

    logger.info(
        f"geometry_generation import completed in {time.time() - import_start:.2f}s"
    )

    # Preload pipeline if requested (serialized across workers by the parent,
    # which starts workers one-at-a-time and waits for ready).
    if preload_pipeline:
        logger.info(f"Starting pipeline preload for {backend}...")
        preload_start = time.time()
        _preload_pipeline(backend=backend, use_mini=use_mini, sam3d_config=sam3d_config)
        logger.info(f"Pipeline preloaded successfully in {time.time() - preload_start:.2f}s")

    # Track processing statistics.
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    processing_times: list[float] = []

    # Signal that this worker is ready for requests.
    send_message(connection, {"type": MSG_READY, "gpu_id": gpu_id})
    logger.info("Worker ready, waiting for requests...")

    while True:
        try:
            message = recv_message(connection)
        except EOFError:
            logger.info("Socket closed by parent, exiting...")
            break

        msg_type = message.get("type")
        if msg_type == MSG_SHUTDOWN:
            logger.info("Received shutdown signal, exiting...")
            break

        if msg_type != MSG_REQUEST:
            logger.warning(f"Received unknown message type: {msg_type!r}")
            continue

        # Process the request.
        total_requests += 1
        start_time = time.time()
        request_id = message["request_id"]
        received_timestamp = message["received_timestamp"]

        try:
            request = _request_from_dict(message["request"])
            result_data = _process_request(
                request=request,
                generate_fn=generate_geometry_from_image,
                use_mini=use_mini,
            )

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            # Keep only last 10000 times.
            if len(processing_times) > 10000:
                processing_times.pop(0)

            completed_requests += 1
            logger.info(f"Completed request {request_id} in {processing_time:.2f}s")

            end_to_end_latency = time.time() - received_timestamp
            send_message(
                connection,
                {
                    "type": MSG_RESULT,
                    "request_id": request_id,
                    "worker_id": gpu_id,
                    "status": "success",
                    "data": {"geometry_path": result_data.geometry_path},
                    "error": None,
                    "processing_time_seconds": processing_time,
                    "end_to_end_latency_seconds": end_to_end_latency,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            end_to_end_latency = time.time() - received_timestamp
            failed_requests += 1
            logger.error(f"Request {request_id} failed: {e}")

            send_message(
                connection,
                {
                    "type": MSG_RESULT,
                    "request_id": request_id,
                    "worker_id": gpu_id,
                    "status": "error",
                    "data": None,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                    "end_to_end_latency_seconds": end_to_end_latency,
                },
            )

    logger.info(
        f"Worker shutting down. Stats: {total_requests} total, "
        f"{completed_requests} completed, {failed_requests} failed"
    )


def _preload_pipeline(backend: str, use_mini: bool, sam3d_config: dict | None) -> None:
    """Preload the generation pipeline to eliminate first-request latency.

    Args:
        backend: Generation backend ("hunyuan3d" or "sam3d").
        use_mini: Whether to use mini model variant.
        sam3d_config: Configuration for SAM3D backend.
    """
    if backend == "hunyuan3d":
        from scenesmith.agent_utils.geometry_generation_server.hunyuan3d_pipeline_manager import (
            Hunyuan3DPipelineManager,
        )

        Hunyuan3DPipelineManager.get_pipelines(use_mini=use_mini)

    elif backend == "sam3d":
        if sam3d_config is None:
            raise ValueError("sam3d_config required for SAM3D backend")

        from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
            SAM3DPipelineManager,
        )

        sam3_checkpoint = Path(sam3d_config["sam3_checkpoint"])
        sam3d_checkpoint = Path(sam3d_config["sam3d_checkpoint"])
        SAM3DPipelineManager.get_pipelines(
            sam3_checkpoint=sam3_checkpoint, sam3d_checkpoint=sam3d_checkpoint
        )


def _request_from_dict(data: dict) -> Any:
    """Rebuild a GeometryGenerationServerRequest from a JSON-safe dict.

    ``GeometryGenerationServerRequest.to_dict()`` only emits primitive fields
    (via ``dataclasses.asdict``), so the request can be reconstructed directly.
    """
    return GeometryGenerationServerRequest(**data)


def _process_request(
    request: GeometryGenerationServerRequest, generate_fn: Any, use_mini: bool
) -> GeometryGenerationServerResponse:
    """Process a single geometry generation request.

    Args:
        request: The generation request.
        generate_fn: The geometry generation function.
        use_mini: Whether to use mini model variant.

    Returns:
        Response containing path to generated geometry.
    """
    image_path = Path(request.image_path)
    output_dir = Path(request.output_dir)
    prompt = request.prompt
    debug_folder = Path(request.debug_folder) if request.debug_folder else None

    # Use provided filename or generate from prompt.
    if request.output_filename:
        output_filename = request.output_filename
    else:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_filename = f"geometry_{prompt_hash}_{timestamp}.glb"

    output_path = output_dir / output_filename

    # Convert sam3d_config paths from strings to Path objects if present.
    processed_sam3d_config = None
    if request.sam3d_config:
        processed_sam3d_config = request.sam3d_config.copy()
        if "sam3_checkpoint" in processed_sam3d_config:
            processed_sam3d_config["sam3_checkpoint"] = Path(
                processed_sam3d_config["sam3_checkpoint"]
            )
        if "sam3d_checkpoint" in processed_sam3d_config:
            processed_sam3d_config["sam3d_checkpoint"] = Path(
                processed_sam3d_config["sam3d_checkpoint"]
            )

    # Generate 3D geometry.
    generate_fn(
        image_path=image_path,
        output_path=output_path,
        debug_folder=debug_folder,
        use_mini=use_mini,
        use_pipeline_caching=True,
        backend=request.backend,
        sam3d_config=processed_sam3d_config,
    )

    return GeometryGenerationServerResponse(geometry_path=str(output_path))
