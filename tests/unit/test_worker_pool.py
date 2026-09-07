"""Unit tests for GPU worker pool (post socket-IPC refactor)."""

import socket
import subprocess
import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.gpu_worker import (
    ShutdownRequest,
    WorkerReady,
    WorkRequest,
    WorkResult,
)
from scenesmith.agent_utils.geometry_generation_server.worker_pool import (
    GPUWorkerPool,
    PoolStats,
    WorkerInfo,
    WorkerToken,
)
from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
    encode_message,
    recv_message,
    send_message,
)


class TestGPUDetection(unittest.TestCase):
    """Test GPU detection logic."""

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_multiple_gpus(self, mock_run):
        """Test detection of multiple GPUs via nvidia-smi."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0\n1\n2\n3\n",
        )

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0, 1, 2, 3])
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertIn("nvidia-smi", call_args[0][0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_single_gpu(self, mock_run):
        """Test detection of single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0\n",
        )

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_nvidia_smi_fails(self, mock_run):
        """Test fallback when nvidia-smi fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        # Should fall back to GPU 0.
        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7"})
    def test_detect_gpu_ids_respects_cuda_visible_devices(self):
        """Test that CUDA_VISIBLE_DEVICES is respected."""
        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [1, 2, 3, 4, 5, 6, 7])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "  2 , 5 , 7  "})
    def test_detect_gpu_ids_handles_whitespace(self):
        """Test that whitespace in CUDA_VISIBLE_DEVICES is handled."""
        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [2, 5, 7])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""})
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_empty_cuda_visible_devices(self, mock_run):
        """Test empty CUDA_VISIBLE_DEVICES falls back to nvidia-smi."""
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n1\n")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0, 1])
        mock_run.assert_called_once()


class TestPoolStats(unittest.TestCase):
    """Test PoolStats dataclass."""

    def test_pool_stats_creation(self):
        """Test PoolStats creation with all fields."""
        stats = PoolStats(
            num_workers=4,
            total_requests=100,
            completed_requests=95,
            failed_requests=5,
            avg_processing_time_s=25.5,
            avg_end_to_end_latency_s=30.0,
            avg_queue_wait_s=4.5,
            max_queue_wait_s=10.0,
            worker_details=[
                {"gpu_id": 0, "pid": 1234, "alive": True},
                {"gpu_id": 1, "pid": 1235, "alive": True},
            ],
        )

        self.assertEqual(stats.num_workers, 4)
        self.assertEqual(stats.total_requests, 100)
        self.assertEqual(stats.completed_requests, 95)
        self.assertEqual(stats.failed_requests, 5)
        self.assertEqual(stats.avg_processing_time_s, 25.5)
        self.assertEqual(stats.avg_end_to_end_latency_s, 30.0)
        self.assertEqual(stats.avg_queue_wait_s, 4.5)
        self.assertEqual(stats.max_queue_wait_s, 10.0)
        self.assertEqual(len(stats.worker_details), 2)

    def test_pool_stats_none_avg_time(self):
        """Test PoolStats with no average processing time."""
        stats = PoolStats(
            num_workers=1,
            total_requests=0,
            completed_requests=0,
            failed_requests=0,
            avg_processing_time_s=None,
            avg_end_to_end_latency_s=None,
            avg_queue_wait_s=None,
            max_queue_wait_s=None,
            worker_details=[],
        )

        self.assertIsNone(stats.avg_processing_time_s)
        self.assertIsNone(stats.avg_end_to_end_latency_s)
        self.assertIsNone(stats.avg_queue_wait_s)
        self.assertIsNone(stats.max_queue_wait_s)


class TestWorkRequestResult(unittest.TestCase):
    """Test WorkRequest and WorkResult dataclasses (kept for compatibility)."""

    def test_work_request_creation(self):
        """Test WorkRequest creation."""
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="A wooden chair",
        )

        work_request = WorkRequest(
            request_id="test-123",
            request=request,
            received_timestamp=1234567890.0,
        )

        self.assertEqual(work_request.request_id, "test-123")
        self.assertEqual(work_request.request.prompt, "A wooden chair")
        self.assertEqual(work_request.received_timestamp, 1234567890.0)

    def test_work_result_success(self):
        """Test WorkResult for successful request."""
        result = WorkResult(
            request_id="test-123",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output/chair.glb"},
            error=None,
        )

        self.assertEqual(result.request_id, "test-123")
        self.assertEqual(result.worker_id, 0)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["geometry_path"], "/test/output/chair.glb")
        self.assertIsNone(result.error)

    def test_work_result_error(self):
        """Test WorkResult for failed request."""
        result = WorkResult(
            request_id="test-456",
            worker_id=1,
            status="error",
            data=None,
            error="Generation failed: out of memory",
        )

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertEqual(result.error, "Generation failed: out of memory")


class TestShutdownRequest(unittest.TestCase):
    """Test ShutdownRequest sentinel class (kept for compatibility)."""

    def test_shutdown_request_is_distinct_type(self):
        """Test that ShutdownRequest is distinguishable from other types."""
        shutdown = ShutdownRequest()
        work_request = WorkRequest(
            request_id="test",
            request=GeometryGenerationServerRequest(
                image_path="/test/image.png",
                output_dir="/test/output",
                prompt="test",
            ),
            received_timestamp=1234567890.0,
        )

        self.assertIsInstance(shutdown, ShutdownRequest)
        self.assertNotIsInstance(work_request, ShutdownRequest)
        self.assertNotIsInstance("string", ShutdownRequest)


class TestWorkerReady(unittest.TestCase):
    """Test WorkerReady signal class (kept for compatibility)."""

    def test_worker_ready_creation(self):
        """Test WorkerReady creation with worker ID."""
        ready = WorkerReady(worker_id=3)

        self.assertEqual(ready.worker_id, 3)

    def test_worker_ready_is_distinct_type(self):
        """Test that WorkerReady is distinguishable from other message types."""
        ready = WorkerReady(worker_id=0)
        shutdown = ShutdownRequest()
        work_result = WorkResult(
            request_id="test",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output.glb"},
            error=None,
        )

        self.assertIsInstance(ready, WorkerReady)
        self.assertNotIsInstance(shutdown, WorkerReady)
        self.assertNotIsInstance(work_result, WorkerReady)


class TestWorkerProtocol(unittest.TestCase):
    """Test length-prefixed JSON framing over a socket."""

    def test_roundtrip(self):
        """A message sent is received intact and equal."""
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        msg = {"type": "hello", "gpu_id": 0, "pid": 1, "token": "abc"}
        send_message(a, msg)
        self.assertEqual(recv_message(b), msg)

    def test_message_boundaries_preserved(self):
        """Multiple back-to-back messages are not merged."""
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        for i in range(5):
            send_message(a, {"type": "result", "index": i, "data": "x" * 500})
        for i in range(5):
            self.assertEqual(recv_message(b)["index"], i)

    def test_partial_delivery(self):
        """A message split across many small reads is reassembled correctly."""
        buf = encode_message({"type": "ready", "gpu_id": 0})
        pos = 0

        class FakeSock:
            def __init__(self, data, chunk):
                self.data = data
                self.chunk = chunk
                self.off = 0

            def recv(self, n):
                c = self.data[self.off : self.off + min(n, self.chunk)]
                self.off += len(c)
                return c

        self.assertEqual(recv_message(FakeSock(buf, 1)), {"type": "ready", "gpu_id": 0})

    def test_payload_cap(self):
        """Encoding a payload over the size cap raises."""
        from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
            MAX_MESSAGE_BYTES,
        )

        with self.assertRaises(ValueError):
            encode_message({"x": "y" * (MAX_MESSAGE_BYTES + 16)})


class TestWorkerInfoToken(unittest.TestCase):
    """Test the new WorkerInfo / WorkerToken dataclasses."""

    def test_worker_token_eq(self):
        """WorkerToken equality follows its identity fields."""
        t1 = WorkerToken(gpu_id=0, pid=100, generation=1)
        t2 = WorkerToken(gpu_id=0, pid=100, generation=1)
        t3 = WorkerToken(gpu_id=0, pid=101, generation=1)
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_worker_info_fields(self):
        """WorkerInfo carries the fields the pool relies on."""
        info = WorkerInfo(
            gpu_id=0,
            generation=3,
            process=None,  # type: ignore[arg-type]
            token="secret",
        )
        self.assertEqual(info.gpu_id, 0)
        self.assertEqual(info.generation, 3)
        self.assertIsNone(info.process)
        self.assertEqual(info.token, "secret")
        self.assertFalse(info.ready)


class TestWorkerPoolInitialization(unittest.TestCase):
    """Test GPUWorkerPool initialization (without starting)."""

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0, 1, 2, 3])
    def test_pool_initialization_defaults(self, mock_detect):
        """Test pool initialization with default parameters."""
        pool = GPUWorkerPool()

        self.assertEqual(pool.num_workers, 4)
        self.assertEqual(pool._use_mini, False)
        self.assertEqual(pool._backend, "hunyuan3d")
        self.assertIsNone(pool._sam3d_config)
        self.assertTrue(pool._preload_pipeline)
        # No multiprocessing context anymore — worker IPC is over sockets.
        self.assertEqual(pool._workers, {})
        self.assertIsNone(pool._listener)

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0, 1])
    def test_pool_initialization_custom_params(self, mock_detect):
        """Test pool initialization with custom parameters."""
        sam3d_config = {"sam3_checkpoint": "/path/to/sam3.pt"}

        pool = GPUWorkerPool(
            use_mini=True,
            backend="sam3d",
            sam3d_config=sam3d_config,
            preload_pipeline=False,
        )

        self.assertEqual(pool.num_workers, 2)
        self.assertTrue(pool._use_mini)
        self.assertEqual(pool._backend, "sam3d")
        self.assertEqual(pool._sam3d_config, sam3d_config)
        self.assertFalse(pool._preload_pipeline)

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_pool_stats_before_start(self, mock_detect):
        """Test getting stats before pool is started."""
        pool = GPUWorkerPool()

        stats = pool.get_stats()

        self.assertEqual(stats.num_workers, 1)
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.completed_requests, 0)
        self.assertEqual(stats.failed_requests, 0)
        self.assertIsNone(stats.avg_processing_time_s)
        self.assertEqual(stats.worker_details, [])

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_build_worker_command_python(self, mock_detect):
        """The worker command launches worker_entry via `python -m`."""
        pool = GPUWorkerPool()
        cmd = pool._build_worker_command("/tmp/s.sock", 0, "/tmp/tok", "/tmp/log")
        self.assertIn("worker_entry", cmd[cmd.index("-m") + 1])
        self.assertIn("--socket-path", cmd)


class TestWorkerPoolFailurePaths(unittest.TestCase):
    """Regression tests for request encoding and worker initialization failures."""

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_oversized_request_fails_before_acquiring_worker(self, _mock_detect):
        from scenesmith.agent_utils.geometry_generation_server.worker_protocol import (
            MAX_MESSAGE_BYTES,
        )

        pool = GPUWorkerPool()
        pool._running = True
        callback = MagicMock()
        request = MagicMock()
        request.to_dict.return_value = {"prompt": "x" * (MAX_MESSAGE_BYTES + 1)}

        with patch.object(pool, "_acquire_available_worker") as mock_acquire:
            pool.submit_request(request, callback, 7, 123.0)

        mock_acquire.assert_not_called()
        callback.assert_called_once()
        index, result = callback.call_args.args
        self.assertEqual(index, 7)
        self.assertEqual(result[0], "error")
        self.assertIn("Message payload too large", result[1])
        self.assertEqual(pool._pending_callbacks, {})

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_non_json_request_fails_before_acquiring_worker(self, _mock_detect):
        pool = GPUWorkerPool()
        pool._running = True
        callback = MagicMock()
        request = MagicMock()
        request.to_dict.return_value = {"not_json": object()}

        with patch.object(pool, "_acquire_available_worker") as mock_acquire:
            pool.submit_request(request, callback, 3, 123.0)

        mock_acquire.assert_not_called()
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[1][0], "error")
        self.assertEqual(pool._pending_callbacks, {})

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_request_conversion_failure_completes_callback(self, _mock_detect):
        pool = GPUWorkerPool()
        pool._running = True
        callback = MagicMock()
        request = MagicMock()
        request.to_dict.side_effect = RuntimeError("conversion failed")

        with patch.object(pool, "_acquire_available_worker") as mock_acquire:
            pool.submit_request(request, callback, 5, 123.0)

        mock_acquire.assert_not_called()
        callback.assert_called_once()
        index, result = callback.call_args.args
        self.assertEqual(index, 5)
        self.assertEqual(result[0], "error")
        self.assertIn("conversion failed", result[1])
        self.assertEqual(pool._pending_callbacks, {})

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_init_send_failure_closes_connection_and_restarts(self, _mock_detect):
        class FailingConnection:
            def __init__(self):
                self.closed = False
                self.timeouts = []

            def settimeout(self, timeout):
                self.timeouts.append(timeout)

            def sendall(self, _frame):
                raise BrokenPipeError("worker disconnected")

            def close(self):
                self.closed = True

        pool = GPUWorkerPool()
        pool._running = True
        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.return_value = 0
        worker = WorkerInfo(
            gpu_id=0,
            generation=1,
            process=process,
            token="secret",
        )
        pool._workers[0] = worker
        conn = FailingConnection()
        hello = {
            "type": "hello",
            "gpu_id": 0,
            "pid": 1234,
            "generation": "1",
            "token": "secret",
        }

        with (
            patch(
                "scenesmith.agent_utils.geometry_generation_server.worker_pool.recv_message",
                return_value=hello,
            ),
            patch.object(pool, "_terminate_process") as mock_terminate,
            patch.object(pool, "_restart_worker") as mock_restart,
        ):
            pool._handle_connection(conn)

        self.assertTrue(conn.closed)
        self.assertNotIn(0, pool._workers)
        mock_terminate.assert_called_once_with(process)
        mock_restart.assert_called_once_with(0)

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_invalid_init_config_disables_restart(self, _mock_detect):
        class RecordingConnection:
            def __init__(self):
                self.closed = False
                self.send_calls = 0

            def settimeout(self, _timeout):
                pass

            def sendall(self, _frame):
                self.send_calls += 1

            def close(self):
                self.closed = True

        pool = GPUWorkerPool(sam3d_config={"not_json": object()})
        pool._running = True
        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.return_value = 0
        worker = WorkerInfo(
            gpu_id=0,
            generation=1,
            process=process,
            token="secret",
        )
        pool._workers[0] = worker
        conn = RecordingConnection()
        hello = {
            "type": "hello",
            "gpu_id": 0,
            "pid": 1234,
            "generation": "1",
            "token": "secret",
        }

        with (
            patch(
                "scenesmith.agent_utils.geometry_generation_server.worker_pool.recv_message",
                return_value=hello,
            ),
            patch.object(pool, "_terminate_process"),
            patch.object(pool, "_restart_worker") as mock_restart,
        ):
            pool._handle_connection(conn)

        self.assertTrue(conn.closed)
        self.assertEqual(conn.send_calls, 0)
        self.assertNotIn(0, pool._workers)
        self.assertIn(0, pool._exhausted_gpus)
        mock_restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
