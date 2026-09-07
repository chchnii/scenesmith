"""Unit tests for the geometry HTTP server lifecycle."""

import threading
import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.geometry_generation_server.server_manager import (
    GeometryGenerationServer,
)


class _FakeHTTPServer:
    """Small controllable stand-in for Werkzeug's BaseWSGIServer."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.closed = False
        self.shutdown_calls = 0

    def serve_forever(self) -> None:
        self.started.set()
        self.stopped.wait(timeout=5)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.stopped.set()

    def server_close(self) -> None:
        self.closed = True


class TestGeometryGenerationServerLifecycle(unittest.TestCase):
    """Verify that the manager owns and closes the Werkzeug server."""

    @patch(
        "scenesmith.agent_utils.geometry_generation_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_stop_shuts_down_http_server_and_joins_thread(self, _mock_port):
        app = MagicMock()
        http_server = _FakeHTTPServer()

        with (
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.GeometryGenerationApp",
                return_value=app,
            ),
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.make_server",
                return_value=http_server,
            ) as mock_make_server,
            patch.object(GeometryGenerationServer, "_wait_until_ready"),
        ):
            server = GeometryGenerationServer(port=7105)
            server.start()
            self.assertTrue(http_server.started.wait(timeout=1))
            server_thread = server._server_thread

            server.stop()

        mock_make_server.assert_called_once_with(
            "127.0.0.1", 7105, app, threaded=True
        )
        app.start_processing.assert_called_once_with()
        app.stop_processing.assert_called_once_with()
        self.assertEqual(http_server.shutdown_calls, 1)
        self.assertTrue(http_server.closed)
        self.assertIsNotNone(server_thread)
        self.assertFalse(server_thread.is_alive())
        self.assertFalse(server.is_running())
        self.assertIsNone(server._http_server)

    @patch(
        "scenesmith.agent_utils.geometry_generation_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_readiness_failure_stops_http_server_and_workers(self, _mock_port):
        app = MagicMock()
        http_server = _FakeHTTPServer()

        with (
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.GeometryGenerationApp",
                return_value=app,
            ),
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.make_server",
                return_value=http_server,
            ),
            patch.object(
                GeometryGenerationServer,
                "_wait_until_ready",
                side_effect=RuntimeError("not ready"),
            ),
        ):
            server = GeometryGenerationServer(port=7105)
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                server.start()

        app.start_processing.assert_called_once_with()
        app.stop_processing.assert_called_once_with()
        self.assertEqual(http_server.shutdown_calls, 1)
        self.assertTrue(http_server.closed)
        self.assertFalse(server.is_running())
        self.assertIsNone(server._http_server)
        self.assertIsNone(server._server_thread)

    @patch(
        "scenesmith.agent_utils.geometry_generation_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_thread_start_failure_closes_server_without_joining(self, _mock_port):
        app = MagicMock()
        http_server = _FakeHTTPServer()

        with (
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.GeometryGenerationApp",
                return_value=app,
            ),
            patch(
                "scenesmith.agent_utils.geometry_generation_server.server_manager.make_server",
                return_value=http_server,
            ),
            patch.object(
                threading.Thread, "start", side_effect=RuntimeError("thread failed")
            ),
        ):
            server = GeometryGenerationServer(port=7105)
            with self.assertRaisesRegex(RuntimeError, "thread failed"):
                server.start()

        app.start_processing.assert_called_once_with()
        app.stop_processing.assert_called_once_with()
        self.assertEqual(http_server.shutdown_calls, 0)
        self.assertTrue(http_server.closed)
        self.assertFalse(server.is_running())


if __name__ == "__main__":
    unittest.main()
