"""Unit tests for in-process retrieval HTTP server lifecycles."""

import threading

from unittest.mock import MagicMock

import pytest

from scenesmith.agent_utils.articulated_retrieval_server import (
    server_manager as articulated,
)
from scenesmith.agent_utils.hssd_retrieval_server import server_manager as hssd
from scenesmith.agent_utils.materials_retrieval_server import (
    server_manager as materials,
)
from scenesmith.agent_utils.objaverse_retrieval_server import (
    server_manager as objaverse,
)


class _FakeHTTPServer:
    """Controllable stand-in for Werkzeug's ``BaseWSGIServer``."""

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


SERVER_CASES = [
    (materials, "MaterialsRetrievalServer", "MaterialsRetrievalApp"),
    (articulated, "ArticulatedRetrievalServer", "ArticulatedRetrievalApp"),
    (hssd, "HssdRetrievalServer", "HssdRetrievalApp"),
    (objaverse, "ObjaverseRetrievalServer", "ObjaverseRetrievalApp"),
]


@pytest.mark.parametrize("module,server_name,app_name", SERVER_CASES)
def test_stop_shuts_down_http_server_and_joins_thread(
    monkeypatch, module, server_name, app_name
):
    """stop() must terminate the HTTP loop instead of calling /shutdown."""
    app = MagicMock()
    http_server = _FakeHTTPServer()
    make_server = MagicMock(return_value=http_server)

    monkeypatch.setattr(module, "is_port_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, app_name, MagicMock(return_value=app))
    monkeypatch.setattr(module, "make_server", make_server)
    monkeypatch.setattr(
        module.requests,
        "post",
        MagicMock(side_effect=AssertionError("stop() must not call /shutdown")),
    )

    server_class = getattr(module, server_name)
    monkeypatch.setattr(server_class, "_wait_until_ready", lambda self: None)
    server = server_class(port=7199)
    server.start()
    assert http_server.started.wait(timeout=1)
    server_thread = server._server_thread

    server.stop()

    make_server.assert_called_once_with("127.0.0.1", 7199, app, threaded=True)
    app.start_processing.assert_called_once_with()
    app.stop_processing.assert_called_once_with()
    assert http_server.shutdown_calls == 1
    assert http_server.closed
    assert server_thread is not None
    assert not server_thread.is_alive()
    assert not server.is_running()
    assert server._http_server is None


@pytest.mark.parametrize("module,server_name,app_name", SERVER_CASES)
def test_readiness_failure_closes_http_server_and_processing(
    monkeypatch, module, server_name, app_name
):
    """A partially started server must not leak its HTTP or worker threads."""
    app = MagicMock()
    http_server = _FakeHTTPServer()

    monkeypatch.setattr(module, "is_port_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, app_name, MagicMock(return_value=app))
    monkeypatch.setattr(module, "make_server", MagicMock(return_value=http_server))

    server_class = getattr(module, server_name)

    def fail_readiness(_self):
        raise RuntimeError("not ready")

    monkeypatch.setattr(server_class, "_wait_until_ready", fail_readiness)
    server = server_class(port=7199)

    with pytest.raises(RuntimeError, match="not ready"):
        server.start()

    app.start_processing.assert_called_once_with()
    app.stop_processing.assert_called_once_with()
    assert http_server.shutdown_calls == 1
    assert http_server.closed
    assert not server.is_running()
    assert server._http_server is None
    assert server._server_thread is None
