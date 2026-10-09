"""The cold-start contract, tested against a real server process.

Why this needs a live process
-----------------------------
Two defects lived in this area and neither was reachable from the in-process test client:

1. The model was loaded inside the ASGI lifespan *before* yielding, so uvicorn did not bind the
   socket until the load finished (~164 s on the reference machine). A launcher that opens a browser
   a few seconds in showed "connection refused".
2. Once binding was fixed, a search during the load called a synchronous loader from the event loop
   and blocked the whole service, so the client saw a timeout rather than a 503.

Both are timing properties of the running server, so they are checked against one. The checkpoint is
deliberately a path that does not exist: that makes "never ready" a *deterministic* state, so the
loading window can be exercised without waiting for a real model. It also pins the fail-fast
behaviour — a service whose model cannot load must still answer /healthz and still report 503
clearly, because that is the difference between "still loading" and "broken" for an operator.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _multipart(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


@pytest.fixture()
def unready_server(tmp_path):
    """A server that binds, then fails to load its model, so it stays not-ready forever.

    A *missing* checkpoint does not work: the entry point validates paths before binding and exits
    with a diagnostic, which is deliberate fail-fast behaviour. A file that exists but is not a valid
    checkpoint passes that validation, binds the port, and fails in the background load — exactly the
    state these tests need, and deterministic rather than a timing race.
    """
    port = _free_port()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_id": "a",
                "identity": "cat-a",
                "path": str(tmp_path / "a.jpg"),
                "source": "test",
                "detector": "whole_image",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "a.jpg").write_bytes(b"not really an image")
    broken_checkpoint = tmp_path / "broken.pt"
    broken_checkpoint.write_bytes(b"this is not a torch checkpoint")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "catface.web",
            "--port",
            str(port),
            "--device",
            "cpu",
            "--no-browser",
            "--checkpoint",
            str(broken_checkpoint),
            "--manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def wait_for(url: str, timeout: float = 30.0) -> float | None:
    """Seconds until ``/healthz`` answers, or ``None`` if it never did."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=2) as reply:
                if reply.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return None


class TestColdStartContract:
    def test_liveness_answers_before_the_model_is_ready(self, unready_server):
        """The socket must be bound while loading, so a launcher can open the page immediately."""
        elapsed = wait_for(unready_server)
        assert elapsed is not None, "/healthz never answered: the port is not bound early"
        assert elapsed < 20, (
            f"/healthz took {elapsed:.1f}s; binding must not wait for the model load, or the "
            "launcher shows a dead page"
        )

    def test_status_reports_not_ready_without_raising(self, unready_server):
        """Readiness is a field, not an error: an operator needs to read it."""
        assert wait_for(unready_server) is not None
        with urllib.request.urlopen(f"{unready_server}/api/status", timeout=10) as reply:
            payload = json.loads(reply.read())
        assert payload["ready"] is False
        assert payload["gallery_images"] == 0

    def test_search_returns_503_immediately_instead_of_blocking(self, unready_server):
        """The measured defect: a search during the load timed out instead of returning 503.

        Returning the 503 only after the load finishes is not a fix: the client would hold the
        connection for minutes, which is indistinguishable from a hang.
        """
        assert wait_for(unready_server) is not None
        body, content_type = _multipart("file", "q.jpg", b"not an image")
        request = urllib.request.Request(
            f"{unready_server}/api/search", data=body, headers={"Content-Type": content_type}
        )
        began = time.monotonic()
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(request, timeout=15)
        elapsed = time.monotonic() - began
        assert info.value.code == 503, f"expected 503 while unready, got {info.value.code}"
        assert elapsed < 5, (
            f"the 503 took {elapsed:.1f}s, so the handler is waiting on the load rather than "
            "reporting readiness"
        )
        assert "still loading" in info.value.read().decode()

    def test_malformed_upload_is_rejected_before_readiness_is_considered(self, unready_server):
        """A client error must not be masked by the not-ready state.

        Validation runs first so that a wrong extension is reported as a wrong extension, not as a
        503 that sends the caller looking at the server.
        """
        assert wait_for(unready_server) is not None
        body, content_type = _multipart("file", "notes.txt", b"hello")
        request = urllib.request.Request(
            f"{unready_server}/api/search", data=body, headers={"Content-Type": content_type}
        )
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(request, timeout=15)
        assert info.value.code == 400, "an unsupported extension must be 400 even while unready"
        assert "accepted" in info.value.read().decode()
