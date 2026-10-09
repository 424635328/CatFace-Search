"""Port selection for the launcher.

A one-click launcher must not be defeated by the default port being busy. On the machine this was
developed on, port 8000 is held by an unrelated product whose listener *answers TCP* — with an HTTP
502 — so a browser shows a gateway error while the server's own bind fails with WinError 10013. The
distinction matters: "something answers on this port" is not the same question as "can I bind it",
and only the second one predicts whether uvicorn will start.

These tests exercise the bind-based check rather than mocking it, because the check's whole value is
that it agrees with what the OS will do.
"""

from __future__ import annotations

import socket

import pytest

from catface.web.__main__ import _choose_port, _port_is_free


def _bound_socket() -> tuple[socket.socket, int]:
    """A socket that is listening on an arbitrary free port."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    return holder, int(holder.getsockname()[1])


class TestPortIsFree:
    def test_reports_an_ephemeral_port_as_free(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()
        assert _port_is_free("127.0.0.1", port) is True

    def test_reports_a_listening_port_as_busy(self):
        """The real case: a port another process is listening on must be detected."""
        holder, port = _bound_socket()
        try:
            assert _port_is_free("127.0.0.1", port) is False, (
                "a port with a live listener must be reported as busy, or uvicorn will fail to bind"
            )
        finally:
            holder.close()

    def test_the_wildcard_host_is_probed_as_loopback(self):
        holder, port = _bound_socket()
        try:
            assert _port_is_free("0.0.0.0", port) is False
        finally:
            holder.close()

    def test_a_privileged_free_port_is_judged_by_the_bind_not_by_policy(self):
        """The answer must come from the OS, since privileges vary by platform and user."""
        result = _port_is_free("127.0.0.1", 80)
        assert isinstance(result, bool)


class TestChoosePort:
    def test_returns_the_preferred_port_when_it_is_free(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()
        assert _choose_port("127.0.0.1", port, strict=False) == port

    def test_moves_to_the_next_free_port_when_the_preferred_one_is_busy(self):
        holder, busy = _bound_socket()
        try:
            chosen = _choose_port("127.0.0.1", busy, strict=False)
            assert chosen is not None
            assert chosen > busy, "the launcher must move on rather than fail"
            assert _port_is_free("127.0.0.1", chosen)
        finally:
            holder.close()

    def test_strict_mode_refuses_to_move(self):
        """A scripted caller that depends on the exact port must be able to say so."""
        holder, busy = _bound_socket()
        try:
            assert _choose_port("127.0.0.1", busy, strict=True) is None
        finally:
            holder.close()

    def test_reports_failure_when_no_port_in_range_is_free(self, monkeypatch):
        """Consecutive busy ports must end in a clear failure, not an endless search."""
        import catface.web.__main__ as module

        monkeypatch.setattr(module, "_port_is_free", lambda host, port: False)
        assert _choose_port("127.0.0.1", 9000, strict=False, attempts=3) is None


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_choose_port_accepts_hostnames(host):
    """``localhost`` must work: getaddrinfo decides the address family, not the caller."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    chosen = _choose_port(host, port, strict=False)
    assert chosen is not None
