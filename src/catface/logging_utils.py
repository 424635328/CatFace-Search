"""Structured, reproducible logging.

Every run gets a ``run_id`` that is stamped onto each log record and, later, onto
every artifact and metrics file. That is what makes an experiment traceable from
a number in a report back to the exact code, config and weights that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_NAME = "catface"
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Machine-readable log lines for log aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        }
        for key in ("stage", "epoch", "step", "metric", "value", "duration_s"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class _HumanFormatter(logging.Formatter):
    """Readable console output."""

    def __init__(self, run_id: str | None) -> None:
        suffix = f" [{run_id}]" if run_id else ""
        super().__init__(fmt=f"%(asctime)s %(levelname)-7s {suffix} %(message)s",
                         datefmt="%H:%M:%S")


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    run_id: str | None = None,
    json_console: bool = False,
) -> logging.Logger:
    """Configure the package logger exactly once and return it.

    Args:
        level: Root level name (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
        log_file: Optional path; a JSON-lines file handler is attached.
        run_id: Correlates every record with one pipeline run.
        json_console: Emit JSON on stdout instead of human-readable lines.

    Returns:
        The configured ``catface`` logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonFormatter() if json_console else _HumanFormatter(run_id))
        logger.addHandler(handler)
        _CONFIGURED = True

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid stacking duplicate file handlers when a run is re-entered.
        for existing in list(logger.handlers):
            if isinstance(existing, logging.FileHandler) and Path(existing.baseFilename) == path.resolve():
                logger.removeHandler(existing)
                existing.close()
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``catface`` namespace."""
    base = logging.getLogger(_LOGGER_NAME)
    if not base.handlers:
        configure_logging()
    return base if not name else base.getChild(name)


def new_run_id() -> str:
    """Generate a sortable, unique identifier for one pipeline run.

    The timestamp prefix keeps run directories in chronological order, which
    matters when scanning ``artifacts/`` during incident review.
    """
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@contextmanager
def timed(logger: logging.Logger, message: str, **extra: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log its duration, even when it raises.

    Yields a mutable dict so the caller can attach outcomes (e.g. row counts).
    """
    start = time.perf_counter()
    payload: dict[str, Any] = dict(extra)
    try:
        yield payload
    finally:
        payload["duration_s"] = round(time.perf_counter() - start, 4)
        logger.info("%s", message, extra=payload)


def environment_report() -> dict[str, Any]:
    """Collect the exact runtime fingerprint that a run depends on.

    Recorded into ``metrics.json`` so a reviewer can tell whether two numbers
    were produced by comparable hardware and library versions.
    """
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }

    try:  # Optional dependencies: report if present, never fail the run.
        import numpy

        report["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover - numpy is a hard dependency in practice
        pass

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info()
            report["cuda_memory_gb"] = round(total / 1024**3, 2)
            report["cuda_memory_free_gb"] = round(free / 1024**3, 2)
    except Exception:
        pass

    try:
        import torchvision

        report["torchvision"] = torchvision.__version__
    except Exception:
        pass

    try:
        import faiss

        report["faiss"] = getattr(faiss, "__version__", "unknown")
    except Exception:
        pass

    try:
        report["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
        report["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip()
        )
    except Exception:
        report["git_commit"] = None

    return report


def configure_utf8_console() -> None:
    """Best-effort UTF-8 console/OS setup for Windows terminals.

    The original scripts printed Chinese status text; on a cp936/cp1252 console
    that raises ``UnicodeEncodeError`` mid-run. Forcing UTF-8 avoids a class of
    environment-specific crashes.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")
    if platform.system() == "Windows":
        try:  # pragma: no cover - platform specific
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def which_or_none(program: str) -> str | None:
    """Return an absolute path for ``program`` if it is on PATH."""
    return shutil.which(program)


def flatten(prefix: str, mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested mappings into ``prefix.key`` form for flat metric tables."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(name, value))
        else:
            out[name] = value
    return out


__all__ = [
    "configure_logging",
    "configure_utf8_console",
    "environment_report",
    "flatten",
    "get_logger",
    "new_run_id",
    "timed",
    "which_or_none",
]
