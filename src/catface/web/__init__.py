"""Web interface for cat-face identity search.

``python -m catface.web`` serves a search page and a JSON API over a trained checkpoint. The
framework-independent part lives in :mod:`catface.web.service`; :mod:`catface.web.api` only
translates HTTP to it.
"""

from __future__ import annotations

from .service import MAX_UPLOAD_BYTES, QueryRejectedError, SearchService

__all__ = ["MAX_UPLOAD_BYTES", "QueryRejectedError", "SearchService"]
