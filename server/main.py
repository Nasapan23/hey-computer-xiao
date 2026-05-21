from __future__ import annotations

try:
    from .wake_server.app import app
except ImportError:
    from wake_server.app import app
