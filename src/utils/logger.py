"""Thin wrapper for module-scoped structlog loggers."""
from __future__ import annotations

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to the given module name."""
    return structlog.get_logger(name)
