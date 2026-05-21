"""
Structured logging configuration.

Uses structlog to emit JSON logs in production (machine-parseable for
SIEM/observability stacks) and pretty console logs in development.
Call configure_logging() once at application startup.
"""
import logging
import sys

import structlog

from config.settings import settings


def configure_logging() -> None:
    """Configure structlog. Idempotent; safe to call multiple times."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.environment == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to the given module name."""
    return structlog.get_logger(name)