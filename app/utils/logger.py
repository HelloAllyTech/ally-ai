import logging
import logging.config
import contextvars

from app.core.config import settings

# Context variable for Trace ID
trace_id_var = contextvars.ContextVar("trace_id", default="N/A")


def get_trace_id():
    """Retrieve the current Trace ID."""
    return trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """Custom logging filter to add Trace ID to log records."""

    def filter(self, record):
        record.trace_id = get_trace_id()
        return True


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": (
                "[%(asctime)s] [%(levelname)s] [%(name)s] "
                "[%(filename)s:%(lineno)d] [TraceID:%(trace_id)s] - %(message)s"
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "filters": {
        "trace_id": {
            "()": TraceIdFilter,
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": ["trace_id"],
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "root": {"level": settings.LOG_LEVEL, "handlers": ["stdout"]},
        "lifeline-ai": {
            "level": settings.LOG_LEVEL,
            "handlers": ["stdout"],
            "propagate": False,
        },
        "uvicorn": {
            "level": settings.LOG_LEVEL,
            "handlers": ["stdout"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("lifeline-ai")


def get_logger(logger_name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(logger_name)
