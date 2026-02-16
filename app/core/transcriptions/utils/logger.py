import contextvars
import logging
import logging.config
import sys
from logging import LogRecord

# Context variable for Trace ID
trace_id_var = contextvars.ContextVar("trace_id", default="N/A")


def get_trace_id():
    """Retrieve the current Trace ID."""
    return trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """Custom logging filter to add Trace ID to log records."""

    def filter(self, record: LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


def setup_logging():
    """Setup logging configuration for Lambda"""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] "
        "[%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific logger levels
    logging.getLogger("lifeline-ai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(logger_name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(logger_name)


# Initialize logging
setup_logging()
logger = logging.getLogger("lifeline-ai")
