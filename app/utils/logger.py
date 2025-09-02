import contextvars
import logging
import logging.config
from logging import LogRecord

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from app.core.config import settings

# Context variable for Trace ID
trace_id_var = contextvars.ContextVar("trace_id", default="N/A")

_slack_alerts_logger = logging.getLogger(__name__)


def get_trace_id():
    """Retrieve the current Trace ID."""
    return trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """Custom logging filter to add Trace ID to log records."""

    def filter(self, record: LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


class SlackSdkHandler(logging.Handler):
    """Custom logging handler to send logs to a Slack channel using Slack SDK."""

    def __init__(self, token, channel_id):
        super().__init__()
        # Slack client is initialized as a Synchronous client
        # since Handler is synchronous.
        # To make it async, can go for asyncio or emitting as a background tasks
        self.client = WebClient(token=token)
        self.channel_id = channel_id

    def emit(self, record: LogRecord) -> None:
        try:
            log_entry = self.format(record)
            self.client.chat_postMessage(
                channel=self.channel_id,
                text=f"*Log Level*: {record.levelname}\n*Message*: {log_entry}",
            )
        except SlackApiError as e:
            _slack_alerts_logger.error(
                f"Failed to send log to Slack: {e.response['error']}"
            )


handlers = {
    "stdout": {
        "class": "logging.StreamHandler",
        "formatter": "simple",
        "filters": ["trace_id"],
        "stream": "ext://sys.stdout",
    },
}

loggers = {
    "root": {"level": settings.LOG.LEVEL, "handlers": ["stdout"]},
    "lifeline-ai": {
        "level": settings.LOG.LEVEL,
        "handlers": ["stdout"],
        "propagate": False,
    },
    "uvicorn": {
        "level": settings.LOG.LEVEL,
        "handlers": ["stdout"],
        "propagate": False,
    },
    "uvicorn.access": {
        "level": settings.LOG.LEVEL,
        "handlers": ["stdout"],
        "propagate": False,
    },
    "slack": {"level": settings.LOG.LEVEL, "handlers": ["stdout"], "propagate": False},
}

if settings.SLACK_ALERTS.ENABLED:
    handlers["slack_alerts"] = {
        "()": SlackSdkHandler,
        "token": settings.SLACK_ALERTS.API_TOKEN,
        "channel_id": settings.SLACK_ALERTS.CHANNEL_ID,
        "level": settings.SLACK_ALERTS.LOG_LEVEL,
        "formatter": "simple",
    }
    for logger_name in loggers:
        if logger_name != "root":  # do not add slack alerts to the root logger.
            loggers[logger_name]["handlers"].append("slack_alerts")

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
    "handlers": handlers,
    "loggers": loggers,
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("lifeline-ai")


def get_logger(logger_name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(logger_name)
