import asyncio
import ipaddress
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import boto3
from botocore.exceptions import ClientError

from app.functions.transcriptions.core.config import settings
from app.functions.transcriptions.utils.execution_manager import ExecutionManager
from app.functions.transcriptions.utils.logger import get_logger
from app.functions.transcriptions.utils.phi_events import PHIEvents

logger = get_logger(__name__)


class PHILogEvent:
    def __init__(
        self,
        event_type: Union[PHIEvents, str],
        chat_id: Optional[str] = None,
        audit_id: Optional[str] = None,
        logged_at: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.chat_id = chat_id
        self.audit_id = audit_id
        self.logged_at = logged_at
        self.details = details or {}


class PHILoggerService:
    _instance: Optional["PHILoggerService"] = None

    def __init__(self):
        self.cloudwatch_client: Optional[Any] = None
        self.log_group_name: Optional[str] = None
        self.log_stream_name: Optional[str] = None
        self.enabled = False
        self._initialize_cloudwatch()

    @classmethod
    def get_instance(cls) -> "PHILoggerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize_cloudwatch(self) -> None:
        try:
            self.enabled = settings.HIPAA_AUDIT_ENABLED
            if not self.enabled:
                return

            # Create boto3 client for AWS CloudWatch
            self.cloudwatch_client = boto3.client("logs")

            self.log_group_name = settings.HIPAA_AUDIT_LOG_GROUP_NAME
            base_stream_name = settings.HIPAA_AUDIT_LOG_STREAM_NAME
            random_id = random.randint(100000, 999999)
            self.log_stream_name = f"{base_stream_name}-{random_id}"

            # Create log group and stream if they don't exist
            self._ensure_log_group_and_stream()

        except Exception as e:
            logger.exception(f"Failed to initialize CloudWatch PHI logging: {e}")
            self.enabled = False

    def _ensure_log_group_and_stream(self) -> None:
        """Ensure log group and stream exist in CloudWatch."""
        try:
            if (
                not self.cloudwatch_client
                or not self.log_group_name
                or not self.log_stream_name
            ):  # noqa: E501
                return

            # Create log group if it doesn't exist
            try:
                self.cloudwatch_client.create_log_group(
                    logGroupName=self.log_group_name
                )  # noqa: E501
                logger.info(f"Created log group: {self.log_group_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                    logger.debug(f"Log group already exists: {self.log_group_name}")
                else:
                    raise

            # Create log stream if it doesn't exist
            try:
                self.cloudwatch_client.create_log_stream(
                    logGroupName=self.log_group_name, logStreamName=self.log_stream_name
                )
                logger.info(f"Created log stream: {self.log_stream_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                    logger.debug(f"Log stream already exists: {self.log_stream_name}")
                else:
                    raise

        except Exception as e:
            logger.exception(f"Failed to create log group/stream: {e}")

    def _is_valid_ip(self, ip: str) -> bool:
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _get_geo_location(self, ip: str) -> str:
        return "Unknown"

    async def log(self, event: PHILogEvent) -> None:
        try:
            request = ExecutionManager.get_request_metadata()

            ip = request.get("ip", "Unknown") if request else "Unknown"
            user_agent = (
                request.get("headers", {}).get("user-agent", "Unknown")
                if request
                else "Unknown"
            )  # noqa: E501
            http_method = request.get("method", "WebSocket") if request else "WebSocket"
            endpoint = (
                request.get("original_url", "WebSocket Connection")
                if request
                else "WebSocket Connection"
            )  # noqa: E501
            request_id = (
                request.get("headers", {}).get("x-request-id", "Unknown")
                if request
                else "Unknown"
            )  # noqa: E501

            is_valid_ip = ip != "Unknown" and self._is_valid_ip(ip)
            location = self._get_geo_location(ip) if is_valid_ip else "Unknown"
            logged_at = event.logged_at if event.logged_at else datetime.utcnow()

            phi_log_dto = {
                "event_type": str(event.event_type),
                "audit_id": event.audit_id,
                "chat_id": event.chat_id,
                "logged_at": logged_at.isoformat(),
                "details": {
                    **(event.details or {}),
                    "ip": ip,
                    "location": location,
                    "user_agent": user_agent,
                    "http_method": http_method,
                    "endpoint": endpoint,
                    "request_id": request_id,
                },
            }

            if self.enabled and self.cloudwatch_client:
                await self._send_to_cloudwatch(phi_log_dto)

        except Exception as error:
            logger.exception(f"Error while creating PHI log for the event: {error}")

    async def _send_to_cloudwatch(self, log_entry: Dict[str, Any]) -> None:
        try:
            if (
                not self.cloudwatch_client
                or not self.log_group_name
                or not self.log_stream_name
            ):  # noqa: E501
                return
            log_event = {
                "timestamp": int(time.time() * 1000),
                "message": json.dumps(log_entry),
            }
            self.cloudwatch_client.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[log_event],
            )
        except ClientError as e:
            logger.exception(f"Failed to send PHI log to CloudWatch: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error sending PHI log to CloudWatch: {e}")


phi_logger = PHILoggerService.get_instance()


def log_sync(event: PHILogEvent) -> None:
    """
    Synchronous wrapper for phi_logger.log() to avoid creating new event loops.

    This function should be used in synchronous contexts where asyncio.run()
    would be inefficient or problematic.

    Args:
        event: The PHI log event to log
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use asyncio.run()
            # Create a task instead
            asyncio.create_task(phi_logger.log(event))
        else:
            # If no loop is running, we can use asyncio.run()
            asyncio.run(phi_logger.log(event))
    except RuntimeError:
        # No event loop exists, create one
        asyncio.run(phi_logger.log(event))
