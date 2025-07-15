import boto3
from botocore.config import Config

from app.core.config import settings
from app.core.constants import ENV
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global SQS client
_sqs_client = None


class SQSQueueClient:
    """
    Utility class for managing the AWS SQS client.
    """

    @staticmethod
    def get_client() -> boto3.client:
        """
        Get the singleton instance of the SQS client.

        Returns:
            boto3.client: The SQS client.

        Raises:
            Exception: If the client has not been created yet.
        """
        global _sqs_client

        if not _sqs_client:
            logger.error("SQS client has not been created. Please create a client first.")
            raise Exception("SQS client has not been created. Please create a client first.")

        return _sqs_client

    @staticmethod
    def create_client() -> None:
        """
        Create the singleton instance of the SQS client if it doesn't exist.
        """
        global _sqs_client

        if not _sqs_client:
            logger.info("Creating a new SQS client...")

            # Configure the client with appropriate timeouts
            config = Config(
                connect_timeout=5,
                read_timeout=60,
                retries={'max_attempts': 3}
            )

            # Create the boto3 client
            if settings.ENV == ENV.DEVELOPMENT:
                _sqs_client = boto3.client(
                    'sqs',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    endpoint_url=settings.AWS_ENDPOINT_URL,
                    config=config
                )
            else:
                # Service should be granted access using IAM
                _sqs_client = boto3.client(
                    'sqs',
                    config=config
                )

            logger.info("SQS client initialized")
        else:
            logger.warning("SQS client already exists. Reusing the existing client.")

    @staticmethod
    def close_client() -> None:
        """
        Close and cleanup the SQS client.
        """
        global _sqs_client

        if _sqs_client:
            logger.info("Closing SQS client...")
            try:
                # Close the underlying HTTP session to shut down ThreadPoolExecutor
                if hasattr(_sqs_client, '_endpoint') and hasattr(_sqs_client._endpoint, 'http_session'):
                    _sqs_client._endpoint.http_session.close()
                    logger.info("SQS client HTTP session closed successfully")
                else:
                    logger.info("SQS client closed (no HTTP session to close)")

            except Exception as e:
                logger.warning(f"Error closing SQS client: {e}")
            finally:
                _sqs_client = None
                logger.info("SQS client reference cleared")
        else:
            logger.info("No SQS client to close")