import asyncio
import json
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.constants import ENV
from app.utils.logger import get_logger

logger = get_logger(__name__)


class S3Service:
    """
    S3 service for uploading data and generating presigned URLs.
    """

    def __init__(self):
        """
        Initialize the S3 service with AWS credentials from settings.
        """
        self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        self.aws_region = settings.AWS_REGION

        if settings.ENV == ENV.DEVELOPMENT:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region,
            )
        else:
            self.s3_client = boto3.client("s3")

    async def upload_to_s3(
        self, bucket_name: str, object_key: str, payload: Dict[str, Any]
    ) -> str:
        """
        Upload payload to S3.

        Args:
            bucket_name (str): The S3 bucket name
            object_key (str): The S3 object key (path)
            payload (Dict[str, Any]): The data to upload

        Returns:
            str: The S3 path to the uploaded object

        Raises:
            Exception: If upload fails
        """
        try:
            # Convert payload to JSON
            json_data = json.dumps(payload, indent=2)

            # Upload to S3 using asyncio.to_thread to avoid blocking
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=bucket_name,
                Key=object_key,
                Body=json_data,
                ContentType="application/json",
            )

            # Create the S3 path
            s3_path = f"s3://{bucket_name}/{object_key}"

            logger.info(f"Successfully uploaded to S3: {s3_path}")
            return s3_path

        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")

    async def generate_presigned_download_url(
        self, bucket_name: str, object_key: str, expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for downloading an S3 object.

        Args:
            bucket_name (str): Name of the S3 bucket
            object_key (str): Key of the S3 object
            expiration (int): Time in seconds for the presigned URL to remain valid

        Returns:
            str: Presigned URL as string. None if error.
        """
        try:
            # Generate presigned URLs (these are fast operations, but let's make
            # them async for consistency)
            response = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=expiration,
            )
            logger.info(
                f"Generated presigned download URL for s3://{bucket_name}/{object_key}"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to generate presigned download URL: {e}")
            return None

    async def generate_presigned_delete_url(
        self, bucket_name: str, object_key: str, expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for deleting an S3 object.

        Args:
            bucket_name (str): Name of the S3 bucket
            object_key (str): Key of the S3 object
            expiration (int): Time in seconds for the presigned URL to remain valid

        Returns:
            str: Presigned URL as string. None if error.
        """
        try:
            # Generate presigned URLs (these are fast operations, but let's make
            # them async for consistency)
            response = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                "delete_object",
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=expiration,
            )
            logger.info(
                f"Generated presigned delete URL for s3://{bucket_name}/{object_key}"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to generate presigned delete URL: {e}")
            return None
