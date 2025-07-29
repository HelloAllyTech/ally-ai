import boto3
import json
import os
from typing import Dict, Any
from botocore.exceptions import ClientError
from app.utils.logger import get_logger

logger = get_logger(__name__)

class S3Service:
    """
    Simple S3 service for uploading data to S3.
    """
    
    def __init__(self):
        """
        Initialize the S3 service with AWS credentials from environment.
        """
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'ap-southeast-1')
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.error("AWS credentials not found in environment variables")
            raise ValueError("AWS credentials not configured")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
    
    async def upload_to_s3(self, bucket_name: str, object_key: str, payload: Dict[str, Any]) -> str:
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
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json_data,
                ContentType='application/json'
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