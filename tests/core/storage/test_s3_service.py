"""Tests for S3Service."""

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.core.storage.s3_service import S3Service


class TestS3Service:
    """Test cases for S3Service."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for S3Service."""
        with patch("app.core.storage.s3_service.settings") as mock_settings:
            mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
            mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
            mock_settings.AWS.REGION = "us-east-1"
            mock_settings.ENV.ENV = "DEVELOPMENT"
            yield mock_settings

    @pytest.fixture
    def s3_service(self, mock_settings):
        """Create S3Service instance for testing."""
        with patch("app.core.storage.s3_service.boto3.client") as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client
            service = S3Service()
            service.s3_client = mock_client
            return service

    def test_init_development_environment(self, mock_settings):
        """Test S3Service initialization in development environment."""
        with patch("app.core.storage.s3_service.boto3.client") as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client

            S3Service()

            # Assert boto3 client was called with credentials
            call_args = mock_boto_client.call_args
            assert call_args[0][0] == "s3"
            # Check that credentials are in the call
            assert "aws_access_key_id" in call_args[1]
            assert "aws_secret_access_key" in call_args[1]
            assert call_args[1]["region_name"] == "us-east-1"

    def test_init_production_environment(self, mock_settings):
        """Test S3Service initialization in production environment."""
        mock_settings.ENV.ENV = "production"

        with patch("app.core.storage.s3_service.boto3.client") as mock_boto_client:
            mock_client = MagicMock()
            mock_boto_client.return_value = mock_client

            S3Service()

            # Assert boto3 client was called without credentials (uses IAM role)
            call_args = mock_boto_client.call_args
            assert call_args[0][0] == "s3"
            assert call_args[1]["region_name"] == "us-east-1"
            assert "aws_access_key_id" not in call_args[1]
            assert "aws_secret_access_key" not in call_args[1]

    @pytest.mark.asyncio
    async def test_upload_to_s3_success(self, s3_service):
        """Test successful S3 upload."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        payload = {"key": "value", "number": 123}

        s3_service.s3_client.put_object = MagicMock()

        # Execute
        result = await s3_service.upload_to_s3(bucket_name, object_key, payload)

        # Assert
        expected_path = f"s3://{bucket_name}/{object_key}"
        assert result == expected_path

        # Verify put_object was called with correct parameters
        s3_service.s3_client.put_object.assert_called_once()
        call_args = s3_service.s3_client.put_object.call_args
        assert call_args[1]["Bucket"] == bucket_name
        assert call_args[1]["Key"] == object_key
        assert call_args[1]["ContentType"] == "application/json"

        # Verify JSON payload
        uploaded_data = json.loads(call_args[1]["Body"])
        assert uploaded_data == payload

    @pytest.mark.asyncio
    async def test_upload_to_s3_client_error(self, s3_service):
        """Test S3 upload with ClientError."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        payload = {"key": "value"}

        s3_service.s3_client.put_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchBucket"}},
            operation_name="PutObject",
        )

        # Execute and assert
        with pytest.raises(Exception, match="S3 upload failed"):
            await s3_service.upload_to_s3(bucket_name, object_key, payload)

    @pytest.mark.asyncio
    async def test_upload_to_s3_general_error(self, s3_service):
        """Test S3 upload with general exception."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        payload = {"key": "value"}

        s3_service.s3_client.put_object.side_effect = Exception("Network error")

        # Execute and assert
        with pytest.raises(Exception, match="S3 upload failed"):
            await s3_service.upload_to_s3(bucket_name, object_key, payload)

    @pytest.mark.asyncio
    async def test_upload_to_s3_complex_payload(self, s3_service):
        """Test S3 upload with complex payload."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/complex.json"
        payload = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        s3_service.s3_client.put_object = MagicMock()

        # Execute
        result = await s3_service.upload_to_s3(bucket_name, object_key, payload)

        # Assert
        expected_path = f"s3://{bucket_name}/{object_key}"
        assert result == expected_path

        # Verify JSON payload
        call_args = s3_service.s3_client.put_object.call_args
        uploaded_data = json.loads(call_args[1]["Body"])
        assert uploaded_data == payload

    @pytest.mark.asyncio
    async def test_generate_presigned_download_url_success(self, s3_service):
        """Test successful presigned download URL generation."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        expiration = 7200
        expected_url = (
            "https://test-bucket.s3.amazonaws.com/test/path/file.json?signature=abc123"
        )

        s3_service.s3_client.generate_presigned_url.return_value = expected_url

        # Execute
        result = await s3_service.generate_presigned_download_url(
            bucket_name, object_key, expiration
        )

        # Assert
        assert result == expected_url
        s3_service.s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )

    @pytest.mark.asyncio
    async def test_generate_presigned_download_url_default_expiration(self, s3_service):
        """Test presigned download URL generation with default expiration."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        expected_url = (
            "https://test-bucket.s3.amazonaws.com/test/path/file.json?signature=abc123"
        )

        s3_service.s3_client.generate_presigned_url.return_value = expected_url

        # Execute
        result = await s3_service.generate_presigned_download_url(
            bucket_name, object_key
        )

        # Assert
        assert result == expected_url
        s3_service.s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=3600,  # Default expiration
        )

    @pytest.mark.asyncio
    async def test_generate_presigned_download_url_client_error(self, s3_service):
        """Test presigned download URL generation with ClientError."""
        from botocore.exceptions import ClientError

        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"

        s3_service.s3_client.generate_presigned_url.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey"}},
            operation_name="GeneratePresignedUrl",
        )

        # Execute
        result = await s3_service.generate_presigned_download_url(
            bucket_name, object_key
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_presigned_delete_url_success(self, s3_service):
        """Test successful presigned delete URL generation."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        expiration = 1800
        expected_url = (
            "https://test-bucket.s3.amazonaws.com/test/path/file.json"
            "?signature=delete123"
        )

        s3_service.s3_client.generate_presigned_url.return_value = expected_url

        # Execute
        result = await s3_service.generate_presigned_delete_url(
            bucket_name, object_key, expiration
        )

        # Assert
        assert result == expected_url
        s3_service.s3_client.generate_presigned_url.assert_called_once_with(
            "delete_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )

    @pytest.mark.asyncio
    async def test_generate_presigned_delete_url_default_expiration(self, s3_service):
        """Test presigned delete URL generation with default expiration."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"
        expected_url = (
            "https://test-bucket.s3.amazonaws.com/test/path/file.json"
            "?signature=delete123"
        )

        s3_service.s3_client.generate_presigned_url.return_value = expected_url

        # Execute
        result = await s3_service.generate_presigned_delete_url(bucket_name, object_key)

        # Assert
        assert result == expected_url
        s3_service.s3_client.generate_presigned_url.assert_called_once_with(
            "delete_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=3600,  # Default expiration
        )

    @pytest.mark.asyncio
    async def test_generate_presigned_delete_url_client_error(self, s3_service):
        """Test presigned delete URL generation with ClientError."""
        from botocore.exceptions import ClientError

        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/file.json"

        s3_service.s3_client.generate_presigned_url.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied"}},
            operation_name="GeneratePresignedUrl",
        )

        # Execute
        result = await s3_service.generate_presigned_delete_url(bucket_name, object_key)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_upload_to_s3_json_serialization(self, s3_service):
        """Test S3 upload with JSON serialization edge cases."""
        # Setup mocks
        bucket_name = "test-bucket"
        object_key = "test/path/edge_cases.json"

        # Test with various data types that should be JSON serializable
        payload = {
            "none_value": None,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "unicode_string": "测试字符串",
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        }

        s3_service.s3_client.put_object = MagicMock()

        # Execute
        result = await s3_service.upload_to_s3(bucket_name, object_key, payload)

        # Assert
        expected_path = f"s3://{bucket_name}/{object_key}"
        assert result == expected_path

        # Verify JSON payload can be deserialized
        call_args = s3_service.s3_client.put_object.call_args
        uploaded_data = json.loads(call_args[1]["Body"])
        assert uploaded_data == payload
