"""Integration tests for llm-bedrock-ks plugin retry logic."""
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

pytestmark = pytest.mark.integration


class TestRetryLogic:
    """Test exponential backoff retry logic."""

    def test_retry_on_throttling(self):
        """Test that throttling errors trigger retries."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_client.list_inference_profiles.side_effect = [
            ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
                "ListInferenceProfiles",
            ),
            {"inferenceProfileSummaries": []},
        ]

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            with patch("time.sleep"):
                from llm_bedrock import discover_bedrock_models
                result = discover_bedrock_models("us-east-1")

        assert result == []
        assert mock_client.list_inference_profiles.call_count == 2

    def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_client.list_inference_profiles.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "ListInferenceProfiles",
        )

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            with patch("time.sleep"):
                from llm_bedrock import discover_bedrock_models
                result = discover_bedrock_models("us-east-1")

        assert result == []
        # 4 attempts (initial + 3 retries)
        assert mock_client.list_inference_profiles.call_count == 4

    def test_no_retry_on_non_throttle_errors(self):
        """Test that non-throttling errors don't trigger retries."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_client.list_inference_profiles.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "ListInferenceProfiles",
        )

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            from llm_bedrock import discover_bedrock_models
            result = discover_bedrock_models("us-east-1")

        assert result == []
        # Only 1 attempt (no retries for non-throttle errors)
        assert mock_client.list_inference_profiles.call_count == 1
