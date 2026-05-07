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


class TestDiscoveryFallback:
    """Test that models without inference profiles are still discovered."""

    def test_foundation_model_without_profile(self):
        """Models with no inference profile are included via foundation model fallback."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "qwen.qwen3-32b-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/qwen.qwen3-32b-v1:0",
                    "modelName": "Qwen3 32B",
                    "providerName": "Qwen",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ]
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            from llm_bedrock import discover_bedrock_models
            result = discover_bedrock_models("us-east-1")

        assert len(result) == 1
        assert result[0]["model_id"] == "qwen.qwen3-32b-v1:0"
        assert result[0]["name"] == "Qwen3 32B"
        assert result[0]["streaming"] is True

    def test_foundation_model_with_profile_not_duplicated(self):
        """Models that have an inference profile are not added again from foundation models."""
        model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "modelArn": model_arn,
                    "modelName": "Claude 3 Sonnet",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["ON_DEMAND", "INFERENCE_PROFILE"],
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ]
        }
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                    "inferenceProfileName": "US Claude 3 Sonnet",
                    "status": "ACTIVE",
                    "models": [{"modelArn": model_arn}],
                },
            ]
        }

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            from llm_bedrock import discover_bedrock_models
            result = discover_bedrock_models("us-east-1")

        assert len(result) == 1
        assert result[0]["model_id"] == "us.anthropic.claude-3-sonnet-20240229-v1:0"

    def test_provisioned_only_models_excluded(self):
        """Context-window variants (PROVISIONED only) are excluded."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.nova-pro-v1:0:24k",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0:24k",
                    "modelName": "Nova Pro",
                    "providerName": "Amazon",
                    "inputModalities": ["TEXT", "IMAGE", "VIDEO"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["PROVISIONED"],
                    "modelLifecycle": {"status": "ACTIVE"},
                },
                {
                    "modelId": "qwen.qwen3-32b-v1:0",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/qwen.qwen3-32b-v1:0",
                    "modelName": "Qwen3 32B",
                    "providerName": "Qwen",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ]
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("llm_bedrock.boto3.client", return_value=mock_client):
            from llm_bedrock import discover_bedrock_models
            result = discover_bedrock_models("us-east-1")

        assert len(result) == 1
        assert result[0]["model_id"] == "qwen.qwen3-32b-v1:0"
