"""Unit tests for llm-bedrock plugin."""
import pytest
from llm_bedrock import simplify_model_id, is_converse_compatible, BedrockModel


def test_simplify_model_id():
    """Test model ID simplification."""
    assert simplify_model_id("anthropic.claude-3-sonnet-20240229-v1:0") == "claude-3-sonnet"
    assert simplify_model_id("amazon.nova-pro-v1:0") == "nova-pro"
    assert simplify_model_id("meta.llama3-70b-instruct-v1:0") == "llama3-70b-instruct"
    assert simplify_model_id("mistral.mistral-large-2407-v1:0") == "mistral-large"


def test_is_converse_compatible():
    """Test Converse API compatibility detection."""
    # Compatible models
    assert is_converse_compatible({"modelId": "anthropic.claude-3-sonnet-20240229-v1:0"})
    assert is_converse_compatible({"modelId": "amazon.nova-pro-v1:0"})
    assert is_converse_compatible({"modelId": "meta.llama3-70b-instruct-v1:0"})
    assert is_converse_compatible({"modelId": "mistral.mistral-large-2407-v1:0"})
    assert is_converse_compatible({"modelId": "cohere.command-r-plus-v1:0"})

    # Incompatible models
    assert not is_converse_compatible({"modelId": "ai21.jamba-1-5-mini-v1:0"})
    assert not is_converse_compatible({"modelId": "stability.stable-diffusion-xl-v1:0"})


def test_bedrock_model_initialization():
    """Test BedrockModel initialization."""
    model = BedrockModel(
        model_id="bedrock/claude-3-sonnet",
        bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
        model_name="Claude 3 Sonnet",
        provider="Anthropic",
        supports_streaming=True,
        input_modalities=["TEXT", "IMAGE"],
        output_modalities=["TEXT"],
    )

    assert model.model_id == "bedrock/claude-3-sonnet"
    assert model.bedrock_model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.region == "us-east-1"
    assert model.can_stream is True
    assert "image/jpeg" in model.attachment_types
    assert "image/png" in model.attachment_types


def test_document_format_mapping():
    """Test MIME type to document format mapping."""
    model = BedrockModel(
        model_id="bedrock/test",
        bedrock_model_id="test-model-v1:0",
    )

    assert model.get_document_format("application/pdf") == "pdf"
    assert model.get_document_format("text/plain") == "txt"
    assert model.get_document_format("text/csv") == "csv"
    assert model.get_document_format("text/html") == "html"
    assert model.get_document_format("unknown/type") == "txt"  # Default


def test_build_content_blocks_text_only():
    """Test building content blocks with text only."""
    model = BedrockModel(
        model_id="bedrock/test",
        bedrock_model_id="test-model-v1:0",
    )

    # Create a mock prompt object
    class MockPrompt:
        def __init__(self, text):
            self.prompt = text
            self.attachments = []

    prompt = MockPrompt("Hello world")
    content = model.build_content_blocks(prompt)

    assert len(content) == 1
    assert content[0] == {"text": "Hello world"}


def test_build_tool_config():
    """Test building tool config from LLM tools."""
    import llm

    model = BedrockModel(
        model_id="bedrock/test",
        bedrock_model_id="test-model-v1:0",
    )

    # Create a simple tool
    class GetWeatherTool:
        name = "get_weather"
        description = "Get current weather"
        input_schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }

    tool = GetWeatherTool()
    config = model.build_tool_config([tool])

    assert "tools" in config
    assert len(config["tools"]) == 1
    assert config["tools"][0]["toolSpec"]["name"] == "get_weather"
    assert config["tools"][0]["toolSpec"]["description"] == "Get current weather"
    assert "json" in config["tools"][0]["toolSpec"]["inputSchema"]
