"""Unit tests for llm-bedrock-ks plugin."""
from llm_bedrock import is_converse_compatible, BedrockModel


def test_is_converse_compatible():
    """Test Converse API compatibility detection."""
    # Compatible models
    assert is_converse_compatible({
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "inputModalities": ["TEXT", "IMAGE"],
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"}
    })
    assert is_converse_compatible({
        "modelId": "amazon.nova-pro-v1:0",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"}
    })
    # No provider filter - AI21 is compatible
    assert is_converse_compatible({
        "modelId": "ai21.jamba-1-5-mini-v1:0",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"}
    })
    # Incompatible: no text modalities
    assert not is_converse_compatible({
        "modelId": "stability.stable-diffusion-xl-v1:0",
        "inputModalities": ["IMAGE"],
        "outputModalities": ["IMAGE"],
        "modelLifecycle": {"status": "ACTIVE"}
    })
    # Incompatible: deprecated
    assert not is_converse_compatible({
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "DEPRECATED"}
    })


def test_bedrock_model_initialization():
    model = BedrockModel(
        model_id="bedrock-ks/anthropic.claude-3-sonnet-20240229-v1:0",
        bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
        model_name="Claude 3 Sonnet",
        provider="Anthropic",
        supports_streaming=True,
        input_modalities=["TEXT", "IMAGE"],
        output_modalities=["TEXT"],
    )

    assert model.model_id == "bedrock-ks/anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.bedrock_model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.region == "us-east-1"
    assert model.can_stream is True
    assert "image/jpeg" in model.attachment_types
    assert "image/png" in model.attachment_types


def test_document_format_mapping():
    assert BedrockModel._doc_format("application/pdf") == "pdf"
    assert BedrockModel._doc_format("text/plain") == "txt"
    assert BedrockModel._doc_format("text/csv") == "csv"
    assert BedrockModel._doc_format("text/html") == "html"
    assert BedrockModel._doc_format("unknown/type") == "txt"


def test_content_blocks_text_only():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockPrompt:
        def __init__(self, text):
            self.prompt = text
            self.attachments = []

    content = model._content_blocks(MockPrompt("Hello world"))
    assert content == [{"text": "Hello world"}]


def test_build_request_with_tools():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockTool:
        name = "get_weather"
        description = "Get current weather"
        input_schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }

    class MockPrompt:
        prompt = "What's the weather?"
        attachments = []
        system = None
        options = None
        tools = [MockTool()]

    request = model._build_request(MockPrompt())
    assert "toolConfig" in request
    assert len(request["toolConfig"]["tools"]) == 1
    assert request["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"


def test_build_request_conversation_history():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockPrompt:
        def __init__(self, text):
            self.prompt = text
            self.attachments = []
            self.system = None
            self.options = None
            self.tools = None

    class MockResponse:
        def __init__(self, prompt_text, response_text):
            self.prompt = MockPrompt(prompt_text)
            self._text = response_text
        def text(self):
            return self._text

    class MockConversation:
        responses = [
            MockResponse("Hello", "Hi there!"),
            MockResponse("How are you?", "Good."),
        ]

    request = model._build_request(MockPrompt("Bye"), MockConversation())
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
