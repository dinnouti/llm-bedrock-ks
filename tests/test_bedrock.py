"""Unit tests for llm-bedrock-ks plugin."""
from llm_bedrock import BedrockModel


def test_bedrock_model_initialization():
    model = BedrockModel(
        model_id="bedrock-ks/us.anthropic.claude-3-sonnet-20240229-v1:0",
        bedrock_model_id="us.anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
        model_name="US Claude 3 Sonnet",
        supports_streaming=True,
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
    )

    assert model.model_id == "bedrock-ks/us.anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.bedrock_model_id == "us.anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.region == "us-east-1"
    assert model.can_stream is True


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


def test_content_blocks_with_image():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockAttachment:
        def resolve_type(self):
            return "image/png"
        def content_bytes(self):
            return b"\x89PNG\r\n\x1a\n"

    class MockPrompt:
        prompt = "What's in this image?"
        attachments = [MockAttachment()]

    content = model._content_blocks(MockPrompt())
    assert len(content) == 2
    assert content[0] == {"text": "What's in this image?"}
    assert content[1] == {"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n\x1a\n"}}}


def test_content_blocks_with_jpeg():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockAttachment:
        def resolve_type(self):
            return "image/jpg"
        def content_bytes(self):
            return b"\xff\xd8\xff"

    class MockPrompt:
        prompt = "Describe this"
        attachments = [MockAttachment()]

    content = model._content_blocks(MockPrompt())
    assert content[1]["image"]["format"] == "jpeg"


def test_content_blocks_with_document():
    model = BedrockModel(
        model_id="bedrock-ks/test-model-v1:0",
        bedrock_model_id="test-model-v1:0",
    )

    class MockAttachment:
        def resolve_type(self):
            return "application/pdf"
        def content_bytes(self):
            return b"%PDF-1.4"

    class MockPrompt:
        prompt = "Summarize this"
        attachments = [MockAttachment()]

    content = model._content_blocks(MockPrompt())
    assert len(content) == 2
    assert content[1]["document"]["format"] == "pdf"
    assert content[1]["document"]["source"] == {"bytes": b"%PDF-1.4"}


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
