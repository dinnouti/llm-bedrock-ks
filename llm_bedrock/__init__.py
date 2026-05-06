"""
LLM plugin for Amazon Bedrock.

Provides access to all Amazon Bedrock models that support the Converse API.
"""
import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
import llm
from pydantic import BaseModel, Field

__version__ = "0.1.0"


# ============================================================================
# Model Discovery & Caching
# ============================================================================


class ModelCache:
    """Cache Bedrock model list locally."""

    def __init__(self, cache_ttl_hours=24):
        self.cache_dir = Path(llm.user_dir()) / "bedrock-models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    def get_cache_path(self, region: str) -> Path:
        return self.cache_dir / f"{region}.json"

    def is_cache_valid(self, region: str) -> bool:
        """Check if cache exists and is not expired."""
        cache_path = self.get_cache_path(region)
        if not cache_path.exists():
            return False

        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < self.cache_ttl

    def load(self, region: str) -> Optional[list]:
        """Load models from cache."""
        cache_path = self.get_cache_path(region)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return data.get("models", [])
        except Exception:
            return None

    def save(self, region: str, models: list):
        """Save models to cache."""
        cache_path = self.get_cache_path(region)
        data = {
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "models": models,
        }
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)


def simplify_model_id(bedrock_model_id: str) -> str:
    """
    Convert Bedrock model ID to plugin model ID.

    Examples:
    - anthropic.claude-3-sonnet-20240229-v1:0 -> claude-3-sonnet
    - amazon.nova-pro-v1:0 -> nova-pro
    - meta.llama3-70b-instruct-v1:0 -> llama3-70b-instruct
    """
    # Remove provider prefix
    without_provider = (
        bedrock_model_id.split(".", 1)[1] if "." in bedrock_model_id else bedrock_model_id
    )

    # Remove version suffix and variant
    name = without_provider.split("-v")[0]  # Remove -v1:0 style suffixes
    name = name.split(":")[0]  # Remove :0 style suffixes

    # Remove date stamps (YYYYMMDD or YYYY-MM-DD patterns)
    name = re.sub(r"-\d{8}", "", name)  # Remove -20240229
    name = re.sub(r"-\d{4}-\d{2}-\d{2}", "", name)  # Remove -2024-02-29

    return name


def is_converse_compatible(model_summary: dict) -> bool:
    """Check if model supports Converse API."""
    model_id = model_summary["modelId"]

    # Known compatible prefixes
    compatible_prefixes = [
        "anthropic.claude",
        "amazon.nova",
        "meta.llama3",
        "mistral.mistral",
        "cohere.command",
    ]

    return any(model_id.startswith(prefix) for prefix in compatible_prefixes)


def discover_bedrock_models(region: str) -> list:
    """Discover Bedrock models via API."""
    try:
        client = boto3.client("bedrock", region_name=region)
        response = client.list_foundation_models()

        models = []
        for summary in response.get("modelSummaries", []):
            # Filter for Converse API support
            if not is_converse_compatible(summary):
                continue

            models.append(
                {
                    "bedrock_id": summary["modelId"],
                    "simplified_id": simplify_model_id(summary["modelId"]),
                    "name": summary.get("modelName"),
                    "provider": summary.get("providerName"),
                    "streaming": summary.get("responseStreamingSupported", False),
                    "input_modalities": summary.get("inputModalities", ["TEXT"]),
                    "output_modalities": summary.get("outputModalities", ["TEXT"]),
                }
            )

        return models

    except Exception as e:
        # Log error but don't crash plugin loading
        import sys

        print(f"Warning: Failed to discover Bedrock models: {e}", file=sys.stderr)
        return []


def get_fallback_models() -> list:
    """Hardcoded fallback list if API discovery fails."""
    return [
        {
            "bedrock_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "simplified_id": "claude-3-sonnet",
            "name": "Claude 3 Sonnet",
            "provider": "Anthropic",
            "streaming": True,
            "input_modalities": ["TEXT", "IMAGE"],
            "output_modalities": ["TEXT"],
        },
        {
            "bedrock_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "simplified_id": "claude-3-haiku",
            "name": "Claude 3 Haiku",
            "provider": "Anthropic",
            "streaming": True,
            "input_modalities": ["TEXT", "IMAGE"],
            "output_modalities": ["TEXT"],
        },
        {
            "bedrock_id": "amazon.nova-pro-v1:0",
            "simplified_id": "nova-pro",
            "name": "Amazon Nova Pro",
            "provider": "Amazon",
            "streaming": True,
            "input_modalities": ["TEXT", "IMAGE"],
            "output_modalities": ["TEXT"],
        },
        {
            "bedrock_id": "amazon.nova-lite-v1:0",
            "simplified_id": "nova-lite",
            "name": "Amazon Nova Lite",
            "provider": "Amazon",
            "streaming": True,
            "input_modalities": ["TEXT", "IMAGE"],
            "output_modalities": ["TEXT"],
        },
    ]


# ============================================================================
# BedrockModel Implementation
# ============================================================================


class BedrockModel(llm.AsyncKeyModel):
    """Amazon Bedrock model using Converse API."""

    needs_key = "bedrock-api-key"
    key_env_var = "AWS_BEARER_TOKEN_BEDROCK"

    def __init__(
        self,
        model_id: str,
        bedrock_model_id: str,
        region: str = "us-east-1",
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        supports_streaming: bool = True,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.bedrock_model_id = bedrock_model_id
        self.region = region
        self.model_name = model_name or bedrock_model_id
        self.provider = provider
        self.can_stream = supports_streaming

        # Determine capabilities from modalities
        self.supports_tools = True  # Most Converse models support tools
        self.supports_schema = False  # Structured output not widely supported yet

        # Set attachment types based on input modalities
        self.attachment_types = set()
        if input_modalities:
            if "IMAGE" in input_modalities:
                self.attachment_types.update(
                    ["image/jpeg", "image/png", "image/gif", "image/webp"]
                )
            if "DOCUMENT" in input_modalities:
                self.attachment_types.update(
                    ["application/pdf", "text/plain", "text/csv", "text/html"]
                )

    class Options(BaseModel):
        """Options for inference configuration."""

        max_tokens: int = Field(
            default=4096,
            description="Maximum number of tokens to generate",
            ge=1,
            le=100000,
        )
        temperature: float = Field(
            default=0.7, description="Sampling temperature (0-1)", ge=0.0, le=1.0
        )
        top_p: float = Field(
            default=0.9, description="Nucleus sampling threshold", ge=0.0, le=1.0
        )
        stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")

    def get_bedrock_client(self, key: Optional[str] = None):
        """Create boto3 Bedrock Runtime client with API key authentication."""

        # 1. Try to get API key from LLM keys store
        if key is None:
            try:
                key = self.get_key()
            except llm.NeedsKeyException:
                pass

        # 2. Try environment variable
        if not key:
            key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

        # 3. Try config file
        if not key:
            config_path = Path.home() / ".aws" / "bedrock-api-key"
            if config_path.exists():
                key = config_path.read_text().strip()

        # 4. Error if still not found
        if not key:
            raise llm.NeedsKeyException(
                name="bedrock-api-key",
                instructions="""
Get a Bedrock API key from AWS Console:
1. Go to https://console.aws.amazon.com/bedrock
2. Navigate to API Keys section
3. Create a new API key
4. Store it with: llm keys set bedrock-api-key

Or set the AWS_BEARER_TOKEN_BEDROCK environment variable.
                """,
            )

        # Get region
        region = self.get_region()

        # Set environment variable for boto3 to pick up
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key

        client = boto3.client(service_name="bedrock-runtime", region_name=region)

        return client

    def get_region(self) -> str:
        """Get AWS region from configuration."""
        # 1. Try LLM config
        try:
            return llm.get_key("bedrock-region", env_var="AWS_DEFAULT_REGION")
        except Exception:
            pass

        # 2. Try environment
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            return region

        # 3. Default to us-east-1
        return "us-east-1"

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation] = None,
    ) -> AsyncIterator[str]:
        """Execute the model using Bedrock Converse API."""

        # Get client
        client = self.get_bedrock_client()

        # Build request
        request = self.build_converse_request(prompt, conversation)

        if stream:
            # Use ConverseStream API
            async for chunk in self.execute_stream(client, request, response):
                yield chunk
        else:
            # Use Converse API
            async for chunk in self.execute_non_stream(client, request, response):
                yield chunk

    def build_converse_request(
        self, prompt: llm.Prompt, conversation: Optional[llm.Conversation] = None
    ) -> dict:
        """Build Converse API request from LLM prompt."""

        request: Dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": [],
            "inferenceConfig": {},
        }

        # Add conversation history
        if conversation:
            for prev_response in conversation.responses:
                # Add user message
                if prev_response.prompt.prompt or prev_response.prompt.attachments:
                    request["messages"].append(
                        {
                            "role": "user",
                            "content": self.build_content_blocks(prev_response.prompt),
                        }
                    )

                # Add assistant response
                request["messages"].append(
                    {"role": "assistant", "content": [{"text": prev_response.text()}]}
                )

        # Add current prompt
        if prompt.prompt or prompt.attachments:
            request["messages"].append(
                {"role": "user", "content": self.build_content_blocks(prompt)}
            )

        # Add system prompt
        if prompt.system:
            request["system"] = [{"text": prompt.system}]

        # Add inference config from options
        options = prompt.options or {}
        if "max_tokens" in options:
            request["inferenceConfig"]["maxTokens"] = options["max_tokens"]
        if "temperature" in options:
            request["inferenceConfig"]["temperature"] = options["temperature"]
        if "top_p" in options:
            request["inferenceConfig"]["topP"] = options["top_p"]
        if "stop_sequences" in options:
            request["inferenceConfig"]["stopSequences"] = options["stop_sequences"]

        # Add tools if present
        if prompt.tools:
            request["toolConfig"] = self.build_tool_config(prompt.tools)

        return request

    def build_content_blocks(self, prompt: llm.Prompt) -> List[dict]:
        """Build content blocks from prompt (text + attachments)."""
        content = []

        # Add text
        if prompt.prompt:
            content.append({"text": prompt.prompt})

        # Add attachments (images, documents)
        if prompt.attachments:
            for attachment in prompt.attachments:
                mime_type = attachment.resolve_type()

                if mime_type.startswith("image/"):
                    # Get format (jpeg, png, gif, webp)
                    image_format = mime_type.split("/")[1]
                    if image_format == "jpg":
                        image_format = "jpeg"

                    content.append(
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": attachment.content_bytes()},
                            }
                        }
                    )

                elif mime_type.startswith("application/pdf") or mime_type.startswith("text/"):
                    doc_format = self.get_document_format(mime_type)
                    doc_name = (
                        attachment.path.name
                        if hasattr(attachment, "path")
                        else f"document.{doc_format}"
                    )

                    content.append(
                        {
                            "document": {
                                "format": doc_format,
                                "name": doc_name,
                                "source": {"bytes": attachment.content_bytes()},
                            }
                        }
                    )

        return content

    def get_document_format(self, mime_type: str) -> str:
        """Map MIME type to Bedrock document format."""
        mapping = {
            "application/pdf": "pdf",
            "text/plain": "txt",
            "text/csv": "csv",
            "text/html": "html",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }
        return mapping.get(mime_type, "txt")

    def build_tool_config(self, tools: List[llm.Tool]) -> dict:
        """Build toolConfig from LLM tools."""
        tool_specs = []
        for tool in tools:
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
            )

        return {"tools": tool_specs}

    async def execute_stream(
        self, client, request: dict, response: llm.Response
    ) -> AsyncIterator[str]:
        """Execute with ConverseStream API."""

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        stream_response = await loop.run_in_executor(
            None, partial(client.converse_stream, **request)
        )

        # Process event stream
        stream = stream_response.get("stream")

        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]

            elif "metadata" in event:
                # Extract usage information
                usage = event["metadata"].get("usage", {})
                if usage:
                    response.set_usage(
                        input=usage.get("inputTokens", 0), output=usage.get("outputTokens", 0)
                    )

    async def execute_non_stream(
        self, client, request: dict, response: llm.Response
    ) -> AsyncIterator[str]:
        """Execute with Converse API (non-streaming)."""

        loop = asyncio.get_event_loop()
        api_response = await loop.run_in_executor(None, partial(client.converse, **request))

        # Extract response
        output = api_response.get("output", {})
        message = output.get("message", {})

        # Extract text content
        for content_block in message.get("content", []):
            if "text" in content_block:
                yield content_block["text"]
            elif "toolUse" in content_block:
                # Handle tool use
                tool_use = content_block["toolUse"]
                response.add_tool_call(
                    tool_name=tool_use["name"],
                    tool_call_id=tool_use["toolUseId"],
                    parameters=tool_use["input"],
                )

        # Extract usage
        usage = api_response.get("usage", {})
        if usage:
            response.set_usage(
                input=usage.get("inputTokens", 0), output=usage.get("outputTokens", 0)
            )


# ============================================================================
# Plugin Hooks
# ============================================================================


@llm.hookimpl
def register_models(register, model_aliases):
    """Register all Bedrock models that support Converse API."""

    # Get default region
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # Try to load from cache first
    cache = ModelCache()
    if cache.is_cache_valid(region):
        models = cache.load(region)
    else:
        # Fetch from API
        try:
            models = discover_bedrock_models(region)
            if models:
                cache.save(region, models)
        except Exception:
            models = None

        # If discovery fails, try to use stale cache
        if not models:
            models = cache.load(region)

        # If no cache, use hardcoded fallback list
        if not models:
            models = get_fallback_models()

    # Register each model
    for model_info in models:
        model = BedrockModel(
            model_id=f"bedrock/{model_info['simplified_id']}",
            bedrock_model_id=model_info["bedrock_id"],
            region=region,
            model_name=model_info.get("name"),
            provider=model_info.get("provider"),
            supports_streaming=model_info.get("streaming", True),
            input_modalities=model_info.get("input_modalities", ["TEXT"]),
            output_modalities=model_info.get("output_modalities", ["TEXT"]),
        )
        register(model)

        # Also register with full Bedrock ID as alias
        model_aliases[f"bedrock/{model_info['bedrock_id']}"] = model.model_id
