"""
LLM plugin for Amazon Bedrock.

Provides access to all Amazon Bedrock models that support the Converse API.
"""
import json
import os
import random
import shutil
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import boto3
import click
import llm
import structlog
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from pydantic import BaseModel, Field

__version__ = "0.1.0"

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer() if os.environ.get("STRUCTLOG_DEV") else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.environ.get("LOG_LEVEL", "30"))
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

DEFAULT_REGION = "us-east-1"
CACHE_TTL_HOURS = 24
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0
RETRY_MULTIPLIER = 2.0
JITTER_FACTOR = 0.1

THROTTLE_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "ServiceQuotaExceededException",
}


def with_retry(max_retries: int = MAX_RETRIES):
    """Decorator that retries on throttling errors with exponential backoff and jitter."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "Unknown")

                    if error_code not in THROTTLE_ERROR_CODES:
                        raise

                    if attempt >= max_retries:
                        raise

                    base_delay = INITIAL_RETRY_DELAY * (RETRY_MULTIPLIER**attempt)
                    jitter = base_delay * JITTER_FACTOR * random.random()
                    delay = min(base_delay + jitter, MAX_RETRY_DELAY)

                    logger.info(
                        "throttled_retrying",
                        error_code=error_code,
                        attempt=attempt + 1,
                        delay_seconds=round(delay, 2),
                    )
                    time.sleep(delay)
                    last_exception = e

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class ModelCache:
    """Cache Bedrock model list locally."""

    def __init__(self, cache_ttl_hours=CACHE_TTL_HOURS):
        self.cache_dir = Path(llm.user_dir()) / "bedrock-ks-models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    def get_cache_path(self, region: str) -> Path:
        return self.cache_dir / f"{region}.json"

    def load(self, region: str) -> Optional[list]:
        """Load models from cache if not expired."""
        cache_path = self.get_cache_path(region)

        if not cache_path.exists():
            return None

        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
            if (datetime.now(tz=timezone.utc) - mtime) >= self.cache_ttl:
                return None
        except OSError:
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            models = data.get("models", [])
            return models if isinstance(models, list) else None
        except (json.JSONDecodeError, IOError):
            return None

    def save(self, region: str, models: list):
        """Save models to cache with secure permissions."""
        cache_path = self.get_cache_path(region)
        data = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "region": region,
            "models": models,
        }
        try:
            fd = os.open(str(cache_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("cache_save_failed", error=str(e))

    def clear(self):
        """Remove all cached data."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def is_converse_compatible(model_summary: dict) -> bool:
    """Check if model supports Converse API (text I/O and active status)."""
    input_modalities = model_summary.get("inputModalities", [])
    output_modalities = model_summary.get("outputModalities", [])
    has_text_io = "TEXT" in input_modalities and "TEXT" in output_modalities

    lifecycle = model_summary.get("modelLifecycle", {})
    is_active = lifecycle.get("status", "ACTIVE") == "ACTIVE"

    return has_text_io and is_active


def discover_bedrock_models(region: str) -> list:
    """Discover Bedrock models via API. Returns empty list on error."""
    try:
        client = boto3.client("bedrock", region_name=region)

        @with_retry(max_retries=MAX_RETRIES)
        def _list_models():
            return client.list_foundation_models()

        response = _list_models()
        models = []
        for summary in response.get("modelSummaries", []):
            if not is_converse_compatible(summary):
                continue
            model_id = summary["modelId"]
            models.append({
                "model_id": model_id,
                "name": summary.get("modelName"),
                "provider": summary.get("providerName"),
                "streaming": summary.get("responseStreamingSupported", False),
                "input_modalities": summary.get("inputModalities", ["TEXT"]),
                "output_modalities": summary.get("outputModalities", ["TEXT"]),
            })

        logger.info("models_discovered", count=len(models), region=region)
        return models

    except (NoCredentialsError, ClientError, BotoCoreError, Exception) as e:
        logger.warning("discovery_failed", error=str(e), error_type=type(e).__name__)
        return []


class BedrockModel(llm.KeyModel):
    """Amazon Bedrock model using Converse API."""

    needs_key = "bedrock-ks-api-key"
    key_env_var = "AWS_BEARER_TOKEN_BEDROCK"

    _client = None
    _client_key = None

    def __init__(
        self,
        model_id: str,
        bedrock_model_id: str,
        region: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        supports_streaming: bool = True,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.bedrock_model_id = bedrock_model_id
        self.region = region if region is not None else self._get_default_region()
        self.model_name = model_name or bedrock_model_id
        self.provider = provider
        self.can_stream = supports_streaming
        self.supports_tools = True
        self.supports_schema = False

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
        max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=100000)
        temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
        top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
        stop_sequences: List[str] = Field(default_factory=list)

    def get_bedrock_client(self, key: Optional[str] = None):
        """Get or create cached boto3 Bedrock Runtime client."""
        if key is None:
            key = self.get_key()

        if self._client is not None and self._client_key == key:
            return self._client

        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key
        self._client = boto3.client(service_name="bedrock-runtime", region_name=self.region)
        self._client_key = key
        return self._client

    @staticmethod
    def _get_default_region() -> str:
        """Resolve region: env vars → llm config → us-east-1."""
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            return region

        try:
            region = llm.get_key("bedrock-ks-region")
            if region and not region.startswith("bedrock-"):
                return region
        except llm.NeedsKeyException:
            pass

        return DEFAULT_REGION

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation] = None,
        key: Optional[str] = None,
    ) -> Iterator[str]:
        client = self.get_bedrock_client(key=key)
        request = self._build_request(prompt, conversation)

        if stream:
            yield from self._execute_stream(client, request, response)
        else:
            yield from self._execute_non_stream(client, request, response)

    def _build_request(
        self, prompt: llm.Prompt, conversation: Optional[llm.Conversation] = None
    ) -> dict:
        """Build Converse API request."""
        request: Dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": [],
        }

        # Conversation history (user/assistant must alternate)
        if conversation:
            for prev in conversation.responses:
                if prev.prompt.prompt or prev.prompt.attachments:
                    request["messages"].append(
                        {"role": "user", "content": self._content_blocks(prev.prompt)}
                    )
                    request["messages"].append(
                        {"role": "assistant", "content": [{"text": prev.text()}]}
                    )

        # Current prompt
        if prompt.prompt or prompt.attachments:
            request["messages"].append(
                {"role": "user", "content": self._content_blocks(prompt)}
            )

        if prompt.system:
            request["system"] = [{"text": prompt.system}]

        # Inference config
        if prompt.options:
            config = {}
            if prompt.options.max_tokens:
                config["maxTokens"] = prompt.options.max_tokens
            if prompt.options.temperature is not None:
                config["temperature"] = prompt.options.temperature
            if prompt.options.top_p is not None:
                config["topP"] = prompt.options.top_p
            if prompt.options.stop_sequences:
                config["stopSequences"] = prompt.options.stop_sequences
            if config:
                request["inferenceConfig"] = config

        if prompt.tools:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": {"json": tool.input_schema},
                        }
                    }
                    for tool in prompt.tools
                ]
            }

        return request

    def _content_blocks(self, prompt: llm.Prompt) -> List[dict]:
        """Build content blocks from prompt text and attachments."""
        content = []

        if prompt.prompt:
            content.append({"text": prompt.prompt})

        if prompt.attachments:
            for attachment in prompt.attachments:
                mime_type = attachment.resolve_type()

                if mime_type.startswith("image/"):
                    fmt = mime_type.split("/")[1]
                    if fmt == "jpg":
                        fmt = "jpeg"
                    content.append({
                        "image": {"format": fmt, "source": {"bytes": attachment.content_bytes()}}
                    })
                elif mime_type.startswith("application/pdf") or mime_type.startswith("text/"):
                    doc_format = self._doc_format(mime_type)
                    doc_name = (
                        attachment.path.name
                        if hasattr(attachment, "path")
                        else f"document.{doc_format}"
                    )
                    content.append({
                        "document": {
                            "format": doc_format,
                            "name": doc_name,
                            "source": {"bytes": attachment.content_bytes()},
                        }
                    })

        return content

    @staticmethod
    def _doc_format(mime_type: str) -> str:
        """Map MIME type to Bedrock document format."""
        return {
            "application/pdf": "pdf",
            "text/plain": "txt",
            "text/csv": "csv",
            "text/html": "html",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }.get(mime_type, "txt")

    def _execute_stream(self, client, request: dict, response: llm.Response) -> Iterator[str]:
        """Execute with ConverseStream API."""
        stream_response = client.converse_stream(**request)
        metadata = {}

        for event in stream_response.get("stream"):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
            elif "metadata" in event:
                metadata = event["metadata"]

        usage = metadata.get("usage", {})
        if usage:
            response.set_usage(
                input=usage.get("inputTokens", 0),
                output=usage.get("outputTokens", 0),
            )

    def _execute_non_stream(self, client, request: dict, response: llm.Response) -> Iterator[str]:
        """Execute with Converse API (non-streaming)."""
        api_response = client.converse(**request)
        message = api_response.get("output", {}).get("message", {})

        for block in message.get("content", []):
            if "text" in block:
                yield block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                response.add_tool_call(
                    tool_name=tool_use["name"],
                    tool_call_id=tool_use["toolUseId"],
                    parameters=tool_use["input"],
                )

        usage = api_response.get("usage", {})
        if usage:
            response.set_usage(
                input=usage.get("inputTokens", 0),
                output=usage.get("outputTokens", 0),
            )


@llm.hookimpl
def register_models(register):
    """Register all Bedrock models that support Converse API."""
    region = BedrockModel._get_default_region()
    cache = ModelCache()
    models = cache.load(region)

    if not models:
        models = discover_bedrock_models(region)
        if models:
            cache.save(region, models)

    if not models:
        logger.warning("no_models_found", reason="discovery_failed_and_no_cache")
        return

    for model_info in models:
        register(BedrockModel(
            model_id=f"bedrock-ks/{model_info['model_id']}",
            bedrock_model_id=model_info["model_id"],
            region=region,
            model_name=model_info.get("name"),
            provider=model_info.get("provider"),
            supports_streaming=model_info.get("streaming", True),
            input_modalities=model_info.get("input_modalities", ["TEXT"]),
            output_modalities=model_info.get("output_modalities", ["TEXT"]),
        ))


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def bedrock_ks():
        """Commands for the Bedrock KS plugin."""

    @bedrock_ks.command()
    def refresh():
        """Clear the model cache and re-discover models."""
        cache = ModelCache()
        cache.clear()
        click.echo("Cache cleared.")

        region = BedrockModel._get_default_region()
        models = discover_bedrock_models(region)
        if models:
            cache.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.save(region, models)
            click.echo(f"Discovered {len(models)} models in {region}.")
        else:
            click.echo("No models discovered. Check your credentials and region.")

    @bedrock_ks.command()
    def models():
        """List all available Bedrock models."""
        region = BedrockModel._get_default_region()
        cache = ModelCache()
        cached_models = cache.load(region)
        if not cached_models:
            click.echo("No cached models. Run 'llm bedrock-ks refresh' first.")
            return
        for m in sorted(cached_models, key=lambda x: x.get("model_id", "")):
            provider = m.get("provider", "Unknown")
            name = m.get("name", m["model_id"])
            click.echo(f"  bedrock-ks/{m['model_id']} ({provider}: {name})")
