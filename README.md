# llm-bedrock-ks

LLM plugin for Amazon Bedrock - access 70+ models through the Converse API.

**Features:**
- 🚀 70+ models from 16+ providers (Anthropic, Amazon, Meta, Mistral, AI21, Google, OpenAI, and more)
- ⚡ Streaming responses
- 🖼️ Multimodal support (images, documents)
- 💬 Conversation history
- 🔧 Tool use / function calling
- 🔄 Automatic retries with exponential backoff

## Installation

```bash
# Clone the repository
git clone https://github.com/dinnouti/llm-bedrock-ks.git
cd llm-bedrock-ks

# Install in editable mode
llm install -e .

# Verify installation
llm models list | grep bedrock-ks/
```

## Setup

### 1. Get AWS Bedrock API Key

1. Go to https://console.aws.amazon.com/bedrock/
2. In the left navigation, select **API keys**
3. Choose **Generate long-term API keys** (30-day expiration)
4. Copy the key (you won't see it again)

> **Note:** Long-term API keys are for development only. For production, use short-term keys or IAM roles.
> Ensure you have model access enabled in AWS Bedrock Console → Model Access.

### 2. Configure Plugin

**Option A: Using llm keys (recommended)**
```bash
# Set your API key
llm keys set bedrock-ks-api-key
# Paste your API key when prompted

# Optional: Set region (defaults to us-east-1)
llm keys set bedrock-ks-region
# Enter your region, e.g., us-west-2
```

**Option B: Using environment variables**
```bash
export AWS_BEARER_TOKEN_BEDROCK="your-api-key-here"
export AWS_DEFAULT_REGION="us-east-1"
```

### 3. Verify

```bash
# Should show 70+ models
llm bedrock-ks refresh
llm models list | grep bedrock-ks/
```

## Usage

### Quick Start

```bash
# List available models
llm models list | grep bedrock-ks/

# Basic prompt
llm -m bedrock-ks/amazon.nova-micro-v1:0 "Hello, world!"

# Simple question
llm -m bedrock-ks/amazon.nova-lite-v1:0 "What is Python?"
```

### More Examples

**Streaming (default):**
```bash
llm -m bedrock-ks/amazon.nova-pro-v1:0 "Write a story about AI"
```

**Non-streaming:**
```bash
llm -m bedrock-ks/amazon.nova-micro-v1:0 --no-stream "What is 2+2?"
```

**Custom options:**
```bash
# More creative (higher temperature)
llm -m bedrock-ks/amazon.nova-pro-v1:0 \
    -o temperature 0.9 \
    "Write a creative poem"

# More focused (lower temperature)
llm -m bedrock-ks/amazon.nova-micro-v1:0 \
    -o temperature 0.3 \
    "List the planets in order"

# Limit output length
llm -m bedrock-ks/amazon.nova-micro-v1:0 \
    -o max_tokens 100 \
    "Explain quantum computing"
```

**System prompts:**
```bash
llm -m bedrock-ks/amazon.nova-micro-v1:0 \
    --system "You are a helpful Python expert" \
    "Explain decorators"
```

**Images and documents:**
```bash
# Analyze an image
llm -m bedrock-ks/anthropic.claude-3-sonnet-20240229-v1:0 \
    -a photo.jpg \
    "What's in this image?"

# Process a PDF
llm -m bedrock-ks/anthropic.claude-3-sonnet-20240229-v1:0 \
    -a document.pdf \
    "Summarize this document"
```

**Conversations:**
```bash
# Start a conversation
llm -m bedrock-ks/amazon.nova-micro-v1:0 "My name is Alice"

# Continue (remembers context)
llm -m bedrock-ks/amazon.nova-micro-v1:0 --continue "What's my name?"
```

## Available Models

The plugin auto-discovers 70+ Converse API-compatible models from all providers:

- **Anthropic:** Claude 4.x, Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- **Amazon:** Nova Pro, Nova Lite, Nova Micro
- **Meta:** Llama 3.3, 3.2, 3.1 (8B, 70B, 405B variants)
- **Mistral AI:** Mistral Large 3, Mistral 7B, Mixtral
- **Cohere:** Command R/R+
- **AI21 Labs:** Jamba models
- **DeepSeek:** DeepSeek V3
- **Google:** Gemini models
- **NVIDIA:** Nemotron models
- **OpenAI:** GPT models (via Bedrock)
- **Qwen:** Qwen models
- **And more:** MiniMax, Moonshot AI, TwelveLabs, Writer, Z.AI

> The plugin shows **all** models that support the Bedrock Converse API, regardless of provider.

Models are cached locally for 24 hours. To refresh:
```bash
llm bedrock-ks refresh
```

To list all cached models with provider info:
```bash
llm bedrock-ks models
```

## Plugin Commands

```bash
# Refresh model cache (re-discover from AWS)
llm bedrock-ks refresh

# List all cached models with provider info
llm bedrock-ks models
```

## Configuration

### Options

Available via `-o` flag:

- `temperature` (0-1, default: 0.7) - Sampling temperature
- `max_tokens` (int, default: 4096) - Maximum tokens to generate
- `top_p` (0-1, default: 0.9) - Nucleus sampling threshold
- `stop_sequences` (list) - Stop generation at these sequences

### Set Default Model

Set a default model to avoid typing `-m` every time:

```bash
# Set default model
llm models default bedrock-ks/amazon.nova-micro-v1:0

# Now you can use it without -m flag
llm "Hello, world!"
llm "What is Python?"

# Check current default
llm models default
```

### Model Aliases

Create shortcuts for frequently used models:

```bash
# Create aliases
llm aliases set nova bedrock-ks/amazon.nova-micro-v1:0
llm aliases set nova-pro bedrock-ks/amazon.nova-pro-v1:0
llm aliases set claude bedrock-ks/anthropic.claude-3-sonnet-20240229-v1:0

# Use aliases
llm -m nova "Hello!"
llm -m claude "Explain quantum computing"

# Set alias as default
llm models default nova
llm "Hello!"  # Uses nova
```

### Logging

The plugin uses structured logging via [structlog](https://www.structlog.org/):

```bash
# Enable human-readable logs
STRUCTLOG_DEV=1 LOG_LEVEL=20 llm models list

# Debug mode
STRUCTLOG_DEV=1 LOG_LEVEL=10 llm -m nova "test"

# JSON logs (for monitoring)
LOG_LEVEL=20 llm -m nova "test" 2> logs.json
```

Log levels: 10=DEBUG, 20=INFO, 30=WARNING (default), 40=ERROR

## Troubleshooting

### No models showing up

```bash
# Verify API key is set
llm keys get bedrock-ks-api-key

# Check with explicit credentials
AWS_BEARER_TOKEN_BEDROCK="your-key" llm models list | grep bedrock-ks/

# Enable debug logging
STRUCTLOG_DEV=1 LOG_LEVEL=10 llm models list
```

### "ValidationException: Model not supported"

Some models require inference profiles in certain regions. Try:
- Different model (e.g., `amazon.nova-micro-v1:0` instead of `claude-sonnet-4`)
- Different AWS region

### "ThrottlingException"

The plugin automatically retries with exponential backoff. If it persists:
- Reduce request frequency
- Request quota increase in AWS Service Quotas
- Use different models

### Clear cache

```bash
llm bedrock-ks refresh
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run manual test suite
python test_manual.py
```

## License

Apache 2.0

## Links

- [GitHub Repository](https://github.com/dinnouti/llm-bedrock-ks)
- [LLM CLI Documentation](https://llm.datasette.io/)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [Bedrock API Keys Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-keys.html)
- [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
