# llm-bedrock

LLM plugin for [Amazon Bedrock](https://aws.amazon.com/bedrock/) - provides access to **all models** that support the Bedrock Converse API.

## Features

- ✅ **Universal model access** - Auto-discovers all Converse API-compatible models
- ✅ **Streaming responses** - Real-time output with ConverseStream API
- ✅ **Multimodal support** - Images and documents via attachments
- ✅ **Tool use** - Function calling capabilities
- ✅ **System prompts** - Custom behavior instructions
- ✅ **Conversation history** - Multi-turn conversations
- ✅ **Modern authentication** - Uses Bedrock API Keys (not IAM)
- ✅ **Multiple regions** - Support for all AWS regions

## Installation

```bash
llm install llm-bedrock
```

Or with pip:

```bash
pip install llm-bedrock
```

## Setup

### 1. Get a Bedrock API Key

1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to "API Keys"
3. Create a new API key (short-term or long-term)
4. Copy the key

### 2. Configure Authentication

```bash
llm keys set bedrock-api-key
# Paste your API key when prompted

llm keys set bedrock-region
# Enter your AWS region (e.g., us-east-1)
```

Or use environment variables:

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-api-key-here"
export AWS_DEFAULT_REGION="us-east-1"
```

## Usage

### Basic Usage

```bash
llm -m bedrock/claude-3-sonnet "Hello, how are you?"
```

### List Available Models

```bash
llm models list | grep bedrock/
```

### Streaming

```bash
llm -m bedrock/claude-3-sonnet "Write a poem" --stream
```

### With Images

```bash
llm -m bedrock/claude-3-sonnet \
    "What's in this image?" \
    -a image.jpg
```

### System Prompt

```bash
llm -m bedrock/claude-3-sonnet \
    --system "You are a helpful coding assistant" \
    "Write a Python function to reverse a string"
```

### With Options

```bash
llm -m bedrock/claude-3-sonnet \
    -o max_tokens 2000 \
    -o temperature 0.8 \
    "Tell me a story"
```

## Available Models

Auto-discovered models include:
- **Anthropic:** Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- **Amazon:** Nova Pro, Nova Lite, Nova Micro
- **Meta:** Llama 3.1/3.2 (8B, 70B, 405B)
- **Mistral AI:** Mistral Large, Mistral 7B
- **Cohere:** Command R/R+

## Configuration

### Options

Available options (via `-o` flag):

- `max_tokens` (int, default: 4096) - Maximum tokens to generate
- `temperature` (float, 0-1, default: 0.7) - Sampling temperature
- `top_p` (float, 0-1, default: 0.9) - Nucleus sampling threshold
- `stop_sequences` (list) - Stop generation at these sequences

## Development

```bash
git clone https://github.com/yourusername/llm-bedrock.git
cd llm-bedrock
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0

## Links

- [LLM CLI](https://llm.datasette.io/)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
