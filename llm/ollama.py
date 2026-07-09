"""Ollama 本地推理。

Ollama 提供 OpenAI 兼容端点 ``http://localhost:11434/v1``,
因此复用 openai SDK,只改 base_url 与占位 api_key。
默认模型 ``llama3.1``,可在 config 中覆盖。
"""

from __future__ import annotations

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    provider = "ollama"
    default_base_url = "http://localhost:11434/v1"
    default_model = "llama3.1"
    placeholder_api_key = "ollama"       # Ollama 不校验 key,但 SDK 要求非空


__all__ = ["OllamaLLM"]
