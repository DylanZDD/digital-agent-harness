"""OpenAI 官方 API。

* base_url 用官方默认(留 None)
* api_key 从 config 或 ``OPENAI_API_KEY`` 环境变量取
"""

from __future__ import annotations

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    provider = "openai"
    default_base_url = None              # openai SDK 默认指向 https://api.openai.com/v1
    default_model = "gpt-4o-mini"
    placeholder_api_key = "EMPTY"        # config 已优先注入环境变量,此处仅兜底


__all__ = ["OpenAILLM"]
