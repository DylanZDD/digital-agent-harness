"""LLM 抽象层:统一 OpenAI 兼容接口,支持 openai / ollama / vllm / zhipu。"""

from .base import BaseLLM, LLMResponse, ToolCall
from .openai import OpenAILLM
from .ollama import OllamaLLM
from .vllm import VLLMLLM
from .zhipu import ZhipuLLM


def build_llm(cfg) -> BaseLLM:
    """根据 config.LLMConfig.provider 构造对应的 LLM 客户端。"""
    provider = (cfg.provider or "openai").lower()
    mapping = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
        "vllm": VLLMLLM,
        "zhipu": ZhipuLLM,
    }
    if provider not in mapping:
        raise ValueError(f"未知的 LLM provider: {provider}(可选: {list(mapping)})")
    return mapping[provider](cfg)


__all__ = [
    "BaseLLM",
    "LLMResponse",
    "ToolCall",
    "OpenAILLM",
    "OllamaLLM",
    "VLLMLLM",
    "ZhipuLLM",
    "build_llm",
]
