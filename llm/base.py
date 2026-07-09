"""基础 LLM 类

统一封装 OpenAI 兼容的 Chat Completions 接口。三个 provider(openai / ollama /
vllm)都通过 ``openai.AsyncOpenAI`` 客户端调用 —— ollama 与 vllm 都提供
OpenAI 兼容端点,因此只需在子类里设置不同的默认 ``base_url``。

本模块定义两个跨层共用的小数据结构:

* ``ToolCall``   —— 一次工具调用(id / name / 已解析的参数 dict)
* ``LLMResponse``—— 一次 chat 调用的归一化结果

消息格式直接采用 OpenAI 的 dict 格式,由 agent 层负责构造与转换,
避免在层间引入循环依赖。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    from openai import NOT_GIVEN
except ImportError as _e:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    NOT_GIVEN = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


@dataclass
class ToolCall:
    """一次工具调用。arguments 已从 JSON 字符串解析为 dict。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM 返回的归一化结果。"""

    content: str | None               # 文本回复(可能为 None,当模型只调用工具时)
    tool_calls: list[ToolCall]        # 本轮要执行的工具调用
    finish_reason: str                # stop | tool_calls | length | ...
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None                   # 原始响应对象,调试用


class BaseLLM:
    """OpenAI 兼容 Chat Completions 客户端的基类。

    子类通过覆盖类属性或构造参数提供 provider 默认值(base_url / 默认 model /
    占位 api_key)。实例化时 ``config`` 里的非空字段优先。
    """

    provider: str = "base"
    default_base_url: str | None = None
    default_model: str = "gpt-4o-mini"
    placeholder_api_key: str = "EMPTY"

    def __init__(self, cfg: Any) -> None:
        if AsyncOpenAI is None:
            raise ImportError(
                "未安装 openai SDK,请先 pip install openai(见 requirements.txt)"
            ) from _IMPORT_ERROR
        self.cfg = cfg
        self.model: str = cfg.model or self.default_model
        self.temperature: float = getattr(cfg, "temperature", 0.2)
        self.max_tokens: int | None = getattr(cfg, "max_tokens", None)

        base_url = cfg.base_url or self.default_base_url
        api_key = cfg.api_key or self.placeholder_api_key
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ── 核心调用 ───────────────────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发起一次 chat completion。

        Args:
            messages: OpenAI 格式的消息列表。
            tools:    OpenAI function-calling 格式的工具 schema(可为空)。
            **kwargs: 透传给底层 API(如 tool_choice)。
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = tools
        request.update(kwargs)

        logger.debug("LLM 请求:model=%s, messages=%d, tools=%d", self.model, len(messages), len(tools or []))

        resp = await self.client.chat.completions.create(**request)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            raw_args = tc.function.arguments if tc.function else None
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                # 模型偶尔返回非法 JSON,保留原始字符串供调试
                args = {"_raw_arguments": raw_args}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=resp,
        )

    async def close(self) -> None:
        """关闭底层 HTTP 客户端。"""
        await self.client.close()


__all__ = ["BaseLLM", "LLMResponse", "ToolCall"]
