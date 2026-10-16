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
from typing import Any, Callable

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


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """把模型给的 arguments JSON 字符串解析成 dict;非法 JSON 保留原文供调试。"""
    try:
        args = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {"_raw_arguments": raw}
    return args if isinstance(args, dict) else {"_raw_arguments": raw}


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
        self.stream: bool = bool(getattr(cfg, "stream", True))
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
        on_delta: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发起一次 chat completion。

        Args:
            messages: OpenAI 格式的消息列表。
            tools:    OpenAI function-calling 格式的工具 schema(可为空)。
            on_delta: 流式回调;给出且 ``llm.stream`` 开启时走流式请求,
                      文本 delta 每段回调一次。工具调用的增量在内部按
                      index 拼装,返回的 LLMResponse 与非流式完全一致。
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

        if self.stream and on_delta is not None:
            return await self._chat_stream(request, on_delta)

        resp = await self.client.chat.completions.create(**request)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            raw_args = tc.function.arguments if tc.function else None
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_arguments(raw_args))
            )

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

    async def _chat_stream(
        self, request: dict[str, Any], on_delta: Callable[[str], None]
    ) -> LLMResponse:
        """流式请求:累积文本与工具调用增量,拼成与非流式一致的 LLMResponse。"""
        stream_req = {**request, "stream": True, "stream_options": {"include_usage": True}}
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}  # delta.index -> 增量累积
        finish_reason = "stop"
        usage: dict[str, int] = {}

        async for chunk in await self.client.chat.completions.create(**stream_req):
            # usage 只在末块给(该块 choices 为空)
            if getattr(chunk, "usage", None):
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                on_delta(delta.content)
            # 工具调用增量:首块带 id/name,后续 arguments 以字符串分片到达
            for tc in delta.tool_calls or []:
                acc = tool_acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    acc["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        acc["name"] += fn.name
                    if fn.arguments:
                        acc["args"] += fn.arguments

        tool_calls = [
            ToolCall(
                id=acc["id"] or f"stream-{idx}",
                name=acc["name"],
                arguments=_parse_arguments(acc["args"]),
            )
            for idx, acc in sorted(tool_acc.items())
        ]
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def close(self) -> None:
        """关闭底层 HTTP 客户端。"""
        await self.client.close()


__all__ = ["BaseLLM", "LLMResponse", "ToolCall"]
