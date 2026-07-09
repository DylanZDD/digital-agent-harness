"""Tool executor

把解析出的 ``ToolCall`` 交给 ``ToolRegistry`` 执行,并把结果包成
``role="tool"`` 的 ``Message`` 回填进对话。

registry 自身已有异常兜底(返回 is_error 的 ToolResult),这里再做一层
防御:即便 registry 出意外,也保证回填一条可读的 tool 消息,不中断 loop。
"""

from __future__ import annotations

import asyncio
import logging

from llm.base import ToolCall
from tools.registry import ToolRegistry

from .state import Message

logger = logging.getLogger(__name__)


async def execute_call(call: ToolCall, registry: ToolRegistry) -> Message:
    """执行单个工具调用,返回对应的 tool 消息。"""
    try:
        result = await registry.execute(call.name, call.arguments)
        output = result.output
    except Exception as e:  # noqa: BLE001 - 最后一道防线
        logger.exception("执行工具 %s 时发生未捕获异常", call.name)
        output = f"执行工具 '{call.name}' 时发生未捕获异常:{type(e).__name__}: {e}"

    return Message(
        role="tool",
        tool_call_id=call.id,
        name=call.name,
        content=output,
    )


async def execute_calls(
    calls: list[ToolCall],
    registry: ToolRegistry,
    max_concurrency: int = 8,
) -> list[Message]:
    """并发执行多个工具调用,结果按调用顺序返回。

    用 ``asyncio.Semaphore`` 限流,避免一轮里几十个 shell 同时开。
    """
    if not calls:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _bounded(c: ToolCall) -> Message:
        async with sem:
            return await execute_call(c, registry)

    return await asyncio.gather(*[_bounded(c) for c in calls])


__all__ = ["execute_call", "execute_calls"]
