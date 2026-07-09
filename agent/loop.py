"""Agent loop

核心推理-执行循环。一轮(agent turn)从当前对话状态出发:

    构建 messages → LLM 调用 → 解析工具调用
        ├─ 有工具调用:执行 → 回填结果 → 回到「构建 messages」
        └─ 无工具调用:返回最终文本,本轮结束

用 ``max_iterations`` 兜底,防止模型陷入无限工具调用。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from llm.base import BaseLLM
from tools.registry import ToolRegistry

from .executor import execute_call
from .parser import extract_tool_calls
from .state import AgentState, to_openai_messages

logger = logging.getLogger(__name__)

# 事件回调: (event_type: str, data: dict) -> None
#   event_type ∈ {"assistant", "tool_call", "tool_result", "final", "limit"}
EventHandler = Callable[[str, dict[str, Any]], None]


async def agent_turn(
    state: AgentState,
    llm: BaseLLM,
    registry: ToolRegistry,
    system_prompt: str,
    max_iterations: int = 25,
    on_event: EventHandler | None = None,
) -> str:
    """运行一轮 agent,直到模型给出不再调用工具的最终答复,或达到迭代上限。

    Args:
        state:         会话状态(已含本轮用户消息)。
        llm:           LLM 客户端。
        registry:      工具注册表。
        system_prompt: 系统提示词。
        max_iterations:最大工具调用轮数。
        on_event:      可选的事件回调,用于 UI 展示进度。

    Returns:
        最终文本答复。
    """

    def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(kind, data)
            except Exception:  # noqa: BLE001 - UI 回调不应中断 loop
                logger.exception("事件回调抛错(kind=%s)", kind)

    schemas = registry.schemas()
    final_text = ""

    for i in range(1, max_iterations + 1):
        state.iteration = i
        messages = to_openai_messages(state.messages, system_prompt)

        response = await llm.chat(messages, tools=schemas or None)

        # 记录 assistant 回复(含本轮工具调用),OpenAI 要求它在 tool 消息之前
        state.add_assistant(response.content, response.tool_calls)
        emit("assistant", {"content": response.content, "iteration": i})

        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # 模型给出最终答复
            final_text = response.content or ""
            break

        # 依次执行工具调用,边执行边回填、边上报(便于 UI 顺序展示)
        for call in tool_calls:
            emit("tool_call", {"name": call.name, "arguments": call.arguments, "iteration": i})
            tool_msg = await execute_call(call, registry)
            state.messages.append(tool_msg)
            emit(
                "tool_result",
                {"name": call.name, "output": tool_msg.content, "iteration": i},
            )
    else:
        # for 循环正常结束 = 达到迭代上限
        emit("limit", {"max_iterations": max_iterations})
        final_text = final_text or f"(已达到最大工具调用轮数 {max_iterations},自动停止。)"

    emit("final", {"content": final_text})
    return final_text


__all__ = ["agent_turn", "EventHandler"]
