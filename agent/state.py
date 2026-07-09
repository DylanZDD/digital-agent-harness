"""Agent 状态

定义对话中流转的 ``Message`` 与聚合状态 ``AgentState``,
以及把内部消息转成 OpenAI 请求格式的 ``to_openai_messages``。

消息角色:

* ``system``    —— 系统提示词(由 planner 构建,发送时拼到最前)
* ``user``      —— 用户输入
* ``assistant`` —— 模型回复(可能带 tool_calls)
* ``tool``      —— 工具执行结果(必须带 tool_call_id 对应上面的调用)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from llm.base import ToolCall


@dataclass
class Message:
    role: str                                   # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] | None = None    # 仅 assistant
    tool_call_id: str | None = None             # 仅 tool
    name: str | None = None                     # tool 使用的工具名

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            d["content"] = self.content
            if self.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in self.tool_calls
                ]
        elif self.role == "tool":
            d["content"] = self.content or ""
            if self.tool_call_id:
                d["tool_call_id"] = self.tool_call_id
        else:  # system / user
            d["content"] = self.content or ""
        return d


@dataclass
class AgentState:
    """单次会话的运行时状态。系统提示词不存这里,发送时由外部拼入。"""

    messages: list[Message] = field(default_factory=list)
    iteration: int = 0

    # ── 追加消息的便捷方法 ─────────────────────────────────────
    def add_user(self, text: str) -> Message:
        m = Message(role="user", content=text)
        self.messages.append(m)
        return m

    def add_assistant(self, content: str | None, tool_calls: list[ToolCall] | None = None) -> Message:
        m = Message(role="assistant", content=content, tool_calls=tool_calls)
        self.messages.append(m)
        return m

    def add_tool_result(self, tool_call_id: str, name: str, output: str) -> Message:
        m = Message(role="tool", tool_call_id=tool_call_id, name=name, content=output)
        self.messages.append(m)
        return m

    def reset(self) -> None:
        self.messages.clear()
        self.iteration = 0


def to_openai_messages(messages: list[Message], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """把内部消息列表转成 OpenAI 请求格式;有 system_prompt 则拼到最前。"""
    out: list[dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        out.append(m.to_openai())
    return out


__all__ = ["Message", "AgentState", "to_openai_messages"]
