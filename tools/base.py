"""基础工具类

所有工具(本地工具、MCP 工具、skill 工具)实现统一的 ``Tool`` 抽象,
这样 ``ToolRegistry`` 和 agent loop 可以无差别地调度它们。

一个工具需要提供:

* ``name``        —— 唯一标识,模型用它来调用
* ``description`` —— 给模型看的说明
* ``parameters``  —— JSON Schema,描述入参
* ``execute(**kwargs) -> ToolResult`` —— 实际执行(异步)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具执行结果。统一以字符串返回给 LLM。"""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output


class Tool(ABC):
    """工具抽象基类。"""

    name: str = ""
    description: str = ""
    # JSON Schema,描述函数入参;默认无参数。
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具,返回 ToolResult。实现方应自行捕获异常并以 is_error=True 返回。"""
        raise NotImplementedError

    # ── 给 OpenAI function-calling 用的 schema ──────────────────
    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI ``tools`` 字段里的一项。"""
        params = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": params,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


__all__ = ["Tool", "ToolResult"]
