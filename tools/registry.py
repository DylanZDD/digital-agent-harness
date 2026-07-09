"""工具注册器

``ToolRegistry`` 聚合所有可用工具,提供:

* ``register(tool)``     —— 注册一个工具
* ``get(name)``          —— 按名取工具
* ``names()``            —— 列出所有工具名
* ``schemas()``          —— 生成 OpenAI ``tools`` 参数
* ``execute(name, args)``—— 调度执行,统一异常兜底
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ── 注册 / 查询 ────────────────────────────────────────────
    def register(self, tool: Tool) -> Tool:
        if not tool.name:
            raise ValueError(f"工具缺少 name:{tool!r}")
        if tool.name in self._tools:
            logger.warning("工具 %s 已存在,将被覆盖", tool.name)
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    # ── 给 LLM 的 schema ───────────────────────────────────────
    def schemas(self) -> list[dict[str, Any]]:
        """生成传给 LLM 的 tools 列表(OpenAI function-calling 格式)。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ── 执行 ───────────────────────────────────────────────────
    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """按名执行工具。工具未找到或抛异常时返回 is_error=True 的 ToolResult,
        而不是向上抛错 —— 保证 agent loop 不会被单个工具打断。"""
        arguments = arguments or {}
        tool = self.get(name)
        if tool is None:
            return ToolResult(output=f"错误:未知的工具 '{name}'。", is_error=True)
        try:
            result = await tool.execute(**arguments)
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, str):  # 允许实现方直接返回字符串
                return ToolResult(output=result)
            return ToolResult(output=str(result))
        except Exception as e:  # noqa: BLE001 - 工具层兜底
            logger.exception("工具 %s 执行异常", name)
            return ToolResult(output=f"工具 '{name}' 执行出错:{type(e).__name__}: {e}", is_error=True)


__all__ = ["ToolRegistry"]
