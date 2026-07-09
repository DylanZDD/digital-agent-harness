"""MCP 客户端

封装官方 ``mcp`` SDK 的 stdio 传输,把一个 MCP server 的生命周期管理好:

* ``connect()`` —— 启动子进程、握手 initialize
* ``list_tools()`` —— 拉取远端工具清单
* ``call_tool(name, args)`` —— 调用远端工具
* ``close()`` —— 关闭子进程与 session

并用 ``MCPTool`` 把远端工具适配成本地 ``Tool`` 抽象,挂进 registry 后
agent loop 可无差别调用。
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as _e:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


class MCPClient:
    """一个 MCP server 的连接。"""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if StdioServerParameters is None:
            raise ImportError(
                "未安装 mcp SDK,请先 pip install mcp(见 requirements.txt)"
            ) from _IMPORT_ERROR
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env) if env else None
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        logger.info("MCP server '%s' 已连接(command=%s)", self.name, self.command)

    async def list_tools(self) -> list[Any]:
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' 尚未 connect()")
        result = await self.session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' 尚未 connect()")
        return await self.session.call_tool(name, arguments or {})

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # noqa: BLE001
                logger.exception("关闭 MCP server '%s' 时出错", self.name)
            finally:
                self._stack = None
                self.session = None


# ──────────────────────────────────────────────────────────────


class MCPTool(Tool):
    """把一个 MCP 远端工具适配成本地 Tool。"""

    def __init__(self, client: MCPClient, tool_info: Any) -> None:
        self._client = client
        self._remote_name: str = tool_info.name
        self.name: str = tool_info.name
        self.description: str = tool_info.description or ""
        self.parameters: dict[str, Any] = tool_info.inputSchema or {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._client.call_tool(self._remote_name, kwargs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                output=f"MCP 工具 '{self.name}' 调用失败:{type(e).__name__}: {e}",
                is_error=True,
            )
        # 把 content blocks 拼成文本
        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            texts.append(text if text is not None else str(block))
        is_error = bool(getattr(result, "isError", False))
        output = "\n".join(texts).strip() or "(MCP 工具无文本输出)"
        return ToolResult(output=output, is_error=is_error)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MCPTool {self.name} @ {self._client.name}>"


__all__ = ["MCPClient", "MCPTool"]
