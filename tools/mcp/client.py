"""MCP 客户端

封装官方 ``mcp`` SDK 的传输,把一个 MCP server 的生命周期管理好:

* ``connect()`` —— 连接、握手 initialize
* ``list_tools()`` —— 拉取远端工具清单
* ``call_tool(name, args)`` —— 调用远端工具
* ``close()`` —— 关闭 session

支持两种传输方式:
* ``stdio`` —— 本地子进程(stdio_client)
* ``streamable_http`` —— HTTP 流式传输(streamablehttp_client)

并用 ``MCPTool`` 把远端工具适配成本地 ``Tool`` 抽象,挂进 registry 后
agent loop 可无差别调用。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError as _e:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    streamablehttp_client = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


class MCPClient:
    """一个 MCP server 的连接。"""

    def __init__(
        self,
        name: str,
        transport: str = "stdio",
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if ClientSession is None:
            raise ImportError(
                "未安装 mcp SDK,请先 pip install mcp(见 requirements.txt)"
            ) from _IMPORT_ERROR
        self.name = name
        self.transport = transport
        # stdio 相关
        self.command = command
        self.args = list(args or [])
        self.env = dict(env) if env else None
        # streamable_http 相关
        self.url = url
        self.headers = dict(headers) if headers else None
        # 公共
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        if self.transport == "streamable_http":
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self.url, headers=self.headers)
            )
            self.session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            logger.info(
                "MCP server '%s' 已连接(transport=streamable_http, url=%s)",
                self.name, self.url,
            )
        else:
            params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self.session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            logger.info(
                "MCP server '%s' 已连接(transport=stdio, command=%s)",
                self.name, self.command,
            )

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
            except (asyncio.CancelledError, Exception) as e:  # noqa: BLE001
                # streamable_http 关闭时,其内部后台 GET 监听任务被取消,会引发
                # CancelledError(BaseException 子类),并常伴生
                # "RuntimeError: exit a cancel scope …"。二者同属 SDK 在 teardown
                # 阶段的已知竞态;close 本身已是终态清理,连接最终都会被回收,
                # 没有可执行的补救动作。故只记一行 DEBUG、不打堆栈:既保证优雅退出
                # (不打断 app.py finally 里的 llm.close() 等后续清理、不以非 0 码
                # 退出),又不在每次关闭时刷一堆吓人的 ERROR 堆栈。
                # KeyboardInterrupt / SystemExit 不在此列,仍会正常向上抛出。
                logger.debug(
                    "关闭 MCP server '%s' 时捕获到 %s(已忽略,属关闭期正常现象): %s",
                    self.name, type(e).__name__, e,
                )
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
        # streamable_http 传输在远端 GET 长连接断开时,会取消在途请求并抛出
        # CancelledError(它是 BaseException 子类,不会被下面的 except Exception
        # 捕获)。这类中断是瞬时的:传输随后会自动重连,因此做有限次重试通常
        # 即可恢复。重试到上限仍失败则返回 is_error 的 ToolResult,而不是向上
        # 抛——否则一次网络抖动就会让 CancelledError 冒泡出 asyncio.run、
        # 打垮整个 agent loop(与本地工具在 registry.execute 里的兜底语义一致)。
        result = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._client.call_tool(self._remote_name, kwargs)
                break
            except asyncio.CancelledError:
                if attempt >= max_attempts:
                    return ToolResult(
                        output=(
                            f"MCP 工具 '{self.name}' 调用因远端连接中断失败"
                            f"(已重试 {max_attempts} 次),请稍后重试。"
                        ),
                        is_error=True,
                    )
                logger.debug(
                    "MCP 工具 '%s' 第 %d/%d 次调用因连接中断失败,退避后重试",
                    self.name, attempt, max_attempts,
                )
                await asyncio.sleep(1.0 * attempt)  # 给传输留出重连时间
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
