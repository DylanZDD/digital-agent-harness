"""MCP 工具加载器

根据 config 里的 ``mcp.servers`` 定义,逐个连接 MCP server,
把远端工具包成 ``MCPTool`` 注册进 registry。返回已连接的客户端列表,
调用方负责在退出时逐个 ``close()``。

服务器定义格式(config.yaml)::

    mcp:
      servers:
        # stdio 传输
        filesystem:
          type: stdio           # 可选,默认 stdio
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
          env: {}
        # streamable_http 传输
        my-http-server:
          type: streamable_http
          url: https://example.com/mcp
          headers: {}

同名工具会被自动加上 ``server__`` 前缀,避免冲突。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..registry import ToolRegistry
from .client import MCPClient, MCPTool

logger = logging.getLogger(__name__)


def _build_client(server_name: str, spec: dict[str, Any]) -> MCPClient:
    transport = spec.get("type", "stdio")

    if transport == "streamable_http":
        if "url" not in spec:
            raise ValueError(f"MCP server '{server_name}' 使用 streamable_http 但缺少 'url'")
        return MCPClient(
            name=server_name,
            transport="streamable_http",
            url=spec["url"],
            headers=spec.get("headers") or None,
        )
    else:
        # stdio 传输
        if "command" not in spec:
            raise ValueError(f"MCP server '{server_name}' 缺少必填字段 'command'")
        return MCPClient(
            name=server_name,
            transport="stdio",
            command=spec["command"],
            args=spec.get("args") or [],
            env=spec.get("env") or None,
        )


async def load_mcp_servers(
    servers: dict[str, dict[str, Any]],
    registry: ToolRegistry,
    *,
    connect_timeout: float = 10.0,
) -> list[MCPClient]:
    """并发连接所有 MCP server 并注册其工具。单个 server 失败(含超时)不影响其他。

    各 server 之间无依赖,故用 ``asyncio.gather`` 同时建连,避免一个等一个把
    启动时间叠成 ``N × 单次耗时``。

    Args:
        servers:          ``{server_name: spec}`` 映射。
        registry:         工具注册表。``register`` 是同步方法、内部无 await,故
                          在多个并发任务里注册不会相互交错,是安全的。
        connect_timeout:  单个 server 的 connect / list_tools 各自的超时(秒)。
                          任一 server 卡住时不会拖垮整体启动。
    """
    if not servers:
        return []

    async def _setup_one(server_name: str, spec: dict[str, Any]) -> MCPClient | None:
        try:
            client = _build_client(server_name, spec)
            await asyncio.wait_for(client.connect(), timeout=connect_timeout)
        except Exception as e:  # noqa: BLE001
            logger.error("MCP server '%s' 连接失败:%s: %s", server_name, type(e).__name__, e)
            return None

        try:
            remote_tools = await asyncio.wait_for(
                client.list_tools(), timeout=connect_timeout
            )
        except Exception as e:  # noqa: BLE001
            logger.error("MCP server '%s' 拉取工具失败:%s", server_name, e)
            await client.close()
            return None

        registered = 0
        for t in remote_tools:
            tool = MCPTool(client, t)
            # 同名冲突时加 server 前缀
            if tool.name in registry:
                tool.name = f"{server_name}__{tool.name}"
            registry.register(tool)
            registered += 1
        logger.info("MCP server '%s' 注册了 %d 个工具", server_name, registered)
        return client

    results = await asyncio.gather(
        *(_setup_one(name, spec) for name, spec in servers.items())
    )
    return [c for c in results if c is not None]


async def close_clients(clients: list[MCPClient]) -> None:
    """关闭全部 MCP 客户端(优雅退出)。"""
    for c in clients:
        await c.close()


__all__ = ["load_mcp_servers", "close_clients"]
