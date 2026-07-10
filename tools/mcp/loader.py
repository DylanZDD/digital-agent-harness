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
) -> list[MCPClient]:
    """连接所有 MCP server 并注册其工具。单个 server 失败不影响其他。"""
    clients: list[MCPClient] = []
    if not servers:
        return clients

    for server_name, spec in servers.items():
        try:
            client = _build_client(server_name, spec)
            await client.connect()
        except Exception as e:  # noqa: BLE001
            logger.error("MCP server '%s' 连接失败:%s: %s", server_name, type(e).__name__, e)
            continue

        try:
            remote_tools = await client.list_tools()
        except Exception as e:  # noqa: BLE001
            logger.error("MCP server '%s' 拉取工具失败:%s", server_name, e)
            await client.close()
            continue

        registered = 0
        for t in remote_tools:
            tool = MCPTool(client, t)
            # 同名冲突时加 server 前缀
            if tool.name in registry:
                tool.name = f"{server_name}__{tool.name}"
            registry.register(tool)
            registered += 1
        logger.info(
            "MCP server '%s' 注册了 %d 个工具", server_name, registered
        )
        clients.append(client)

    return clients


async def close_clients(clients: list[MCPClient]) -> None:
    """关闭全部 MCP 客户端(优雅退出)。"""
    for c in clients:
        await c.close()


__all__ = ["load_mcp_servers", "close_clients"]
