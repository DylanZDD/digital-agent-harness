"""最小 MCP server 示例(模板)

用官方 ``mcp`` SDK 的 FastMCP 写一个 stdio server,暴露两个工具:
``add(a, b)`` 与 ``get_time()``。

运行方式(作为子进程被 harness 拉起,或单独调试)::

    python3 examples/mcp_demo_server.py

在 config.yaml 里接入::

    mcp:
      servers:
        demo:
          command: python3
          args: ["<项目绝对路径>/examples/mcp_demo_server.py"]

把这个文件当模板:加一个 ``@mcp.tool()`` 装饰的函数,就是一个新工具,
harness 启动时会自动发现并注册。
"""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加,返回它们的和。"""
    return a + b


@mcp.tool()
def get_time() -> str:
    """返回当前时间(ISO 8601 字符串)。"""
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输
