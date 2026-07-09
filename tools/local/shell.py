"""Shell 工具

通过系统 shell 执行命令,返回 stdout/stderr/退出码。

安全策略(可配置):

* ``shell_allowlist`` 为空 —— 允许任意命令(开发态便利,生产慎用)
* ``shell_allowlist`` 非空 —— 只允许命令第一个 token 命中白名单前缀
* 每条命令有独立超时,超时则杀进程
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any

from ..base import Tool, ToolResult


class ShellTool(Tool):
    name = "run_shell"
    description = (
        "在系统 shell 中执行一条命令,返回 stdout、stderr 与退出码。"
        "用于运行脚本、构建、git 等任务。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的完整 shell 命令"},
        },
        "required": ["command"],
    }

    def __init__(
        self,
        workdir: str | os.PathLike = ".",
        timeout: float = 120.0,
        allowlist: list[str] | None = None,
    ) -> None:
        self.workdir = str(workdir)
        self.timeout = timeout
        self.allowlist = [a.strip() for a in (allowlist or []) if a.strip()]

    def _check_allowed(self, command: str) -> str | None:
        """返回拒绝原因,None 表示放行。"""
        if not self.allowlist:
            return None
        try:
            first = shlex.split(command, posix=True)[0]
        except Exception:  # noqa: BLE001 - 解析失败按原始 token 判断
            first = command.split()[0] if command.split() else ""
        first_base = os.path.basename(first)
        for allowed in self.allowlist:
            if first == allowed or first_base == allowed:
                return None
        return f"命令 '{first}' 不在白名单内(允许:{self.allowlist})"

    async def execute(self, command: str, **_: Any) -> ToolResult:
        if not command or not command.strip():
            return ToolResult(output="空命令。", is_error=True)

        denied = self._check_allowed(command)
        if denied:
            return ToolResult(output=denied, is_error=True)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"启动命令失败:{e}", is_error=True)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ToolResult(
                output=f"命令超时({self.timeout}s),已终止:{command}",
                is_error=True,
            )

        stdout = stdout_b.decode("utf-8", errors="replace").rstrip()
        stderr = stderr_b.decode("utf-8", errors="replace").rstrip()
        parts = [f"退出码: {proc.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        is_error = proc.returncode != 0
        return ToolResult(output="\n".join(parts), is_error=is_error)


def shell_tool(workdir: str | os.PathLike = ".", timeout: float = 120.0, allowlist: list[str] | None = None) -> ShellTool:
    return ShellTool(workdir=workdir, timeout=timeout, allowlist=allowlist)


__all__ = ["ShellTool", "shell_tool"]
