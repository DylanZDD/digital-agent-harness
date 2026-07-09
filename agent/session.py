"""session 管理类

``Session`` 把 LLM、工具注册表、系统提示词、状态串起来,对外提供:

* ``run(user_input)`` —— 单轮:塞入用户消息 → 跑 agent_turn → 返回最终文本
* ``repl()``          —— 交互式 REPL,支持 ``/help`` ``/reset`` ``/exit``
* ``reset()``         —— 清空历史,开新对话

同时实现 ``agent_turn`` 的事件回调,用 rich 把工具调用 / 结果 / 最终答复
漂亮地打印到终端。
"""

from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLM
from tools.registry import ToolRegistry

from .loop import agent_turn
from .state import AgentState

logger = logging.getLogger(__name__)

# ── 终端 UI:优先 rich,缺失则降级到 print ─────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    _console = Console()

    def _print_assistant(content: str) -> None:
        _console.print(Text(content, style="dim"))

    def _print_tool_call(name: str, args: dict) -> None:
        _console.print(f"  [bold cyan]→ 工具[/] [yellow]{name}[/] {args}")

    def _print_tool_result(name: str, output: str, is_error: bool = False) -> None:
        style = "red" if is_error else "green"
        preview = output if len(output) <= 1200 else output[:1200] + "\n…(已截断)"
        _console.print(Panel(preview, title=f"{name}", border_style=style, expand=False))

    def _print_final(content: str) -> None:
        _console.print()
        _console.print(Panel(Markdown(content), title="助手", border_style="blue"))

    def _print_system(msg: str) -> None:
        _console.print(msg, style="magenta")

except Exception:  # pragma: no cover - rich 不可用时降级
    def _print_assistant(content: str) -> None:
        print(content)

    def _print_tool_call(name: str, args: dict) -> None:
        print(f"  → 工具 {name} {args}")

    def _print_tool_result(name: str, output: str, is_error: bool = False) -> None:
        tag = "[错误]" if is_error else ""
        print(f"  [{name}{tag}] {output}")

    def _print_final(content: str) -> None:
        print("\n=== 助手 ===")
        print(content)

    def _print_system(msg: str) -> None:
        print(msg)


class Session:
    def __init__(
        self,
        config: Any,
        llm: BaseLLM,
        registry: ToolRegistry,
        system_prompt: str,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.state = AgentState()

    # ── 事件回调 ───────────────────────────────────────────────
    def _on_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "assistant":
            if data.get("content"):
                _print_assistant(data["content"])
        elif kind == "tool_call":
            _print_tool_call(data["name"], data["arguments"])
        elif kind == "tool_result":
            output = data.get("output", "")
            # registry 兜底返回的错误文本以「错误:」开头,据此判断样式
            is_error = str(output).lstrip().startswith(("错误:", "执行工具"))
            _print_tool_result(data["name"], output, is_error=is_error)
        elif kind == "limit":
            _print_system(f"⚠ 达到最大迭代数 {data['max_iterations']},已停止。")

    # ── 单轮 ───────────────────────────────────────────────────
    async def run(self, user_input: str, show_final: bool = True) -> str:
        """运行一轮。返回最终文本。"""
        self.state.add_user(user_input)
        final = await agent_turn(
            state=self.state,
            llm=self.llm,
            registry=self.registry,
            system_prompt=self.system_prompt,
            max_iterations=self.config.agent.max_iterations,
            on_event=self._on_event,
        )
        if show_final and final:
            _print_final(final)
        return final

    # ── 交互式 REPL ────────────────────────────────────────────
    async def repl(self) -> None:
        _print_system("进入交互模式。输入 /help 查看命令,/exit 退出。\n")
        while True:
            try:
                line = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                _print_system("\n再见。")
                break
            if not line:
                continue

            if line.startswith("/"):
                if self._handle_command(line):
                    break
                continue

            try:
                await self.run(line)
            except Exception as e:  # noqa: BLE001 - 单轮失败不杀 REPL
                logger.exception("本轮出错")
                _print_system(f"⚠ 本轮出错:{type(e).__name__}: {e}")
            _print_system("")  # 空行分隔

    def _handle_command(self, line: str) -> bool:
        """处理 / 命令。返回 True 表示请求退出 REPL。"""
        cmd = line.lower()
        if cmd in ("/exit", "/quit"):
            _print_system("再见。")
            return True
        if cmd == "/reset":
            self.reset()
            _print_system("已清空对话历史。")
        elif cmd == "/help":
            _print_system("命令:/help  帮助   /reset  清空历史   /exit  退出")
        else:
            _print_system(f"未知命令:{line}(试试 /help)")
        return False

    def reset(self) -> None:
        self.state = AgentState()


__all__ = ["Session"]
