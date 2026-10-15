"""session 管理类

``Session`` 把 LLM、工具注册表、系统提示词、状态串起来,对外提供:

* ``run(user_input)`` —— 单轮:塞入用户消息 → 跑 agent_turn → 返回最终文本
* ``repl()``          —— 交互式 REPL,支持 ``/help`` ``/reset`` ``/model`` ``/exit``
* ``reset()``         —— 清空历史,开新对话

同时实现 ``agent_turn`` 的事件回调,负责终端展示。布局是无框扁平风格
(参考 Claude Code),靠符号 + 颜色 + 缩进区分三类内容:

* 模型正文 —— 顶格,delta 到达时实时按 Markdown 渲染(rich Live)
* 工具调用 —— ``⏺`` 青色符号 + 粗体工具名 + dim 参数,执行期间转 spinner
* 工具返回 —— 缩进 + ``│`` 竖线 gutter(成功绿 / 出错红),超长截断

输入框使用 prompt_toolkit,正确处理 CJK 宽字符,告别中文删除残留。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from llm import build_llm
from llm.base import BaseLLM
from tools.registry import ToolRegistry

from .loop import agent_turn
from .state import AgentState

logger = logging.getLogger(__name__)

# ── 终端 UI:优先 rich,缺失则降级到 print ─────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text

    _console = Console()

    class _StreamRenderer:
        """一段模型回复的流式渲染:Live 区域里随 delta 实时按 Markdown 成形。"""

        def __init__(self) -> None:
            self._parts: list[str] = []
            self._live = Live(
                console=_console, vertical_overflow="visible", refresh_per_second=8
            )
            self._live.start()

        def add(self, text: str) -> None:
            self._parts.append(text)
            self._live.update(Markdown("".join(self._parts)))

        def stop(self) -> None:
            self._live.stop()

    def _fmt_args(args: dict) -> str:
        s = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        return s if len(s) <= 120 else s[:117] + "…"

    def _print_assistant(content: str) -> None:
        _console.print(content, style="dim")

    def _tool_spinner(name: str, args: dict) -> Any:
        """工具执行期间的 spinner,返回给调用方在结果到达时 stop()。"""
        return _console.status(f"[bold cyan]⏺[/] [bold]{name}[/] [dim]({_fmt_args(args)})[/]")

    def _print_tool_call(name: str, args: dict) -> None:
        _console.print(f"[bold cyan]⏺[/] [bold]{name}[/] [dim]({_fmt_args(args)})[/]")

    def _print_tool_result(name: str, output: str, is_error: bool = False) -> None:
        lines = output.splitlines() or ["(无输出)"]
        shown, hidden = lines[:12], max(len(lines) - 12, 0)
        bar_style = "red" if is_error else "green"
        for ln in shown:
            ln = ln[:200] + "…" if len(ln) > 200 else ln
            line = Text("  │ ", style=bar_style)
            line.append(ln, style="dim")
            _console.print(line)
        if hidden:
            tail = Text("  │ ", style=bar_style)
            tail.append(f"… +{hidden} 行", style="dim")
            _console.print(tail)

    def _print_final(content: str) -> None:
        _console.print()
        _console.print(Markdown(content))

    def _print_system(msg: str) -> None:
        _console.print(msg, style="magenta")

except Exception:  # pragma: no cover - rich 不可用时降级

    class _StreamRenderer:
        def __init__(self) -> None:
            self._printed = False

        def add(self, text: str) -> None:
            print(text, end="", flush=True)
            self._printed = True

        def stop(self) -> None:
            if self._printed:
                print()

    def _fmt_args(args: dict) -> str:
        return str(args)

    def _print_assistant(content: str) -> None:
        print(content)

    def _tool_spinner(name: str, args: dict) -> Any:
        return None

    def _print_tool_call(name: str, args: dict) -> None:
        print(f"  ⏺ {name} {args}")

    def _print_tool_result(name: str, output: str, is_error: bool = False) -> None:
        tag = "[错误]" if is_error else ""
        print(f"  │ {name}{tag}: {output[:1200]}")

    def _print_final(content: str) -> None:
        print("\n" + content)

    def _print_system(msg: str) -> None:
        print(msg)


# ── 输入框:优先 prompt_toolkit,缺失则降级到 input() ──────────
# 注意:REPL 跑在 asyncio 事件循环里,必须用 prompt_async;
# 同步的 prompt() 内部会调 asyncio.run(),嵌套事件循环会抛 RuntimeError。
try:
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts import PromptSession

    async def _read_input(prompt_str: str, history: InMemoryHistory) -> str:
        """prompt_toolkit 接管终端输入,正确处理 CJK 宽字符。"""
        return await PromptSession(history=history).prompt_async(
            prompt_str,
            multiline=False,
            enable_history_search=True,
        )

except Exception:  # pragma: no cover

    async def _read_input(prompt_str: str, history: Any) -> str:
        # input() 是阻塞的,丢到线程里避免卡住事件循环。
        return await asyncio.to_thread(input, prompt_str)


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
        self._history = InMemoryHistory()  # 历史记录,↑/↓ 翻阅
        # 展示层运行时状态
        self._stream: Any = None            # 进行中的流式渲染区(_StreamRenderer)
        self._spinner: Any = None           # 进行中的工具 spinner
        self._pending_call: tuple[str, dict] | None = None  # spinner 对应的工具调用

    # ── 事件回调 ───────────────────────────────────────────────
    def _on_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "delta":
            if self._stream is None:
                _print_system("")  # 与上方输入 / 工具结果空开一行
                self._stream = _StreamRenderer()
            self._stream.add(data["content"])
        elif kind == "assistant":
            # 非流式兜底:整段进度文本。流式模式下文字已通过 delta 显示过,跳过。
            if self._stream is None and data.get("content"):
                _print_assistant(data["content"])
        elif kind == "tool_call":
            self._close_stream()
            self._pending_call = (data["name"], data["arguments"])
            self._spinner = _tool_spinner(*self._pending_call)
            if self._spinner is not None:
                self._spinner.start()
        elif kind == "tool_result":
            if self._spinner is not None:
                self._spinner.stop()
                self._spinner = None
            if self._pending_call is not None:
                _print_tool_call(*self._pending_call)
                self._pending_call = None
            output = data.get("output", "")
            # registry 兜底返回的错误文本以「错误:」开头,据此判断样式
            is_error = str(output).lstrip().startswith(("错误:", "执行工具"))
            _print_tool_result(data["name"], output, is_error=is_error)
        elif kind == "limit":
            self._close_stream()
            _print_system(f"⚠ 达到最大迭代数 {data['max_iterations']},已停止。")

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    # ── 单轮 ───────────────────────────────────────────────────
    async def run(self, user_input: str, show_final: bool = True) -> str:
        """运行一轮。返回最终文本。"""
        self.state.add_user(user_input)
        self._stream = None
        try:
            final = await agent_turn(
                state=self.state,
                llm=self.llm,
                registry=self.registry,
                system_prompt=self.system_prompt,
                max_iterations=self.config.agent.max_iterations,
                on_event=self._on_event,
            )
        finally:
            streamed = self._stream is not None
            self._close_stream()
        # 流式模式下文字已通过 delta 逐段显示过,不再整体重渲染
        if show_final and final and not streamed:
            _print_final(final)
        return final

    # ── 交互式 REPL ────────────────────────────────────────────
    async def repl(self) -> None:
        _print_system("进入交互模式。输入 /help 查看命令,/exit 退出。\n")
        while True:
            try:
                line = (await _read_input("你 > ", self._history)).strip()
            except (EOFError, KeyboardInterrupt):
                _print_system("\n再见。")
                break
            if not line:
                continue

            if line.startswith("/"):
                if await self._handle_command(line):
                    break
                continue

            try:
                await self.run(line)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl+C 只中断当前轮。asyncio.run 会把 SIGINT 转成对主任务的
                # 取消,协程内表现为 CancelledError;竞态下则直接是
                # KeyboardInterrupt。二者都吞掉,回到输入框,不退出 REPL。
                _print_system("⚠ 已中断本轮。")
            except Exception as e:  # noqa: BLE001 - 单轮失败不杀 REPL
                logger.exception("本轮出错")
                _print_system(f"⚠ 本轮出错:{type(e).__name__}: {e}")
            _print_system("")  # 空行分隔

    async def _handle_command(self, line: str) -> bool:
        """处理 / 命令。返回 True 表示请求退出 REPL。"""
        cmd = line.lower()
        if cmd in ("/exit", "/quit"):
            _print_system("再见。")
            return True
        if cmd == "/reset":
            self.reset()
            _print_system("已清空对话历史。")
        elif cmd == "/model" or line.split()[0].lower() == "/model":
            await self._handle_model(line.split()[1:])
        elif cmd == "/help":
            _print_system(
                "命令:/help  帮助   /reset  清空历史   /exit  退出\n"
                "     /model            选择/切换模型(菜单,或 /model <名称> [base_url])"
            )
        else:
            _print_system(f"未知命令:{line}(试试 /help)")
        return False

    async def _handle_model(self, args: list[str]) -> None:
        """`/model` 子命令:菜单选择或手动切换模型。历史与工具不受影响。"""
        if args:  # 手动:/model <名称> [base_url]
            await self._swap_model(args[0], args[1] if len(args) > 1 else None)
            return
        base = getattr(self.llm, "client", None) and self.llm.client.base_url
        _print_system(
            f"当前模型:{getattr(self.llm, 'provider', '?')} / {self.llm.model}"
            f"(base_url={base or '默认'})"
        )
        models = self.config.llm.quick_models
        if not models:
            _print_system(
                "未配置快捷模型列表(config.yaml 的 llm.quick_models),"
                "用法:/model <名称> [base_url]"
            )
            return
        for i, m in enumerate(models, 1):
            tag = f"(端点 {m['base_url']})" if m.get("base_url") else ""
            _print_system(f"  {i}. {m['model']} {tag}")
        choice = (await _read_input("选择编号(0=手动输入,回车取消)> ", self._history)).strip()
        if not choice:
            return
        if choice == "0":
            name = (await _read_input("模型名称 > ", self._history)).strip()
            if name:
                await self._swap_model(name, None)
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            m = models[int(choice) - 1]
            await self._swap_model(m["model"], m.get("base_url"))
        else:
            _print_system(f"无效选择:{choice},已取消。")

    async def _swap_model(self, model: str, base_url: str | None) -> None:
        """重建 LLM 实例并替换;失败保留原模型。"""
        prev = (self.config.llm.model, self.config.llm.base_url)
        self.config.llm.model = model
        if base_url:
            self.config.llm.base_url = base_url
        try:
            new_llm = build_llm(self.config.llm)
        except Exception as e:  # noqa: BLE001 - 切换失败保留原模型
            self.config.llm.model, self.config.llm.base_url = prev
            _print_system(f"⚠ 切换失败,继续用 {self.llm.model}:{type(e).__name__}: {e}")
            return
        await self.llm.close()
        self.llm = new_llm
        _print_system(f"已切换:{new_llm.model}(base_url={base_url or '沿用原端点'})")

    def reset(self) -> None:
        self.state = AgentState()


__all__ = ["Session"]
