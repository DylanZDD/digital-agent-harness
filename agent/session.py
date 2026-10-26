"""session 管理类

``Session`` 把 LLM、工具注册表、系统提示词、状态串起来,对外提供:

* ``run(user_input)`` —— 单轮:塞入用户消息 → 跑 agent_turn → 返回最终文本
* ``repl()``          —— 交互式 REPL,支持 ``/help`` ``/reset`` ``/model``
                        ``/resume`` ``/export`` ``/exit``
* ``reset()``         —— 清空历史,开新对话

会话持久化(见 persistence.py):每轮结束自动落盘到
``agent.session_dir``(默认 ``~/.digital-agent-harness/sessions``),
``/resume`` 恢复历史会话,``/export`` 导出 Markdown 分享。

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
from pathlib import Path
from typing import Any

from llm import build_llm
from llm.base import BaseLLM
from tools.registry import ToolRegistry

from .loop import agent_turn
from .persistence import (
    default_session_dir,
    export_markdown,
    list_sessions,
    load_session,
    new_session_id,
    resolve_session_id,
    save_session,
)
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

    def _think_spinner() -> Any:
        """等待模型输出(首个 token 到达前)的 spinner。"""
        return _console.status("[dim]思考中…[/]")

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

    def _think_spinner() -> Any:
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

    class InMemoryHistory:
        """降级版历史:只保留条目列表,不支持 ↑/↓ 搜索。"""

        def __init__(self) -> None:
            self.strings: list[str] = []

        def append_string(self, s: str) -> None:
            self.strings.append(s)

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
        # 会话持久化:目录来自配置(agent.session_dir),id 在首轮落盘时生成
        self.session_id: str | None = None
        self._session_dir = Path(
            getattr(config.agent, "session_dir", None) or default_session_dir()
        ).expanduser()
        # 展示层运行时状态
        self._stream: Any = None            # 进行中的流式渲染区(_StreamRenderer)
        self._spinner: Any = None           # 进行中的工具 spinner
        self._pending_call: tuple[str, dict] | None = None  # spinner 对应的工具调用

    # ── 事件回调 ───────────────────────────────────────────────
    def _on_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "delta":
            self._stop_spinner()
            if self._stream is None:
                _print_system("")  # 与上方输入 / 工具结果空开一行
                self._stream = _StreamRenderer()
            self._stream.add(data["content"])
        elif kind == "assistant":
            # 非流式兜底:整段进度文本。流式模式下文字已通过 delta 显示过,跳过。
            self._stop_spinner()
            if self._stream is None and data.get("content"):
                _print_assistant(data["content"])
        elif kind == "tool_call":
            self._stop_spinner()
            self._close_stream()
            self._pending_call = (data["name"], data["arguments"])
            self._spinner = _tool_spinner(*self._pending_call)
            if self._spinner is not None:
                self._spinner.start()
        elif kind == "tool_result":
            self._stop_spinner()
            if self._pending_call is not None:
                _print_tool_call(*self._pending_call)
                self._pending_call = None
            output = data.get("output", "")
            # registry 兜底返回的错误文本以「错误:」开头,据此判断样式
            is_error = str(output).lstrip().startswith(("错误:", "执行工具"))
            _print_tool_result(data["name"], output, is_error=is_error)
            # 结果已回填,下一轮又在等模型 → 继续「思考中」
            self._spinner = _think_spinner()
            if self._spinner is not None:
                self._spinner.start()
        elif kind == "limit":
            self._stop_spinner()
            self._close_stream()
            _print_system(f"⚠ 达到最大迭代数 {data['max_iterations']},已停止。")

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    # ── 单轮 ───────────────────────────────────────────────────
    async def run(self, user_input: str, show_final: bool = True) -> str:
        """运行一轮。返回最终文本。"""
        self.state.add_user(user_input)
        self._stream = None
        # 等待首个 token 期间转「思考中」,首个 delta 到达时由事件回调停掉
        self._spinner = _think_spinner()
        if self._spinner is not None:
            self._spinner.start()
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
            self._stop_spinner()
            self._autosave()  # 中断的轮次也落盘,恢复时由 sanitize 清洗
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
        head, _, rest = line.partition(" ")
        cmd = head.lower()
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            _print_system("再见。")
            return True
        if cmd == "/reset":
            self.reset()
            _print_system("已清空对话历史,新对话将存为新会话。")
        elif cmd == "/model":
            await self._handle_model(rest.split())
        elif cmd == "/resume":
            await self._handle_resume(rest.split())
        elif cmd == "/export":
            await self._handle_export(rest)
        elif cmd == "/help":
            _print_system(
                "命令:/help  帮助   /reset  清空历史   /exit  退出\n"
                "     /model            选择/切换模型(菜单,或 /model <名称> [base_url])\n"
                "     /resume [id]      恢复历史会话(菜单,或 /resume <id 前缀>)\n"
                "     /export [路径]    当前对话导出为 Markdown(默认 ./session-<id>.md)"
            )
        else:
            _print_system(f"未知命令:{line}(试试 /help)")
        return False

    # ── 会话持久化 ─────────────────────────────────────────────
    def _autosave(self) -> None:
        """每轮结束落盘;磁盘异常只记日志,不影响对话。"""
        if not self.state.messages:
            return
        if self.session_id is None:
            self.session_id = new_session_id()
        try:
            save_session(
                self.state,
                self.session_id,
                self._session_dir,
                model=self.llm.model,
                provider=getattr(self.llm, "provider", ""),
            )
        except OSError as e:
            logger.warning("会话自动保存失败:%s", e)

    def _load_session(self, session_id: str) -> None:
        """按完整 id 恢复历史会话(恢复前已由 persistence 清洗中断轮)。"""
        try:
            stored = load_session(session_id, self._session_dir)
        except (FileNotFoundError, ValueError) as e:
            _print_system(f"⚠ 恢复失败:{e}")
            return
        self.state = AgentState(messages=stored.messages)
        self.session_id = stored.id  # 续写的轮次落回同一文件
        updated = stored.updated_at.replace("T", " ")
        _print_system(
            f"已恢复会话 {stored.id}"
            f"({sum(1 for m in stored.messages if m.role == 'user')} 轮对话,更新于 {updated}),"
            "继续输入即可接着聊。"
        )

    def resume(self, session_id: str | None = None) -> bool:
        """恢复历史会话,返回是否成功。``None`` = 最近一个。

        供 CLI ``--resume`` 与 REPL ``/resume <id>`` 共用;失败时提示并
        保持当前会话不变。
        """
        if session_id is None:
            latest = list_sessions(self._session_dir, limit=1)
            if not latest:
                _print_system("没有可恢复的会话。")
                return False
            session_id = latest[0].id
        else:
            resolved = resolve_session_id(session_id, self._session_dir)
            if resolved is None:
                _print_system(f"⚠ 找不到会话(或前缀有歧义):{session_id}")
                return False
            session_id = resolved
        self._load_session(session_id)
        return True

    async def _handle_resume(self, args: list[str]) -> None:
        """`/resume` 子命令:菜单选择,或按 id / 唯一前缀直接恢复。"""
        if args:
            self.resume(args[0])
            return

        sessions = list_sessions(self._session_dir)
        if not sessions:
            _print_system(f"没有可恢复的会话(目录为空:{self._session_dir})。")
            return
        _print_system(f"最近的会话(存于 {self._session_dir}):")
        for i, s in enumerate(sessions, 1):
            _print_system(
                f"  {i}. [{s.id}]  {s.updated_at.replace('T', ' ')}  "
                f"{s.n_messages} 条消息  {s.preview or '(空)'}"
            )
        choice = (await _read_input("选择编号(回车取消)> ", self._history)).strip()
        if not choice:
            return
        if not (choice.isdigit() and 1 <= int(choice) <= len(sessions)):
            _print_system(f"无效选择:{choice},已取消。")
            return
        self._load_session(sessions[int(choice) - 1].id)

    async def _handle_export(self, rest: str) -> None:
        """`/export` 子命令:当前对话导出为 Markdown,方便发给其他人。"""
        if not self.state.messages:
            _print_system("当前没有可导出的对话。")
            return
        default_name = f"session-{self.session_id or new_session_id()}.md"
        path = Path(rest.strip('"') or default_name).expanduser()
        try:
            out = export_markdown(
                self.state,
                path,
                session_id=self.session_id or "",
                model=self.llm.model,
                provider=getattr(self.llm, "provider", ""),
            )
        except OSError as e:
            _print_system(f"⚠ 导出失败:{e}")
            return
        _print_system(f"已导出:{out.resolve()}")

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
        self.session_id = None  # 换新 id:旧会话文件保留,可随时 /resume 回来


__all__ = ["Session"]
