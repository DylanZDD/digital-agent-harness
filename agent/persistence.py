"""会话持久化

把 ``AgentState`` 的消息历史落盘为 JSON,支撑三个能力:

* 自动保存 —— Session 每轮结束调用 ``save_session``,退出不丢
* 恢复    —— ``/resume`` 或 ``--resume`` 用 ``load_session`` 还原历史
* 导出    —— ``/export`` / ``--export`` 用 ``export_markdown`` 生成可分享的 Markdown

存储布局:``<session_dir>/<session_id>.json``,默认目录
``~/.digital-agent-harness/sessions``(放 home 而不是 workdir,
避免被文件系统/shell 工具误伤,也与工具沙箱边界解耦)。

恢复历史前先过 ``sanitize_messages``:Ctrl+C 中断的轮次可能留下
「assistant 带 tool_calls 但结果没回填」或孤儿 tool 消息,直接发给
严格校验的端点会 400,这里统一清洗成合法序列。
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import AgentState, Message

logger = logging.getLogger(__name__)

SESSIONS_VERSION = 1  # 文件格式版本,未来字段变更时用于迁移判断


@dataclass
class StoredSession:
    """load_session 的返回:元数据 + 还原好的消息列表。"""

    id: str
    created_at: str
    updated_at: str
    model: str
    provider: str
    messages: list[Message]


@dataclass
class SessionSummary:
    """list_sessions 的返回:列表页展示用的摘要。"""

    id: str
    updated_at: str
    n_messages: int
    preview: str        # 第一条用户消息(截断)
    path: Path


# ── 基础 ───────────────────────────────────────────────────────

def default_session_dir() -> Path:
    return Path.home() / ".digital-agent-harness" / "sessions"


def new_session_id() -> str:
    """时间戳 + 4 位随机后缀,可读且基本不碰撞。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def _session_path(session_dir: Path, session_id: str) -> Path:
    return session_dir / f"{session_id}.json"


def resolve_session_id(prefix: str, session_dir: Path) -> str | None:
    """把用户给的 id(或前缀)解析成完整 id;唯一前缀即可命中。"""
    if not prefix:
        return None
    if _session_path(session_dir, prefix).exists():
        return prefix
    matches = sorted(session_dir.glob(f"{prefix}*.json"))
    if len(matches) == 1:
        return matches[0].stem
    return None  # 无命中或有歧义


# ── 保存 / 加载 ────────────────────────────────────────────────

def save_session(
    state: AgentState,
    session_id: str,
    session_dir: Path,
    model: str = "",
    provider: str = "",
) -> Path:
    """保存会话到 ``<session_dir>/<session_id>.json``,返回文件路径。

    文件已存在时沿用其 created_at,只刷新 updated_at。
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_dir, session_id)

    created_at = datetime.now().isoformat(timespec="seconds")
    if path.exists():
        try:
            created_at = json.loads(path.read_text(encoding="utf-8")).get("created_at", created_at)
        except (OSError, json.JSONDecodeError):
            pass

    doc = {
        "version": SESSIONS_VERSION,
        "id": session_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "provider": provider,
        "messages": [m.to_dict() for m in state.messages],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 原子替换,写一半不会被读到
    return path


def load_session(session_id: str, session_dir: Path) -> StoredSession:
    """加载会话;文件不存在或损坏抛 ``FileNotFoundError`` / ``ValueError``。"""
    path = _session_path(session_dir, session_id)
    if not path.exists():
        raise FileNotFoundError(f"会话不存在:{session_id}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"会话文件损坏(非合法 JSON):{path}") from e
    if not isinstance(doc, dict) or "messages" not in doc:
        raise ValueError(f"会话文件格式不对:{path}")

    raw_msgs = doc.get("messages") or []
    messages = [
        Message.from_dict(d) for d in raw_msgs if isinstance(d, dict) and d.get("role")
    ]
    return StoredSession(
        id=doc.get("id", session_id),
        created_at=doc.get("created_at", ""),
        updated_at=doc.get("updated_at", ""),
        model=doc.get("model", ""),
        provider=doc.get("provider", ""),
        messages=sanitize_messages(messages),
    )


def list_sessions(session_dir: Path, limit: int = 10) -> list[SessionSummary]:
    """按最近更新排序列出会话摘要;单个文件损坏则跳过。"""
    if not session_dir.exists():
        return []
    summaries: list[SessionSummary] = []
    for path in sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            msgs = [m for m in doc.get("messages") or [] if isinstance(m, dict)]
            preview = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
        except (OSError, json.JSONDecodeError):
            logger.warning("跳过无法解析的会话文件:%s", path)
            continue
        summaries.append(
            SessionSummary(
                id=doc.get("id", path.stem),
                updated_at=doc.get("updated_at", ""),
                n_messages=len(msgs),
                preview=preview[:40] + ("…" if len(preview) > 40 else ""),
                path=path,
            )
        )
        if len(summaries) >= limit:
            break
    return summaries


# ── 历史清洗 ───────────────────────────────────────────────────

def sanitize_messages(msgs: list[Message]) -> list[Message]:
    """把可能被中断的历史清洗成 OpenAI 合法序列:

    * assistant 的 tool_calls 若只有部分结果回填(或全没有)→
      剥掉整组未完成的调用,只留正文(没正文整条丢弃)
    * 对不上任何 assistant 调用的孤儿 tool 消息 → 丢弃
    """
    out: list[Message] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == "assistant" and m.tool_calls:
            # 收集紧随其后的、能对上 id 的 tool 结果
            call_ids = {tc.id for tc in m.tool_calls}
            answered: dict[str, Message] = {}
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                if msgs[j].tool_call_id in call_ids:
                    answered[msgs[j].tool_call_id] = msgs[j]
                j += 1
            if len(answered) == len(m.tool_calls):
                out.append(m)
                out.extend(answered[tc.id] for tc in m.tool_calls)  # 保持调用顺序
            elif m.content:  # 组不完整:只留正文
                m.tool_calls = None
                out.append(m)
            i = j
            continue
        if m.role == "tool":
            i += 1  # 孤儿 tool 消息(前面没有 assistant 调用组)直接跳过
            continue
        out.append(m)
        i += 1
    return out


# ── 导出 Markdown ──────────────────────────────────────────────

_ROLE_TITLES = {"user": "👤 用户", "assistant": "🤖 助手"}
_RESULT_MAX = 2000  # 单个工具结果在导出里的最大字符数


def _fmt_tool_args(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False)


def export_markdown(
    state: AgentState,
    path: Path,
    session_id: str = "",
    model: str = "",
    provider: str = "",
) -> Path:
    """把当前对话导出为人类可读的 Markdown,返回文件路径。

    结构:头部元信息 + 按序的消息体。工具调用以引用行呈现,
    结果放进折叠的代码块(超长截断),方便直接发给其他人看。
    """
    lines: list[str] = [
        "# Agent Harness 会话导出",
        "",
        f"- 会话 ID:`{session_id or '(未保存)'}`",
        f"- 导出时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型:{provider or '?'}/{model or '?'}",
        f"- 消息数:{len(state.messages)}",
        "",
        "---",
        "",
    ]

    for m in state.messages:
        if m.role in _ROLE_TITLES:
            lines.append(f"## {_ROLE_TITLES[m.role]}")
            lines.append("")
            lines.append(m.content or "*(无文本)*")
            lines.append("")
            for tc in m.tool_calls or []:
                lines.append(f"> ⚙️ 调用工具 `{tc.name}`,参数:`{_fmt_tool_args(tc.arguments)}`")
                lines.append("")
        elif m.role == "tool":
            result = m.content or "(无输出)"
            if len(result) > _RESULT_MAX:
                result = result[:_RESULT_MAX] + f"\n…(截断,共 {len(m.content or '')} 字符)"
            lines.append(f"<details><summary>📋 <code>{m.name or 'tool'}</code> 返回结果</summary>")
            lines.append("")
            lines.append("```")
            lines.append(result)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


__all__ = [
    "StoredSession",
    "SessionSummary",
    "default_session_dir",
    "new_session_id",
    "resolve_session_id",
    "save_session",
    "load_session",
    "list_sessions",
    "sanitize_messages",
    "export_markdown",
]
