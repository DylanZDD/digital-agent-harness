"""文件系统工具

提供受限的文件读写能力。所有路径都被约束在 ``workdir`` 沙箱内,
阻止 ``../../`` 之类的目录穿越。

包含工具:

* ``read_file``        —— 读取文本文件
* ``write_file``       —— 写入文本文件(可覆盖)
* ``list_directory``   —— 列出目录内容
* ``create_directory`` —— 递归创建目录
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import Tool, ToolResult


def _safe_path(workdir: Path, user_path: str) -> Path:
    """把用户给的相对/绝对路径解析到 workdir 之内,越界则抛错。"""
    workdir = workdir.resolve()
    p = (workdir / user_path).resolve() if not os.path.isabs(user_path) else Path(user_path).resolve()
    # 允许恰好等于 workdir 自身,或位于其下
    if p != workdir and workdir not in p.parents:
        raise PermissionError(f"路径越界,被限制在 {workdir} 之内:{user_path}")
    return p


# ──────────────────────────────────────────────────────────────


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取指定文本文件的内容。路径相对于工作目录。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取的文件路径"},
        },
        "required": ["path"],
    }

    def __init__(self, workdir: str | os.PathLike = ".") -> None:
        self.workdir = Path(workdir).resolve()

    async def execute(self, path: str, **_: Any) -> ToolResult:
        try:
            p = _safe_path(self.workdir, path)
        except PermissionError as e:
            return ToolResult(output=str(e), is_error=True)
        if not p.exists():
            return ToolResult(output=f"文件不存在:{path}", is_error=True)
        if p.is_dir():
            return ToolResult(output=f"路径是目录,不是文件:{path}", is_error=True)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            # 防止把超大文件整段塞给 LLM
            if len(text) > 200_000:
                text = text[:200_000] + f"\n\n…(已截断,文件共 {len(text)} 字符)"
            return ToolResult(output=text)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"读取失败:{e}", is_error=True)


class WriteFileTool(Tool):
    name = "write_file"
    description = "把文本内容写入文件;文件不存在则创建,已存在则覆盖。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目标文件路径"},
            "content": {"type": "string", "description": "要写入的文本内容"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workdir: str | os.PathLike = ".") -> None:
        self.workdir = Path(workdir).resolve()

    async def execute(self, path: str, content: str, **_: Any) -> ToolResult:
        try:
            p = _safe_path(self.workdir, path)
        except PermissionError as e:
            return ToolResult(output=str(e), is_error=True)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(output=f"已写入 {len(content)} 字符到 {path}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"写入失败:{e}", is_error=True)


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "列出目录下的条目(文件/目录),标注类型。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径,默认为工作目录"},
        },
    }

    def __init__(self, workdir: str | os.PathLike = ".") -> None:
        self.workdir = Path(workdir).resolve()

    async def execute(self, path: str = ".", **_: Any) -> ToolResult:
        try:
            p = _safe_path(self.workdir, path)
        except PermissionError as e:
            return ToolResult(output=str(e), is_error=True)
        if not p.exists():
            return ToolResult(output=f"目录不存在:{path}", is_error=True)
        if not p.is_dir():
            return ToolResult(output=f"路径不是目录:{path}", is_error=True)
        try:
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            lines = []
            for e in entries:
                kind = "目录" if e.is_dir() else "文件"
                lines.append(f"[{kind}] {e.name}")
            return ToolResult(output="\n".join(lines) if lines else "(空目录)")
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"列目录失败:{e}", is_error=True)


class CreateDirectoryTool(Tool):
    name = "create_directory"
    description = "递归创建目录(类似 mkdir -p)。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要创建的目录路径"},
        },
        "required": ["path"],
    }

    def __init__(self, workdir: str | os.PathLike = ".") -> None:
        self.workdir = Path(workdir).resolve()

    async def execute(self, path: str, **_: Any) -> ToolResult:
        try:
            p = _safe_path(self.workdir, path)
        except PermissionError as e:
            return ToolResult(output=str(e), is_error=True)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return ToolResult(output=f"已创建目录 {path}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"创建目录失败:{e}", is_error=True)


def filesystem_tools(workdir: str | os.PathLike = ".") -> list[Tool]:
    """一次性产出全部文件系统工具实例。"""
    return [
        ReadFileTool(workdir),
        WriteFileTool(workdir),
        ListDirectoryTool(workdir),
        CreateDirectoryTool(workdir),
    ]


__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "CreateDirectoryTool",
    "filesystem_tools",
]
