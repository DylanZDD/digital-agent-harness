"""Tool call parser

从 ``LLMResponse`` 中解析出要执行的工具调用。

主路径:OpenAI 原生 function-calling,``response.tool_calls`` 已是结构化的。
兜底路径:部分模型(尤其本地小模型)不支持 function-calling,会把调用
写在正文里,常见两种写法::

    <tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>

或裸 JSON 代码块。``_parse_text_tool_calls`` 负责把这些抠出来,统一成 ToolCall。
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm.base import LLMResponse, ToolCall

# <tool_call>{...}</tool_call>  或  <tool_call name="x">{...}</tool_call>
_TAG_RE = re.compile(r"<tool_call(?:\s+name=[\"'](?P<name>[^\"']+)[\"'])?>(?P<body>.*?)</tool_call>", re.DOTALL)
# 兜底:正文里疑似工具调用的 JSON 对象(需同时含 "name" 与 "arguments"/"args")
_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _safe_load(s: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """从纯文本里尽力解析工具调用(兜底用)。"""
    calls: list[ToolCall] = []

    # 1) <tool_call> 标签
    for m in _TAG_RE.finditer(text):
        body = m.group("body").strip()
        tag_name = m.group("name")
        obj = _safe_load(body)
        if obj is None:
            continue
        name = obj.get("name") or tag_name
        if not name:
            continue
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj.get("args", {})
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=f"text-{len(calls)}", name=str(name), arguments=args))

    if calls:
        return calls

    # 2) 含 "name" 的裸 JSON 对象
    #    要求 arguments/args 键同时在场才认定是调用意图:只看 "name" 会把
    #    正文里举例的 {"name": "张三", ...} 误判成工具调用。
    for m in _JSON_OBJ_RE.finditer(text):
        obj = _safe_load(m.group(0))
        if not obj or "name" not in obj:
            continue
        if "arguments" not in obj and "args" not in obj:
            continue
        name = obj["name"]
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj.get("args", {})
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=f"text-{len(calls)}", name=str(name), arguments=args))

    return calls


def extract_tool_calls(response: LLMResponse) -> list[ToolCall]:
    """从一次 LLM 响应中取出工具调用;无则返回空列表。"""
    if response.tool_calls:
        return list(response.tool_calls)
    return _parse_text_tool_calls(response.content or "")


__all__ = ["extract_tool_calls", "_parse_text_tool_calls"]
