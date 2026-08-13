"""Context 构建

负责组装发给 LLM 的系统提示词(system prompt),把:

* 人设(默认是「通用编程助手」,可在 config 覆盖)
* 工作目录、可用工具清单
* 可用技能(skill)摘要

拼成一段稳定、信息充分的上下文,供每轮调用复用。
"""

from __future__ import annotations

import os
import platform
from datetime import date
from typing import Any

from tools.local.skillsloader import skills_catalog
from tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
你是一名由Dylan设计的通用个人工作助手，叫香辣小龙虾。

工作方式:
1. 先理解用户意图;信息不足时,用一两个问题澄清,而不是瞎猜。
2. 把复杂任务拆成小步,优先调用工具去查证真实信息(读文件、跑命令),再下结论。
3. 修改文件前先读、改完再核对;给出结论时引用 `文件名:行号` 等可验证的依据。
4. 一次回复里可以连续调用多个工具;当任务完成、无需再调用工具时,直接给出最终答复。
5. 命令执行出错时,读 stderr、修正后重试,而不是把错误抛给用户。

守则:
- 实事求是:没查证过的不要编造;不确定就说明不确定。
- 最小改动:只改需要改的,不顺手重构无关代码。
- 简洁:最终答复直击要点,避免冗长复述。
"""


def _tool_catalog(registry: ToolRegistry) -> str:
    if len(registry) == 0:
        return "(未注册任何工具)"
    lines = []
    for t in registry:
        desc = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"- {t.name}:{desc}")
    return "\n".join(lines)


def build_system_prompt(
    config: Any,
    registry: ToolRegistry,
    skills: list[Any] | None = None,
) -> str:
    """组装系统提示词。``config.agent.system_prompt`` 非空则覆盖默认人设。"""
    persona = config.agent.system_prompt or DEFAULT_SYSTEM_PROMPT

    parts: list[str] = [persona.strip()]

    # 环境信息
    env_lines = [
        f"- 操作系统:{platform.system()} {platform.release()}",
        f"- 工作目录:{os.path.abspath(config.tools.workdir)}",
        f"- 当前日期:{date.today().isoformat()}",
    ]
    parts.append("运行环境:\n" + "\n".join(env_lines))

    # 工具清单
    parts.append("可用工具(名称:说明):\n" + _tool_catalog(registry))

    # 技能清单
    if skills is not None:
        parts.append("可用技能(需要细节时调用 get_skill 取回正文):\n" + skills_catalog(skills))

    return "\n\n".join(parts)


__all__ = ["DEFAULT_SYSTEM_PROMPT", "build_system_prompt"]
