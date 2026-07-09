"""Skills 加载器

技能(Skill)是 ``skills/`` 目录下、每个子目录含一个 ``SKILL.md`` 的知识包,
参考 Claude Code 的 skill 机制:

* ``SKILL.md`` 头部用 YAML frontmatter 声明 ``name`` 与 ``description``
* 正文是该技能的详细指令 / 知识

加载策略(按需取用,省 token):

1. 启动时扫描全部技能,只把 ``name: description`` 摘要注入系统提示词
2. 注册一个 ``get_skill`` 工具,模型在需要时取回某个技能的完整正文
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..base import Tool, ToolResult

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str          # 去掉 frontmatter 后的正文
    path: Path

    def summary(self) -> str:
        return f"- {self.name}:{self.description}"


def _parse_skill_md(text: str) -> tuple[dict, str]:
    """拆出 YAML frontmatter 与正文。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.strip()
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2).strip()


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """扫描 ``skills_dir/*/SKILL.md``,返回 Skill 列表。目录不存在则返回空。"""
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        meta, body = _parse_skill_md(text)
        name = meta.get("name") or skill_md.parent.name
        description = meta.get("description") or "(无描述)"
        skills.append(
            Skill(
                name=str(name),
                description=str(description),
                body=body or text,
                path=skill_md,
            )
        )
    return skills


# ──────────────────────────────────────────────────────────────


class GetSkillTool(Tool):
    """按名取回某个技能的完整正文。"""

    name = "get_skill"
    description = "获取指定技能(skill)的完整指令内容。先用它了解技能细节,再据此行动。"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "技能名称"},
        },
        "required": ["name"],
    }

    def __init__(self, skills: list[Skill]) -> None:
        self._by_name = {s.name: s for s in skills}

    async def execute(self, name: str, **_: Any) -> ToolResult:
        skill = self._by_name.get(name)
        if skill is None:
            available = ", ".join(self._by_name.keys()) or "(无)"
            return ToolResult(
                output=f"未知技能 '{name}'。可用技能:{available}",
                is_error=True,
            )
        return ToolResult(output=f"# 技能:{skill.name}\n\n{skill.body}")


def skills_catalog(skills: list[Skill]) -> str:
    """生成给系统提示词用的技能摘要清单。"""
    if not skills:
        return "(暂无可用技能)"
    return "\n".join(s.summary() for s in skills)


__all__ = ["Skill", "load_skills", "GetSkillTool", "skills_catalog"]
