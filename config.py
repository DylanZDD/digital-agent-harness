"""配置文件

集中管理整个 harness 的运行配置。加载优先级:

    CLI 参数  >  环境变量  >  config.yaml  >  代码默认值

用法::

    from config import load_config
    cfg = load_config("config.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # 可选依赖:从 .env 读取环境变量
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv 不存在也不影响
    pass


# ──────────────────────────────────────────────────────────────
# 数据类定义
# ──────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    provider: str = "openai"            # openai | ollama | vllm
    model: str = "gpt-4o-mini"
    base_url: str | None = None         # None = 用 provider 默认
    api_key: str | None = None          # None = 读环境变量
    temperature: float = 0.2
    max_tokens: int | None = 2048


@dataclass
class ToolsConfig:
    workdir: str = "."
    enable_filesystem: bool = True
    enable_shell: bool = True
    shell_timeout: float = 120.0
    shell_allowlist: list[str] = field(default_factory=list)


@dataclass
class SkillsConfig:
    dir: str = "skills"


@dataclass
class AgentConfig:
    max_iterations: int = 25
    system_prompt: str | None = None    # None = 用 planner 内置人设


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    mcp: dict[str, dict[str, Any]] = field(default_factory=dict)  # {name: {command, args, env}}


# ──────────────────────────────────────────────────────────────
# 加载逻辑
# ──────────────────────────────────────────────────────────────


def _dict_to_dataclass(cls, data: dict | None):
    """把 dict 填进 dataclass,忽略多余字段。"""
    if not data:
        return cls()
    valid = set(cls.__dataclass_fields__.keys())
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | os.PathLike | None = None) -> Config:
    """加载配置。

    Args:
        path: YAML 配置文件路径;不存在则用全部默认值。

    Returns:
        Config 实例,其中 LLM 的 api_key/base_url/model 会叠加环境变量覆盖。
    """
    raw: dict[str, Any] = {}
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件顶层必须是映射(dict),得到 {type(loaded).__name__}")
        raw = loaded

    cfg = Config()
    cfg.llm = _dict_to_dataclass(LLMConfig, raw.get("llm"))
    cfg.tools = _dict_to_dataclass(ToolsConfig, raw.get("tools"))
    cfg.skills = _dict_to_dataclass(SkillsConfig, raw.get("skills"))
    cfg.agent = _dict_to_dataclass(AgentConfig, raw.get("agent"))
    cfg.mcp = raw.get("mcp", {}).get("servers", {}) or {}

    # 环境变量覆盖(仅 LLM 相关)
    # api_key:支持多家 provider 的常见变量名,取第一个非空
    if not cfg.llm.api_key:
        for _var in ("OPENAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY", "API_KEY"):
            _val = os.getenv(_var)
            if _val:
                cfg.llm.api_key = _val
                break
    if os.getenv("OPENAI_BASE_URL") and not cfg.llm.base_url:
        cfg.llm.base_url = os.environ["OPENAI_BASE_URL"]
    if cfg.llm.model == LLMConfig.model:
        for _mvar in ("OPENAI_MODEL", "ZHIPU_MODEL", "GLM_MODEL", "MODEL"):
            _mval = os.getenv(_mvar)
            if _mval:
                cfg.llm.model = _mval
                break

    return cfg


__all__ = [
    "LLMConfig",
    "ToolsConfig",
    "SkillsConfig",
    "AgentConfig",
    "Config",
    "load_config",
]
