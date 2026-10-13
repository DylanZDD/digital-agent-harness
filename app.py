"""程序入口

把各层装配起来并运行。支持两种模式:

* 单次:``python app.py -p "列出当前目录的文件"``
* 交互:``python app.py``(进入 REPL)

装配顺序:加载配置 → 构建 LLM → 注册本地工具 → 加载 skills → 连接 MCP
→ 构建系统提示词 → 创建 Session → 运行 → 优雅关闭。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from agent.planner import build_system_prompt
from agent.session import Session
from config import Config, load_config
from llm import build_llm
from tools.local.filesystem import filesystem_tools
from tools.local.shell import shell_tool
from tools.local.skillsloader import GetSkillTool, load_skills
from tools.mcp.loader import close_clients, load_mcp_servers
from tools.registry import ToolRegistry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="harness",
        description="一个可扩展的 agent harness(OpenAI 兼容 / 工具 / MCP / skills)",
    )
    p.add_argument("-c", "--config", default="config.yaml", help="配置文件路径(默认 config.yaml)")
    p.add_argument("-p", "--prompt", default=None, help="单次执行模式:直接给一条 prompt;不给则进入交互 REPL")
    p.add_argument("--provider", default=None, choices=["openai", "ollama", "vllm", "zhipu"], help="覆盖 LLM provider")
    p.add_argument("--model", default=None, help="覆盖 LLM 模型名")
    p.add_argument("--base-url", default=None, help="覆盖 LLM base_url")
    p.add_argument("--workdir", default=None, help="覆盖工具工作目录(文件系统/shell 的根)")
    p.add_argument("--no-mcp", action="store_true", help="跳过 MCP 服务器连接")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG")
    return p.parse_args(argv)


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """把 CLI 参数覆盖到 config 上。"""
    if args.provider:
        cfg.llm.provider = args.provider
    if args.model:
        cfg.llm.model = args.model
    if args.base_url:
        cfg.llm.base_url = args.base_url
    if args.workdir:
        cfg.tools.workdir = args.workdir


def _build_registry(cfg: Config) -> ToolRegistry:
    """注册内置本地工具。"""
    registry = ToolRegistry()
    if cfg.tools.enable_filesystem:
        for tool in filesystem_tools(cfg.tools.workdir):
            registry.register(tool)
    if cfg.tools.enable_shell:
        registry.register(
            shell_tool(
                workdir=cfg.tools.workdir,
                timeout=cfg.tools.shell_timeout,
                allowlist=cfg.tools.shell_allowlist,
            )
        )
    return registry


async def _run(cfg: Config, args: argparse.Namespace) -> int:
    llm = build_llm(cfg.llm)
    registry = _build_registry(cfg)

    # skills:摘要进系统提示词,并提供 get_skill 工具
    skills = load_skills(cfg.skills.dir)
    if skills:
        registry.register(GetSkillTool(skills))

    # MCP:连接远端服务器,工具挂进 registry;保留 client 以便退出关闭
    mcp_clients: list[Any] = []
    if not args.no_mcp and cfg.mcp:
        mcp_clients = await load_mcp_servers(cfg.mcp, registry)

    system_prompt = build_system_prompt(cfg, registry, skills)
    session = Session(cfg, llm, registry, system_prompt)

    exit_code = 0
    try:
        if args.prompt:
            await session.run(args.prompt)
        else:
            await session.repl()
    except KeyboardInterrupt:
        print()
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("运行失败")
        print(f"运行失败:{type(e).__name__}: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        await close_clients(mcp_clients)
        await llm.close()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    return asyncio.run(_run(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
