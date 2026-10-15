<div align="center">

# Agent Harness

**用一千余行 Python 打造的、类 Claude Code 的命令行 AI Agent**

LLM 推理 → 工具调用 → 结果回填 → 再推理的异步循环,
开箱支持 OpenAI / Ollama / vLLM / 智谱 GLM,
内置文件、Shell 等本地工具与 MCP 协议扩展。

[快速开始](#-快速开始) · [架构](#-架构) · [扩展](#-扩展) · [配置](#%EF%B8%8F-配置)

</div>

---

## ✨ 特性

- 🧠 **双路径工具调用解析** —— 优先 OpenAI 原生 `tool_calls`;模型不支持时自动从正文 `<tool_call>{...}</tool_call>` 或裸 JSON 兜底解析,本地小模型也能用
- 🔌 **多 LLM Provider** —— OpenAI / Ollama / vLLM / 智谱 GLM,统一走 OpenAI 兼容接口,子类只设默认 `base_url`/`model`,零重复 HTTP 代码
- 🔄 **运行时切换模型** —— `/model` 命令无需重启即可换模型/端点,对话历史保留,切换失败自动回退
- 🛠 **内置安全工具** —— 文件读写、目录、Shell:统一沙箱根、路径穿越拦截、命令白名单、执行超时
- 📡 **MCP 协议** —— 接入任意 [Model Context Protocol](https://modelcontextprotocol.io/) 服务器,工具自动发现注册,单个服务器可用 `enabled: false` 关闭
- 📚 **Skills 知识包** —— `SKILL.md` 知识包,摘要进系统提示词、按需取回正文,控制上下文成本
- ⛔ **稳定性设计** —— 单个工具出错不打断 agent 循环;`max_iterations` 防死循环;UI 事件回调异常不影响主流程

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 一个 LLM API Key(或本地 [Ollama](https://ollama.com/))

### 安装

```bash
git clone https://github.com/DylanZDD/digital-agent-harness.git
cd digital-agent-harness

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # 填入你的 API key
```

### 运行

```bash
python app.py                                     # 交互式 REPL
python app.py -p "解释这个项目的架构"              # 单次提问
python app.py --provider ollama --model llama3.1  # 切换本地 Ollama
python app.py --no-mcp                            # 本次启动不连 MCP 服务器
python app.py -vv                                 # DEBUG 日志
```

**REPL 内置命令**:

| 命令 | 说明 |
|------|------|
| `/help` | 帮助 |
| `/reset` | 清空对话历史 |
| `/model` | 切换模型(菜单,或 `/model <名称> [base_url]`) |
| `/exit` | 退出 |

### 运行时切换模型

无需重启、历史保留、失败自动回退:

```
/model                                                              # 查看当前模型 + 快捷菜单
/model glm-4.6                                                      # 同端点换模型
/model glm-4.6v-flash https://open.bigmodel.cn/api/paas/v4          # 换模型 + 换端点
```

快捷菜单在 `config.yaml` 的 `llm.quick_models` 中配置,支持 `"名称"` 与 `{model, base_url}` 混排。

## 🧩 架构

```
                    ┌──────────────┐
                    │    app.py    │   CLI 装配 + 单次/交互模式
                    └──────┬───────┘
             ┌─────────────┼─────────────┐
      ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
      │   agent/    │ │  llm/   │ │   tools/    │
      │  loop       │ │ openai  │ │ local/      │
      │  parser     │ │ zhipu   │ │  filesystem │
      │  executor   │ │ ollama  │ │  shell      │
      │  state      │ │ vllm    │ │ mcp/        │
      │  planner    │ │         │ │  client     │
      │  session    │ │         │ │ skillsloader│
      └─────────────┘ └─────────┘ └─────────────┘
```

**层间依赖单向无环**:`app → {agent, llm, tools, config}`、`agent → {llm, tools}`,`llm` 与 `tools` 相互独立。

### 核心循环

```
构建 messages → LLM 调用 → 解析工具调用
    ├─ 有工具调用: 执行 → 回填结果 → 回到「构建 messages」
    └─ 无工具调用: 返回最终文本,本轮结束
```

`max_iterations` 兜底,防止模型陷入无限工具调用。

### 目录结构

```
digital-agent-harness/
├── app.py                 # 入口:CLI 装配 + 单次/交互模式
├── config.py              # 配置加载(YAML + 环境变量 + CLI 覆盖)
├── config.yaml            # 配置示例
├── agent/
│   ├── loop.py            # 核心 agent 循环(LLM → 解析 → 执行 → 回填)
│   ├── planner.py         # 系统提示词 / 上下文构建
│   ├── session.py         # 会话编排 + 交互式 REPL
│   ├── parser.py          # 工具调用解析(原生 + 文本兜底)
│   ├── executor.py        # 工具执行(统一异常兜底)
│   └── state.py           # Message / AgentState / OpenAI 格式转换
├── llm/
│   ├── base.py            # BaseLLM 抽象 + ToolCall/LLMResponse
│   ├── __init__.py        # build_llm 工厂
│   ├── openai.py          # OpenAI 官方
│   ├── zhipu.py           # 智谱 GLM(Coding Plan / 通用端点)
│   ├── ollama.py          # Ollama 本地
│   └── vllm.py            # vLLM 自建
├── tools/
│   ├── base.py            # Tool 抽象 + ToolResult
│   ├── registry.py        # ToolRegistry(注册/查询/schema/执行)
│   ├── local/             # filesystem / shell / skillsloader
│   └── mcp/               # MCP client + loader
├── skills/                # SKILL.md 知识包(code-review / weather)
└── examples/
    └── mcp_demo_server.py # MCP 服务器示例
```

## ⚙️ 配置

优先级:**CLI 参数 > `config.yaml` > 环境变量 > 代码默认值**(环境变量仅作缺省:LLM 的 api_key / base_url / model 在 yaml 未设置时才读)。完整示例见 [`config.yaml`](./config.yaml)。

### LLM

| 配置 | 说明 |
|------|------|
| `llm.provider` | `openai` \| `zhipu` \| `ollama` \| `vllm` |
| `llm.model` / `llm.base_url` | 模型与端点;留空用各 provider 默认值 |
| `llm.api_key` | 留空则读环境变量(`ZHIPU_API_KEY` / `OPENAI_API_KEY` 等) |
| `llm.quick_models` | `/model` 快捷菜单;条目为 `"名称"` 或 `{model, base_url}` |
| `llm.temperature` / `llm.max_tokens` | 采样参数 |

### 工具与安全

| 配置 | 说明 |
|------|------|
| `tools.workdir` | 文件系统/Shell 的沙箱根,路径穿越会被拦截 |
| `tools.enable_filesystem` / `tools.enable_shell` | 关闭对应内置工具组 |
| `tools.shell_allowlist` | 空=允许任意命令;非空=只放行白名单前缀(如 `["git", "ls", "python"]`) |
| `tools.shell_timeout` | 单次 shell 执行超时(秒) |
| `agent.max_iterations` | 单轮最大工具调用轮数,防死循环 |

### MCP 服务器

```yaml
mcp:
  servers:
    sqlite:                    # stdio 方式:拉起本地子进程
      command: uvx
      args: ["mcp-server-sqlite", "--db-path", "/tmp/demo.db"]
    time:                      # HTTP 方式:连远端服务器
      enabled: false           # false 则启动时不连接
      type: streamable_http
      url: https://example.com/mcp
```

启动时自动连接、列出工具并注册(同名工具自动加 `服务器名__` 前缀避免冲突),退出时优雅关闭。

## 🔧 扩展

### 新增本地工具

```python
from tools.base import Tool, ToolResult

class PingTool(Tool):
    name = "ping"
    description = "回显输入,测试工具链是否通"
    parameters = {
        "type": "object",
        "properties": {"msg": {"type": "string", "description": "要回显的内容"}},
        "required": ["msg"],
    }

    async def execute(self, msg: str, **_) -> ToolResult:
        return ToolResult(output=f"pong: {msg}")

# 在 app._build_registry 中:
registry.register(PingTool())
```

### 新增 Skill

在 `skills/` 下建子目录,放入带 frontmatter 的 `SKILL.md`:

```markdown
---
name: my-skill
description: 一句话说明这个技能做什么
---
正文:详细的指令 / 知识……
```

启动时摘要自动进系统提示词,模型通过 `get_skill` 工具按需取回正文。

## 📄 License

[MIT](./LICENSE)
