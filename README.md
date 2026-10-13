# Agent Harness

一个可扩展的、类 Claude Code 的 agent harness。核心是一个 **推理 → 工具调用 → 回填 → 再推理** 的异步循环,支持:

- 🔌 **多 LLM provider** —— OpenAI / Ollama / vLLM,统一走 OpenAI 兼容接口
- 🛠 **本地工具** —— 文件读写、目录、Shell(沙箱 + 白名单 + 超时)
- 📡 **MCP** —— 接入任意 [Model Context Protocol](https://modelcontextprotocol.io/) 服务器,工具自动注册
- 📚 **Skills** —— `SKILL.md` 知识包,摘要进上下文、按需取回正文
- 🧩 **原生 function-calling** + 文本兜底解析(本地小模型也能用)

## 目录结构

```
harness/
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
│   ├── base.py            # BaseLLM + ToolCall/LLMResponse
│   ├── openai.py          # OpenAI 官方
│   ├── ollama.py          # Ollama 本地
│   └── vllm.py            # vLLM 自建
├── tools/
│   ├── base.py            # Tool 抽象 + ToolResult
│   ├── registry.py        # ToolRegistry(注册/查询/schema/执行)
│   ├── local/             # filesystem / shell / skillsloader
│   └── mcp/               # MCP client + loader
└── skills/
    └── code-review/SKILL.md   # 示例技能
```

**层间依赖**(单向,无环):`app → {agent, llm, tools, config}`、`agent → {llm, tools}`、`llm`/`tools` 各自独立。

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 OPENAI_API_KEY
```

## 使用

```bash
# 交互式 REPL
python app.py

# 单次提问
python app.py -p "列出当前目录并解释每个文件的作用"

# 切换到本地 Ollama
python app.py --provider ollama --model llama3.1

# 本次启动不连任何 MCP 服务器
python app.py --no-mcp

# 详细日志
python app.py -vv
```

REPL 内置命令:`/help` `/reset` `/exit` `/model`。

**运行时切换模型**(无需重启,对话历史保留):

```
/model                                    # 查看当前模型
/model glm-5.3-flash                      # 同端点换模型
/model glm-5.3-flash https://open.bigmodel.cn/api/paas/v4   # 换模型 + 换端点
```

## 配置

优先级 **CLI > 环境变量 > config.yaml > 代码默认值**。关键项见 `config.yaml`:

| 配置 | 说明 |
|------|------|
| `llm.provider` / `model` / `base_url` | LLM 接入点;`openai` 走官方,`ollama`/`vllm` 填 `base_url` |
| `tools.workdir` | 文件系统/Shell 的沙箱根,路径穿越会被拦截 |
| `tools.shell_allowlist` | 空=允许任意命令;非空=只放行白名单前缀 |
| `agent.max_iterations` | 单轮最大工具调用轮数,防死循环 |
| `mcp.servers` | MCP 服务器定义(见下);单个服务器可加 `enabled: false` 关闭 |

### 接入 MCP 服务器

```yaml
mcp:
  servers:
    filesystem:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    sqlite:
      command: uvx
      args: ["mcp-server-sqlite", "--db-path", "/tmp/demo.db"]
```

启动时自动连接、列出工具并注册到 registry(同名工具自动加 `server__` 前缀)。退出时优雅关闭子进程。

### 新增一个本地工具

```python
from tools.base import Tool, ToolResult

class PingTool(Tool):
    name = "ping"
    description = "回显输入,测试工具链是否通"
    parameters = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    async def execute(self, msg, **_):
        return ToolResult(output=f"pong: {msg}")

# 在 app._build_registry 里:registry.register(PingTool())
```

### 新增一个技能

在 `skills/` 下建子目录,放 `SKILL.md`,头加 frontmatter:

```markdown
---
name: my-skill
description: 一句话说明这个技能做什么
---
正文:详细的指令 / 知识……
```

启动时摘要自动进系统提示词,模型用 `get_skill` 工具按需取回正文。

## 架构要点

- **统一 OpenAI 接口**:三个 provider 共用 `openai.AsyncOpenAI` 客户端,子类只设默认 `base_url`/`model`,无重复 HTTP 代码。
- **工具零散落**:所有工具(本地 / MCP / skill)实现同一个 `Tool` 抽象,`registry.schemas()` 直接产出 function-calling schema,`registry.execute()` 统一异常兜底——单个工具出错不会打断 agent 循环。
- **解析双路径**:优先 OpenAI 原生 `tool_calls`;模型不支持时,从正文 `<tool_call>{...}</tool_call>` 或裸 JSON 兜底解析。
- **消息可重放**:`Message` ⇄ OpenAI dict 双向转换,assistant 轮带 `tool_calls`、tool 轮带 `tool_call_id`,严格符合 API 要求。
