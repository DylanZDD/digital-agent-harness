"""智谱 AI(Zhipu / BigModel)

智谱提供 OpenAI 兼容接口,可直接复用 openai SDK。注意 base_url 分两种:

* **Coding Plan(编程套餐专用)**:``https://open.bigmodel.cn/api/coding/paas/v4``
  —— 必须**带 ``/coding/``**,否则调用失败(Claude-Code-Router 等工具的常见坑)。
* **通用 API**:``https://open.bigmodel.cn/api/paas/v4``

本 provider 默认指向 **Coding Plan**。模型 id 需与你的套餐权限一致
(如 ``glm-4.6`` / ``glm-4.7`` 等,以控制台「模型」页为准),在 config 里覆盖即可。
API key 从 config 或环境变量 ``ZHIPU_API_KEY`` / ``GLM_API_KEY`` 读取。
智谱 OpenAI 兼容端点接受原始 API key 作为 Bearer,无需手动签 JWT。
"""

from __future__ import annotations

from .base import BaseLLM


class ZhipuLLM(BaseLLM):
    provider = "zhipu"
    # Coding Plan 专用路径;若用通用 API 改成 https://open.bigmodel.cn/api/paas/v4
    default_base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
    default_model = "glm-4.6"          # 改成你套餐里的实际模型 id
    placeholder_api_key = "EMPTY"


__all__ = ["ZhipuLLM"]
