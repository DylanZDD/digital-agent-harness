"""vLLM 自建推理服务。

vLLM 同样暴露 OpenAI 兼容端点。``base_url`` 必须在 config 里显式给出
(例如 ``http://localhost:8000/v1``),模型名也需与服务端启动的 ``--served-model-name`` 一致。
"""

from __future__ import annotations

from .base import BaseLLM


class VLLMLLM(BaseLLM):
    provider = "vllm"
    default_base_url = None              # 必须由 config.base_url 提供
    default_model = "default-model"      # 必须由 config.model 覆盖
    placeholder_api_key = "EMPTY"


__all__ = ["VLLMLLM"]
