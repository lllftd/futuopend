from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


DEFAULT_API_KEY = "sk-b4f330cffab146aea98adcc7db938879"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


@dataclass
class QwenResponse:
    content: str
    raw: dict[str, Any]


class QwenClient:
    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> QwenResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if enable_thinking:
            payload["enable_thinking"] = True
        if extra_body:
            payload.update(extra_body)

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Qwen API connection failed: {exc}") from exc

        choices = raw.get("choices", [])
        if not choices:
            raise RuntimeError(f"Unexpected Qwen response: {raw}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        return QwenResponse(content=content, raw=raw)


def example_quant_analysis() -> str:
    client = QwenClient()
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a quantitative research assistant focused on intraday strategy analysis.",
            },
            {
                "role": "user",
                "content": (
                    "请从日内期权交易视角，分析为什么一个1分钟趋势策略的Sharpe不高，"
                    "并给出3个具体优化方向。"
                ),
            },
        ],
        temperature=0.2,
    )
    return response.content


if __name__ == "__main__":
    print(example_quant_analysis())
