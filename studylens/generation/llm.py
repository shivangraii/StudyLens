"""One thin client for any OpenAI-compatible provider (Groq, Gemini, OpenRouter, ...)."""
from __future__ import annotations

import time


class LLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key, self.base_url, self.model = api_key, base_url, model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False) -> str:
        from openai import BadRequestError, RateLimitError

        kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except RateLimitError:
                time.sleep(2**attempt)
            except BadRequestError:
                if "response_format" in kwargs:  # provider without JSON mode -> retry plain
                    kwargs.pop("response_format")
                    continue
                raise
        raise RuntimeError("LLM kept rate-limiting. Wait a minute or switch model/provider.")
