"""Thin LLM client used by the perf agent.

One client talks to OpenAI, Anthropic (via OpenAI-compatible endpoint), or
Gemini (via OpenAI-compatible endpoint) -- the choice is made entirely by
environment variables, so no code change is needed to switch providers:

  LLM_BASE_URL   e.g. https://api.openai.com/v1
                      https://api.anthropic.com/v1/
                      https://generativelanguage.googleapis.com/v1beta/openai/
  LLM_API_KEY    placeholder OK for now -- the request will fail at call time
                 if the key is invalid, which is the right failure mode.
  LLM_MODEL      e.g. gpt-4o-mini, claude-sonnet-4-6, gemini-2.5-pro
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

load_dotenv()

RETRYABLE = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("LLM_API_KEY", "PLACEHOLDER_REPLACE_ME"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )


class LLMError(RuntimeError):
    """Raised when the LLM returns no usable content or invalid JSON."""


def chat_json(
    system: str,
    user: str,
    *,
    cfg: LLMConfig | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> dict:
    """Send a chat request and parse the response as JSON.

    `max_tokens` defaults to 8192 because reasoning models (gemini-2.5-pro,
    o-series) consume part of the budget on hidden thinking tokens before
    emitting visible output.  Lower it explicitly if cost matters.
    """
    cfg = cfg or LLMConfig.from_env()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    # Retry transient errors (Gemini and others routinely return 503/429
    # during traffic spikes).  Exponential backoff with jitter.
    attempts = 6
    resp = None
    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            break
        except RETRYABLE as e:
            if i == attempts - 1:
                raise
            delay = min(60.0, (2 ** i) + random.uniform(0, 1))
            print(f"  [LLM] {type(e).__name__}; retrying in {delay:.1f}s "
                  f"(attempt {i+2}/{attempts})", flush=True)
            time.sleep(delay)
    assert resp is not None  # loop either succeeded or raised
    choice = resp.choices[0]
    content = choice.message.content

    if not content:
        finish = getattr(choice, "finish_reason", "?")
        usage = getattr(resp, "usage", None)
        raise LLMError(
            f"empty content from LLM (finish_reason={finish}, usage={usage}). "
            "If finish_reason='length', the model ran out of tokens before "
            "emitting JSON -- raise max_tokens."
        )
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        finish = getattr(choice, "finish_reason", "?")
        hint = ""
        if finish == "length":
            hint = (" -- the response was TRUNCATED (finish_reason='length'); "
                    "raise max_tokens.")
        raise LLMError(
            f"invalid JSON from LLM: {e}{hint}\nraw content:\n{content}"
        ) from e
