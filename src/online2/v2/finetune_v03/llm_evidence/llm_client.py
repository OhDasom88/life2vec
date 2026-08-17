"""OpenAI-compatible chat-completion client.

Generalized from `scripts/online2_v2/v03/generate_cf1s_case_analysis_v03.py`'s
`_http_chat` (the only place in the repo that actually calls an LLM) -- same
request shape and retry loop, extracted so `llm_evidence/` doesn't duplicate
it. That script calls a LOCAL server (`--api-base http://127.0.0.1:8080`,
model `gemma-4-31b-it`); no cloud API key is configured in this environment
(only `ANTHROPIC_BASE_URL`, which is the default endpoint, not a credential).
This client is transport-only and works with whatever OpenAI-compatible
endpoint is passed in -- it does not assume or require a specific provider.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional


class LLMCallError(RuntimeError):
    pass


def chat_completion(
    *,
    api_base: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 2200,
    timeout: int = 300,
    retries: int = 3,
    api_key: Optional[str] = None,
) -> str:
    """POST {api_base}/chat/completions, OpenAI-compatible payload. Raises
    LLMCallError with the last underlying error after exhausting retries."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Optional[str] = None
    for _attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                api_base.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            time.sleep(2)
    raise LLMCallError(f"chat_completion failed after {retries} attempts: {last_err}")
