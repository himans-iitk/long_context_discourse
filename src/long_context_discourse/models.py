"""Thin OpenRouter chat-completions client used by every chat-based experiment.

We keep the wrapper minimal:

* deterministic decoding (``temperature=0`` by default);
* exponential backoff on transient failures (network/timeout/rate-limit/5xx);
* explicit dataclass response so callers can inspect both the generated text
  and metadata (model id, latency).

The class is intentionally **not** a singleton — instantiate one per
experiment so timeouts/headers can be tuned independently.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import openai
from dotenv import load_dotenv

from .config import OpenRouterConfig
from .logging_utils import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatResponse:
    """Result of a single chat-completion call."""

    text: str | None
    model: str
    latency_seconds: float
    raw: Any | None = None

    @property
    def succeeded(self) -> bool:
        return self.text is not None


_TRANSIENT = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# OpenRouter free-tier / upstream providers often return 429 with ~30s Retry-After; a few
# exponential-backoff tries at 1.5→30s are not enough. Keep separate budget for 429s.
_MAX_RATE_LIMIT_RETRIES = 120


def _retry_after_seconds_from_error(exc: BaseException) -> float | None:
    """Parse OpenRouter ``retry_after_seconds`` / ``Retry-After`` from the SDK exception text."""
    text = str(exc)
    for pattern in (
        r"['\"]retry_after_seconds['\"]:\s*(\d+(?:\.\d+)?)",
        r"retry_after_seconds['\"]:\s*(\d+(?:\.\d+)?)",
        r"['\"]Retry-After['\"]:\s*['\"](\d+)['\"]",
    ):
        m = re.search(pattern, text)
        if m:
            return float(m.group(1))
    return None


class OpenRouterClient:
    """OpenRouter wrapper with deterministic defaults and bounded retries."""

    def __init__(self, config: OpenRouterConfig, *, env_path: str | os.PathLike[str] | None = None) -> None:
        load_dotenv(env_path) if env_path else load_dotenv()
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env!r} is not set. "
                "Copy .env.example → .env and fill in your OpenRouter key."
            )
        self._config = config
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.request_timeout_seconds,
        )
        self._headers = {
            "HTTP-Referer": os.environ.get(config.referer_env, ""),
            "X-Title": os.environ.get(config.title_env, ""),
        }

    @property
    def config(self) -> OpenRouterConfig:
        return self._config

    def chat(
        self,
        model_id: str,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send one chat-completion request, returning a :class:`ChatResponse`.

        On transient failures we retry with exponential backoff. OpenRouter 429s are
        retried many more times, sleeping for ``retry_after_seconds`` from the error
        when present (free-tier / upstream often requires ~30s). HTTP 402 from a
        provider is retried a few times, then the call returns empty text. A persistent
        failure returns ``text=None`` so a long sweep is not aborted.
        """
        payload = [m.as_openai() if isinstance(m, ChatMessage) else dict(m) for m in messages]

        attempt = 0
        backoff = self._config.retry_base_seconds
        rate_attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers=self._headers,
                    **(extra or {}),
                )
            except openai.RateLimitError as exc:
                rate_attempt += 1
                if rate_attempt > _MAX_RATE_LIMIT_RETRIES:
                    _log.warning(
                        "model=%s gave up after %d rate-limit retries: %s",
                        model_id,
                        rate_attempt - 1,
                        exc,
                    )
                    return ChatResponse(text=None, model=model_id, latency_seconds=time.perf_counter() - t0)
                hint = _retry_after_seconds_from_error(exc)
                wait = float(hint) + 2.0 if hint is not None else min(120.0, max(25.0, backoff))
                wait = min(wait, 300.0)
                _log.info(
                    "model=%s rate limited (attempt %d/%d); sleeping %.1fs: %s",
                    model_id,
                    rate_attempt,
                    _MAX_RATE_LIMIT_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
                backoff = min(backoff * 1.5, 120.0)
                continue
            except _TRANSIENT as exc:
                if attempt > self._config.max_retries:
                    _log.warning("model=%s gave up after %d retries: %s", model_id, attempt - 1, exc)
                    return ChatResponse(text=None, model=model_id, latency_seconds=time.perf_counter() - t0)
                _log.info("model=%s transient error attempt %d: %s", model_id, attempt, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue
            except openai.APIStatusError as exc:
                # Provider HTTP 402 (e.g. BYOK spend cap) — retry a few times in case it is transient.
                code = getattr(exc, "status_code", None)
                if code == 402 and attempt <= 8:
                    wait = _retry_after_seconds_from_error(exc)
                    wait = (wait + 5.0) if wait is not None else 45.0
                    _log.warning(
                        "model=%s HTTP 402 (billing/provider limit); retry in %.0fs (attempt %d/8). "
                        "If this persists, raise the key spend cap at https://openrouter.ai/settings/keys — %s",
                        model_id,
                        wait,
                        attempt,
                        exc,
                    )
                    time.sleep(min(wait, 120.0))
                    continue
                _log.warning("model=%s permanent error: %s", model_id, exc)
                return ChatResponse(text=None, model=model_id, latency_seconds=time.perf_counter() - t0)
            except openai.OpenAIError as exc:
                _log.warning("model=%s permanent error: %s", model_id, exc)
                return ChatResponse(text=None, model=model_id, latency_seconds=time.perf_counter() - t0)

            elapsed = time.perf_counter() - t0
            choice = response.choices[0] if response.choices else None
            text = choice.message.content if choice and choice.message else None
            return ChatResponse(text=text, model=model_id, latency_seconds=elapsed, raw=response)

    def sleep_between_calls(self) -> None:
        """Honour the configured per-call rate-limit pause."""
        if self._config.rate_limit_sleep_seconds > 0:
            time.sleep(self._config.rate_limit_sleep_seconds)
