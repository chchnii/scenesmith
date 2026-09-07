"""Compatibility patches for pinned dependency versions.

The openai-agents SDK 0.6.9 builds ``InputTokensDetails(cached_tokens=0)`` in
``agents/usage.py``, but openai >=2.54 declares ``cache_write_tokens`` as a
required field on that model. Without a patch every agent run (in any entry
point) dies with:

    pydantic_core._pydantic_core.ValidationError: 1 validation error for
    InputTokensDetails, cache_write_tokens Field required

We give the field a default of 0 so the agents SDK's single-argument
construction succeeds. This patch is idempotent and safe to call at import.

Plus a rate-limit retry patch: the OpenAI SDK's default backoff caps at
``MAX_RETRY_DELAY`` (8s), but the relay used here enforces a *per-minute*
token quota with no ``Retry-After`` header, so transient 429s never recover.
This backend waits 65s (just over the quota window) before each retry.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI
from openai.types.responses.response_usage import InputTokensDetails

_PATCHED = False


class _RateLimitRetryMixin:
    """Retry backoff that outlasts a per-minute quota window.

    The stock SDK retries 429s with exponential backoff capped at 8s, which is
    shorter than the ~60s reset window of the JD relay, so every retry re-hits
    the rate limit and the pipeline hard-fails with ``openai.RateLimitError``.
    We extend that delay to 65s and allow several retries so a burst that trips
    the per-minute cap recovers on its own once the window elapses.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_retries", 4)
        super().__init__(*args, **kwargs)

    def _calculate_retry_timeout(
        self,
        remaining_retries: int,
        options: Any,
        response_headers: httpx.Headers | None = None,
    ) -> float:
        # Honor an explicit server Retry-After when present (always respected
        # for a sane window), otherwise wait the full quota-reset interval.
        retry_after = self._parse_retry_after_header(response_headers)
        if (
            retry_after is not None
            and math.isfinite(retry_after)
            and 0 < retry_after <= 120
        ):
            return retry_after
        return 65.0


class RateLimitOpenAI(_RateLimitRetryMixin, OpenAI):
    """Synchronous OpenAI client with per-minute-quota-safe backoff."""


class RateLimitRetryOpenAI(_RateLimitRetryMixin, AsyncOpenAI):
    """Async OpenAI client with per-minute-quota-safe backoff."""


def apply_compat_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    for _field_name in ("cache_write_tokens", "cached_tokens"):
        _field = InputTokensDetails.model_fields.get(_field_name)
        if _field is not None:
            _field.default = 0
    InputTokensDetails.model_rebuild(force=True)

    # Point both OpenAI Agents SDK client-construction paths at the
    # retry-tolerant client before any model is constructed:
    #
    # * A directly constructed OpenAIChatCompletionsModel lazily creates its
    #   client through openai_chatcompletions.AsyncOpenAI.
    # * The normal Runner path asks OpenAIProvider to create a client first and
    #   inject it into the model, bypassing the model's lazy constructor.
    #
    # Patch both module-local imports; patching only the first path leaves the
    # normal Runner path on the stock sub-second retry schedule.
    import agents.models.openai_chatcompletions as _ocm
    import agents.models.openai_provider as _provider

    _ocm.AsyncOpenAI = RateLimitRetryOpenAI  # type: ignore[assignment]
    _provider.AsyncOpenAI = RateLimitRetryOpenAI  # type: ignore[assignment]

    _PATCHED = True
