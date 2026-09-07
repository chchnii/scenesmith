"""Regression tests for compatibility patches around third-party SDKs."""

from agents.models import openai_chatcompletions, openai_provider
from agents.models.openai_provider import OpenAIProvider

from scenesmith._compat import RateLimitRetryOpenAI, apply_compat_patches


def test_openai_provider_uses_rate_limit_retry_client() -> None:
    """The normal Runner model path must receive the 65-second retry client."""
    apply_compat_patches()

    assert openai_chatcompletions.AsyncOpenAI is RateLimitRetryOpenAI
    assert openai_provider.AsyncOpenAI is RateLimitRetryOpenAI

    provider = OpenAIProvider(api_key="test-only", use_responses=False)
    model = provider.get_model("gpt-5")

    assert isinstance(model._client, RateLimitRetryOpenAI)
    assert model._client.max_retries == 4
    assert model._client._calculate_retry_timeout(4, None, None) == 65.0
