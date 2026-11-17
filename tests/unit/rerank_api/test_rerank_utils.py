from typing import Final

from litellm.rerank_api.rerank_utils import get_optional_rerank_params


def test_get_optional_rerank_params_without_config_returns_request_params() -> None:
    """Providers reranked through a dedicated handler (Together AI, Bedrock)
    have no provider config, so their request params pass through unmapped
    instead of being mapped through another provider's config."""
    params: Final = get_optional_rerank_params(
        rerank_provider_config=None,
        model="my_model",
        drop_params=False,
        query="my_query",
        documents=["doc_a", "doc_b"],
        custom_llm_provider="together_ai",
        top_n=2,
    )

    assert params == {
        "query": "my_query",
        "top_n": 2,
        "documents": ["doc_a", "doc_b"],
        "return_documents": True,
    }
