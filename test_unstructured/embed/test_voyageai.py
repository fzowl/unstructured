from unittest.mock import Mock

import pytest

from unstructured.documents.elements import Text
from unstructured.embed.voyageai import VoyageAIEmbeddingConfig, VoyageAIEmbeddingEncoder


def test_embed_documents_does_not_break_element_to_dict(mocker):
    # Mocked client with the desired behavior for embed_documents
    embed_response = Mock()
    embed_response.embeddings = [[1], [2]]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = embed_response
    mock_client.tokenize.return_value = [[1], [1]]  # Mock token counts

    # Mock get_client to return our mock_client
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3-large")
    )
    elements = encoder.embed_documents(
        elements=[Text("This is sentence 1"), Text("This is sentence 2")],
    )
    assert len(elements) == 2
    assert elements[0].to_dict()["text"] == "This is sentence 1"
    assert elements[1].to_dict()["text"] == "This is sentence 2"


def test_embed_documents_voyage_3_5(mocker):
    """Test embedding with voyage-3.5 model."""
    embed_response = Mock()
    embed_response.embeddings = [[1.0] * 1024, [2.0] * 1024]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = embed_response
    mock_client.tokenize.return_value = [[1, 2, 3], [1, 2]]  # Mock token counts

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    elements = encoder.embed_documents(
        elements=[Text("Test document 1"), Text("Test document 2")],
    )
    assert len(elements) == 2
    assert len(elements[0].embeddings) == 1024
    assert len(elements[1].embeddings) == 1024


def test_embed_documents_voyage_3_5_lite(mocker):
    """Test embedding with voyage-3.5-lite model."""
    embed_response = Mock()
    embed_response.embeddings = [[1.0] * 512, [2.0] * 512, [3.0] * 512]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = embed_response
    mock_client.tokenize.return_value = [[1], [1], [1]]  # Mock token counts

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5-lite")
    )
    elements = encoder.embed_documents(
        elements=[Text("Test 1"), Text("Test 2"), Text("Test 3")],
    )
    assert len(elements) == 3
    assert all(len(e.embeddings) == 512 for e in elements)


def _contextualized_response(n_chunks_per_doc):
    response = Mock()
    items = []
    for n_chunks in n_chunks_per_doc:
        item = Mock()
        item.embeddings = [[float(i)] * 1024 for i in range(n_chunks)]
        items.append(item)
    response.results = items
    return response


@pytest.mark.parametrize("model_name", ["voyage-context-3", "voyage-context-4"])
def test_embed_documents_contextual_model_uses_auto_chunking(mocker, model_name):
    mock_client = mocker.MagicMock()
    mock_client.contextualized_embed.return_value = _contextualized_response([1, 1])
    mock_client.tokenize.return_value = [[1, 2], [1, 2, 3]]

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name=model_name)
    )
    elements = encoder.embed_documents(
        elements=[Text("Context document 1"), Text("Context document 2")],
    )
    assert len(elements) == 2
    assert len(elements[0].embeddings) == 1024
    mock_client.contextualized_embed.assert_called_once()
    call_kwargs = mock_client.contextualized_embed.call_args[1]
    # Auto-chunking requires one flat string per document (nested lists are rejected by the API).
    assert call_kwargs["inputs"] == ["Context document 1", "Context document 2"]
    assert call_kwargs["input_type"] == "document"
    assert call_kwargs["enable_auto_chunking"] is True
    assert call_kwargs["chunk_size"] == 32_000


def test_embed_documents_contextual_model_rejects_multi_chunk_result(mocker):
    mock_client = mocker.MagicMock()
    mock_client.contextualized_embed.return_value = _contextualized_response([1, 2])
    mock_client.tokenize.return_value = [[1], [1]]

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-context-4")
    )
    with pytest.raises(ValueError, match="Text 1 was split into 2 chunks"):
        encoder.embed_documents(elements=[Text("a"), Text("b")])


def test_embed_query_contextual_model_does_not_auto_chunk(mocker):
    mock_client = mocker.MagicMock()
    mock_client.contextualized_embed.return_value = _contextualized_response([1])

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-context-4")
    )
    embedding = encoder.embed_query("test query")

    assert len(embedding) == 1024
    call_kwargs = mock_client.contextualized_embed.call_args[1]
    assert call_kwargs["inputs"] == [["test query"]]
    assert call_kwargs["input_type"] == "query"
    assert "enable_auto_chunking" not in call_kwargs
    assert "chunk_size" not in call_kwargs


def test_count_tokens(mocker):
    """Test token counting functionality."""
    mock_client = mocker.MagicMock()
    mock_client.tokenize.return_value = [[1, 2], [1, 2, 3, 4, 5]]  # Different token counts

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    texts = ["short text", "this is a longer text with more tokens"]
    token_counts = encoder.count_tokens(texts)

    assert len(token_counts) == 2
    assert token_counts[0] == 2
    assert token_counts[1] == 5


def test_count_tokens_empty_list(mocker):
    """Test token counting with empty list."""
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mocker.MagicMock())

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    token_counts = encoder.count_tokens([])
    assert token_counts == []


def test_get_token_limit(mocker):
    """Test getting token limit for different models."""
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mocker.MagicMock())

    # Test voyage-3.5 model
    config = VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    assert config.get_token_limit() == 320_000

    # Test voyage-3.5-lite model
    config_lite = VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5-lite")
    assert config_lite.get_token_limit() == 1_000_000

    # Test contextualized models (auto-chunking allows 120K tokens per request)
    for context_model in ("voyage-context-3", "voyage-context-4"):
        config_context = VoyageAIEmbeddingConfig(api_key="api_key", model_name=context_model)
        assert config_context.get_token_limit() == 120_000

    # Test voyage-4 family
    expected_limits = {"voyage-4-large": 120_000, "voyage-4": 320_000, "voyage-4-lite": 1_000_000}
    for model_name, limit in expected_limits.items():
        config_v4 = VoyageAIEmbeddingConfig(api_key="api_key", model_name=model_name)
        assert config_v4.get_token_limit() == limit

    # Test voyage-2 model
    config_v2 = VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-2")
    assert config_v2.get_token_limit() == 320_000

    # Test unknown model (should use default)
    config_unknown = VoyageAIEmbeddingConfig(api_key="api_key", model_name="unknown-model")
    assert config_unknown.get_token_limit() == 120_000


def test_is_context_model(mocker):
    """Test the _is_context_model helper method."""
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mocker.MagicMock())

    # Test with context model
    encoder_context = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-context-3")
    )
    assert encoder_context._is_context_model() is True

    encoder_context_4 = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-context-4")
    )
    assert encoder_context_4._is_context_model() is True

    # Test with regular models
    for model_name in [
        "voyage-3.5",
        "voyage-4",
        "voyage-4-lite",
        "voyage-4-large",
        "voyage-code-4",
    ]:
        encoder_regular = VoyageAIEmbeddingEncoder(
            config=VoyageAIEmbeddingConfig(api_key="api_key", model_name=model_name)
        )
        assert encoder_regular._is_context_model() is False


def test_build_batches_respects_token_limit_and_order(mocker):
    mock_client = mocker.MagicMock()
    # voyage-3.5 has a 320_000 token limit: 200K + 100K fit, the third text starts a new batch
    mock_client.tokenize.return_value = [[1] * 200_000, [1] * 100_000, [1] * 30_000, [1] * 5]
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    texts = ["text1", "text2", "text3", "text4"]
    batches = list(encoder._build_batches(texts, mock_client))

    assert batches == [["text1", "text2"], ["text3", "text4"]]


def test_build_batches_respects_max_inputs(mocker):
    mock_client = mocker.MagicMock()
    mock_client.tokenize.return_value = [[1]] * 2_001
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    texts = [f"text{i}" for i in range(2_001)]
    batches = list(encoder._build_batches(texts, mock_client))

    assert [len(batch) for batch in batches] == [1_000, 1_000, 1]
    assert [text for batch in batches for text in batch] == texts


def test_embed_query(mocker):
    """Test embedding a single query."""
    embed_response = Mock()
    embed_response.embeddings = [[1.0] * 1024]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = embed_response

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    embedding = encoder.embed_query("test query")

    assert len(embedding) == 1024
    # Verify embed was called with input_type="query"
    mock_client.embed.assert_called_once()
    call_kwargs = mock_client.embed.call_args[1]
    assert call_kwargs["input_type"] == "query"


def test_embed_documents_with_output_dimension(mocker):
    """Test embedding with custom output dimension."""
    embed_response = Mock()
    embed_response.embeddings = [[1.0] * 512, [2.0] * 512]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = embed_response
    mock_client.tokenize.return_value = [[1], [1]]

    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mock_client)

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(
            api_key="api_key", model_name="voyage-3.5", output_dimension=512
        )
    )
    elements = encoder.embed_documents(
        elements=[Text("Test 1"), Text("Test 2")],
    )
    assert len(elements) == 2
    # Verify output_dimension was passed
    call_kwargs = mock_client.embed.call_args[1]
    assert call_kwargs["output_dimension"] == 512


def test_embed_documents_empty_list(mocker):
    """Test embedding empty list of documents."""
    mocker.patch.object(VoyageAIEmbeddingConfig, "get_client", return_value=mocker.MagicMock())

    encoder = VoyageAIEmbeddingEncoder(
        config=VoyageAIEmbeddingConfig(api_key="api_key", model_name="voyage-3.5")
    )
    elements = encoder.embed_documents(elements=[])
    assert elements == []
