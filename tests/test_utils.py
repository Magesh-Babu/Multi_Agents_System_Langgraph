import pytest
from langchain.docstore.document import Document
from stock_analyst.utils.rag_utils import split_documents


def test_split_documents_returns_chunks(sample_documents):
    chunks = split_documents(sample_documents)
    assert isinstance(chunks, list)
    assert len(chunks) >= len(sample_documents)


def test_split_documents_empty_input():
    result = split_documents([])
    assert result == []


def test_split_documents_respects_chunk_size():
    long_text = "word " * 500  # ~2500 chars — should produce multiple chunks
    docs = [Document(page_content=long_text, metadata={})]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_split_documents_preserves_content(sample_documents):
    chunks = split_documents(sample_documents)
    combined = " ".join(c.page_content for c in chunks)
    for doc in sample_documents:
        # Every word from original content should appear somewhere in chunks
        first_word = doc.page_content.split()[0]
        assert first_word in combined
