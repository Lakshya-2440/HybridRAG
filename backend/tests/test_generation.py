from __future__ import annotations

import pytest


def _chunk() -> dict:
    return {
        "text": "Alpha policy lives on page one.",
        "metadata": {
            "chunk_id": "doc_1_0",
            "doc_id": "doc",
            "source": "tiny.pdf",
            "page": 1,
            "chunk_index": 0,
        },
    }


@pytest.mark.asyncio
async def test_chain_returns_answer_and_citations_for_grounded_query(monkeypatch, mock_openrouter_llm):
    import generation.chain as chain

    chunk = _chunk()
    monkeypatch.setattr(chain.vector_store, "query", lambda query, top_k, doc_ids=None: [chunk])
    monkeypatch.setattr(chain.bm25_store, "query", lambda query, top_k: [chunk])
    monkeypatch.setattr(chain, "rerank", lambda query, chunks, top_n: chunks[:top_n])

    result = await chain.run_rag_chain("Where is alpha?")

    assert "Alpha policy" in result["answer"]
    assert result["citations"][0]["verified"] is True


@pytest.mark.asyncio
async def test_chain_sets_has_sufficient_context_false(monkeypatch):
    import types
    import generation.chain as chain

    monkeypatch.setattr(chain.vector_store, "query", lambda query, top_k, doc_ids=None: [])
    monkeypatch.setattr(chain.bm25_store, "query", lambda query, top_k: [])
    monkeypatch.setattr(chain, "rerank", lambda query, chunks, top_n: [])
    monkeypatch.setattr(
        chain,
        "get_llm",
        lambda: types.SimpleNamespace(
            invoke=lambda prompt: types.SimpleNamespace(
                content="INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."
            )
        ),
    )

    result = await chain.run_rag_chain("Unknown?")

    assert result["has_sufficient_context"] is False


def test_extract_citations_parses_citation_correctly():
    from generation.citations import extract_citations

    answer = "Claim. [Source: doc.pdf, Page: 3, Chunk: 2]"
    chunks = [
        {
            "text": "Evidence excerpt.",
            "metadata": {"source": "doc.pdf", "page": 3, "chunk_index": 2},
        }
    ]

    citations = extract_citations(answer, chunks)

    assert citations[0]["source"] == "doc.pdf"
    assert citations[0]["page"] == 3
    assert citations[0]["chunk_index"] == 2
    assert citations[0]["verified"] is True


def test_unverified_citations_flagged_false():
    from generation.citations import extract_citations

    citations = extract_citations("Claim. [Source: doc.pdf, Page: 3, Chunk: 2]", [])

    assert citations[0]["verified"] is False
