from __future__ import annotations


def _chunk(text: str, chunk_id: str) -> dict:
    doc_id, page, index = chunk_id.split("_")
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": f"{doc_id}.txt",
            "page": int(page),
            "chunk_index": int(index),
        },
    }


def test_vector_store_upsert_then_query_returns_correct_doc(tmp_path, mock_openrouter_embed):
    import retrieval.vector_store as vector_module

    store = vector_module.VectorStore(str(tmp_path / "chroma"))
    store.upsert_chunks([_chunk("alpha searchable content", "d1_1_0")])

    result = store.query("alpha", top_k=1)

    assert result[0]["metadata"]["chunk_id"] == "d1_1_0"


def test_bm25_query_returns_results_for_known_term():
    from retrieval.bm25_store import BM25Store

    store = BM25Store()
    store.rebuild_from_vector_store(
        [
            _chunk("alpha unique searchable content", "d1_1_0"),
            _chunk("beta unrelated content", "d2_1_0"),
            _chunk("gamma unrelated content", "d3_1_0"),
        ]
    )

    result = store.query("alpha", top_k=1)

    assert result[0]["metadata"]["chunk_id"] == "d1_1_0"


def test_rrf_merges_deduplicates_chunk_ids():
    from retrieval.hybrid import reciprocal_rank_fusion

    first = _chunk("alpha", "d1_1_0")
    second = _chunk("beta", "d2_1_0")

    result = reciprocal_rank_fusion([first], [first, second])

    assert [item["metadata"]["chunk_id"] for item in result].count("d1_1_0") == 1
    assert len(result) == 2


def test_reranker_returns_top_n_sorted_by_score_descending(mock_cohere_rerank):
    from retrieval.reranker import rerank

    chunks = [_chunk("alpha", "d1_1_0"), _chunk("beta", "d2_1_0"), _chunk("gamma", "d3_1_0")]

    result = rerank("alpha", chunks, top_n=2)

    assert len(result) == 2
    assert result[0]["rerank_score"] >= result[1]["rerank_score"]
