def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, dict] = {}
    for rank, result in enumerate(vector_results):
        cid = result["metadata"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "data": result}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(bm25_results):
        cid = result["metadata"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "data": result}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
    sorted_results = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [r["data"] for r in sorted_results]
