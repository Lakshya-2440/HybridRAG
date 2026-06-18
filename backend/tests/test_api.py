from __future__ import annotations

from fastapi.testclient import TestClient


def test_upload_list_query_delete_and_health(tiny_pdf, monkeypatch):
    import api.routes.documents as documents_route
    import api.routes.query as query_route
    import main

    monkeypatch.setattr(documents_route, "ingest_document_sync", lambda doc_id, file_path, file_type: [])
    monkeypatch.setattr(documents_route.vector_store, "delete_document", lambda doc_id: None)
    monkeypatch.setattr(documents_route.vector_store, "get_all_chunks", lambda: [])
    monkeypatch.setattr(documents_route.bm25_store, "rebuild_from_vector_store", lambda chunks: None)

    async def fake_chain(query: str, doc_ids=None):
        return {
            "answer": "Answer. [Source: tiny.pdf, Page: 1, Chunk: 0]",
            "has_sufficient_context": True,
            "citations": [
                {
                    "source": "tiny.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "verified": True,
                    "excerpt": "Alpha policy lives on page one.",
                }
            ],
            "chunks_used": [
                {
                    "text": "Alpha policy lives on page one.",
                    "metadata": {"source": "tiny.pdf", "page": 1, "chunk_index": 0},
                    "rerank_score": 0.92,
                }
            ],
            "latency_ms": 10,
        }

    monkeypatch.setattr(query_route, "run_rag_chain", fake_chain)
    client = TestClient(main.app)

    with tiny_pdf.open("rb") as fh:
        upload = client.post("/documents/upload", files={"file": ("tiny.pdf", fh, "application/pdf")})

    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    assert upload.json()["status"] == "processing"

    documents = client.get("/documents")
    assert documents.status_code == 200
    assert any(item["id"] == doc_id for item in documents.json())

    query = client.post("/query", json={"query": "Where is alpha?", "doc_ids": [doc_id]})
    assert query.status_code == 200
    assert query.json()["citations"][0]["verified"] is True

    delete = client.delete(f"/documents/{doc_id}")
    assert delete.status_code == 200
    assert delete.json() == {"deleted": True}

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
