from __future__ import annotations


def test_pdf_loader_extracts_text_and_page_metadata(tiny_pdf):
    from ingestion.loader import load_document

    pages = load_document(str(tiny_pdf), "pdf", "doc-1")

    assert len(pages) == 2
    assert "Alpha policy" in pages[0]["text"]
    assert pages[0]["metadata"]["page"] == 1
    assert pages[0]["metadata"]["doc_id"] == "doc-1"


def test_chunker_produces_chunks_within_limit_and_unique_ids():
    from ingestion.chunker import chunk_documents

    pages = [
        {
            "text": "Alpha beta gamma. " * 40,
            "metadata": {"source": "doc.txt", "page": 1, "doc_id": "doc-2", "file_type": "txt"},
        }
    ]

    chunks = chunk_documents(pages, chunk_size=80, chunk_overlap=10)
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]

    assert chunks
    assert all(len(chunk["text"]) <= 80 for chunk in chunks)
    assert len(ids) == len(set(ids))


def test_full_pipeline_pdf_to_temp_chroma(tiny_pdf, tmp_path, mock_openrouter_embed, monkeypatch):
    import ingestion.pipeline as pipeline
    import retrieval.bm25_store as bm25_module
    import retrieval.vector_store as vector_module

    store = vector_module.VectorStore(str(tmp_path / "chroma"))
    bm25 = bm25_module.BM25Store()
    monkeypatch.setattr(vector_module, "vector_store", store)
    monkeypatch.setattr(bm25_module, "bm25_store", bm25)
    monkeypatch.setattr(pipeline, "_maybe_generate_dataset", lambda: None)

    chunks = pipeline.ingest_document_sync("doc-pdf", str(tiny_pdf), "pdf")

    assert chunks
    stored = store.get_all_chunks()
    assert len(stored) == len(chunks)
    assert stored[0]["metadata"]["doc_id"] == "doc-pdf"
