from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

try:
    from config import settings
    from database import Document, create_session
    from ingestion.chunker import chunk_documents
    from ingestion.loader import load_document
except ImportError:  # pragma: no cover - package import path
    from ..config import settings
    from ..database import Document, create_session
    from .chunker import chunk_documents
    from .loader import load_document

logger = logging.getLogger(__name__)


def _set_document_status(doc_id: str, status: str, chunk_count: int = 0, error_message: str | None = None) -> None:
    db = create_session()
    try:
        document = db.get(Document, doc_id)
        if document:
            document.status = status
            document.chunk_count = chunk_count
            document.error_message = error_message
            db.commit()
    finally:
        db.close()


def _maybe_generate_dataset() -> None:
    if not settings.auto_generate_eval_dataset:
        return

    try:
        from evaluation.dataset_generator import build_langchain_docs_from_vector_store, generate_golden_dataset_sync, load_golden_dataset
    except ImportError:
        from ..evaluation.dataset_generator import build_langchain_docs_from_vector_store, generate_golden_dataset_sync, load_golden_dataset

    dataset = load_golden_dataset()
    if dataset.get("num_questions", 0) < 10:
        docs = build_langchain_docs_from_vector_store()
        if docs:
            generate_golden_dataset_sync(docs)


def ingest_document_sync(
    doc_id: str,
    file_path: str,
    file_type: str,
    *,
    after_success: Callable[[], None] | None = None,
) -> list[dict]:
    try:
        pages = load_document(file_path, file_type, doc_id)
        chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)

        try:
            from retrieval.vector_store import vector_store
            from retrieval.bm25_store import bm25_store
        except ImportError:
            from ..retrieval.vector_store import vector_store
            from ..retrieval.bm25_store import bm25_store

        vector_store.upsert_chunks(chunks)
        bm25_store.rebuild_from_vector_store(vector_store.get_all_chunks())
        _set_document_status(doc_id, "ready", len(chunks))
        if after_success:
            after_success()
        else:
            _maybe_generate_dataset()
        logger.info("Ingested document %s from %s with %s chunks", doc_id, Path(file_path).name, len(chunks))
        return chunks
    except Exception as exc:
        _set_document_status(doc_id, "failed", 0, str(exc))
        logger.exception("Document ingestion failed for %s", doc_id)
        raise


async def ingest_document(doc_id: str, file_path: str, file_type: str) -> list[dict]:
    return await asyncio.to_thread(ingest_document_sync, doc_id, file_path, file_type)
