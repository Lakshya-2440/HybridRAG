from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import math
import re

import chromadb
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import httpx

try:
    from config import settings
except ImportError:  # pragma: no cover
    from ..config import settings


COLLECTION_NAME = "rag_documents"
BATCH_SIZE = 100
EMBEDDING_DIMENSION = 1536
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _local_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def get_embedding_client() -> OpenAI:
    headers = {
        "HTTP-Referer": "https://rag-system.app",
        "X-Title": "RAG System",
        "Authorization": f"Bearer {settings.openrouter_api_key}",
    }
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers,
        http_client=httpx.Client(base_url="https://openrouter.ai/api/v1", headers=headers),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not settings.use_remote_models:
        return [_local_embedding(text) for text in texts]

    client = get_embedding_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def get_langchain_embeddings() -> OpenAIEmbeddings:
    from openai import OpenAI, AsyncOpenAI
    headers = {
        "HTTP-Referer": "https://rag-system.app",
        "X-Title": "RAG System",
        "Authorization": f"Bearer {settings.openrouter_api_key}",
    }
    sync_client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers,
        http_client=httpx.Client(base_url="https://openrouter.ai/api/v1", headers=headers),
    )
    async_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers,
        http_client=httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", headers=headers),
    )
    return OpenAIEmbeddings(
        client=sync_client.embeddings,
        async_client=async_client.embeddings,
        api_key=settings.openrouter_api_key,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.embedding_model,
        default_headers=headers,
        model_kwargs={"extra_headers": headers},
    )


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
        )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)
        ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
        query_embedding = embed_texts([query_text])[0]
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        return [
            {
                "text": text,
                "metadata": metadata,
                "score": float(1 - distance) if distance is not None else 0.0,
            }
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def get_all_chunks(self) -> list[dict]:
        result: dict[str, Any] = self.collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            {"text": text, "metadata": metadata, "chunk_id": metadata.get("chunk_id")}
            for text, metadata in zip(documents, metadatas)
        ]


vector_store = VectorStore()
