from __future__ import annotations

import os
import sys
from pathlib import Path

import fitz
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter")
os.environ.setdefault("COHERE_API_KEY", "test-cohere")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/rag_tests.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/rag_tests_chroma")


@pytest.fixture
def settings_override(monkeypatch, tmp_path):
    from config import settings

    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'rag.db'}")
    monkeypatch.setattr(settings, "chunk_size", 120)
    monkeypatch.setattr(settings, "chunk_overlap", 10)
    monkeypatch.setattr(settings, "top_k_retrieval", 5)
    monkeypatch.setattr(settings, "top_k_rerank", 2)
    return settings


@pytest.fixture
def tiny_pdf(tmp_path):
    path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Alpha policy lives on page one.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Beta procedure lives on page two.")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def mock_openrouter_embed(monkeypatch):
    import retrieval.vector_store as vector_store

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]

    monkeypatch.setattr(vector_store, "embed_texts", fake_embed_texts)
    return fake_embed_texts


@pytest.fixture
def mock_openrouter_llm(monkeypatch):
    import types
    import generation.chain as chain

    fake_llm = types.SimpleNamespace(
        invoke=lambda prompt: types.SimpleNamespace(
            content="Alpha policy lives on page one. [Source: tiny.pdf, Page: 1, Chunk: 0]"
        )
    )
    monkeypatch.setattr(chain, "get_llm", lambda: fake_llm)
    return fake_llm


@pytest.fixture
def mock_cohere_rerank(monkeypatch):
    import retrieval.reranker as reranker

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def rerank(self, query, documents, top_n, model):
            class Result:
                def __init__(self, index, score):
                    self.index = index
                    self.relevance_score = score

            class Response:
                results = [Result(i, 1.0 - (i * 0.1)) for i in range(min(top_n, len(documents)))]

            return Response()

    monkeypatch.setattr(reranker.cohere, "Client", FakeClient)
    return FakeClient
