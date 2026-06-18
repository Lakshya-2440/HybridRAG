from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class URLIngestRequest(BaseModel):
    url: str


class UploadResponse(BaseModel):
    doc_id: str
    status: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    upload_timestamp: datetime
    error_message: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    source: str
    page: int
    chunk_index: int
    verified: bool
    excerpt: str


class Chunk(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    has_sufficient_context: bool
    citations: list[Citation]
    chunks_used: list[Chunk]
    latency_ms: int


class EvalRunRequest(BaseModel):
    triggered_by: Literal["manual", "ci", "auto_startup"] = "manual"


class EvalRunResponse(BaseModel):
    eval_id: str
    status: str


class EvalGenerateResponse(BaseModel):
    status: str


class EvalResultResponse(BaseModel):
    id: str
    run_timestamp: datetime
    faithfulness_score: float
    answer_relevancy_score: float
    num_questions: int
    passed: bool
    raw_results: str
    triggered_by: str


class DeleteResponse(BaseModel):
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    chroma: bool
    bm25_docs: int
    sqlite: bool
