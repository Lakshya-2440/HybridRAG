from __future__ import annotations

from fastapi import APIRouter, HTTPException

try:
    from api.schemas import QueryRequest, QueryResponse
    from generation.chain import run_rag_chain
except ImportError:  # pragma: no cover
    from ..schemas import QueryRequest, QueryResponse
    from ...generation.chain import run_rag_chain

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        result = await run_rag_chain(payload.query, payload.doc_ids)
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
