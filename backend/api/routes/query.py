from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

try:
    from api.schemas import QueryRequest, QueryResponse
    from database import EvalRun, create_session
    from evaluation.eval_runner import evaluate_single_response
    from generation.chain import run_rag_chain
except ImportError:  # pragma: no cover
    from ..schemas import QueryRequest, QueryResponse
    from ...database import EvalRun, create_session
    from ...evaluation.eval_runner import evaluate_single_response
    from ...generation.chain import run_rag_chain

router = APIRouter()


def _evaluate_live_response_background(user_input: str, response: str, retrieved_contexts: list[str], eval_id: str) -> None:
    import asyncio
    db = create_session()
    try:
        try:
            results = asyncio.run(evaluate_single_response(user_input, response, retrieved_contexts))
            run = EvalRun(
                id=eval_id,
                faithfulness_score=results["faithfulness"],
                answer_relevancy_score=results["answer_relevancy"],
                num_questions=1,
                passed=results["passed"],
                raw_results=json.dumps(results),
                triggered_by="live_query",
            )
        except Exception as exc:
            run = EvalRun(
                id=eval_id,
                faithfulness_score=0.0,
                answer_relevancy_score=0.0,
                num_questions=1,
                passed=False,
                raw_results=json.dumps({"error": str(exc)}),
                triggered_by="live_query",
            )
        db.add(run)
        db.commit()
    finally:
        db.close()


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    try:
        result = await run_rag_chain(payload.query, payload.doc_ids)
        eval_id = str(uuid.uuid4())
        retrieved_contexts = [c["text"] for c in result.get("chunks_used", [])]
        background_tasks.add_task(
            _evaluate_live_response_background,
            payload.query,
            result.get("answer", ""),
            retrieved_contexts,
            eval_id,
        )
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

