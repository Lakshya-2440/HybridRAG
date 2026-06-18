from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

try:
    from api.schemas import EvalGenerateResponse, EvalResultResponse, EvalRunRequest, EvalRunResponse
    from database import EvalRun, create_session, get_db
    from evaluation.dataset_generator import generate_golden_dataset_from_store, load_golden_dataset
    from evaluation.eval_runner import run_evaluation
    from generation.chain import run_rag_chain
except ImportError:  # pragma: no cover
    from ..schemas import EvalGenerateResponse, EvalResultResponse, EvalRunRequest, EvalRunResponse
    from ...database import EvalRun, create_session, get_db
    from ...evaluation.dataset_generator import generate_golden_dataset_from_store, load_golden_dataset
    from ...evaluation.eval_runner import run_evaluation
    from ...generation.chain import run_rag_chain

router = APIRouter()


def _eval_to_response(run: EvalRun) -> EvalResultResponse:
    return EvalResultResponse(
        id=run.id,
        run_timestamp=run.run_timestamp,
        faithfulness_score=run.faithfulness_score,
        answer_relevancy_score=run.answer_relevancy_score,
        num_questions=run.num_questions,
        passed=run.passed,
        raw_results=run.raw_results,
        triggered_by=run.triggered_by,
    )


def _run_eval_background(eval_id: str, triggered_by: str) -> None:
    import asyncio

    db = create_session()
    try:
        try:
            results = asyncio.run(run_evaluation(lambda question: run_rag_chain(question)))
            run = EvalRun(
                id=eval_id,
                faithfulness_score=results["faithfulness"],
                answer_relevancy_score=results["answer_relevancy"],
                num_questions=results["num_questions"],
                passed=results["passed"],
                raw_results=results["raw_results"],
                triggered_by=triggered_by,
            )
        except Exception as exc:
            run = EvalRun(
                id=eval_id,
                faithfulness_score=0.0,
                answer_relevancy_score=0.0,
                num_questions=0,
                passed=False,
                raw_results=json.dumps({"error": str(exc)}),
                triggered_by=triggered_by,
            )
        db.add(run)
        db.commit()
    finally:
        db.close()


def _generate_dataset_background() -> None:
    import asyncio

    asyncio.run(generate_golden_dataset_from_store())


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(payload: EvalRunRequest, background_tasks: BackgroundTasks) -> EvalRunResponse:
    dataset = load_golden_dataset()
    if dataset.get("num_questions", 0) == 0:
        raise HTTPException(
            status_code=422,
            detail="Golden dataset not yet generated. POST /eval/generate first.",
        )
    eval_id = str(uuid.uuid4())
    background_tasks.add_task(_run_eval_background, eval_id, payload.triggered_by)
    return EvalRunResponse(eval_id=eval_id, status="running")


@router.get("/results", response_model=list[EvalResultResponse])
async def get_eval_results(db: Session = Depends(get_db)) -> list[EvalResultResponse]:
    runs = db.query(EvalRun).order_by(EvalRun.run_timestamp.desc()).limit(20).all()
    return [_eval_to_response(run) for run in runs]


@router.post("/generate", response_model=EvalGenerateResponse)
async def generate_eval_dataset(background_tasks: BackgroundTasks) -> EvalGenerateResponse:
    background_tasks.add_task(_generate_dataset_background)
    return EvalGenerateResponse(status="generating")
