from __future__ import annotations

import json
import logging

try:
    from config import settings
    from evaluation.dataset_generator import get_ragas_embeddings, get_ragas_llm, load_golden_dataset
except ImportError:  # pragma: no cover
    from ..config import settings
    from .dataset_generator import get_ragas_embeddings, get_ragas_llm, load_golden_dataset

logger = logging.getLogger(__name__)


async def run_evaluation(rag_chain_fn) -> dict:
    """
    rag_chain_fn: async callable that takes a question string and returns
    {"answer": str, "chunks_used": list[dict], "has_sufficient_context": bool}
    """
    dataset_dict = load_golden_dataset()
    questions = dataset_dict.get("questions", [])

    if not questions:
        raise ValueError("Golden dataset is empty. Run POST /eval/generate first.")

    results = []
    for item in questions:
        try:
            output = await rag_chain_fn(item["question"])
            results.append({
                "user_input": item["question"],
                "response": output["answer"],
                "retrieved_contexts": [c["text"] for c in output.get("chunks_used", [])],
                "reference": item["answer"],
            })
        except Exception as e:
            logger.warning(f"Skipping question {item['id']}: {e}")

    if not results:
        raise ValueError("No evaluation results produced.")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )

    faithfulness_score = float(scores["faithfulness"])
    relevancy_score = float(scores["answer_relevancy"])
    passed = (
        faithfulness_score >= settings.min_faithfulness_score and
        relevancy_score >= settings.min_answer_relevancy_score
    )

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "num_questions": len(results),
        "passed": passed,
        "raw_results": scores.to_pandas().to_json(),
    }
