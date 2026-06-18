from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from langchain_core.documents import Document

try:
    from config import settings
except ImportError:  # pragma: no cover
    from ..config import settings

logger = logging.getLogger(__name__)
GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
TARGET_SIZE = 50


def get_ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=settings.llm_model,
        default_headers={
            "HTTP-Referer": "https://rag-system.app",
            "X-Title": "RAG System",
        }
    ))


def get_ragas_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.embedding_model,
    ))


def build_langchain_docs_from_vector_store() -> list[Document]:
    try:
        from retrieval.vector_store import vector_store
    except ImportError:  # pragma: no cover
        from ..retrieval.vector_store import vector_store

    all_chunks = vector_store.get_all_chunks()
    return [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in all_chunks
    ]


async def generate_golden_dataset(langchain_docs: list) -> dict:
    """
    Takes a list of LangChain Document objects (from already-ingested chunks).
    Generates 50 synthetic QA pairs using RAGAS TestsetGenerator.
    Saves to golden_dataset.json and returns the dataset dict.
    """
    if len(langchain_docs) < 5:
        logger.warning("Not enough documents to generate golden dataset (need at least 5)")
        dataset = _empty_dataset()
        _save_dataset(dataset)
        return dataset

    try:
        from ragas.testset import TestsetGenerator
    except ImportError:  # pragma: no cover - ragas version compatibility
        from ragas.testset.synthesizers.generate import TestsetGenerator

    generator = TestsetGenerator(
        llm=get_ragas_llm(),
        embedding_model=get_ragas_embeddings(),
    )

    try:
        testset = await asyncio.to_thread(
            generator.generate_with_langchain_docs,
            langchain_docs,
            testset_size=TARGET_SIZE,
        )
        df = testset.to_pandas()
        questions = []
        for i, row in df.iterrows():
            questions.append({
                "id": f"q{str(i+1).zfill(3)}",
                "question": str(row.get("user_input", row.get("question", ""))),
                "answer": str(row.get("reference", row.get("ground_truth", ""))),
                "context": str(row.get("reference_contexts", "")),
                "verified": True,
                "source": "auto_generated_ragas",
            })

        dataset = {
            "version": "1.0",
            "description": "Auto-generated golden dataset via RAGAS TestsetGenerator",
            "num_questions": len(questions),
            "questions": questions,
        }

        _save_dataset(dataset)
        logger.info(f"Golden dataset generated: {len(questions)} QA pairs saved to {GOLDEN_PATH}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to generate golden dataset: {e}")
        dataset = _empty_dataset()
        _save_dataset(dataset)
        return dataset


async def generate_golden_dataset_from_store() -> dict:
    return await generate_golden_dataset(build_langchain_docs_from_vector_store())


def generate_golden_dataset_sync(langchain_docs: list) -> dict:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(generate_golden_dataset(langchain_docs))
    return loop.create_task(generate_golden_dataset(langchain_docs))  # type: ignore[return-value]


def _save_dataset(dataset: dict) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(dataset, f, indent=2)


def _empty_dataset() -> dict:
    return {"version": "1.0", "description": "Empty", "num_questions": 0, "questions": []}


def load_golden_dataset() -> dict:
    if GOLDEN_PATH.exists():
        with open(GOLDEN_PATH) as f:
            return json.load(f)
    return _empty_dataset()
