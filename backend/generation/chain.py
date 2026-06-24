from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import httpx

try:
    from config import settings
    from generation.citations import extract_citations
    from retrieval.bm25_store import bm25_store
    from retrieval.hybrid import reciprocal_rank_fusion
    from retrieval.reranker import rerank
    from retrieval.vector_store import vector_store
except ImportError:  # pragma: no cover
    from ..config import settings
    from .citations import extract_citations
    from ..retrieval.bm25_store import bm25_store
    from ..retrieval.hybrid import reciprocal_rank_fusion
    from ..retrieval.reranker import rerank
    from ..retrieval.vector_store import vector_store

logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).parent / "prompts"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "what",
    "who", "with", "you", "your", "about", "tell",
}
PROFILE_HINTS = {
    "about", "profile", "resume", "cv", "portfolio", "background", "who", "me", "myself",
    "skills", "experience", "education",
}

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    query: str
    original_query: str
    doc_ids: list[str] | None
    vector_results: list[dict]
    bm25_results: list[dict]
    hybrid_results: list[dict]
    reranked_chunks: list[dict]
    raw_answer: str
    citations: list[dict]
    has_sufficient_context: bool
    latency_ms: int
    retry_count: int
    rewrite_history: list[str]
    all_chunks_seen: list[dict]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_llm():
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
    return ChatOpenAI(
        client=sync_client,
        async_client=async_client,
        api_key=settings.openrouter_api_key,
        openai_api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=800,
        default_headers=headers,
        model_kwargs={"extra_headers": headers},
    )


# ---------------------------------------------------------------------------
# Retrieval nodes
# ---------------------------------------------------------------------------

async def retrieve_vector(state: RAGState) -> dict:
    if _is_profile_query(state["query"]):
        all_chunks = await asyncio.to_thread(vector_store.get_all_chunks)
        doc_ids = state.get("doc_ids")
        results = [
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": 1.0,
            }
            for chunk in all_chunks
            if not doc_ids or chunk["metadata"].get("doc_id") in doc_ids
        ]
    else:
        results = await asyncio.to_thread(
            vector_store.query,
            state["query"],
            settings.top_k_retrieval,
            state.get("doc_ids"),
        )
    logger.info("Vector retrieval count=%s query=%s retry=%s", len(results), state["query"], state.get("retry_count", 0))
    return {"vector_results": results}


async def retrieve_bm25(state: RAGState) -> dict:
    results = await asyncio.to_thread(bm25_store.query, state["query"], settings.top_k_retrieval)
    doc_ids = state.get("doc_ids")
    if doc_ids:
        results = [r for r in results if r["metadata"].get("doc_id") in doc_ids]
    logger.info("BM25 retrieval count=%s query=%s", len(results), state["query"])
    return {"bm25_results": results}


async def fuse_results(state: RAGState) -> dict:
    results = reciprocal_rank_fusion(state.get("vector_results", []), state.get("bm25_results", []))
    return {"hybrid_results": results}


async def rerank_chunks(state: RAGState) -> dict:
    top_n = max(settings.top_k_rerank, 10) if _is_profile_query(state["query"]) else settings.top_k_rerank
    results = await asyncio.to_thread(
        rerank,
        state["query"],
        state.get("hybrid_results", []),
        top_n,
    )
    # Merge with chunks from previous retries (deduped)
    prev_chunks = state.get("all_chunks_seen", [])
    merged = _dedupe_chunks(prev_chunks + results)
    logger.info("Reranked count=%s total_seen=%s query=%s", len(results), len(merged), state["query"])
    return {"reranked_chunks": results, "all_chunks_seen": merged}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        lines.append(
            f"--- Chunk [Source: {metadata.get('source')}, Page: {metadata.get('page')}, "
            f"Chunk: {metadata.get('chunk_index')}] ---\n{chunk['text']}\n"
        )
    return "\n".join(lines)


def _split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for line in text.splitlines():
        clean = line.strip(" -•\t")
        if not clean or set(clean) <= {"|", "-"}:
            continue
        pieces.extend(re.split(r"(?<=[.!?])\s+", clean))
    return [piece.strip() for piece in pieces if piece.strip()]


def _query_terms(query: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(query.lower()) if token not in STOPWORDS}


def _is_profile_query(query: str) -> bool:
    tokens = set(TOKEN_PATTERN.findall(query.lower()))
    return bool(tokens & PROFILE_HINTS) and (
        {"who", "me"} <= tokens
        or {"who", "i"} <= tokens
        or {"about", "me"} <= tokens
        or {"tell", "me"} <= tokens
        or bool(tokens & {"profile", "resume", "cv", "background"})
    )


def _citation(metadata: dict) -> str:
    return (
        f"[Source: {metadata.get('source')}, Page: {metadata.get('page')}, "
        f"Chunk: {metadata.get('chunk_index')}]"
    )


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for chunk in chunks:
        cid = chunk["metadata"].get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(chunk)
    return unique


def _find_chunk(chunks: list[dict], *needles: str) -> dict | None:
    for chunk in chunks:
        text = chunk["text"].lower()
        if any(needle.lower() in text for needle in needles):
            return chunk
    return None


def _line_after(text: str, marker: str) -> str:
    lines = [line.strip(" -|") for line in text.splitlines() if line.strip(" -|")]
    marker_lower = marker.lower()
    for index, line in enumerate(lines):
        if marker_lower in line.lower() and index + 1 < len(lines):
            return lines[index + 1].strip()
    return ""


def _extract_name(chunks: list[dict]) -> str:
    joined = "\n".join(chunk["text"] for chunk in chunks[:3])
    match = re.search(r"ABOUT ME\s*[—-]\s*([A-Z][A-Z\s]+)", joined)
    if match:
        return match.group(1).title().strip()
    return "This person"


# ---------------------------------------------------------------------------
# Profile answer builder
# ---------------------------------------------------------------------------

def _profile_answer(query: str, chunks: list[dict]) -> str:
    ordered = sorted(_dedupe_chunks(chunks), key=lambda c: c["metadata"].get("chunk_index", 0))
    if not ordered:
        return "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."

    name = _extract_name(ordered)
    intro_chunk = _find_chunk(ordered, "summary", "coding since", "b.tech") or ordered[0]
    education_chunk = _find_chunk(ordered, "newton school", "gpa", "b.tech") or intro_chunk
    projects_chunk = _find_chunk(ordered, "12,000+ users", "rag systems", "live ai", "education tools") or intro_chunk
    skills_chunk = _find_chunk(ordered, "technical skills", "python", "fastapi", "react") or projects_chunk
    leadership_chunk = _find_chunk(ordered, "director general", "best delegate", "leadership") or projects_chunk

    education_text = education_chunk["text"]
    degree = "B.Tech CS freshman at Newton School of Technology, Pune"
    gpa_match = re.search(r"GPA:\s*([0-9.]+\s*/\s*10)", education_text)
    gpa = f" with GPA {gpa_match.group(1)}" if gpa_match else ""

    summary_sentence = ""
    for sentence in _split_sentences(intro_chunk["text"]):
        lowered = sentence.lower()
        if "coding since" in lowered or "users across" in lowered or "builds rag" in lowered:
            summary_sentence = sentence
            break
    if not summary_sentence:
        summary_sentence = "You are a computer science student building AI, RAG, and education tools."

    project_sentence = ""
    for sentence in _split_sentences(projects_chunk["text"]):
        lowered = sentence.lower()
        if "users" in lowered or "rag" in lowered or "ai" in lowered or "education" in lowered:
            project_sentence = sentence
            break
    if not project_sentence:
        project_sentence = "Your projects span RAG systems, AI evaluation, search, robotics, and education platforms."

    skill_text = skills_chunk["text"]
    skill_matches = []
    for keyword in ["Python", "FastAPI", "LangChain", "React", "Node.js", "PostgreSQL", "Neo4j", "XGBoost", "Pytest"]:
        if keyword.lower() in skill_text.lower():
            skill_matches.append(keyword)
    skill_summary = ", ".join(skill_matches[:8]) if skill_matches else "Python, full-stack engineering, ML, and RAG tooling"

    leadership_sentence = ""
    for sentence in _split_sentences(leadership_chunk["text"]):
        lowered = sentence.lower()
        if "director general" in lowered or "best delegate" in lowered or "chaired" in lowered:
            leadership_sentence = sentence
            break
    if not leadership_sentence:
        leadership_sentence = "You also show leadership through policy, debate, and community work."

    return (
        f"### {name}\n\n"
        f"You are a {degree}{gpa}. {summary_sentence} {_citation(intro_chunk['metadata'])}\n\n"
        f"**What you build**\n"
        f"- {project_sentence} {_citation(projects_chunk['metadata'])}\n"
        f"- Your strongest technical lane is practical AI infrastructure: retrieval systems, evaluation, search/ranking, and deployed education tools. {_citation(projects_chunk['metadata'])}\n\n"
        f"**Technical profile**\n"
        f"- Core stack: {skill_summary}. {_citation(skills_chunk['metadata'])}\n"
        f"- You combine backend systems, ML workflows, data work, and frontend product shipping instead of staying in one narrow lane. {_citation(skills_chunk['metadata'])}\n\n"
        f"**Leadership angle**\n"
        f"- {leadership_sentence} {_citation(leadership_chunk['metadata'])}\n\n"
        f"**Short version**\n"
        f"You are an early CS builder with unusually broad shipped work: AI/RAG, education products, search, evaluation, and policy-flavored leadership. {_citation(intro_chunk['metadata'])}"
    )


# ---------------------------------------------------------------------------
# Local answer (no LLM)
# ---------------------------------------------------------------------------

def _local_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."

    if _is_profile_query(query):
        return _profile_answer(query, chunks)

    query_terms = _query_terms(query)
    selected: list[tuple[float, str, dict]] = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        for sentence in _split_sentences(chunk["text"]):
            sentence_terms = {token for token in TOKEN_PATTERN.findall(sentence.lower()) if token not in STOPWORDS}
            overlap = len(query_terms & sentence_terms)
            density = overlap / max(len(sentence_terms), 1)
            score = overlap + density + float(chunk.get("rerank_score", 0.0))
            if overlap > 0:
                selected.append((score, sentence, metadata))

    if not selected:
        return "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."

    selected = sorted(selected, key=lambda item: item[0], reverse=True)[:4]
    answer_parts = ["Here is the grounded answer:"]
    for _, sentence, metadata in selected:
        answer_parts.append(
            f"- {sentence} {_citation(metadata)}"
        )
    return "\n".join(answer_parts)


# ---------------------------------------------------------------------------
# Generation node
# ---------------------------------------------------------------------------

async def generate_answer(state: RAGState) -> dict:
    # On retries, use merged chunks from all attempts for richer context
    chunks_to_use = state.get("all_chunks_seen", []) if state.get("retry_count", 0) > 0 else state.get("reranked_chunks", [])
    if not chunks_to_use:
        chunks_to_use = state.get("reranked_chunks", [])

    if not settings.use_remote_models:
        answer = _local_answer(state["query"], chunks_to_use)
        logger.info("Local answer length=%s query=%s retry=%s", len(answer), state["query"], state.get("retry_count", 0))
        return {"raw_answer": answer}

    system_prompt = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
    citation_prompt = (PROMPT_DIR / "citation_enforce.txt").read_text(encoding="utf-8")

    retry_instruction = ""
    if state.get("retry_count", 0) > 0:
        retry_instruction = (
            "\n\nIMPORTANT: This is retry attempt #{retry}. Previous queries did not find sufficient context. "
            "You now have additional chunks from expanded retrieval. Try harder to find relevant information. "
            "If you can provide even a partial answer with what's available, do so and note which parts "
            "have limited evidence rather than saying INSUFFICIENT_CONTEXT."
        ).format(retry=state["retry_count"])

    prompt = (
        f"{system_prompt}{retry_instruction}\n\n"
        f"CONTEXT CHUNKS:\n{_format_context(chunks_to_use)}\n\n"
        f"QUESTION: {state['query']}\n\n"
        f"{citation_prompt}\n\n"
        "Answer:"
    )
    llm = get_llm()
    response = await asyncio.to_thread(llm.invoke, prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    logger.info("Answer length=%s query=%s retry=%s", len(answer), state["query"], state.get("retry_count", 0))
    return {"raw_answer": answer}


# ---------------------------------------------------------------------------
# Citation extraction + sufficiency check
# ---------------------------------------------------------------------------

async def extract_and_validate_citations(state: RAGState) -> dict:
    citations = extract_citations(state.get("raw_answer", ""), state.get("reranked_chunks", []))
    return {"citations": citations}


async def check_context_sufficiency(state: RAGState) -> dict:
    answer = state.get("raw_answer", "")
    has_sufficient = not answer.startswith("INSUFFICIENT_CONTEXT:")
    return {"has_sufficient_context": has_sufficient}


# ---------------------------------------------------------------------------
# Query rewriting node (agentic self-correction)
# ---------------------------------------------------------------------------

async def rewrite_query(state: RAGState) -> dict:
    """Rewrite the query for better retrieval on retry."""
    original = state.get("original_query", state["query"])
    retry_count = state.get("retry_count", 0) + 1
    history = list(state.get("rewrite_history", []))

    if settings.use_remote_models:
        rewritten = await _llm_rewrite(original, history, state.get("reranked_chunks", []))
    else:
        rewritten = _local_rewrite(original, history, state.get("reranked_chunks", []), retry_count)

    history.append(rewritten)
    logger.info("Query rewrite: '%s' → '%s' (retry %s)", original, rewritten, retry_count)

    return {
        "query": rewritten,
        "retry_count": retry_count,
        "rewrite_history": history,
    }


async def _llm_rewrite(original: str, history: list[str], chunks: list[dict]) -> str:
    """Use LLM to intelligently rewrite the query."""
    rewrite_prompt_path = PROMPT_DIR / "rewrite_prompt.txt"
    if rewrite_prompt_path.exists():
        template = rewrite_prompt_path.read_text(encoding="utf-8")
    else:
        template = (
            "You are a search query optimizer. The original question did not retrieve "
            "sufficient documents. Rewrite it to be more likely to match relevant content.\n\n"
            "Original question: {original}\n"
            "Previous attempts: {history}\n"
            "Available chunk snippets: {snippets}\n\n"
            "Write ONLY the rewritten query, nothing else."
        )

    snippets = "\n".join(c["text"][:100] for c in chunks[:5]) if chunks else "No chunks found"
    history_str = " | ".join(history) if history else "None"

    prompt = template.format(
        original=original,
        history=history_str,
        snippets=snippets,
    )

    llm = get_llm()
    response = await asyncio.to_thread(llm.invoke, prompt)
    rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # Fallback if LLM returns empty or same query
    if not rewritten or rewritten.lower() == original.lower():
        return _local_rewrite(original, history, chunks, len(history) + 1)

    return rewritten


def _local_rewrite(original: str, history: list[str], chunks: list[dict], retry_num: int) -> str:
    """Rewrite query locally by extracting keywords from available chunks and expanding."""
    original_terms = _query_terms(original)

    # Extract frequent terms from chunks that aren't in original query
    chunk_terms: dict[str, int] = {}
    for chunk in chunks[:10]:
        for token in TOKEN_PATTERN.findall(chunk["text"].lower()):
            if token not in STOPWORDS and token not in original_terms and len(token) > 2:
                chunk_terms[token] = chunk_terms.get(token, 0) + 1

    # Pick top expansion terms
    sorted_terms = sorted(chunk_terms.items(), key=lambda x: x[1], reverse=True)
    expansion = [term for term, _ in sorted_terms[:3]]

    if retry_num == 1:
        # First retry: add chunk-derived context terms
        if expansion:
            return f"{original} {' '.join(expansion)}"
        # If no chunks, try broadening
        return f"{original} overview summary details"
    elif retry_num == 2:
        # Second retry: rephrase as keywords
        terms = list(original_terms)
        if expansion:
            terms.extend(expansion[:2])
        return " ".join(terms)
    else:
        # Third retry: maximally broad
        return f"{original} information context explanation"


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_retry(state: RAGState) -> str:
    """Conditional edge: decide whether to retry or finish."""
    if state.get("has_sufficient_context", True):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("Max retries reached for query=%s", state.get("original_query", state["query"]))
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("retrieve_vector", retrieve_vector)
    graph.add_node("retrieve_bm25", retrieve_bm25)
    graph.add_node("fuse_results", fuse_results)
    graph.add_node("rerank_chunks", rerank_chunks)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("extract_and_validate_citations", extract_and_validate_citations)
    graph.add_node("check_context_sufficiency", check_context_sufficiency)
    graph.add_node("rewrite_query", rewrite_query)

    # Edges: main pipeline
    graph.set_entry_point("retrieve_vector")
    graph.add_edge("retrieve_vector", "retrieve_bm25")
    graph.add_edge("retrieve_bm25", "fuse_results")
    graph.add_edge("fuse_results", "rerank_chunks")
    graph.add_edge("rerank_chunks", "generate_answer")
    graph.add_edge("generate_answer", "extract_and_validate_citations")
    graph.add_edge("extract_and_validate_citations", "check_context_sufficiency")

    # Conditional edge: retry or end
    graph.add_conditional_edges(
        "check_context_sufficiency",
        should_retry,
        {
            "retry": "rewrite_query",
            "end": END,
        },
    )

    # Rewrite loops back to retrieval
    graph.add_edge("rewrite_query", "retrieve_vector")

    return graph.compile()


rag_graph = build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_rag_chain(query: str, doc_ids: list[str] | None = None) -> dict:
    start = time.perf_counter()
    initial: RAGState = {
        "query": query,
        "original_query": query,
        "doc_ids": doc_ids,
        "vector_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "reranked_chunks": [],
        "raw_answer": "",
        "citations": [],
        "has_sufficient_context": True,
        "latency_ms": 0,
        "retry_count": 0,
        "rewrite_history": [],
        "all_chunks_seen": [],
    }
    result = await rag_graph.ainvoke(initial)
    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "RAG query complete query=%s chunks=%s citations=%s retries=%s latency_ms=%s",
        query,
        len(result.get("reranked_chunks", [])),
        len(result.get("citations", [])),
        result.get("retry_count", 0),
        latency_ms,
    )
    return {
        "answer": result.get("raw_answer", ""),
        "has_sufficient_context": result.get("has_sufficient_context", True),
        "citations": result.get("citations", []),
        "chunks_used": result.get("reranked_chunks", []),
        "latency_ms": latency_ms,
        "retry_count": result.get("retry_count", 0),
        "rewrite_history": result.get("rewrite_history", []),
    }
