import cohere
import re

try:
    from config import settings
except ImportError:  # pragma: no cover
    from ..config import settings

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
PROFILE_SECTION_BOOSTS = {
    "summary": 1.2,
    "education": 1.0,
    "experience": 0.8,
    "projects": 0.8,
    "technical skills": 0.9,
    "leadership": 0.8,
    "contact": 0.4,
}


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


def _lexical_rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    query_terms = _query_terms(query)
    profile_query = _is_profile_query(query)
    scored = []
    for chunk in chunks:
        text = chunk["text"].lower()
        text_terms = TOKEN_PATTERN.findall(text)
        text_term_set = set(text_terms)
        overlap = len(query_terms & text_term_set)
        density = overlap / max(len(query_terms), 1)
        phrase_bonus = 0.25 if query.lower() in text else 0.0
        section_bonus = 0.0
        if profile_query:
            section_bonus = sum(weight for marker, weight in PROFILE_SECTION_BOOSTS.items() if marker in text)
        score = density + phrase_bonus + section_bonus
        next_chunk = chunk.copy()
        next_chunk["rerank_score"] = float(score)
        scored.append(next_chunk)
    return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []
    if not settings.use_remote_models:
        return _lexical_rerank(query, chunks, top_n)

    co = cohere.Client(settings.cohere_api_key)
    docs = [c["text"] for c in chunks]
    response = co.rerank(
        query=query,
        documents=docs,
        top_n=top_n,
        model=settings.rerank_model,
    )
    reranked = []
    for r in response.results:
        chunk = chunks[r.index].copy()
        chunk["rerank_score"] = float(r.relevance_score)
        reranked.append(chunk)
    return reranked
