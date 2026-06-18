import re

CITATION_PATTERN = re.compile(
    r'\[Source:\s*(.+?),\s*Page:\s*(\d+),\s*Chunk:\s*(\d+)\]'
)


def extract_citations(answer: str, reranked_chunks: list[dict]) -> list[dict]:
    found = CITATION_PATTERN.findall(answer)
    citations = []
    for source, page, chunk_index in found:
        source = source.strip()
        page = int(page)
        chunk_index = int(chunk_index)
        verified = any(
            c["metadata"].get("source") == source and
            c["metadata"].get("page") == page and
            c["metadata"].get("chunk_index") == chunk_index
            for c in reranked_chunks
        )
        excerpt = ""
        if verified:
            for c in reranked_chunks:
                if (c["metadata"].get("source") == source and
                    c["metadata"].get("page") == page and
                    c["metadata"].get("chunk_index") == chunk_index):
                    excerpt = c["text"][:200]
                    break
        citations.append({
            "source": source, "page": page, "chunk_index": chunk_index,
            "verified": verified, "excerpt": excerpt
        })
    return citations
