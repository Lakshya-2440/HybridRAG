import json
from rank_bm25 import BM25Okapi

try:
    from config import settings
except ImportError:  # pragma: no cover
    from ..config import settings


class BM25Store:
    def __init__(self):
        self.corpus: list[dict] = []
        self.index: BM25Okapi | None = None
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            with open(settings.bm25_corpus_path) as f:
                self.corpus = json.load(f)
            self._build_index()
        except FileNotFoundError:
            pass

    def _build_index(self):
        if self.corpus:
            tokenized = [doc["text"].lower().split() for doc in self.corpus]
            self.index = BM25Okapi(tokenized)

    def rebuild_from_vector_store(self, chunks: list[dict]):
        self.corpus = chunks
        self._build_index()
        with open(settings.bm25_corpus_path, "w") as f:
            json.dump(self.corpus, f)

    def query(self, query_text: str, top_k: int) -> list[dict]:
        if not self.index or not self.corpus:
            return []
        tokens = query_text.lower().split()
        scores = self.index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self.corpus[i], "bm25_score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]


bm25_store = BM25Store()
