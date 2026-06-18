PRD: Production-Grade RAG System (Ask-My-Docs)
For: OpenAI Codex / Claude Code / any agentic coding system
 Goal: Build this completely, end-to-end, production-ready, in one shot. No placeholders. No TODOs. Every file complete. No human intervention required.

1. Project Overview
Build a production-grade Retrieval-Augmented Generation (RAG) system — a domain-specific "Ask My Docs" application. Users upload documents, ask questions, and receive cited answers grounded in actual retrieved content.
Deployment:
Frontend → Vercel (Next.js, automatic via vercel.json)
Backend → Render (FastAPI, automatic via render.yaml)
Stack (exact, no substitutions unless noted):
Backend: Python 3.11+, FastAPI
Orchestration: LangChain + LangGraph
LLM + Embeddings: OpenRouter API (OpenAI-compatible, single key for everything)
LLM model: google/gemini-2.5-flash (fast, cheap, excellent for RAG)
Embedding model: openai/text-embedding-3-small (via OpenRouter embeddings endpoint)
Vector Store: ChromaDB (persistent disk on Render)
Keyword Search: BM25 via rank_bm25
Reranker: Cohere Rerank API (rerank-english-v3.0)
Evaluation: RAGAS with TestsetGenerator — auto-generates 50 QA pairs from ingested docs
Frontend: Next.js 14 (App Router) + Tailwind CSS
Database: SQLite via SQLAlchemy (metadata + eval logs)
File Parsing: PyMuPDF (PDF), python-markdown (MD), BeautifulSoup (HTML/web)
Config: Pydantic Settings + .env
Testing: pytest + pytest-asyncio
CI: GitHub Actions

2. Repo Structure
Build exactly this structure. Every file listed must be created and fully implemented.
rag-system/
├── .env.example
├── .gitignore
├── README.md
├── render.yaml                        # Render Blueprint (backend deployment)
├── vercel.json                        # Vercel config (frontend deployment)
├── pyproject.toml
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                        # FastAPI app entrypoint
│   ├── config.py                      # Pydantic settings (reads .env)
│   ├── database.py                    # SQLAlchemy setup + models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                  # PDF, MD, HTML, TXT, URL loaders
│   │   ├── chunker.py                 # RecursiveCharacterTextSplitter
│   │   └── pipeline.py                # load → chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py            # ChromaDB + OpenRouter embeddings
│   │   ├── bm25_store.py              # BM25Okapi index
│   │   ├── hybrid.py                  # RRF fusion
│   │   └── reranker.py                # Cohere rerank
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts/
│   │   │   ├── system.txt
│   │   │   └── citation_enforce.txt
│   │   ├── chain.py                   # LangGraph RAG chain
│   │   └── citations.py               # Parse + verify citations
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── golden_dataset.json        # Auto-generated 50 QA pairs (see §8)
│   │   ├── dataset_generator.py       # RAGAS TestsetGenerator — runs at startup
│   │   ├── eval_runner.py             # RAGAS evaluation runner
│   │   └── metrics.py                 # Faithfulness + answer relevancy
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── documents.py
│   │   │   ├── query.py
│   │   │   └── eval.py
│   │   └── schemas.py
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_ingestion.py
│       ├── test_retrieval.py
│       ├── test_generation.py
│       └── test_api.py
│
├── frontend/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── next.config.ts
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   ├── eval/
│       │   │   └── page.tsx
│       │   └── globals.css
│       ├── components/
│       │   ├── ChatInterface.tsx
│       │   ├── DocumentUpload.tsx
│       │   ├── DocumentList.tsx
│       │   ├── CitationCard.tsx
│       │   ├── EvalDashboard.tsx
│       │   └── LoadingDots.tsx
│       ├── hooks/
│       │   ├── useChat.ts
│       │   └── useDocuments.ts
│       └── lib/
│           └── api.ts
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── scripts/
    └── run_eval.sh


3. Environment Variables
.env.example — exact keys:
# OpenRouter (single key for BOTH LLM inference AND embeddings)
OPENROUTER_API_KEY=sk-or-v1-...

# Cohere (reranking only)
COHERE_API_KEY=...

# App
APP_ENV=development
LOG_LEVEL=INFO
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:3000

# ChromaDB (local dev) / mounted disk path (Render)
CHROMA_PERSIST_DIR=./chroma_db

# SQLite
DATABASE_URL=sqlite:///./rag.db

# RAG tuning
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5

# Models (all via OpenRouter)
LLM_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=openai/text-embedding-3-small
RERANK_MODEL=rerank-english-v3.0

# Eval thresholds — CI fails below these
MIN_FAITHFULNESS_SCORE=0.80
MIN_ANSWER_RELEVANCY_SCORE=0.75

# Production URLs (set in Render + Vercel dashboards)
RENDER_BACKEND_URL=https://rag-backend.onrender.com
NEXT_PUBLIC_API_URL=https://rag-backend.onrender.com


4. OpenRouter Integration — Critical Details
OpenRouter is fully OpenAI-API-compatible. Use it everywhere in place of direct OpenAI calls. One key handles everything.
4.1 LLM via LangChain
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=settings.llm_model,          # "google/gemini-2.5-flash"
        temperature=0.0,
        default_headers={
            "HTTP-Referer": "https://rag-system.app",
            "X-Title": "RAG System",
        }
    )

4.2 Embeddings via OpenRouter
OpenRouter supports embeddings at https://openrouter.ai/api/v1/embeddings — same schema as OpenAI. Use the openai Python SDK pointed at OpenRouter:
from openai import OpenAI

def get_embedding_client():
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_embedding_client()
    response = client.embeddings.create(
        model=settings.embedding_model,   # "openai/text-embedding-3-small"
        input=texts,
    )
    return [item.embedding for item in response.data]

Batch in groups of 100. Embedding dimension for text-embedding-3-small = 1536. Set this as ChromaDB collection dimension.
4.3 LangChain Embeddings wrapper (for RAGAS TestsetGenerator)
from langchain_openai import OpenAIEmbeddings

def get_langchain_embeddings():
    return OpenAIEmbeddings(
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.embedding_model,
    )


5. Backend — Full Implementation Spec
5.1 config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openrouter_api_key: str
    cohere_api_key: str
    app_env: str = "development"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    chroma_persist_dir: str = "./chroma_db"
    database_url: str = "sqlite:///./rag.db"
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    llm_model: str = "google/gemini-2.5-flash"
    embedding_model: str = "openai/text-embedding-3-small"
    rerank_model: str = "rerank-english-v3.0"
    min_faithfulness_score: float = 0.80
    min_answer_relevancy_score: float = 0.75
    render_backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

settings = Settings()

5.2 database.py
Two SQLAlchemy models:
Document: id (UUID str), filename (str), file_type (str), upload_timestamp (datetime), chunk_count (int), status (str: processing / ready / failed), error_message (str, nullable)
EvalRun: id (UUID str), run_timestamp (datetime), faithfulness_score (float), answer_relevancy_score (float), num_questions (int), passed (bool), raw_results (JSON str), triggered_by (str: manual / ci / auto_startup)
Call Base.metadata.create_all(bind=engine) on app startup.
5.3 ingestion/loader.py
load_document(file_path: str, file_type: str, doc_id: str) -> list[dict]
PDF: fitz.open() — extract text per page, include {page_num, source_file, doc_id}
MD: markdown lib + BeautifulSoup to strip tags → single block
HTML: BeautifulSoup → extract <article> or <main> or <body>, strip scripts/styles
TXT: raw open().read()
URL (detected by http:// or https:// prefix): requests.get() with 10s timeout + BeautifulSoup body extraction
Return: list[{"text": str, "metadata": {"source": str, "page": int, "doc_id": str, "file_type": str}}]
5.4 ingestion/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunk_id = f"{page['metadata']['doc_id']}_{page['metadata'].get('page', 0)}_{i}"
            chunks.append({
                "text": split,
                "chunk_id": chunk_id,
                "metadata": {**page["metadata"], "chunk_index": i, "chunk_id": chunk_id}
            })
    return chunks

5.5 retrieval/vector_store.py
Init chromadb.PersistentClient(path=settings.chroma_persist_dir)
Collection: "rag_documents" with embedding_function=None (manual embeddings)
upsert_chunks(chunks): embed in batches of 100 via embed_texts(), call collection.upsert(ids, embeddings, documents, metadatas)
query(query_text, top_k, doc_ids=None): embed query, call collection.query(query_embeddings, n_results, where={"doc_id": {"$in": doc_ids}} if doc_ids else None) → return [{text, metadata, score}]
delete_document(doc_id): collection.delete(where={"doc_id": doc_id})
get_all_chunks(): for BM25 index rebuild — collection.get(include=["documents", "metadatas"])
5.6 retrieval/bm25_store.py
import json
from rank_bm25 import BM25Okapi

CORPUS_PATH = "./bm25_corpus.json"

class BM25Store:
    def __init__(self):
        self.corpus: list[dict] = []
        self.index: BM25Okapi | None = None
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            with open(CORPUS_PATH) as f:
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
        with open(CORPUS_PATH, "w") as f:
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

5.7 retrieval/hybrid.py
Reciprocal Rank Fusion (RRF) — k=60:
def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, dict] = {}
    for rank, result in enumerate(vector_results):
        cid = result["metadata"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "data": result}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(bm25_results):
        cid = result["metadata"]["chunk_id"]
        if cid not in scores:
            scores[cid] = {"rrf_score": 0.0, "data": result}
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
    sorted_results = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [r["data"] for r in sorted_results]

5.8 retrieval/reranker.py
import cohere
from config import settings

def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []
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

5.9 Prompt Files
generation/prompts/system.txt — load at runtime, never hardcode:
You are a precise document assistant. Answer ONLY using the provided context chunks.

STRICT RULES:
1. Every factual claim MUST have a citation in this exact format: [Source: {filename}, Page: {page}, Chunk: {chunk_index}]
2. Place the citation immediately after the claim it supports.
3. NEVER introduce information not present in the provided chunks.
4. NEVER extrapolate or infer beyond what the chunks explicitly state.
5. If the chunks do not contain sufficient information to answer the question, respond EXACTLY with: "INSUFFICIENT_CONTEXT: The provided documents do not contain enough information to answer this question."
6. Be concise: answer in 2-4 paragraphs maximum.
7. Use plain language. No filler phrases.

generation/prompts/citation_enforce.txt:
Before finalizing your response, verify each claim:
- Does it have a citation?
- Does the citation reference a chunk actually provided to you?
- Is there any claim without grounding in the chunks?
Remove any ungrounded claims. Add missing citations. Only then output your final answer.

5.10 generation/chain.py — LangGraph RAG Chain
State:
class RAGState(TypedDict):
    query: str
    doc_ids: list[str] | None
    vector_results: list[dict]
    bm25_results: list[dict]
    hybrid_results: list[dict]
    reranked_chunks: list[dict]
    raw_answer: str
    citations: list[dict]
    has_sufficient_context: bool
    latency_ms: int

Nodes (implement all fully):
retrieve_vector(state) → call vector_store.query
retrieve_bm25(state) → call bm25_store.query
fuse_results(state) → RRF hybrid merge
rerank_chunks(state) → Cohere rerank → top K
generate_answer(state) → load system.txt + citation_enforce.txt, build prompt with chunks + query, call OpenRouter LLM via LangChain, return raw answer
extract_and_validate_citations(state) → parse citations from answer, verify against reranked_chunks
check_context_sufficiency(state) → set has_sufficient_context = False if answer starts with INSUFFICIENT_CONTEXT:
Wire as linear graph: 1→2→3→4→5→6→7. Compile with graph.compile().
Generate answer prompt template (inside generate_answer node):
CONTEXT CHUNKS:
{for each chunk: "--- Chunk [Source: {source}, Page: {page}, Chunk: {chunk_index}] ---\n{text}\n"}

QUESTION: {query}

{contents of citation_enforce.txt}

Answer:

5.11 generation/citations.py
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

5.12 API Routes
POST /documents/upload
Multipart file (PDF/TXT/MD/HTML) OR JSON {"url": "https://..."}
Validate extension. Save file to ./uploads/{doc_id}/original.{ext}
Create Document record (status=processing)
Run ingestion pipeline as BackgroundTask
Pipeline completion: update chunk_count, status=ready. On exception: status=failed, error_message=str(e)
Rebuild BM25 index after successful ingestion
Return: {"doc_id": str, "status": "processing"}
GET /documents
Return: [{id, filename, file_type, status, chunk_count, upload_timestamp, error_message}]
GET /documents/{doc_id}
Return single document status (used for polling)
DELETE /documents/{doc_id}
Delete from ChromaDB, SQLite. Rebuild BM25. Return {"deleted": true}
POST /query
Body: {"query": str, "doc_ids": list[str] | null}
Run RAG chain. Track start/end time for latency.
Return:
{
  "answer": "string",
  "has_sufficient_context": true,
  "citations": [{"source": "file.pdf", "page": 3, "chunk_index": 2, "verified": true, "excerpt": "..."}],
  "chunks_used": [{"text": "...", "metadata": {...}, "rerank_score": 0.92}],
  "latency_ms": 1240
}

POST /eval/run
Body: {"triggered_by": "manual"} (optional)
If golden_dataset.json has 0 questions, return 422 with message "Golden dataset not yet generated. POST /eval/generate first."
Run evaluation as BackgroundTask, return {"eval_id": str, "status": "running"}
GET /eval/results
Return last 20 eval runs sorted by timestamp desc
POST /eval/generate
Triggers dataset_generator.py to (re)generate the golden dataset from current ingested docs
Runs as BackgroundTask; return {"status": "generating"}
GET /health
Return {"status": "ok", "chroma": bool, "bm25_docs": int, "sqlite": bool}
5.13 main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings
from database import engine, Base
from api.routes import documents, query, eval

app = FastAPI(title="RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, tags=["query"])
app.include_router(eval.router, prefix="/eval", tags=["eval"])

@app.get("/health")
async def health(): ...

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.backend_host, port=settings.backend_port, reload=True)


6. Golden Dataset — Fully Automated Generation (No Human Input)
The Problem
RAGAS evaluation needs a golden dataset of verified QA pairs. The system must generate this itself from whatever documents are ingested, with no human involvement.
6.1 evaluation/dataset_generator.py
This module uses RAGAS TestsetGenerator to synthesize the golden dataset automatically from ingested documents.
import json
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import settings

logger = logging.getLogger(__name__)
GOLDEN_PATH = Path("./evaluation/golden_dataset.json")
TARGET_SIZE = 50

def get_ragas_llm():
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
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.embedding_model,
    ))

async def generate_golden_dataset(langchain_docs: list) -> dict:
    """
    Takes a list of LangChain Document objects (from already-ingested chunks).
    Generates 50 synthetic QA pairs using RAGAS TestsetGenerator.
    Saves to golden_dataset.json and returns the dataset dict.
    """
    if len(langchain_docs) < 5:
        logger.warning("Not enough documents to generate golden dataset (need at least 5)")
        return _empty_dataset()

    generator = TestsetGenerator(
        llm=get_ragas_llm(),
        embedding_model=get_ragas_embeddings(),
    )

    try:
        testset = generator.generate_with_langchain_docs(
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

        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Golden dataset generated: {len(questions)} QA pairs saved to {GOLDEN_PATH}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to generate golden dataset: {e}")
        return _empty_dataset()

def _empty_dataset() -> dict:
    return {"version": "1.0", "description": "Empty", "num_questions": 0, "questions": []}

def load_golden_dataset() -> dict:
    if GOLDEN_PATH.exists():
        with open(GOLDEN_PATH) as f:
            return json.load(f)
    return _empty_dataset()

6.2 Auto-generation Trigger
In POST /documents/upload background task pipeline:
# After successful ingestion of a document:
# 1. If total ingested doc count >= 1 AND golden_dataset.json has < 10 questions:
#    → Auto-trigger generate_golden_dataset() in background
# 2. This ensures the first document upload auto-builds the evaluation set
# 3. POST /eval/generate can be called manually to regenerate at any time

In generate_golden_dataset(), build langchain_docs from ALL currently stored chunks in ChromaDB:
# Get all chunks from vector store
all_chunks = vector_store.get_all_chunks()  # returns list of {text, metadata}
# Convert to LangChain Document objects
from langchain_core.documents import Document
langchain_docs = [
    Document(page_content=c["text"], metadata=c["metadata"])
    for c in all_chunks
]

6.3 evaluation/golden_dataset.json — Initial State
Ship this as the initial empty file. Gets populated automatically:
{
  "version": "1.0",
  "description": "Auto-generated golden dataset via RAGAS TestsetGenerator. Upload documents and call POST /eval/generate to populate.",
  "num_questions": 0,
  "questions": []
}


7. Evaluation — evaluation/eval_runner.py
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from evaluation.dataset_generator import load_golden_dataset, get_ragas_llm, get_ragas_embeddings
from config import settings

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


8. Frontend — Full Implementation Spec
8.1 Three-Panel Layout (page.tsx)
┌──────────────┬─────────────────────────────┬────────────────────┐
│ Documents    │       Chat                  │  Citations         │
│ (250px)      │       (flex-1)              │  (300px, toggle)   │
│              │                             │                    │
│ [Upload]     │  User: How does X work?     │ 📄 doc.pdf p.3 ✅  │
│              │                             │ "relevant excerpt" │
│ • doc1.pdf ✅│  AI: [answer with cites]    │                    │
│ • doc2.txt ✅│  Sources ▶                  │ 📄 doc.pdf p.7 ✅  │
│              │                             │ "another excerpt"  │
│              │ [Type your question...] [▶] │                    │
└──────────────┴─────────────────────────────┴────────────────────┘

8.2 ChatInterface.tsx
Message history with user (right) and assistant (left) bubbles
Show ⏳ Searching documents... spinner during API call
If has_sufficient_context === false: yellow banner "⚠️ Documents don't contain enough information to answer this."
Each assistant message: "📎 Sources (N)" toggle → show citation cards in right panel
Show latency badge: ⚡ 1.2s
Input: textarea + send button, Enter to send (Shift+Enter for newline)
8.3 DocumentUpload.tsx
Drag-and-drop zone: accept .pdf, .txt, .md, .html
URL input with "Ingest URL" button
Per-file upload progress bar
Poll GET /documents/{id} every 2s until status === "ready" or "failed"
Show ✅ ready / ❌ failed / ⏳ processing badge per document
After first doc reaches ready: show banner "🧠 Generating evaluation dataset in background..."
8.4 CitationCard.tsx
┌──────────────────────────────────────┐
│ 📄 filename.pdf — Page 3, Chunk 2   │
│ ✅ Verified  |  Score: 0.92          │
├──────────────────────────────────────┤
│ "First 200 characters of the chunk  │
│  text used as evidence..."           │
└──────────────────────────────────────┘

Unverified: ⚠️ badge + orange border.
8.5 EvalDashboard.tsx (/eval route)
Table of last 20 eval runs: timestamp, faithfulness, answer_relevancy, pass/fail
Green row = passed, red = failed
Progress bars for score columns
"🔄 Run Evaluation" button → POST /eval/run → poll for result every 3s
"🧠 Regenerate Dataset" button → POST /eval/generate
8.6 lib/api.ts
Fully typed TypeScript client:
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface QueryResponse {
  answer: string;
  has_sufficient_context: boolean;
  citations: Citation[];
  chunks_used: Chunk[];
  latency_ms: number;
}
// ... all interfaces matching backend Pydantic schemas
// ... typed fetch functions for all endpoints with proper error handling


9. Deployment Files
9.1 render.yaml (Render Blueprint)
services:
  - type: web
    name: rag-backend
    runtime: python
    region: oregon
    plan: starter                         # Starter required for persistent disk
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    rootDir: .
    healthCheckPath: /health
    envVars:
      - key: OPENROUTER_API_KEY
        sync: false                       # Must be set manually in Render dashboard
      - key: COHERE_API_KEY
        sync: false
      - key: APP_ENV
        value: production
      - key: LOG_LEVEL
        value: INFO
      - key: CHROMA_PERSIST_DIR
        value: /data/chroma_db
      - key: DATABASE_URL
        value: sqlite:////data/rag.db
      - key: FRONTEND_URL
        sync: false                       # Set to your Vercel URL after deploy
      - key: LLM_MODEL
        value: google/gemini-2.5-flash
      - key: EMBEDDING_MODEL
        value: openai/text-embedding-3-small
      - key: RERANK_MODEL
        value: rerank-english-v3.0
      - key: MIN_FAITHFULNESS_SCORE
        value: "0.80"
      - key: MIN_ANSWER_RELEVANCY_SCORE
        value: "0.75"
      - key: CHUNK_SIZE
        value: "700"
      - key: CHUNK_OVERLAP
        value: "100"
      - key: TOP_K_RETRIEVAL
        value: "20"
      - key: TOP_K_RERANK
        value: "5"
    disk:
      name: rag-persistent-storage
      mountPath: /data
      sizeGB: 10

Note: Persistent disk at /data stores both ChromaDB and SQLite. CHROMA_PERSIST_DIR=/data/chroma_db and DATABASE_URL=sqlite:////data/rag.db — note four slashes for absolute path in SQLAlchemy.
9.2 vercel.json (Vercel Config)
{
  "buildCommand": "cd frontend && npm install && npm run build",
  "outputDirectory": "frontend/.next",
  "installCommand": "cd frontend && npm install",
  "framework": "nextjs",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://rag-backend.onrender.com/:path*"
    }
  ],
  "env": {
    "NEXT_PUBLIC_API_URL": "https://rag-backend.onrender.com"
  }
}

Update https://rag-backend.onrender.com to actual Render service URL after first deploy.
9.3 backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data/chroma_db uploads
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

9.4 frontend/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;


10. CI/CD — .github/workflows/ci.yml
name: RAG CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r backend/requirements.txt
      - name: Run tests
        run: pytest backend/tests/ -v --tb=short
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}
          APP_ENV: test
          CHROMA_PERSIST_DIR: /tmp/test_chroma
          DATABASE_URL: sqlite:////tmp/test_rag.db

  eval:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r backend/requirements.txt
      - name: Run RAG Evaluation
        working-directory: backend
        run: |
          python -c "
          import asyncio, json, sys
          from evaluation.eval_runner import run_evaluation
          from evaluation.dataset_generator import load_golden_dataset
          from generation.chain import run_rag_chain

          dataset = load_golden_dataset()
          if dataset['num_questions'] == 0:
              print('No golden dataset available — skipping eval')
              sys.exit(0)

          async def main():
              results = await run_evaluation(run_rag_chain)
              with open('eval_results.json', 'w') as f:
                  json.dump(results, f)
              print('Eval results:', results)
              if not results['passed']:
                  print(f'EVAL FAILED: faithfulness={results[\"faithfulness\"]:.2f}, relevancy={results[\"answer_relevancy\"]:.2f}')
                  sys.exit(1)
              print('EVAL PASSED')

          asyncio.run(main())
          "
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}
          CHROMA_PERSIST_DIR: /tmp/ci_chroma
          DATABASE_URL: sqlite:////tmp/ci_rag.db


11. pyproject.toml / requirements.txt
Create backend/requirements.txt:
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
langchain>=0.2.0
langgraph>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langchain-core>=0.2.0
chromadb>=0.5.0
openai>=1.30.0
cohere>=5.5.0
rank-bm25>=0.2.2
PyMuPDF>=1.24.0
markdown>=3.6
beautifulsoup4>=4.12.0
requests>=2.32.0
sqlalchemy>=2.0.0
pydantic-settings>=2.2.0
ragas>=0.2.0
datasets>=2.19.0
python-multipart>=0.0.9
aiofiles>=23.2.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
httpx>=0.27.0


12. Tests — Minimum Coverage
tests/conftest.py
settings_override fixture: inject test env vars
tiny_pdf fixture: create a 2-page in-memory PDF via PyMuPDF
mock_openrouter_embed fixture: mock embed_texts() → returns list of 1536-dim zero vectors
mock_openrouter_llm fixture: mock LangChain ChatOpenAI → returns deterministic string
mock_cohere_rerank fixture: mock Cohere client → return chunks in same order
test_ingestion.py
PDF loader extracts text + page metadata
Chunker produces chunks ≤ chunk_size chars
Chunker generates unique chunk_id per chunk
Full pipeline fixture PDF → chunks stored in temp ChromaDB
test_retrieval.py
Vector store upsert then query returns correct doc
BM25 query returns results for known term
RRF merges deduplicates chunk_ids correctly
Reranker returns top_n results sorted by score descending
test_generation.py
Chain returns answer + citations for grounded query (mocked LLM returns text with citation tags)
Chain sets has_sufficient_context=False when answer is INSUFFICIENT_CONTEXT:...
extract_citations() parses [Source: doc.pdf, Page: 3, Chunk: 2] correctly
Unverified citations flagged with verified=False
test_api.py
POST /documents/upload accepts PDF, returns {doc_id, status: "processing"}
GET /documents returns list with doc
POST /query returns correct schema (mock chain)
DELETE /documents/{id} returns {deleted: true}
GET /health returns 200

13. README.md
Generate a complete README.md with:
# RAG System — Ask My Docs

## Architecture
[ASCII diagram showing: User → Vercel Frontend → Render Backend → OpenRouter → ChromaDB + BM25 + Cohere]

## Quick Start (Local)
cp .env.example .env    # fill OPENROUTER_API_KEY + COHERE_API_KEY
cd backend && pip install -r requirements.txt
uvicorn main:app --reload
cd frontend && npm install && npm run dev

## Deploy to Production
### Backend → Render
1. Push repo to GitHub
2. render.com → New → Blueprint → connect repo → detect render.yaml
3. In Render dashboard → rag-backend → Environment → add OPENROUTER_API_KEY + COHERE_API_KEY
4. Copy your Render URL (e.g. https://rag-backend.onrender.com)

### Frontend → Vercel
1. vercel.com → New Project → import repo
2. Set NEXT_PUBLIC_API_URL = https://rag-backend.onrender.com
3. Deploy

## Using the System
1. Upload PDFs/docs via the UI drag-drop zone
2. Wait for ✅ status
3. Ask questions in the chat — answers come with source citations
4. POST /eval/generate to auto-build evaluation dataset from your docs
5. POST /eval/run or visit /eval to see quality scores

## Evaluation
System auto-generates a 50-QA golden dataset using RAGAS TestsetGenerator from your documents.
No manual QA pairs needed. Re-generate any time by calling POST /eval/generate.
CI runs eval on every push to main. Build fails if faithfulness < 0.80 or relevancy < 0.75.


14. Implementation Notes for Codex
No placeholders. No TODOs. Every function fully implemented.
OpenRouter replaces OpenAI directly. Base URL = https://openrouter.ai/api/v1. API key env var = OPENROUTER_API_KEY. Headers must include HTTP-Referer and X-Title.
Prompts are files in generation/prompts/. Load with Path(__file__).parent / "prompts" / "system.txt".read_text()`. Never hardcode.
BM25 rebuilds after every ingest/delete. Persist corpus to ./bm25_corpus.json.
CORS must include both settings.frontend_url AND "https://*.vercel.app" to allow Vercel preview deployments.
Render disk mount at /data. Both ChromaDB and SQLite live there. SQLite URL uses four slashes: sqlite:////data/rag.db.
Render start command uses $PORT env var (Render injects this): uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Vercel rewrites proxy /api/* to the Render backend — frontend calls /api/query not the full Render URL directly.
Golden dataset auto-generates from first uploaded document batch. No user action needed.
RAGAS TestsetGenerator uses the same OpenRouter LLM and embeddings. Distribution: simple: 0.5, multi_context: 0.3, reasoning: 0.2.
Chunk IDs format: {doc_id}_{page}_{chunk_index} — deterministic, unique, used as ChromaDB document IDs.
Async everywhere: FastAPI async def, background tasks for ingestion and eval, asyncio.to_thread() for sync Cohere/ChromaDB calls.
Logging: Use Python logging module. Log: query text, vector count, BM25 count, after-rerank count, answer length, latency. Never print().
Error handling: 422 for validation, 404 for missing doc, 500 with {"detail": "..."} for unexpected. Never expose stack traces in production (APP_ENV=production → suppress detail).
uploads/ directory: create on startup if not exists. Files saved as uploads/{doc_id}/original.{ext}.
.gitignore must include: .env, chroma_db/, uploads/, *.db, bm25_corpus.json, __pycache__/, .next/, node_modules/, eval_results.json.

15. Credential Guide
🔑 Credential 1: OpenRouter API Key
Where: .env → OPENROUTER_API_KEY=sk-or-v1-...
 Also set in: Render dashboard → rag-backend → Environment Variables
How to get:
Go to https://openrouter.ai → Sign Up (Google/GitHub or email)
Top-right → Settings → API Keys
Click "Create Key" → name it rag-system → click Create
Copy the key (shown once — starts with sk-or-v1-)
Paste into .env
Add credits: Settings → Credits → Add credits ($5 minimum, ~5,000 queries with Gemini Flash)
 Free tier: OpenRouter has some free models. google/gemini-2.5-flash is pay-per-use (~$0.0001/query at low volume).
 GitHub Actions secret: Repo → Settings → Secrets → Actions → OPENROUTER_API_KEY

🔑 Credential 2: Cohere API Key
Where: .env → COHERE_API_KEY=...
 Also set in: Render dashboard → rag-backend → Environment Variables
How to get:
Go to https://dashboard.cohere.com → Sign Up (free, no credit card)
Left sidebar → API Keys
Click "New Trial Key" → name it rag-system → Create
Copy the key
Paste into .env
Free tier: Trial key = 100 reranks/min, more than enough for dev + demo. Upgrade to production key for >1000 req/day.
 GitHub Actions secret: Repo → Settings → Secrets → Actions → COHERE_API_KEY

🔑 No other credentials needed
Service
Credential
Reason
ChromaDB
❌ None
Local filesystem / Render disk
SQLite
❌ None
Local filesystem / Render disk
BM25
❌ None
In-memory + local JSON
Vercel
❌ None (for deploy)
GitHub OAuth during setup — no API key needed
Render
❌ None (for deploy)
GitHub OAuth during setup — no API key needed


16. Deployment Checklist (Post-Build)
# LOCAL DEV
cp .env.example .env
# → Fill OPENROUTER_API_KEY and COHERE_API_KEY
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000
cd frontend && npm install && npm run dev
# → Open http://localhost:3000

# PRODUCTION DEPLOY
# Step 1: Push to GitHub

# Step 2: Deploy backend on Render
# → render.com → New → Blueprint → select repo → it reads render.yaml automatically
# → In Render dashboard: rag-backend → Environment → Add:
#    OPENROUTER_API_KEY = sk-or-v1-...
#    COHERE_API_KEY = ...
#    FRONTEND_URL = https://your-app.vercel.app  (fill after Vercel deploy)
# → Render auto-deploys. Copy your URL: https://rag-backend.onrender.com

# Step 3: Deploy frontend on Vercel
# → vercel.com → Add New Project → Import from GitHub
# → Framework: Next.js (auto-detected)
# → Root directory: frontend
# → Environment Variables → Add: NEXT_PUBLIC_API_URL = https://rag-backend.onrender.com
# → Deploy. Copy your URL: https://your-app.vercel.app

# Step 4: Update Render FRONTEND_URL
# → Render dashboard → rag-backend → Environment → FRONTEND_URL = https://your-app.vercel.app
# → Trigger redeploy

# Step 5: Test
# → Open https://your-app.vercel.app
# → Upload a PDF
# → Ask a question
# → POST https://rag-backend.onrender.com/eval/generate  (auto-builds golden dataset)
# → Visit https://your-app.vercel.app/eval → Run Evaluation


PRD Version: 2.0 | OpenRouter + Vercel + Render | Auto-Generated Golden Dataset | Ready for Codex

