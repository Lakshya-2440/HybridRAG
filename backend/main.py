from __future__ import annotations

import logging
from pathlib import Path

# Pre-import all openai resources to prevent Python 3.14 import lock deadlocks in multithreaded async paths
try:
    import openai
    import openai.resources
    import openai.resources.chat
    import openai.resources.embeddings
    import openai.resources.models
    import openai.resources.files
    import openai.resources.batches
    import langchain_openai
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

try:
    from api.routes import documents, eval, query
    from config import settings
    from database import Base, create_session, engine
    from retrieval.bm25_store import bm25_store
    from retrieval.vector_store import vector_store
except ImportError:  # pragma: no cover
    from .api.routes import documents, eval, query
    from .config import settings
    from .database import Base, create_session, engine
    from .retrieval.bm25_store import bm25_store
    from .retrieval.vector_store import vector_store

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

# Ensure storage directories exist
for dir_path in [settings.upload_dir, settings.chroma_persist_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
# Ensure parent dir for sqlite db and bm25 corpus
Path(settings.bm25_corpus_path).parent.mkdir(parents=True, exist_ok=True)
db_path = settings.database_url.replace("sqlite:///", "")
if db_path:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG System", version="1.0.0")

# CORS: allow frontend origin + any localhost for dev + Vercel preview URLs
allowed_origins = [settings.frontend_url]
if settings.app_env != "production":
    allowed_origins += [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, tags=["query"])
app.include_router(eval.router, prefix="/eval", tags=["eval"])


@app.get("/health")
async def health():
    sqlite_ok = False
    db = create_session()
    try:
        db.execute(text("SELECT 1"))
        sqlite_ok = True
    finally:
        db.close()

    chroma_ok = False
    try:
        vector_store.collection.count()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    return {
        "status": "ok",
        "chroma": chroma_ok,
        "bm25_docs": len(bm25_store.corpus),
        "sqlite": sqlite_ok,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
