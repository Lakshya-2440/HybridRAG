from __future__ import annotations

import logging
from pathlib import Path

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
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
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
