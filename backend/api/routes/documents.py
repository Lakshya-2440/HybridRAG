from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

try:
    from api.schemas import DeleteResponse, DocumentResponse, UploadResponse
    from database import Document, get_db
    from ingestion.pipeline import ingest_document_sync
    from retrieval.bm25_store import bm25_store
    from retrieval.vector_store import vector_store
    from config import settings
except ImportError:  # pragma: no cover
    from ..schemas import DeleteResponse, DocumentResponse, UploadResponse
    from ...database import Document, get_db
    from ...ingestion.pipeline import ingest_document_sync
    from ...retrieval.bm25_store import bm25_store
    from ...retrieval.vector_store import vector_store
    from ...config import settings

router = APIRouter()
UPLOAD_DIR = Path(settings.upload_dir)
ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "html", "htm"}


def _document_to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        status=document.status,
        chunk_count=document.chunk_count,
        upload_timestamp=document.upload_timestamp,
        error_message=document.error_message,
    )


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {ext or 'unknown'}",
        )
    return "html" if ext == "htm" else ext


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    doc_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if file is not None:
        file_type = _validate_extension(file.filename or "")
        doc_dir = UPLOAD_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / f"original.{file_type}"
        file_path.write_bytes(await file.read())
        filename = file.filename or file_path.name
        source_path = str(file_path)
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Provide multipart file or JSON {'url': 'https://...'}") from exc
        url = payload.get("url", "")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=422, detail="Invalid URL")
        file_type = "url"
        filename = parsed.netloc
        source_path = url

    document = Document(
        id=doc_id,
        filename=filename,
        file_type=file_type,
        status="processing",
        chunk_count=0,
    )
    db.add(document)
    db.commit()

    background_tasks.add_task(ingest_document_sync, doc_id, source_path, file_type)
    return UploadResponse(doc_id=doc_id, status="processing")


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)) -> list[DocumentResponse]:
    documents = db.query(Document).order_by(Document.upload_timestamp.desc()).all()
    return [_document_to_response(document) for document in documents]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    document = db.get(Document, doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return _document_to_response(document)


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    document = db.get(Document, doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    vector_store.delete_document(doc_id)
    db.delete(document)
    db.commit()
    bm25_store.rebuild_from_vector_store(vector_store.get_all_chunks())
    return DeleteResponse(deleted=True)
