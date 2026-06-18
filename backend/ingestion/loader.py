from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import fitz
import markdown
import requests
from bs4 import BeautifulSoup


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    return root.get_text("\n", strip=True)


def _metadata(source: str, page: int, doc_id: str, file_type: str) -> dict:
    return {"source": source, "page": page, "doc_id": doc_id, "file_type": file_type}


def _normalize_type(file_path: str, file_type: str) -> str:
    if file_path.startswith(("http://", "https://")):
        return "url"
    ft = file_type.lower().lstrip(".")
    if not ft:
        ft = Path(file_path).suffix.lower().lstrip(".")
    if ft == "htm":
        return "html"
    return ft


def load_document(file_path: str, file_type: str, doc_id: str) -> list[dict]:
    normalized_type = _normalize_type(file_path, file_type)

    if normalized_type == "url":
        response = requests.get(file_path, timeout=10)
        response.raise_for_status()
        text = _clean_html(response.text)
        source = urlparse(file_path).netloc or file_path
        return [{"text": text, "metadata": _metadata(source, 1, doc_id, "url")}]

    path = Path(file_path)
    source = path.name

    if normalized_type == "pdf":
        pages = []
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                pages.append(
                    {
                        "text": page.get_text("text"),
                        "metadata": _metadata(source, page_index, doc_id, "pdf"),
                    }
                )
        return pages

    if normalized_type == "md":
        raw = path.read_text(encoding="utf-8")
        text = _clean_html(markdown.markdown(raw))
        return [{"text": text, "metadata": _metadata(source, 1, doc_id, "md")}]

    if normalized_type == "html":
        text = _clean_html(path.read_text(encoding="utf-8"))
        return [{"text": text, "metadata": _metadata(source, 1, doc_id, "html")}]

    if normalized_type == "txt":
        return [
            {
                "text": path.read_text(encoding="utf-8"),
                "metadata": _metadata(source, 1, doc_id, "txt"),
            }
        ]

    raise ValueError(f"Unsupported file type: {file_type}")
