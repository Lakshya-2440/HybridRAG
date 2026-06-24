import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    cohere_api_key: str = ""
    app_env: str = "development"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    chroma_persist_dir: str = "./chroma_db"
    database_url: str = "sqlite:///./rag.db"
    upload_dir: str = "./uploads"
    bm25_corpus_path: str = "./bm25_corpus.json"
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
    use_remote_models: bool = False
    auto_generate_eval_dataset: bool = False

    class Config:
        env_file = (".env", "../.env")
        extra = "ignore"


settings = Settings()

# Force relative paths for persistent storage only in production (Render)
if settings.app_env == "production":
    settings.chroma_persist_dir = "./data/chroma_db"
    settings.database_url = "sqlite:///./data/rag.db"
    settings.upload_dir = "./data/uploads"
    settings.bm25_corpus_path = "./data/bm25_corpus.json"
