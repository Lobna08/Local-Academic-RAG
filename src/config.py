"""Central configuration: every path, model name, and tunable parameter lives here.

Phase 0 baseline. The numeric parameters below are deliberate placeholders --
they are meant to be measured and tuned in a later phase, not optimized now.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# --- Environment overrides ---------------------------------------------------
# Every field can be overridden by an environment variable of the same name
# (case-insensitive) or by a .env file at the project root. The defaults are
# the Phase 0 values, so nothing changes when no variable is set.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "llama3.1:8b"
    embed_model: str = "bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    chroma_dir: Path = BASE_DIR / "chroma_store"
    top_k: int = 8
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_page_chars: int = 50
    guide_table_pages: list[int] = [7, 13, 19]


settings = Settings()

JSONL_PATH = DATA_DIR / "plan_etudes_sdia.jsonl"
PDF_PATH = DATA_DIR / "Students_Guide.pdf"

# Persistent Chroma store (single store for both ingestion paths).
CHROMA_DIR = settings.chroma_dir
COLLECTION_NAME = "academic_rag"

# --- Chunking / retrieval (tunable later -- do NOT optimize in Phase 0) ------
CHUNK_SIZE = settings.chunk_size          # characters per PDF chunk
CHUNK_OVERLAP = settings.chunk_overlap    # character overlap between consecutive PDF chunks
TOP_K = settings.top_k                    # number of chunks retrieved per question
MIN_PAGE_CHARS = settings.min_page_chars  # PDF pages with less text than this are skipped (image-only)

# --- Guide tables (Phase 2: row-to-sentence) ---------------------------------
GUIDE_TABLE_PAGES = settings.guide_table_pages  # 1-indexed pages whose tables become sentences

# --- Models (served by local Ollama) ---------------------------------------
EMBED_MODEL = settings.embed_model          # embeddings for both ingestion and queries
LLM_MODEL = settings.llm_model              # answer generation
OLLAMA_BASE_URL = settings.ollama_base_url  # Ollama server for both models
