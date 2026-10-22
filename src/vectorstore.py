"""Build (persist) and load the single on-disk Chroma store.

Embeddings are served by Ollama (bge-m3) -- no local torch / sentence-transformers.
"""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def build_vectorstore(documents: list[Document]) -> Chroma:
    """Rebuild the store from scratch: wipe the directory, embed, persist."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    return Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )


def load_vectorstore() -> Chroma:
    """Open the already-built store for querying (no re-embedding of documents)."""
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma store not found at {config.CHROMA_DIR}. "
            "Run `python scripts/build_index.py` first."
        )

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
