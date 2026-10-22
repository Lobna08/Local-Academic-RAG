"""Phase 0 RAG core: retrieval and generation, kept separately callable.

Interface (relied on by the next phase):
    retrieve(question, k=TOP_K) -> list[Document]      # raw chunks, no LLM call
    answer(question)            -> {"answer": str, "chunks": list[Document],
                                    "citation": {"source": str, "page": int} | None}
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

import config
from vectorstore import load_vectorstore

PROMPT_TEMPLATE = (
    "Answer the question using only the context below.\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

_REFUSAL_MARKERS = (
    # FR
    "je ne sais pas", "ne sais pas", "pas d'information", "aucune information",
    "aucune mention", "ne figure pas", "n'est pas mentionne", "ne mentionne pas",
    "il n'y a pas",
    # EN (the prompt is in English, the system is bilingual)
    "i don't know", "i do not know", "not mentioned", "no information", "not available",
)


@lru_cache(maxsize=1)
def _vectorstore():
    """Load the persisted store once per process."""
    return load_vectorstore()


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)


def retrieve(question: str, k: int = config.TOP_K) -> list[Document]:
    """Return the top-k retrieved chunks (with metadata). No LLM involved."""
    return _vectorstore().similarity_search(question, k=k)


def _format_context(chunks: list[Document]) -> str:
    """One block per chunk, each tagged with its source file and page."""
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", "?")
        blocks.append(f"[{source} p.{page}]\n{chunk.page_content}")
    return "\n\n".join(blocks)


def _normalize(text: str) -> str:
    """Lowercase, strip accents (NFKD), unify typographic apostrophes."""
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c))


def _is_refusal(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in _REFUSAL_MARKERS)


def _tokens(text: str) -> set[str]:
    """Significant tokens: length >= 3 or purely numeric, punctuation as separator."""
    words = re.sub(r"[^\w]|_", " ", _normalize(text)).split()
    return {w for w in words if len(w) >= 3 or w.isdigit()}


def _best_chunk(answer_text: str, chunks: list[Document]) -> Document | None:
    """Chunk sharing the most rarity-weighted tokens with the answer.

    Each shared token weighs 1/df, df = number of retrieved chunks containing it,
    so a distinctive token outweighs words common to many chunks (and stopwords
    fade out on their own). Ties go to the better-ranked chunk; score 0 -> None.
    """
    answer_tokens = _tokens(answer_text)
    if not answer_tokens or not chunks:
        return None
    chunk_tokens = [_tokens(chunk.page_content) for chunk in chunks]
    df = Counter(token for tokens in chunk_tokens for token in tokens)

    best, best_score = None, 0.0
    for chunk, tokens in zip(chunks, chunk_tokens):
        score = sum(1 / df[token] for token in answer_tokens & tokens)
        if score > best_score:
            best, best_score = chunk, score
    return best


def answer(question: str) -> dict:
    """Retrieve, build context, call the LLM; return the answer, chunks and citation.

    The citation comes from the metadata of the retrieved chunk that best matches
    the answer text, never from the LLM. None on a refusal or when nothing matches.
    """
    chunks = retrieve(question)
    prompt = PROMPT_TEMPLATE.format(
        context=_format_context(chunks),
        question=question,
    )
    text = _llm().invoke(prompt).content

    citation = None
    if not _is_refusal(text):
        best = _best_chunk(text, chunks)
        if best is not None:
            citation = {
                "source": best.metadata.get("source_file"),
                "page": best.metadata.get("page"),
            }
    return {"answer": text, "chunks": chunks, "citation": citation}
