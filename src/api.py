"""Thin HTTP layer over the RAG core: reuses rag.answer() as-is, no RAG logic here.

Run from the project root:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# rag.py uses flat imports (`import config`), so src/ must be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import answer  # noqa: E402

SNIPPET_CHARS = 160

app = FastAPI(
    title="TEK-UP RAG API",
    description="Assistant RAG local de conseil académique — SDIA",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source: str
    page: int


class RetrievedSource(BaseModel):
    source: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citation: Citation | None
    retrieved: list[RetrievedSource]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, summary="Poser une question au RAG")
def ask(req: AskRequest) -> AskResponse:
    """Répond à une question à partir du plan d'études SDIA et du Students Guide.

    Renvoie la réponse du LLM, la citation (source + page) du chunk qui l'appuie
    — `null` en cas de refus — et les chunks récupérés avec un extrait.
    """
    try:
        result = answer(req.question)
        return AskResponse(
            answer=result["answer"],
            citation=Citation(**result["citation"]) if result["citation"] else None,
            retrieved=[
                RetrievedSource(
                    source=chunk.metadata["source_file"],
                    page=chunk.metadata["page"],
                    snippet=chunk.page_content[:SNIPPET_CHARS],
                )
                for chunk in result["chunks"]
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}") from exc
