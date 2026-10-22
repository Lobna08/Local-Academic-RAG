"""Rebuild the vector index from scratch.

    python scripts/build_index.py

Loads both sources, prints a chunk count per source, embeds everything into the
single Chroma store, and prints the skipped-pages summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import load_jsonl_documents, load_pdf_documents  # noqa: E402
from vectorstore import build_vectorstore  # noqa: E402


def main() -> None:
    print("Loading JSONL (plan d'etudes)...")
    jsonl_docs = load_jsonl_documents()
    print(f"  JSONL documents: {len(jsonl_docs)}")

    print("Parsing + splitting PDF (Students_Guide.pdf)...")
    pdf_chunks, skipped_pages = load_pdf_documents()
    print(f"  PDF chunks: {len(pdf_chunks)}")

    all_docs = jsonl_docs + pdf_chunks
    print(f"Embedding {len(all_docs)} documents into Chroma...")
    build_vectorstore(all_docs)

    print("Done.")
    print(
        f"Skipped PDF pages ({len(skipped_pages)}): "
        f"{skipped_pages if skipped_pages else 'none'}"
    )


if __name__ == "__main__":
    main()
