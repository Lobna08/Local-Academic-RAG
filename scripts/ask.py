"""Ask one question against the persisted index (no rebuild).

    python scripts/ask.py "Quelle moyenne faut-il pour valider un panier ?"

Prints the answer, then each retrieved chunk as [source_file p.X] with a preview.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import answer  # noqa: E402

PREVIEW_CHARS = 200


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python scripts/ask.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer(question)

    print("\n=== Answer ===")
    print(result["answer"])
    citation = result["citation"]
    if citation is not None:
        print(f"Source : {citation['source']}, page {citation['page']}")

    print("\n=== Retrieved chunks ===")
    for chunk in result["chunks"]:
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", "?")
        preview = " ".join(chunk.page_content.split())[:PREVIEW_CHARS]
        print(f"[{source} p.{page}] {preview}")


if __name__ == "__main__":
    main()
