"""Phase 1 evaluation harness -- MEASURES the Phase 0 baseline, changes nothing.

Runs every row of eval/eval_set.csv through the existing pipeline
(src/rag.py: retrieve()/answer()) and scores retrieval and generation
SEPARATELY. This file only observes the current bot's behaviour (including
whether it refuses on hors_corpus questions); it does not add any refusal
or anti-hallucination logic, does not touch chunking/top_k/the prompt, and
does not rebuild the index.

Usage:
    python eval/run_eval.py                  # full run: retrieval + generation
    python eval/run_eval.py --retrieval-only  # retrieve() only, no LLM, fast
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from rag import answer, retrieve  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

KNOWN_UNITS = {"tnd", "ects", "heures", "hours", "weeks", "semaines"}

REFUSAL_MARKERS = [
    "je ne sais pas",
    "pas d'information",
    "aucune information",
    "aucune mention",
    "il n'y a pas",
    "ne figure pas",
    "n'est pas mentionne",
    "not available",
    "no information",
    "not mentioned",
    "cannot find",
    "no mention",
    "n'est mentionne",
    "aucun module",
    "ne contient pas",
    "ne mentionne",
    "pas de mention",
    "n'est pas disponible",
]


def normalize(text: str | None) -> str:
    """lowercase; strip accents; drop '>='/'≥'; decimal comma -> dot;
    drop thousands-separator commas; drop all whitespace."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace(">=", "").replace("≥", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # decimal comma: comma followed by 1-2 digits then a non-digit/end (e.g. "31,5")
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?:\D|$))", ".", s)
    # remaining digit-to-digit commas are thousands separators (e.g. "1,500")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"\s+", "", s)
    return s


_NUMERIC_FORM_RE = re.compile(r"^(\d+(?:\.\d+)?)([a-z]*)$")


def classify_form(raw_form: str) -> tuple[str, str]:
    """Return ("numeric", digits) or ("text", normalized_string).

    A form is "numeric" only if, once normalized, it is a plain number
    optionally glued to a known unit word (TND, ECTS, heures, hours, weeks,
    semaines). Everything else (emails, "9eme semestre", "S9", "08/20",
    "Good", dates...) is matched as a normalized substring instead.
    """
    norm = normalize(raw_form)
    m = _NUMERIC_FORM_RE.match(norm)
    if m:
        num, unit = m.group(1), m.group(2)
        if unit == "" or unit in KNOWN_UNITS:
            return ("numeric", num)
    return ("text", norm)


def numeric_match(num: str, haystack_norm: str) -> bool:
    """Standalone-digit match: '4' must not match inside '42' or '2024'.

    A trailing sentence-ending dot (e.g. "...2024.") is allowed; only a
    decimal dot (dot followed by a digit, e.g. "31.5") blocks the match.
    """
    pattern = r"(?<![\d.])" + re.escape(num) + r"(?!\d)(?!\.\d)"
    return re.search(pattern, haystack_norm) is not None


def text_match(form_norm: str, haystack_norm: str) -> bool:
    return form_norm != "" and form_norm in haystack_norm


def is_refusal(answer_text: str) -> bool:
    norm = normalize(answer_text)
    return any(normalize(marker) in norm for marker in REFUSAL_MARKERS)


def check_generation(
    answer_text: str, valeur_attendue: str, category: str, reject_if: str
) -> tuple[bool, str]:
    """Generation metric is a PROXY: "the correct value appears in the
    answer", NOT "the answer is perfect". An answer can contain the right
    value plus extra wrong content and still pass -- a known, accepted
    limitation. The full answer is always saved so failures (and the
    hors_corpus refusal heuristic, which can miss a refusal that then
    volunteers unrelated info) can be reviewed by a human.

    ``reject_if`` is an escape hatch for that proxy limitation: a set of
    "almost-right" enclosing labels (e.g. "Pretty good" enclosing "Good")
    that must fail the row even though the accepted value is a substring
    of them. Checked before the value match, non-hors_corpus rows only.
    """
    if category == "hors_corpus":
        return is_refusal(answer_text), ("refusal" if is_refusal(answer_text) else "")

    haystack_norm = normalize(answer_text)

    if reject_if:
        for rejected in reject_if.split("|"):
            rejected = rejected.strip()
            if rejected and normalize(rejected) in haystack_norm:
                return False, ""

    for alt in valeur_attendue.split("|"):
        alt = alt.strip()
        if not alt:
            continue
        kind, key = classify_form(alt)
        if kind == "numeric":
            if numeric_match(key, haystack_norm):
                return True, alt
        else:
            if text_match(key, haystack_norm):
                return True, alt
    return False, ""


def item_coverage(expected_items: str, haystack: str) -> tuple[int, int, list[str]]:
    """guide_list rows: how many of the '|'-separated expected items appear
    (normalized substring, same normalize() as the other rows) in haystack.
    Returns (found, N, missing_items)."""
    items = [it.strip() for it in expected_items.split("|") if it.strip()]
    haystack_norm = normalize(haystack)
    missing = [it for it in items if not text_match(normalize(it), haystack_norm)]
    return len(items) - len(missing), len(items), missing


def _expected_page(source_attendue: str, page_attendue: str) -> int | None:
    """Expected page as int, or None when the row has no usable source/page."""
    if not source_attendue or not page_attendue:
        return None
    try:
        return int(page_attendue)
    except ValueError:
        return None


def _same_source_page(source, page, source_attendue: str, expected_page: int) -> bool:
    """Shared source+page comparison for retrieval and citation scoring."""
    if source != source_attendue:
        return False
    try:
        return int(page) == expected_page
    except (TypeError, ValueError):
        return False


def check_retrieval(chunks, source_attendue: str, page_attendue: str, category: str) -> str:
    if category == "hors_corpus":
        return "n/a"
    expected_page = _expected_page(source_attendue, page_attendue)
    if expected_page is None:
        return "n/a"
    for chunk in chunks:
        meta = chunk.metadata
        if _same_source_page(meta.get("source_file"), meta.get("page"), source_attendue, expected_page):
            return "yes"
    return "no"


def check_citation(citation, source_attendue: str, page_attendue: str, category: str) -> str:
    """hors_corpus: pass iff no citation. Other rows: pass iff the citation
    points at the expected source+page (same comparison as check_retrieval)."""
    if category == "hors_corpus":
        return "yes" if citation is None else "no"
    expected_page = _expected_page(source_attendue, page_attendue)
    if expected_page is None:
        return "n/a"
    if citation is None:
        return "no"
    ok = _same_source_page(citation.get("source"), citation.get("page"), source_attendue, expected_page)
    return "yes" if ok else "no"


def format_retrieved_pages(chunks) -> str:
    parts = []
    for c in chunks:
        source = c.metadata.get("source_file", "unknown")
        page = c.metadata.get("page", "?")
        parts.append(f"{source}:{page}")
    return "[" + ", ".join(parts) + "]"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_results(results: list[dict], path: Path) -> None:
    fieldnames = [
        "id",
        "category",
        "retrieval_ok",
        "generation_ok",
        "matched_value",
        "retrieved_pages",
        "answer",
        "cited_source",
        "cited_page",
        "citation_ok",
        "retrieval_coverage",
        "answer_coverage",
        "missing_retrieval",
        "missing_answer",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def print_summary(results: list[dict], retrieval_only: bool) -> None:
    categories = ["jsonl_plan", "guide_prose", "guide_table", "hors_corpus"]
    print("\n=== Summary by category ===")
    header = f"{'category':<12} {'N':>3}  {'retrieval':>12}" + ("" if retrieval_only else f"  {'generation':>12}")
    print(header)

    total_n = 0
    total_retrieval_pass = 0
    total_retrieval_n = 0
    total_generation_pass = 0
    total_generation_n = 0

    for cat in categories:
        rows = [r for r in results if r["category"] == cat]
        if not rows:
            continue
        n = len(rows)
        total_n += n

        retrieval_rows = [r for r in rows if r["retrieval_ok"] in ("yes", "no")]
        retrieval_pass = sum(1 for r in retrieval_rows if r["retrieval_ok"] == "yes")
        total_retrieval_pass += retrieval_pass
        total_retrieval_n += len(retrieval_rows)
        retrieval_str = f"{retrieval_pass}/{len(retrieval_rows)}" if retrieval_rows else "n/a"

        line = f"{cat:<12} {n:>3}  {retrieval_str:>12}"

        if not retrieval_only:
            generation_rows = [r for r in rows if r["generation_ok"] in ("yes", "no")]
            generation_pass = sum(1 for r in generation_rows if r["generation_ok"] == "yes")
            total_generation_pass += generation_pass
            total_generation_n += len(generation_rows)
            generation_str = f"{generation_pass}/{len(generation_rows)}" if generation_rows else "n/a"
            line += f"  {generation_str:>12}"

        print(line)

    total_line = f"{'TOTAL':<12} {total_n:>3}  {f'{total_retrieval_pass}/{total_retrieval_n}':>12}"
    if not retrieval_only:
        total_line += f"  {f'{total_generation_pass}/{total_generation_n}':>12}"
    print(total_line)

    if not retrieval_only:
        hors_corpus_rows = [r for r in results if r["category"] == "hors_corpus"]
        refusal_pass = sum(1 for r in hors_corpus_rows if r["generation_ok"] == "yes")
        print(f"\nrefusal (hors_corpus): {refusal_pass}/{len(hors_corpus_rows)}")

        cited_rows = [
            r for r in results
            if r["category"] != "hors_corpus" and r["citation_ok"] in ("yes", "no")
        ]
        cited_pass = sum(1 for r in cited_rows if r["citation_ok"] == "yes")
        uncited_pass = sum(1 for r in hors_corpus_rows if r["citation_ok"] == "yes")
        print(f"citations correctes : {cited_pass}/{len(cited_rows)}")
        print(f"refus sans citation : {uncited_pass}/{len(hors_corpus_rows)}")

    list_rows = [r for r in results if r["category"] == "guide_list"]
    if list_rows:
        print(f"\n=== Couverture listes (guide_list, TOP_K={config.TOP_K}) ===")
        for r in list_rows:
            line = f"Q{r['id']} -- recup {r['retrieval_coverage']}"
            if not retrieval_only:
                line += f", reponse {r['answer_coverage']}"
            print(f"{line} -- {r['question']}")
            if r["missing_retrieval"]:
                print(f"    manquants recup  : {r['missing_retrieval']}")
            if not retrieval_only and r["missing_answer"]:
                print(f"    manquants reponse: {r['missing_answer']}")

    failed = [
        r
        for r in results
        if r["retrieval_ok"] == "no" or (not retrieval_only and r["generation_ok"] == "no")
    ]
    print("\n=== Failed rows ===")
    if not failed:
        print("(none)")
    for r in failed:
        print(
            f"id {r['id']} [{r['category']}] "
            f"retrieval={r['retrieval_ok']} generation={r['generation_ok']} "
            f"-- {r['question']}"
        )


def run(retrieval_only: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_eval_set()
    n = len(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"results_{timestamp}_k{config.TOP_K}.csv"
    print(f"TOP_K={config.TOP_K}")

    results: list[dict] = []

    for i, row in enumerate(rows, start=1):
        row_id = row["id"]
        question = row["question"]
        category = row["category"]
        source_attendue = row["source_attendue"]
        page_attendue = row["page_attendue"]
        valeur_attendue = row["valeur_attendue"]
        reject_if = row.get("reject_if", "")
        expected_items = row.get("expected_items", "")
        retrieval_coverage = answer_coverage = ""
        missing_retrieval = missing_answer = ""

        try:
            if retrieval_only:
                chunks = retrieve(question)
                answer_text = ""
                generation_ok = "skipped"
                matched_value = ""
                cited_source = cited_page = ""
                citation_ok = "skipped"
            else:
                result = answer(question)
                chunks = result["chunks"]
                answer_text = result["answer"]
                gen_ok, matched_value = check_generation(
                    answer_text, valeur_attendue, category, reject_if
                )
                generation_ok = "yes" if gen_ok else "no"
                citation = result["citation"]
                cited_source = citation["source"] if citation else ""
                cited_page = citation["page"] if citation else ""
                citation_ok = check_citation(citation, source_attendue, page_attendue, category)

            retrieval_ok = check_retrieval(chunks, source_attendue, page_attendue, category)
            retrieved_pages = format_retrieved_pages(chunks)

            if category == "guide_list":
                # Coverage replaces the single-value metrics: keep these rows
                # out of the 19-row summary (retrieval/generation/citations).
                retrieval_ok = generation_ok = citation_ok = "n/a"
                found, total, missing = item_coverage(
                    expected_items, "\n".join(c.page_content for c in chunks)
                )
                retrieval_coverage = f"{found}/{total}"
                missing_retrieval = "|".join(missing)
                if not retrieval_only:
                    found, total, missing = item_coverage(expected_items, answer_text)
                    answer_coverage = f"{found}/{total}"
                    missing_answer = "|".join(missing)

        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the run
            retrieval_ok = "error"
            generation_ok = "error"
            matched_value = ""
            retrieved_pages = ""
            answer_text = f"error: {exc}"
            cited_source = cited_page = ""
            citation_ok = "error"

        results.append(
            {
                "id": row_id,
                "category": category,
                "question": question,
                "retrieval_ok": retrieval_ok,
                "generation_ok": generation_ok,
                "matched_value": matched_value,
                "retrieved_pages": retrieved_pages,
                "answer": answer_text,
                "cited_source": cited_source,
                "cited_page": cited_page,
                "citation_ok": citation_ok,
                "retrieval_coverage": retrieval_coverage,
                "answer_coverage": answer_coverage,
                "missing_retrieval": missing_retrieval,
                "missing_answer": missing_answer,
            }
        )

        write_results(results, out_path)

        progress = f"[{i}/{n}] id={row_id} retrieval={retrieval_ok} generation={generation_ok}"
        if category == "guide_list":
            progress += f" coverage recup={retrieval_coverage} reponse={answer_coverage or 'skipped'}"
        print(progress)

    print(f"\nResults saved to {out_path}")
    print_summary(results, retrieval_only)


def _selftest() -> None:
    """Quick regression check for numeric_match's dot-boundary handling."""
    assert numeric_match("2024", normalize("lance en juin 2024.")) is True, (
        "sentence-ending dot after a number must still match"
    )
    assert numeric_match("2024", normalize("42")) is False, (
        "'2024' must not match inside '42'"
    )
    assert numeric_match("31", normalize("31.5")) is False, (
        "decimal dot must still block a partial match ('31' in '31.5')"
    )

    reject_if = "pretty good|very good|assez bien|tres bien"
    ok, _ = check_generation('correspond a "Pretty good"', "Good|Bien", "guide_table", reject_if)
    assert ok is False, "an enclosing 'almost-right' label in reject_if must fail the row"
    ok, _ = check_generation("la mention est Good", "Good|Bien", "guide_table", reject_if)
    assert ok is True, "the accepted value alone (no reject_if label) must still pass"

    assert item_coverage("CESI|INP Bordeaux|INP Lorraine", "CESI (France), INP  bordeaux") == (
        2, 3, ["INP Lorraine"]
    ), "item_coverage must use normalize() substring matching"

    print("numeric_match selftest: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 evaluation harness")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="retrieve() only, no LLM call -- fast loop for tuning retrieval",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run quick numeric_match boundary checks and exit (no LLM, no retrieval)",
    )
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    run(retrieval_only=args.retrieval_only)


if __name__ == "__main__":
    main()
