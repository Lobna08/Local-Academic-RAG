"""Turn the two raw sources into a flat ``list[Document]`` ready for embedding.

Two independent paths, no embedding here:

1. ``plan_etudes_sdia.jsonl`` -- already processed. One line -> one Document,
   consumed as-is. No splitting, no re-parsing.
2. ``Students_Guide.pdf`` -- English prose. Text extracted page by page with
   PyMuPDF, then split with a single naive RecursiveCharacterTextSplitter.
   Exception (Phase 2): the tables on ``config.GUIDE_TABLE_PAGES`` are detected
   with ``page.find_tables()``, validated, and turned into one self-contained
   sentence per row (like the JSONL). Their text is removed from the prose so
   it is not indexed twice.
"""

from __future__ import annotations

import json
import re

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

PDF_SOURCE_NAME = "Students_Guide.pdf"

# A text block is dropped from the prose when at least this share of its area
# lies inside a detected table bbox.
TABLE_OVERLAP_RATIO = 0.5

EMAIL_RE = re.compile(r"^[\w.+&-]+@[\w-]+(\.[\w-]+)+$")

# Tables expected on each page, in detection order (top to bottom).
# Each entry: (table id used in logs, metadata "table" value).
EXPECTED_TABLES: dict[int, list[tuple[str, str]]] = {
    7: [("pfe_grades", "pfe_grades")],
    13: [("partners", "partners")],
    19: [
        ("contacts_1", "contacts"),
        ("contacts_2", "contacts"),
        ("contacts_3", "contacts"),
    ],
}

# Verified fallback data, used for a table that fails validation.
VERIFIED_TABLES: dict[str, list[list[str]]] = {
    "pfe_grades": [
        ["10 - 12", "Fair"],
        ["12 - 14", 'Pretty good "Assez Bien"'],
        ["14 - 16", "Good"],
        ["16 - 18", "Very good"],
        ["18 - 20", "Excellent"],
    ],
    "partners": [
        ["SUAS: Schmalkalden University", "Germany", "2",
         "Winter Semester: October, Summer Semester: April"],
        ["CESI", "France", "2",
         "Winter Semester: September, Summer Semester: March"],
        ["INP Bordeaux", "France", "1", "Winter Semester: September"],
        ["INP Lorraine", "France", "1", "Winter Semester: September"],
    ],
    "contacts_1": [
        ["Internship", "Stages@tek-up.tn", "Office N° 015"],
        ["End of year Study project", "PFE@tek-up.tn", "Office N° 015"],
        ["Certification", "Certif@mailing.tek-up.de", "Office N° 016"],
        ["Internal Communication Department",
         "tek-up-university@mailing.tek-up.de", "Office N° 027"],
        ["Clubs and student life", "Clubs@mailing.tek-up.de", "Office N° 027"],
        ["International Office", "international@mailing.tek-up.de",
         "Office N° 027"],
    ],
    "contacts_2": [
        ["Academic Affairs", "Etudes@mailing.tek-up.de", "Office N° 011"],
        ["ING-1&2 Department", "ING.PREPA@mailing.tek-up.de", "Office N° 110"],
        ["ING-3 Department", "ING-3@mailing.tek-up.de", "Office N° 037"],
        ["GLSI Department", "GLSI@mailing.tek-up.de", "Office N° 037"],
        ["SSIR Department", "SSIR@mailing.tek-up.de", "Office N° 110"],
        ["SDIA Department", "SDIA@mailing.tek-up.de", "Office N° 110"],
        ["DMWM Department", "DMWM@mailing.tek-up.de", "Office N° 037"],
    ],
    "contacts_3": [
        ["Complaints", "SPOC@mailing.tek-up.de", ""],
        ["Logistics", "Logistics@mailing.tek-up.de", "Office N° 015"],
        ["Compliances", "Compliance@mailing.tek-up.de", ""],
        ["Finance", "Finances@mailing.tek-up.de", "Office N° 021"],
        ["Request for administrative documents/student services",
         "Documents@mailing.tek-up.de", "Office N° 021"],
    ],
}

# Filled by load_pdf_documents(): table id -> "extracted" | "fallback" | "missing".
TABLE_STATUS: dict[str, str] = {}


def load_jsonl_documents() -> list[Document]:
    """Load the pre-processed plan d'etudes: one JSONL line == one Document."""
    documents: list[Document] = []
    with open(config.JSONL_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["text"],
                    metadata=record["metadata"],
                )
            )
    return documents


# --- Guide tables: cleaning ---------------------------------------------------

def _clean_cell(cell: str | None) -> str:
    """Flatten a cell: a hyphen at a line end is a word split ("tek-\\nup")."""
    if not cell:
        return ""
    cell = cell.replace("-\n", "-")
    return " ".join(cell.split())


def _rows_pfe_grades(raw: list[list[str | None]]) -> list[list[str]]:
    """[Scale, Meaning]; the first row is the header."""
    rows = []
    for raw_row in raw[1:]:
        cells = [c for c in (_clean_cell(x) for x in raw_row) if c]
        rows.append(cells)
    return rows


def _rows_partners(raw: list[list[str | None]]) -> list[list[str]]:
    """[University, Country, Semesters, Period].

    The header row is offset from the data columns, so it is dropped first;
    then columns empty in every data row are removed. A vertically merged
    cell (``France`` for both INP rows) only appears on its first row, so
    empty cells are filled from the row above.
    """
    data = [[_clean_cell(x) for x in raw_row] for raw_row in raw[1:]]
    keep = [i for i in range(len(data[0])) if any(row[i] for row in data)]
    rows = [[row[i] for i in keep] for row in data]
    for previous, row in zip(rows, rows[1:]):
        for i, cell in enumerate(row):
            if not cell:
                row[i] = previous[i]
    for row in rows:
        if len(row) == 4:
            # "Winter Semester: October Summer Semester: April" -> comma-separated
            row[3] = re.sub(r" (?=\w+ Semester:)", ", ", row[3])
    return rows


def _rows_contacts(raw: list[list[str | None]]) -> list[list[str]]:
    """[Name, Email, Office]; no header row (row 0 is data).

    The email is located by pattern rather than by column index, because
    merged cells shift the columns. A row without an email is the
    continuation of the previous row's name.
    """
    rows: list[list[str]] = []
    for raw_row in raw:
        cells = [_clean_cell(x) for x in raw_row]
        email_index = next(
            (i for i, c in enumerate(cells) if EMAIL_RE.match(c)), None
        )
        if email_index is None:
            continuation = " ".join(c for c in cells if c)
            if rows and continuation:
                rows[-1][0] = f"{rows[-1][0]} {continuation}"
            continue
        name = " ".join(c for c in cells[:email_index] if c)
        office = " ".join(c for c in cells[email_index + 1:] if c)
        rows.append([name, cells[email_index], office])
    return rows


ROW_CLEANERS = {
    "pfe_grades": _rows_pfe_grades,
    "partners": _rows_partners,
    "contacts": _rows_contacts,
}


# --- Guide tables: validation ---------------------------------------------------

def _parse_scale(scale: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", scale)
    return (match.group(1), match.group(2)) if match else None


def _validate(table_id: str, rows: list[list[str]]) -> bool:
    if table_id == "pfe_grades":
        if len(rows) != 5 or any(len(r) != 2 or not _parse_scale(r[0]) for r in rows):
            return False
        good = [r for r in rows if r[1] == "Good"]
        return len(good) == 1 and _parse_scale(good[0][0]) == ("14", "16")

    if table_id == "partners":
        if len(rows) != 4 or any(len(r) != 4 or not all(r) for r in rows):
            return False
        if any(not r[2].isdigit() for r in rows):
            return False
        return any(r[0] == "CESI" and r[2] == "2" for r in rows)

    # contacts_N: row count of the verified slice, a name and a valid email per row.
    if len(rows) != len(VERIFIED_TABLES[table_id]):
        return False
    return all(len(r) == 3 and r[0] and EMAIL_RE.match(r[1]) for r in rows)


def _validate_contacts_global(rows: list[list[str]]) -> bool:
    mapping = {r[0]: r[1] for r in rows}
    return (
        mapping.get("SDIA Department") == "SDIA@mailing.tek-up.de"
        and mapping.get("SSIR Department") == "SSIR@mailing.tek-up.de"
    )


# --- Guide tables: sentences ------------------------------------------------------

def _sentence(table: str, row: list[str]) -> str:
    if table == "pfe_grades":
        low, high = _parse_scale(row[0])
        return (
            f"A PFE grade between {low} and {high} out of 20 corresponds to "
            f'the mention "{row[1]}".'
        )
    if table == "partners":
        university, country, semesters, period = row
        period_part = f", {period}" if period else ""
        return (
            f"The partner university {university} ({country}) allows "
            f"{semesters} exchange semester(s){period_part}."
        )
    name, email, office = row
    office_part = f", {office}" if office else ""
    return f"The contact email for {name} is {email}{office_part}."


def _table_rows_for_page(
    page: pymupdf.Page, page_number: int
) -> tuple[list[tuple[str, str, list[list[str]]]], list[pymupdf.Rect]]:
    """Detect, clean and validate the tables of one page.

    Returns ``([(table_id, table, rows), ...], table_bboxes)``. A table that
    fails validation (or is not detected) is replaced by its verified rows.
    """
    detected = page.find_tables().tables
    expected = EXPECTED_TABLES.get(page_number, [])
    if len(detected) != len(expected):
        print(
            f"[warn] {PDF_SOURCE_NAME} page {page_number}: "
            f"{len(detected)} tables detected, {len(expected)} expected"
        )

    results = []
    for index, (table_id, table) in enumerate(expected):
        rows: list[list[str]] | None = None
        if index < len(detected):
            rows = ROW_CLEANERS[table](detected[index].extract())
            if not _validate(table_id, rows):
                rows = None
        if rows is None:
            print(f"[fallback] table {table_id} (page {page_number})")
            TABLE_STATUS[table_id] = "fallback"
            rows = [list(r) for r in VERIFIED_TABLES[table_id]]
        else:
            TABLE_STATUS[table_id] = "extracted"
        results.append((table_id, table, rows))

    # The SDIA/SSIR rule spans the contact sub-tables: if it fails on the
    # extracted data, fall back for every extracted contact sub-table.
    contact_rows = [r for _, t, rows in results if t == "contacts" for r in rows]
    if contact_rows and not _validate_contacts_global(contact_rows):
        for i, (table_id, table, _) in enumerate(results):
            if table == "contacts" and TABLE_STATUS[table_id] == "extracted":
                print(f"[fallback] table {table_id} (page {page_number}, SDIA/SSIR rule)")
                TABLE_STATUS[table_id] = "fallback"
                results[i] = (table_id, table, [list(r) for r in VERIFIED_TABLES[table_id]])

    bboxes = [pymupdf.Rect(t.bbox) for t in detected]
    return results, bboxes


def _prose_outside_tables(page: pymupdf.Page, bboxes: list[pymupdf.Rect]) -> str:
    """Page text without the blocks that lie (>= TABLE_OVERLAP_RATIO) in a table."""
    kept: list[str] = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 0 = text, 1 = image
            continue
        block = pymupdf.Rect(x0, y0, x1, y1)
        area = block.width * block.height
        in_table = False
        for bbox in bboxes:
            overlap_w = min(block.x1, bbox.x1) - max(block.x0, bbox.x0)
            overlap_h = min(block.y1, bbox.y1) - max(block.y0, bbox.y0)
            if overlap_w > 0 and overlap_h > 0 and area > 0:
                if overlap_w * overlap_h / area >= TABLE_OVERLAP_RATIO:
                    in_table = True
                    break
        if not in_table:
            kept.append(text.strip())
    return "\n".join(t for t in kept if t)


def load_pdf_documents() -> tuple[list[Document], list[int]]:
    """Extract the guide page by page, then split it into naive chunks.

    Pages listed in ``config.GUIDE_TABLE_PAGES`` additionally yield one
    sentence Document per table row (``content_type="table_row"``), and their
    prose excludes the table text.

    Returns ``(prose_chunks + table_sentences, skipped_pages)`` where
    ``skipped_pages`` lists the 1-indexed pages that yielded fewer than
    ``MIN_PAGE_CHARS`` characters (image-only pages -- skipping them is intended).
    """
    page_documents: list[Document] = []
    table_documents: list[Document] = []
    skipped_pages: list[int] = []
    TABLE_STATUS.clear()

    with pymupdf.open(config.PDF_PATH) as pdf:
        for zero_based_index, page in enumerate(pdf):
            page_number = zero_based_index + 1  # 1-indexed for humans
            text = page.get_text().strip()

            if len(text) < config.MIN_PAGE_CHARS:
                print(
                    f"[skip] {PDF_SOURCE_NAME} page {page_number}: "
                    f"only {len(text)} chars of text"
                )
                skipped_pages.append(page_number)
                continue

            if page_number in config.GUIDE_TABLE_PAGES:
                tables, bboxes = _table_rows_for_page(page, page_number)
                for _table_id, table, rows in tables:
                    for row in rows:
                        table_documents.append(
                            Document(
                                page_content=_sentence(table, row),
                                metadata={
                                    "source_file": PDF_SOURCE_NAME,
                                    "page": page_number,
                                    "content_type": "table_row",
                                    "table": table,
                                },
                            )
                        )
                if bboxes:
                    text = _prose_outside_tables(page, bboxes)
                else:
                    print(
                        f"[warn] {PDF_SOURCE_NAME} page {page_number}: no table "
                        "bbox detected, page kept whole in prose"
                    )

            if text:
                page_documents.append(
                    Document(
                        page_content=text,
                        metadata={"source_file": PDF_SOURCE_NAME, "page": page_number},
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    # split_documents copies each page's metadata onto every chunk it produces.
    prose_chunks = splitter.split_documents(page_documents)

    return prose_chunks + table_documents, skipped_pages
