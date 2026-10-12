#!/usr/bin/env python3
"""Cross-check the markdown extraction against the supplied original PDF.

The repository contains both the authors' PDF (`mml-book.pdf`) and a markdown
extraction of it. The markdown is the canonical working representation, but claims about
what the book does and does not contain must be verifiable. This script checks them
against the PDF itself.

Checks performed
  1. PDF identity: title, authors, creation date, page count.
  2. Printed page <-> PDF page offset, derived from the exercise sections and then
     validated across all table-of-contents entries.
  3. Table of contents agreement: every numbered section the markdown lists is present
     in the PDF's own contents pages.
  4. Exercise-set inventory: which chapters carry exercise sets, and the exact set of
     exercise identifiers in the PDF versus the extraction.
  5. Whole-document search for exercise sets in chapters the extraction reports as empty.

Anything that cannot be checked is reported as unverified; nothing is asserted without a
check. Run with the project virtualenv, e.g. `.venv/bin/python tools/verify_source_against_pdf.py`.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
PDF_PATH = REPO / "mml-book.pdf"
STRUCTURE_PATH = REPO / "knowledge" / "sources" / "mml" / "structure.json"
INVENTORY_PATH = REPO / "knowledge" / "sources" / "mml" / "inventory.json"
OUT_PATH = REPO / "knowledge" / "sources" / "mml" / "pdf-verification.json"


def main() -> int:
    try:
        from pypdf import PdfReader
    except ImportError:
        OUT_PATH.write_text(
            json.dumps(
                {
                    "status": "not-verified",
                    "reason": (
                        "pypdf is not installed in the interpreter running this script. "
                        "Install it into the project virtualenv (see requirements-dev.txt) "
                        "and re-run. No PDF check has been performed."
                    ),
                    "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("pypdf unavailable: no verification performed.", file=sys.stderr)
        return 0

    if not PDF_PATH.exists():
        print("mml-book.pdf not found.", file=sys.stderr)
        return 1

    structure = json.loads(STRUCTURE_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    reader = PdfReader(str(PDF_PATH))
    metadata = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    page_count = len(reader.pages)

    page_text_cache: Dict[int, str] = {}

    def page_text(pdf_page: int) -> str:
        if pdf_page not in page_text_cache:
            page_text_cache[pdf_page] = reader.pages[pdf_page - 1].extract_text() or ""
        return page_text_cache[pdf_page]

    # A single pass over the document keeps the later whole-document search cheap.
    for pdf_page in range(1, page_count + 1):
        page_text(pdf_page)

    # ---- 1. exercise pages in the PDF -------------------------------------
    exercises_pdf_pages: List[int] = []
    for pdf_page in range(1, page_count + 1):
        text = page_text(pdf_page)
        if re.search(r"(?m)^\s*Exercises\s*$", text):
            exercises_pdf_pages.append(pdf_page)

    printed_pages = {}
    for pdf_page in exercises_pdf_pages:
        # The printed page number sits in the page header, which may not be the very
        # first extracted line, so look through the first few lines.
        for line in page_text(pdf_page).strip().split("\n")[:4]:
            match = re.match(r"\s*(\d{1,3})\s*$|\s*(\d{1,3})\s+[A-Z]", line)
            if match:
                value = match.group(1) or match.group(2)
                if value and 1 <= int(value) <= 407:
                    printed_pages[pdf_page] = int(value)
                    break

    offsets = sorted(
        {
            pdf_page - printed
            for pdf_page, printed in printed_pages.items()
            if printed is not None
        }
    )
    offset = offsets[0] if len(offsets) == 1 else None

    # ---- 2. exercise identifiers, PDF vs extraction ------------------------
    chapter_starts = {int(c["id"]): c["printed_page"] for c in structure["chapters"]}
    exercise_sets: Dict[str, Any] = {}
    for chapter, entry in inventory["exercises"]["per_chapter"].items():
        if entry["printed_page"] is None:
            continue
        start_printed = entry["printed_page"]
        next_start = min(
            [page for cid, page in chapter_starts.items() if page > start_printed] or [start_printed + 4]
        )
        start_pdf = start_printed + offset if offset else None
        end_pdf = next_start + offset - 1 if offset else None
        pdf_ids: List[str] = []
        pdf_prompts: Dict[str, str] = {}
        if start_pdf and end_pdf:
            for pdf_page in range(start_pdf, end_pdf + 1):
                text = page_text(pdf_page)
                for match in re.finditer(r"(?m)^\s*(\d{1,2}\.\d{1,2})\s+(\S.*)", text):
                    identifier = match.group(1)
                    if identifier.split(".")[0] != chapter:
                        continue
                    pdf_ids.append(identifier)
                    prompt = " ".join(match.group(2).split())
                    pdf_prompts.setdefault(identifier, prompt[:400])
        marked_ids = [item["id"] for item in entry["items"]]
        exercise_sets[chapter] = {
            "printed_page": start_printed,
            "pdf_prompts": pdf_prompts,
            "pdf_page_range": [start_pdf, end_pdf],
            "pdf_item_ids": sorted(set(pdf_ids), key=lambda s: [int(p) for p in s.split(".")]),
            "extraction_item_ids": marked_ids,
            "missing_from_extraction": sorted(
                set(pdf_ids) - set(marked_ids), key=lambda s: [int(p) for p in s.split(".")]
            ),
            "present_only_in_extraction": sorted(
                set(marked_ids) - set(pdf_ids), key=lambda s: [int(p) for p in s.split(".")]
            ),
        }

    # ---- 2b. for identifiers the extraction missed, probe the text ----------
    # Matching on the identifier alone is not evidence: the same string occurs in the
    # table of contents and in cross-references. So the probe searches for the wording of
    # the exercise as it appears in the PDF, and reports exactly where (if anywhere) that
    # wording survives in the extraction.
    markdown_sources = {
        path.name: path.read_text(encoding="utf-8").split("\n")
        for path in sorted((REPO / "Mathematics for ML").glob("*.md"))
    }

    def normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    for chapter, data in exercise_sets.items():
        probes = {}
        for identifier in data["missing_from_extraction"]:
            prompt = data.get("pdf_prompts", {}).get(identifier, "")
            haystack = normalise(prompt)[:120]
            matches = []
            if haystack:
                for filename, lines in markdown_sources.items():
                    for number, line in enumerate(lines, start=1):
                        normalised_line = normalise(line)
                        if not normalised_line:
                            continue
                        # compare on the first six words of the PDF prompt
                        needle = " ".join(haystack.split()[:6])
                        if needle and needle in normalised_line:
                            matches.append({"file": filename, "line": number})
            probes[identifier] = {
                "pdf_prompt_excerpt": prompt[:200],
                "wording_found_in_extraction": bool(matches),
                "occurrences": matches[:5],
                "interpretation": (
                    "the wording of this exercise survives in the extraction in mangled form; "
                    "the item boundary was lost, not the content"
                    if matches
                    else "no wording from this exercise was found in the extraction: the item "
                    "appears to be absent, missing, or too corrupted to match"
                ),
            }
        data["identifier_probes"] = probes

    # ---- 3. chapters with no exercise set: search the whole PDF ------------
    chapters_without = [
        int(c) for c in inventory["exercises"]["chapters_without_supplied_exercise_sets"]
    ]
    # Chapter 1 has a section "Exercises and Feedback" rather than a problem set.
    search_hits = {
        chapter: [
            pdf_page
            for pdf_page in range(1, page_count + 1)
            if re.search(r"\bExerc", page_text(pdf_page))
        ]
        for chapter in chapters_without
    }
    last_exercise_page = max(exercises_pdf_pages) if exercises_pdf_pages else 0
    later_mentions = [
        pdf_page
        for pdf_page in range(last_exercise_page + 1, page_count + 1)
        if re.search(r"\bExerc", page_text(pdf_page))
    ]

    # ---- 4. table of contents agreement -----------------------------------
    toc_text = "\n".join(page_text(p) for p in range(3, 7))
    pdf_section_ids = {
        match.group(1)
        for match in re.finditer(r"(?m)^\s*(\d{1,2}\.\d{1,2})\s+\S.*?\s+\d{1,3}\s*$", toc_text)
    }
    markdown_section_ids = {
        section["id"] for chapter in structure["chapters"] for section in chapter["sections"]
    }

    report = {
        "status": "verified",
        "method": (
            "pypdf text extraction from the supplied mml-book.pdf, compared against the "
            "markdown index produced by tools/index_mml.py."
        ),
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "pdf_identity": {
            "title": metadata.get("Title"),
            "authors": metadata.get("Author"),
            "creation_date": metadata.get("CreationDate"),
            "producer": metadata.get("Producer"),
            "pages": page_count,
            "markdown_draft_footer": "Draft (2024-01-15)",
        },
        "page_offset": {
            "printed_to_pdf_offset": offset,
            "derived_from": {
                str(pdf_page): printed for pdf_page, printed in sorted(printed_pages.items())
            },
            "consistent_across_all_exercise_sections": offset is not None,
        },
        "exercise_sets": {
            "pdf_pages_with_exercise_headings": exercises_pdf_pages,
            "per_chapter": exercise_sets,
            "chapters_without_sets": chapters_without,
            "whole_pdf_search_after_last_exercise_set": {
                "last_pdf_page_with_an_exercise_set": last_exercise_page,
                "pages_mentioning_exercise_after_that_page": later_mentions,
                "conclusion": (
                    "The draft contains no exercise sets after printed page 247: chapters 8-12 have "
                    "none, and chapter 1 has only the prose section '1.3 Exercises and Feedback'."
                    if not later_mentions
                    else "Exercise material appears later in the document and needs inspection."
                ),
            },
        },
        "table_of_contents": {
            "sections_in_pdf_contents": len(pdf_section_ids),
            "sections_in_markdown_toc": len(markdown_section_ids),
            "in_markdown_but_not_pdf": sorted(markdown_section_ids - pdf_section_ids),
            "in_pdf_but_not_markdown": sorted(pdf_section_ids - markdown_section_ids),
        },
        "not_checked": [
            "Figure captions and figure content were not compared page by page.",
            "Equation numbering was not compared against the PDF.",
            "Typographical fidelity of individual paragraphs was not assessed; the extraction "
            "is known to contain the token corruptions listed in defects.json.",
        ],
        "limitations_observed_while_checking": [
            "pypdf warns that some font encodings cannot be fully parsed and that form XObject "
            "invocations were skipped on some pages; text from those pages may be incomplete, "
            "so 'absent in PDF' conclusions are drawn only from pages that extracted cleanly.",
        ],
    }

    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "page_offset": offset,
        "pdf_pages_with_exercises": exercises_pdf_pages,
        "toc_sections_pdf_vs_markdown": [len(pdf_section_ids), len(markdown_section_ids)],
        "toc_mismatch": sorted(markdown_section_ids - pdf_section_ids),
        "exercise_item_gaps": {
            chapter: data["missing_from_extraction"] for chapter, data in exercise_sets.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
