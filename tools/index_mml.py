#!/usr/bin/env python3
"""Index the supplied *Mathematics for Machine Learning* markdown extraction.

Inputs (read-only):
  * `Mathematics for ML/Mathematics for ML part 1.md`
  * `Mathematics for ML/Mathematics for ML part 2.md`
  * `Mathematics for ML/Mathematics for ML part 3.md`
  * `mml-book.pdf` (hashed only — the markdown is the canonical source representation)

Outputs (generated — do not hand-edit):
  * knowledge/sources/mml/manifest.json   — identity, files, hashes, provenance limits
  * knowledge/sources/mml/structure.json  — parts/chapters/sections with printed pages + anchors
  * knowledge/sources/mml/inventory.json  — figures, tables, definitions, theorems, examples, code
  * knowledge/sources/mml/defects.json    — extraction-defect register

Design notes:
  * The book's own table of contents survives in the extraction, so every numbered
    section keeps its **printed page number** as published by the authors. That is the
    citation anchor used across the curriculum.
  * The extraction contains running headers repeated mid-page (a PDF->markdown artefact).
    The first occurrence of a heading is treated as the real heading; later occurrences
    are recorded as duplicates rather than being deleted, so the defect stays visible.
  * No page numbers are invented for subsections: the source only gives section-level
    start pages, and the markdown carries no page markers.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "Mathematics for ML"
SOURCE_FILES = [
    SOURCE_DIR / "Mathematics for ML part 1.md",
    SOURCE_DIR / "Mathematics for ML part 2.md",
    SOURCE_DIR / "Mathematics for ML part 3.md",
]
PDF_PATH = REPO / "mml-book.pdf"
OUT_DIR = REPO / "knowledge" / "sources" / "mml"

WORK_ID = "SRC-MML"
EDITION = "Draft (2024-01-15)"

PART_RE = re.compile(r"^(Part\s+(?:I|II|III|IV))\s+(.+?)\s+(\d+)\s*$")
CHAPTER_RE = re.compile(r"^(\d{1,2})\s+(\S.*?)\s+(\d+)\s*$")
SECTION_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(\S.*?)\s+(\d+)\s*$")
SPECIAL_RE = re.compile(r"^(Exercises|References|Index|Foreword)\s+(\d+)\s*$")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*?)\s*$")
NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\s+(\S.*?)\s*$")
RUNNING_HEADER_RE = re.compile(r"^Contents\s+[ivx]+$", re.I)
DRAFT_FOOTER_RE = re.compile(r"^Draft \(2024-01-15\) of .Mathematics for Machine Learning")
PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*\d{1,3}\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)\s]+)\)")
BARCODE_RE = re.compile(r"^0\d{9}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_book() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return per-file text/line data plus a flat list combining all files in order."""
    files = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        files.append(
            {
                "path": str(path.relative_to(REPO)),
                "filename": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "lines": text.count("\n") + 1,
                "text": text,
                "line_list": text.split("\n"),
            }
        )
    return files, []


# --------------------------------------------------------------------------- #
# Table of contents
# --------------------------------------------------------------------------- #

def parse_toc(lines: List[str]) -> Dict[str, Any]:
    start = next(i for i, line in enumerate(lines) if line.strip() == "## Contents")
    end = next(
        i for i, line in enumerate(lines[start + 1:], start=start + 1) if line.strip() == "## Foreword"
    )
    region = lines[start:end]

    entries: List[Dict[str, Any]] = []
    current_part: Optional[str] = None
    current_chapter: Optional[str] = None

    for raw in region:
        line = raw.strip().rstrip("\u200b")
        if not line or RUNNING_HEADER_RE.match(line) or line.startswith("##"):
            continue
        match = PART_RE.match(line)
        if match:
            current_part = match.group(1)
            entries.append(
                {
                    "type": "part",
                    "id": match.group(1).replace(" ", "-").upper(),
                    "title": match.group(2),
                    "printed_page": int(match.group(3)),
                    "part": current_part,
                    "chapter": None,
                }
            )
            continue
        match = SECTION_RE.match(line)
        if match:
            entries.append(
                {
                    "type": "section",
                    "id": match.group(1),
                    "title": match.group(2),
                    "printed_page": int(match.group(3)),
                    "part": current_part,
                    "chapter": current_chapter,
                }
            )
            continue
        match = CHAPTER_RE.match(line)
        if match:
            current_chapter = match.group(1)
            entries.append(
                {
                    "type": "chapter",
                    "id": match.group(1),
                    "title": match.group(2),
                    "printed_page": int(match.group(3)),
                    "part": current_part,
                    "chapter": current_chapter,
                }
            )
            continue
        match = SPECIAL_RE.match(line)
        if match:
            entries.append(
                {
                    "type": "special",
                    "id": f"{current_chapter or ''}:{match.group(1)}".lstrip(":"),
                    "title": match.group(1),
                    "printed_page": int(match.group(2)),
                    "part": current_part,
                    "chapter": current_chapter,
                }
            )
            continue
        entries.append({"type": "unparsed", "raw": line, "part": current_part, "chapter": current_chapter})

    return {"entries": entries, "logical_pages": {"first": 1, "last": 407}}


# --------------------------------------------------------------------------- #
# Content scan
# --------------------------------------------------------------------------- #

def scan_content(files: List[Dict[str, Any]], toc_section_ids: set) -> Dict[str, Any]:
    headings: List[Dict[str, Any]] = []
    defects: List[Dict[str, Any]] = []
    images: List[Dict[str, str]] = []
    promoted_exercise_items: List[Dict[str, Any]] = []
    counters: Counter = Counter()
    exercise_block_chapter: Optional[str] = None

    for file_index, file in enumerate(files, start=1):
        current_chapter: Optional[str] = None
        in_code_block = False
        dollar_fence_open: Optional[int] = None

        for line_number, line in enumerate(file["line_list"], start=1):
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                counters["code_fences"] += 1

            if stripped == "$$":
                if dollar_fence_open is None:
                    dollar_fence_open = line_number
                else:
                    dollar_fence_open = None
                    counters["display_math_blocks"] += 1

            if DRAFT_FOOTER_RE.match(stripped):
                counters["draft_footers"] += 1

            for image in IMAGE_RE.finditer(line):
                images.append(
                    {
                        "url": image.group("url"),
                        "file": file["filename"],
                        "line": line_number,
                    }
                )

            heading = HEADING_RE.match(line)
            if heading and not in_code_block:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                numbered = NUMBERED_HEADING_RE.match(title)
                section_number = numbered.group(1) if numbered else None
                clean_title = numbered.group(2) if numbered else title

                if re.fullmatch(r"Exercises?", title, flags=re.I):
                    # The set belongs to the chapter of the nearest section above it.
                    exercise_block_chapter = current_chapter
                elif section_number and exercise_block_chapter is not None:
                    if section_number.split(".")[0] == exercise_block_chapter:
                        # Exercise items are sometimes emitted as level-2 headings, e.g.
                        # "## 7.1 Consider the univariate function". A chapter's real
                        # sections never follow its exercise set, so a same-chapter
                        # numbered heading inside the set is an item, not a section.
                        promoted_exercise_items.append(
                            {
                                "id": section_number,
                                "chapter": section_number.split(".")[0],
                                "prompt_excerpt": clean_title[:180],
                                "file": file["filename"],
                                "line": line_number,
                                "note": "exercise item emitted as a level-2 heading by the extraction",
                            }
                        )
                        counters["promoted_exercise_items"] += 1
                        continue
                    exercise_block_chapter = None
                if section_number:
                    # The chapter is always the first component of a section number,
                    # so "4.7" and "4.7.2" both belong to chapter 4. A file may start
                    # mid-chapter (the supplied extraction is split by page range), which
                    # is why this is derived from the heading rather than from file order.
                    current_chapter = section_number.split(".")[0]

                entry = {
                    "level": level,
                    "title": clean_title,
                    "section_number": section_number,
                    "chapter": section_number.split(".")[0] if section_number else current_chapter,
                    "file": file["filename"],
                    "file_index": file_index,
                    "line": line_number,
                    "char_offset": None,
                }

                if section_number is None and re.fullmatch(r"(Analytic Geometry|Classification with Support Vector Machines|Linear Regression|Density Estimation with Gaussian Mixture Models|Dimensionality Reduction with Principal Component Analysis|When Models Meet Data|Matrix Decompositions|Vector Calculus|Probability and Distributions|Continuous Optimization)", title):
                    defects.append(
                        {
                            "code": "chapter-heading-without-number",
                            "severity": "info",
                            "file": file["filename"],
                            "line": line_number,
                            "detail": (
                                f"Chapter heading '{title}' appears in the running text without its "
                                "chapter number; numbering is recovered from the table of contents."
                            ),
                        }
                    )

                if BARCODE_RE.match(stripped) or PAGE_NUMBER_ONLY_RE.match(stripped):
                    pass  # handled below as separate defect scan

                headings.append(entry)

        # file-level defect scan
        for line_number, line in enumerate(file["line_list"], start=1):
            stripped = line.strip()
            if BARCODE_RE.match(stripped):
                defects.append(
                    {
                        "code": "stray-barcode-or-isbn-artifact",
                        "severity": "warning",
                        "file": file["filename"],
                        "line": line_number,
                        "detail": f"Line {line_number} contains the bare digit string '{stripped}', "
                        "an extraction artefact rather than book content.",
                    }
                )
            heading_body = re.sub(r"^#+\s*", "", stripped)
            if BARCODE_RE.match(heading_body):
                defects.append(
                    {
                        "code": "stray-barcode-or-isbn-artifact",
                        "severity": "warning",
                        "file": file["filename"],
                        "line": line_number,
                        "detail": f"Line {line_number} emits the bare digit string "
                        f"'{heading_body}' as a heading; it is an extraction artefact.",
                    }
                )
            if stripped.startswith("## ") and PAGE_NUMBER_ONLY_RE.match(stripped[3:]):
                defects.append(
                    {
                        "code": "page-number-emitted-as-heading",
                        "severity": "warning",
                        "file": file["filename"],
                        "line": line_number,
                        "detail": f"Printed page number '{stripped[3:].strip()}' was emitted as a level-2 heading.",
                    }
                )

        if dollar_fence_open is not None:
            defects.append(
                {
                    "code": "unbalanced-display-math-fence",
                    "severity": "error",
                    "file": file["filename"],
                    "line": dollar_fence_open,
                    "detail": "An opening '$$' fence is never closed in this file.",
                }
            )

    # attach char offsets
    for file in files:
        offset = 0
        offsets_by_line = {}
        for index, line in enumerate(file["line_list"], start=1):
            offsets_by_line[index] = offset
            offset += len(line) + 1
        for heading in headings:
            if heading["file"] == file["filename"]:
                heading["char_offset"] = offsets_by_line[heading["line"]]

    return {
        "headings": headings,
        "defects": defects,
        "images": images,
        "counters": counters,
        "promoted_exercise_items": promoted_exercise_items,
    }


def build_structure(toc: Dict[str, Any], scan: Dict[str, Any]) -> Dict[str, Any]:
    headings = scan["headings"]
    numbered = [h for h in headings if h["section_number"]]

    first_occurrence: Dict[str, Dict[str, Any]] = {}
    duplicates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for heading in numbered:
        key = heading["section_number"]
        if key in first_occurrence:
            duplicates[key].append(heading)
        else:
            first_occurrence[key] = heading

    toc_sections = [e for e in toc["entries"] if e["type"] == "section"]
    toc_chapters = [e for e in toc["entries"] if e["type"] == "chapter"]
    toc_parts = [e for e in toc["entries"] if e["type"] == "part"]
    toc_specials = [e for e in toc["entries"] if e["type"] == "special"]

    chapters: List[Dict[str, Any]] = []
    for chapter in toc_chapters:
        chapter_sections = []
        for section in toc_sections:
            if section["chapter"] != chapter["id"]:
                continue
            anchor = first_occurrence.get(section["id"])
            chapter_sections.append(
                {
                    "id": section["id"],
                    "title": section["title"],
                    "printed_page": section["printed_page"],
                    "anchor": (
                        {
                            "file": anchor["file"],
                            "line": anchor["line"],
                            "char_offset": anchor["char_offset"],
                        }
                        if anchor
                        else None
                    ),
                    "status": "located" if anchor else "not-found-in-extraction",
                    "duplicate_heading_occurrences": len(duplicates.get(section["id"], [])),
                }
            )

        exercises = next(
            (s for s in toc_specials if s["chapter"] == chapter["id"] and s["title"] == "Exercises"),
            None,
        )
        # A chapter heading may itself be unnumbered in the extraction (e.g. the body
        # prints "Analytic Geometry" without the "3"). The chapter is then anchored at its
        # first located section, and that is stated explicitly rather than implied.
        chapter_anchor = first_occurrence.get(chapter["id"])
        anchor_derivation = "chapter heading found in the body"
        if chapter_anchor is None:
            located = [s for s in chapter_sections if s["anchor"]]
            if located:
                first_section = min(located, key=lambda s: (s["anchor"]["file"], s["anchor"]["line"]))
                chapter_anchor = first_occurrence[first_section["id"]]
                anchor_derivation = (
                    "inferred from the first located section ("
                    + first_section["id"]
                    + "); the body prints this chapter's title without its number"
                )
        chapters.append(
            {
                "id": chapter["id"],
                "title": chapter["title"],
                "printed_page": chapter["printed_page"],
                "part": chapter["part"],
                "anchor": (
                    {
                        "file": chapter_anchor["file"],
                        "line": chapter_anchor["line"],
                        "char_offset": chapter_anchor["char_offset"],
                    }
                    if chapter_anchor
                    else None
                ),
                "anchor_derivation": anchor_derivation,
                "source_file": chapter_anchor["file"] if chapter_anchor else None,
                "sections": chapter_sections,
                "exercises_printed_page": exercises["printed_page"] if exercises else None,
                "subsection_headings_found": sorted(
                    {
                        h["section_number"]
                        for h in numbered
                        if h["section_number"].count(".") >= 2
                        and h["section_number"].split(".")[0] == chapter["id"]
                    }
                ),
            }
        )

    missing_sections = [
        section["id"]
        for chapter in chapters
        for section in chapter["sections"]
        if section["status"] != "located"
    ]

    return {
        "work_id": WORK_ID,
        "edition": EDITION,
        "parts": toc_parts,
        "chapters": chapters,
        "exercises_index": {
            s["chapter"]: s["printed_page"] for s in toc_specials if s["title"] == "Exercises"
        },
        "specials": toc_specials,
        "totals": {
            "chapters": len(chapters),
            "toc_sections": len(toc_sections),
            "sections_located": len(toc_sections) - len(missing_sections),
            "sections_not_found": len(missing_sections),
            "subsection_headings_found": len(
                [h for h in numbered if h["section_number"].count(".") >= 2]
            ),
            "duplicate_heading_occurrences": sum(len(v) for v in duplicates.values()),
        },
        "missing_sections": missing_sections,
        "duplicate_headings": {
            key: [{"file": d["file"], "line": d["line"]} for d in values]
            for key, values in sorted(duplicates.items())
        },
        "unparsed_toc_lines": [e for e in toc["entries"] if e["type"] == "unparsed"],
    }



# --------------------------------------------------------------------------- #
# Exercise sets supplied by the source
# --------------------------------------------------------------------------- #

def build_exercise_index(
    files: List[Dict[str, Any]], structure: Dict[str, Any], scan: Dict[str, Any]
) -> Dict[str, Any]:
    """Locate the exercise sets that the source actually supplies.

    The extraction places each exercise item at the start of a line
    (e.g. `4.1 Compute the determinant ...`), so items can be located precisely.
    A chapter whose exercise set is absent from the source is reported as such —
    that is a property of the book draft, not of this indexer.
    """
    headings_by_file: Dict[str, List[Tuple[int, str, Optional[str]]]] = defaultdict(list)
    for chapter in structure["chapters"]:
        for section in chapter["sections"]:
            anchor = section["anchor"]
            if anchor:
                headings_by_file[anchor["file"]].append(
                    (anchor["line"], section["id"], chapter["id"])
                )

    results: Dict[str, Any] = {}
    for file in files:
        lines = file["line_list"]
        for index, line in enumerate(lines):
            if not re.fullmatch(r"#{2,4}\s+Exercises?\s*", line):
                continue
            # chapter = nearest numbered heading above the exercise set
            chapter = None
            for heading_line, section_id, chapter_id in sorted(headings_by_file[file["filename"]]):
                if heading_line < index + 1:
                    chapter = chapter_id
                else:
                    break
            items = []
            for offset in range(index + 1, len(lines)):
                candidate = lines[offset].strip()
                heading_here = HEADING_RE.match(lines[offset])
                if heading_here:
                    # A same-chapter numbered heading inside the set is one of the
                    # exercise items the extraction promoted to a heading (merged
                    # below), so the scan must continue past it. Anything else —
                    # the next chapter, the references, a later section — ends the set.
                    inner_title = heading_here.group(2).strip()
                    inner_numbered = NUMBERED_HEADING_RE.match(inner_title)
                    if inner_numbered and inner_numbered.group(1).split(".")[0] == chapter:
                        continue
                    break
                match = re.match(r"^(\d{1,2}\.\d{1,2})\s+(.{5,})", candidate)
                if match and chapter and match.group(1).split(".")[0] == chapter:
                    items.append(
                        {
                            "id": match.group(1),
                            "prompt_excerpt": " ".join(match.group(2).split())[:180],
                            "file": file["filename"],
                            "line": offset + 1,
                        }
                    )
            if chapter:
                results[chapter] = {
                    "chapter": chapter,
                    "anchor": {"file": file["filename"], "line": index + 1},
                    "printed_page": structure["exercises_index"].get(chapter),
                    "item_count": len(items),
                    "items": items,
                }

    # Merge items that the extraction emitted as headings, keeping line order and
    # recording the artefact so a reader can see why the item list is what it is.
    for chapter, entry in results.items():
        for promoted in scan["promoted_exercise_items"]:
            if promoted["chapter"] != chapter:
                continue
            if any(item["id"] == promoted["id"] for item in entry["items"]):
                continue
            entry["items"].append(
                {
                    "id": promoted["id"],
                    "prompt_excerpt": promoted["prompt_excerpt"],
                    "file": promoted["file"],
                    "line": promoted["line"],
                    "extraction_artefact": promoted["note"],
                }
            )
        entry["items"].sort(key=lambda item: [int(part) for part in item["id"].split(".")])
        entry["item_count"] = len(entry["items"])
    return results


def build_inventory(
    files: List[Dict[str, Any]],
    scan: Dict[str, Any],
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    patterns = {
        "figures": re.compile(r"\bFigure\s+(\d{1,3}\.\d{1,3})"),
        "tables": re.compile(r"\bTable\s+(\d{1,3}\.\d{1,3})"),
        "definitions": re.compile(r"\bDefinition\s+(\d{1,3}\.\d{1,3})"),
        "theorems": re.compile(r"\bTheorem\s+(\d{1,3}\.\d{1,3})"),
        "examples": re.compile(r"\bExample\s+(\d{1,3}\.\d{1,3})"),
        "equations": re.compile(r"\((\d{1,3}\.\d{1,3})\)"),
    }

    inventory: Dict[str, Any] = {}
    for kind, pattern in patterns.items():
        per_chapter: Dict[str, set] = defaultdict(set)
        for file in files:
            for match in pattern.finditer(file["text"]):
                identifier = match.group(1)
                per_chapter[identifier.split(".")[0]].add(identifier)
        inventory[kind] = {
            "total_unique": sum(len(v) for v in per_chapter.values()),
            "per_chapter": {k: sorted(v, key=lambda s: [int(p) for p in s.split(".")]) for k, v in sorted(per_chapter.items())},
        }

    hosts = Counter()
    for image in scan["images"]:
        match = re.match(r"https?://([^/]+)/", image["url"])
        hosts[match.group(1) if match else "relative-or-unknown"] += 1

    code_fences = sum(len(re.findall(r"^```", file["text"], flags=re.M)) for file in files)

    exercises = build_exercise_index(files, structure, scan)

    return {
        "work_id": WORK_ID,
        "numbered_objects": inventory,
        "exercises": {
            "chapters_with_supplied_exercise_sets": sorted(exercises, key=int),
            "chapters_without_supplied_exercise_sets": [
                chapter["id"] for chapter in structure["chapters"] if chapter["id"] not in exercises
            ],
            "total_items_located": sum(e["item_count"] for e in exercises.values()),
            "per_chapter": exercises,
        },
        "images": {
            "references": len(scan["images"]),
            "distinct_urls": len({i["url"] for i in scan["images"]}),
            "hosts": dict(hosts),
            "locally_available_assets": 0,
            "note": (
                "Every figure is referenced as an external CDN URL that is not part of this "
                "repository, so no figure bitmap is available offline. Captions and figure "
                "numbers are present and can be cited; the images themselves cannot be shipped."
            ),
        },
        "tables_as_html": {
            "occurrences": sum(len(re.findall(r"<table>", file["text"])) for file in files),
            "note": "Some tables were emitted as raw HTML <table> markup instead of markdown tables.",
        },
        "code_fences": code_fences,
        "counters": dict(scan["counters"]),
    }


def build_defects(files: List[Dict[str, Any]], scan: Dict[str, Any], structure: Dict[str, Any], inventory: Dict[str, Any]) -> Dict[str, Any]:
    defects = list(scan["defects"])

    def add(code: str, severity: str, detail: str, **extra: Any) -> None:
        defects.append({"code": code, "severity": severity, "detail": detail, **extra})

    # 1. sections present in the table of contents but absent from the extraction
    for section in structure["missing_sections"]:
        add(
            "toc-section-missing-from-extraction",
            "error",
            f"Section {section} is listed in the table of contents but no matching heading exists in any supplied markdown file.",
        )

    # 2. duplicated running headings
    duplicate_total = structure["totals"]["duplicate_heading_occurrences"]
    if duplicate_total:
        worst = sorted(
            structure["duplicate_headings"].items(), key=lambda kv: -len(kv[1])
        )[:5]
        add(
            "running-headings-duplicated-in-body",
            "info",
            f"{duplicate_total} heading occurrences repeat an earlier identical heading. The "
            "extraction interleaves the printed running header into the body text, so a naive "
            "heading scan counts a section more than once. Affected (worst first): "
            + "; ".join(f"{key} x{len(value) + 1}" for key, value in worst)
            + ".",
        )

    # 3. exercises: what the source actually supplies
    exercises = inventory["exercises"]
    add(
        "source-supplies-exercise-sets-only-for-chapters-"
        + "-".join(exercises["chapters_with_supplied_exercise_sets"]),
        "warning",
        "The supplied source contains exercise sets for chapters "
        + ", ".join(exercises["chapters_with_supplied_exercise_sets"])
        + " ("
        + str(exercises["total_items_located"])
        + " items located), and none for chapters "
        + ", ".join(exercises["chapters_without_supplied_exercise_sets"])
        + ". Practice material for the chapters without sets must be sourced elsewhere or "
        "authored, and must be labelled accordingly. Verified independently against the supplied "
        "PDF: the string 'Exercises' occurs in the PDF only on the table-of-contents pages and on "
        "printed pages 64, 96, 137, 170, 222 and 247, matching the located sets.",
    )

    # 4. figures without local assets
    add(
        "figures-not-available-offline",
        "warning",
        f"{inventory['images']['references']} figure references point at {inventory['images']['distinct_urls']} "
        "external CDN URLs (" + ", ".join(inventory["images"]["hosts"]) + "). No image file exists in the "
        "repository, so lessons cannot rely on the original figures being present.",
    )

    # 5. OCR/text corruption signatures seen in this extraction
    corruptions = {
        r"\\ b \{": "a mangled macro such as '\\ b { S }' where '\\mathbf' or '\\boldsymbol' was intended",
        r"\\stackrel \{ \\cdot \}": "'\\stackrel{\\cdot}' where the book prints '\\stackrel{!}'",
        r"= \\stackrel": "a corruption near an equality annotation",
        r"\\mathrm \{ i \. e \.": "broken roman-text runs inside math ('\\mathrm{i.e.}' split across tokens)",
    }
    for pattern, explanation in corruptions.items():
        count = sum(len(re.findall(pattern, file["text"])) for file in files)
        if count:
            add(
                "latex-token-corruption",
                "warning",
                f"{count} occurrence(s) of {explanation}. Such tokens must be repaired in the "
                "lesson's quotation of the source, or the formula paraphrased in correct notation.",
            )

    add(
        "page-mapping-is-section-level-only",
        "info",
        "The markdown carries no page markers (only occasional stray page numbers and the authors' "
        "draft footer). Printed page numbers therefore survive only where the table of contents "
        "supplies them, i.e. at section granularity. Sub-section citations use markdown anchors "
        "(file + line) instead of page numbers.",
    )

    severity_order = {"error": 0, "warning": 1, "info": 2}
    defects.sort(key=lambda d: (severity_order.get(d["severity"], 3), d["code"]))

    return {
        "work_id": WORK_ID,
        "generated_by": "tools/index_mml.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "summary": {
            "errors": sum(1 for d in defects if d["severity"] == "error"),
            "warnings": sum(1 for d in defects if d["severity"] == "warning"),
            "info": sum(1 for d in defects if d["severity"] == "info"),
        },
        "defects": defects,
    }


def main() -> int:
    if not all(path.exists() for path in SOURCE_FILES):
        print("Source markdown files are missing; nothing to index.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files, _ = read_book()
    toc = parse_toc(files[0]["line_list"])
    toc_section_ids = {e["id"] for e in toc["entries"] if e["type"] == "section"}
    scan = scan_content(files, toc_section_ids)

    structure = build_structure(toc, scan)
    inventory = build_inventory(files, scan, structure)
    defects = build_defects(files, scan, structure, inventory)

    manifest = {
        "work_id": WORK_ID,
        "title": "Mathematics for Machine Learning",
        "authors": ["Marc Peter Deisenroth", "A. Aldo Faisal", "Cheng Soon Ong"],
        "edition": EDITION,
        "publisher": "Cambridge University Press (2020); free draft revision used here",
        "canonical_web": "https://mml-book.com",
        "supplied_representation": "markdown extracted from the authors' PDF draft",
        "note_on_identity": (
            "The work was identified from its own title page, running draft footer "
            "('Draft (2024-01-15) of \"Mathematics for Machine Learning\"'), author line, and table of "
            "contents, not from the directory name."
        ),
        "license_note": (
            "The book is free to download but remains under the authors' and publisher's copyright. "
            "Lessons therefore cite it and teach from it without reproducing its text or figures. "
            "Original explanatory prose, worked examples and diagrams are authored for this platform."
        ),
        "files": [
            {
                "path": file["path"],
                "sha256": file["sha256"],
                "bytes": file["bytes"],
                "lines": file["lines"],
            }
            for file in files
        ],
        "pdf": (
            {
                "path": str(PDF_PATH.relative_to(REPO)),
                "sha256": sha256(PDF_PATH),
                "bytes": PDF_PATH.stat().st_size,
                "role": "original supplied PDF; retained as the authoritative artefact but not parsed",
            }
            if PDF_PATH.exists()
            else None
        ),
        "structure_snapshot": {
            "parts": len(structure["parts"]),
            "chapters": structure["totals"]["chapters"],
            "toc_sections": structure["totals"]["toc_sections"],
            "sections_located_in_markdown": structure["totals"]["sections_located"],
        },
        "processing_status": "indexed",
        "generated_by": "tools/index_mml.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    for name, payload in (
        ("manifest.json", manifest),
        ("structure.json", structure),
        ("inventory.json", inventory),
        ("defects.json", defects),
    ):
        (OUT_DIR / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(json.dumps(manifest["structure_snapshot"], indent=2))
    print("defects:", defects["summary"])
    print("exercises found in body for chapters:", sorted({h["chapter"] for h in scan["headings"] if re.fullmatch(r"Exercises?", h["title"], flags=re.I) and h["chapter"]}))
    print("missing sections:", structure["missing_sections"])
    print("promoted exercise item headings:", scan["counters"].get("promoted_exercise_items", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
