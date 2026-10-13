#!/usr/bin/env python3
"""Build toc.json + coverage.json from a book's real Markdown headings (R16).

Usage: python3 scripts/build_toc.py <book-id>
Headings levels 1-3. Duplicate running-header headings are merged into the
first occurrence (their words are added to it). Never uses memory of the book.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:80]

def kind_of(title, level):
    t = title.lower()
    if any(k in t for k in ["further reading", "references", "index"]):
        return "reference"
    if t.startswith("exercises") or re.match(r"^\d+\.\d+\s", t) and "exercise" in t:
        return "exercise"
    if any(k in t for k in ["contents", "foreword", "acknowledgment", "abbreviations",
                            "introduction to notation", "notation", "part i", "part ii"]):
        return "front-matter"
    if title.startswith("Example"):
        return "example"
    return "content"

def main():
    if len(sys.argv) != 2:
        print("usage: build_toc.py <book-id>"); return 2
    book_id = sys.argv[1]
    book_md = ROOT / "books" / book_id / "book.md"
    if not book_md.exists():
        print(f"FAIL: {book_md} missing"); return 1
    lines = book_md.read_text(encoding="utf-8", errors="replace").splitlines()
    # collect headings with line numbers
    heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.+?)\s*$", ln)
        if m:
            heads.append({"line": i + 1, "level": len(m.group(1)), "title": m.group(2).strip()})
    if not heads:
        print("FAIL: no parseable headings"); return 1
    # word count of segment after each heading until next heading
    rows, seen = [], {}
    current_chapter = 0
    # chapter titles from the book's own Contents (lines like "2 Linear Algebra 17")
    chapter_titles = {}
    for ln in lines[:150]:
        m = re.match(r"^(\d{1,2}) ([A-Z][A-Za-z ,\-]{3,60}?) \d+\s*$", ln.strip())
        if m: chapter_titles[m.group(2).strip().lower()] = int(m.group(1))
    for idx, h in enumerate(heads):
        end = heads[idx + 1]["line"] - 1 if idx + 1 < len(heads) else len(lines)
        seg = "\n".join(lines[h["line"]:end])
        words = len(re.findall(r"\S+", seg))
        title = h["title"]
        num = re.match(r"^(\d+)\.(\d+)(?:\.\d+)?\b", title)
        chap = re.match(r"^(\d+)\.(\d+)\b", title)
        if h["level"] == 1 and title.lower() in chapter_titles:
            current_chapter = chapter_titles[title.lower()]
        if chap:
            current_chapter = int(chap.group(1))
        key = (current_chapter, title.lower())
        if key in seen:  # running header duplicate: merge words
            rows[seen[key]]["source_words"] += words
            continue
        seen[key] = len(rows)
        rows.append({
            "chapter": current_chapter, "section": title, "level": h["level"],
            "source_words": words, "kind": kind_of(title, h["level"]),
            "status": "todo", "topic_files": [], "reason": "",
            "line": h["line"],
        })
    # level-1 chapter-title headings take the chapter of the next numbered section
    for i, r in enumerate(rows):
        if r["level"] == 1 or r["section"].lower() in chapter_titles:
            for nxt in rows[i+1:]:
                m = re.match(r"^(\d+)\.", nxt["section"])
                if m: r["chapter"] = int(m.group(1)); break
    # infer chapter number for level-1 chapter headings from following numbered section
    for r in rows:  # anything before chapter 1 is front matter
        if r["chapter"] == 0 and r["kind"] == "content": r["kind"] = "front-matter"
    chapters = sorted({r["chapter"] for r in rows if r["chapter"]})
    kb = ROOT / "knowledge-base" / book_id
    kb.mkdir(parents=True, exist_ok=True)
    toc = {"book": book_id, "headings_total": len(rows),
           "by_kind": {}, "chapters": chapters, "headings": rows}
    for r in rows:
        toc["by_kind"][r["kind"]] = toc["by_kind"].get(r["kind"], 0) + 1
    (kb / "toc.json").write_text(json.dumps(toc, indent=1, ensure_ascii=False))
    # coverage.json preserves existing statuses if re-run
    cov_path = kb / "coverage.json"
    old = {}
    if cov_path.exists():
        try:
            for r in json.loads(cov_path.read_text()).get("rows", []):
                old[(r["chapter"], r["section"].lower())] = r
        except Exception:
            pass
    for r in rows:
        o = old.get((r["chapter"], r["section"].lower()))
        if o and o.get("status") != "todo":
            r["status"], r["topic_files"], r["reason"] = o["status"], o.get("topic_files", []), o.get("reason", "")
    cov_path.write_text(json.dumps({"book": book_id, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"book={book_id} headings={len(rows)} by_kind={toc['by_kind']} chapters={chapters}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
