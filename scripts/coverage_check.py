#!/usr/bin/env python3
"""Usage: coverage_check.py --book <id> | --skills
--book: every content heading is covered/merged/skipped(reason) and topic files exist.
        example-kind headings must be merged (they live inside a section topic).
--skills: every must_know topic in skills-map.json has a topic file in knowledge-base/.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def book_check(book):
    p = ROOT / "knowledge-base" / book / "coverage.json"
    if not p.exists(): print(f"FAIL: {p} missing"); return 1
    rows = json.loads(p.read_text())["rows"]
    fails = []
    for r in rows:
        if r["kind"] in ("content", "example"):
            if r["status"] == "todo":
                fails.append(f"todo: ch{r['chapter']} {r['section']}")
            if r["status"] == "skipped" and not r.get("reason"):
                fails.append(f"skipped without reason: {r['section']}")
            if r["status"] == "covered":
                for tf in r["topic_files"]:
                    if not (ROOT / "knowledge-base" / book / tf).exists() and not (ROOT / tf).exists():
                        # topic_files stored relative to knowledge-base/<book>/
                        if not (ROOT / "knowledge-base" / tf).exists():
                            fails.append(f"missing topic file {tf} for {r['section']}")
            if r["status"] == "merged" and not r["topic_files"]:
                fails.append(f"merged without target topic file: {r['section']}")
    done = sum(1 for r in rows if r["kind"] in ("content", "example") and r["status"] != "todo")
    tot = sum(1 for r in rows if r["kind"] in ("content", "example"))
    print(f"book={book} content+example headings accounted {done}/{tot}")
    for f in fails: print("FAIL:", f)
    print("coverage_check --book:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1

def skills_check():
    sm = ROOT / "skills-map" / "skills-map.json"
    if not sm.exists(): print("FAIL: skills-map.json missing"); return 1
    data = json.loads(sm.read_text())
    topics = []
    def walk(node):
        for t in node.get("topics", []): topics.append(t)
        for s in node.get("skills", []): walk(s)
    for a in data.get("areas", []):
        for s in a.get("skills", []): walk(s)
    kb_files = list((ROOT / "knowledge-base").rglob("topic-*.md"))
    # map topic id -> file via frontmatter id
    ids = set()
    for f in kb_files:
        txt = f.read_text(encoding="utf-8", errors="replace")[:2000]
        m = __import__("re").search(r"^id:\s*(\S+)", txt, __import__("re").M)
        if m: ids.add(m.group(1))
    missing = [t["id"] for t in topics if t["tag"] == "must_know" and t["id"] not in ids]
    print(f"must_know topics={sum(1 for t in topics if t['tag']=='must_know')} with_topic_file={len([t for t in topics if t['tag']=='must_know' and t['id'] in ids])} missing={len(missing)}")
    for m in missing: print("MISSING:", m)
    print("coverage_check --skills:", "PASS" if not missing else "FAIL")
    return 0 if not missing else 1

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--book": sys.exit(book_check(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--skills": sys.exit(skills_check())
    print(__doc__); sys.exit(2)
