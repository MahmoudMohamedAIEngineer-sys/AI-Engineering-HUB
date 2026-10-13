#!/usr/bin/env python3
"""Print exact counts (R19). Never fails. All reporting must quote this verbatim."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def count(pat, base="knowledge-base"):
    b = ROOT / base
    return len(list(b.rglob(pat))) if b.exists() else 0

def main():
    books = [d.name for d in sorted((ROOT / "books").iterdir()) if d.is_dir()] if (ROOT / "books").exists() else []
    print("INVENTORY")
    print(f"books: {len(books)} {books}")
    for b in books:
        cov = ROOT / "knowledge-base" / b / "coverage.json"
        if cov.exists():
            rows = json.loads(cov.read_text())["rows"]
            ce = [r for r in rows if r["kind"] in ("content", "example")]
            acc = [r for r in ce if r["status"] != "todo"]
            print(f"book {b}: headings_total={len(rows)} content+example={len(ce)} accounted={len(acc)} todo={len(ce)-len(acc)}")
            for ch in sorted({r['chapter'] for r in rows}):
                c = [r for r in ce if r["chapter"] == ch]; a = [r for r in c if r["status"] != "todo"]
                if c: print(f"  ch{ch}: accounted {len(a)}/{len(c)}")
    print(f"topic_files: {count('topic-*.md')}")
    print(f"practice_files: {count('practice-*.md')}")
    print(f"web_topic_files: {len(list((ROOT/'knowledge-base'/'web').glob('topic-*.md'))) if (ROOT/'knowledge-base'/'web').exists() else 0}")
    probs = 0
    for f in (ROOT / "knowledge-base").rglob("practice-*.md"):
        probs += len(re.findall(r"^### Problem", f.read_text(encoding='utf-8', errors='replace'), flags=re.M))
    print(f"practice_problems: {probs}")
    mods = 0
    cm = ROOT / "course" / "course-map.json"
    if cm.exists():
        try: mods = len(json.loads(cm.read_text()).get("modules", []))
        except Exception: pass
    print(f"modules_in_course_map: {mods}")
    pdfs = sorted((ROOT / "exports").glob("*.pdf")) if (ROOT / "exports").exists() else []
    print(f"pdfs: {len(pdfs)} {[ (p.name, p.stat().st_size) for p in pdfs ]}")
    sm = ROOT / "skills-map" / "skills-map.json"
    if sm.exists():
        d = json.loads(sm.read_text()); topics = []
        def walk(sk):
            topics.extend(sk.get("topics", []))
            for s in sk.get("skills", []): walk(s)
        for a in d.get("areas", []):
            for s in a.get("skills", []): walk(s)
        print(f"skills_map_topics: total={len(topics)} must_know={sum(1 for t in topics if t['tag']=='must_know')} good_to_know={sum(1 for t in topics if t['tag']=='good_to_know')} skip={sum(1 for t in topics if t['tag']=='skip')}")
    prog = ROOT / "progress.json"
    if prog.exists(): print("progress.json:", prog.read_text().strip().replace("\n", " ")[:400])

if __name__ == "__main__":
    main()
