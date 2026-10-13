#!/usr/bin/env python3
"""Depth check (R17/R18). For every topic-*.md under knowledge-base/:
- template sections 1..11 present; sections 1-5 each >=80 words
- length parity: for book topics, sum of sections1-5 words over the topic files
  mapped to a book section in coverage.json >= that section's source_words (ratio>=1.0)
- web topics: sections1-5 >=1200 words
- no banned phrases; paired practice-<slug>.md exists with >=8 problems (### Problem)
Usage: depth_check.py [--book <id>]  (default: all)
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BANNED = ["it can be shown", "obviously", "clearly", "trivially", "left as an exercise",
          "tbd", "todo", "lorem", "example only"]

def sections(txt):
    parts = re.split(r"^## (\d+)\.\s+", txt, flags=re.M)
    out = {}
    # parts: [pre, num, body, num, body, ...]
    for i in range(1, len(parts) - 1, 2):
        out[int(parts[i])] = parts[i + 1]
    return out

def check_file(f, fails):
    txt = f.read_text(encoding="utf-8", errors="replace")
    rel = f.relative_to(ROOT)
    secs = sections(txt)
    for n in range(1, 12):
        if n not in secs: fails.append(f"{rel}: missing template section {n}")
    w = {}
    for n in range(1, 6):
        if n in secs:
            body = re.split(r"^## ", secs[n], flags=re.M)[0]
            w[n] = len(re.findall(r"\S+", body))
            if w[n] < 80: fails.append(f"{rel}: section {n} has {w[n]} words (<80)")
    low = txt.lower()
    for b in BANNED:
        if b in low: fails.append(f"{rel}: banned phrase '{b}'")
    # practice pair
    slug = f.name[len("topic-"):-len(".md")]
    prac = f.parent / f"practice-{slug}.md"
    if not prac.exists():
        fails.append(f"{rel}: missing paired practice-{slug}.md")
    else:
        probs = len(re.findall(r"^### Problem", prac.read_text(encoding="utf-8", errors="replace"), flags=re.M))
        if probs < 8: fails.append(f"{prac.relative_to(ROOT)}: {probs} problems (<8)")
    return sum(w.values())

def main():
    book_filter = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--book": book_filter = sys.argv[2]
    fails, files = [], []
    base = ROOT / "knowledge-base"
    for f in sorted(base.rglob("topic-*.md")):
        if book_filter and book_filter not in str(f): continue
        files.append(f); check_file(f, fails)
    # parity per book section
    for covp in sorted(base.glob("*/coverage.json")):
        if book_filter and book_filter not in str(covp): continue
        cov = json.loads(covp.read_text())
        bookdir = covp.parent
        # word counts per file
        wc = {}
        for f in files:
            if f.parent.parent == bookdir or bookdir in f.parents:
                wc[str(f.relative_to(bookdir))] = wc.get(str(f.relative_to(bookdir)), 0)
        # recompute sections1-5 words per topic file in this book
        per_file = {}
        for f in sorted(bookdir.rglob("topic-*.md")):
            secs = sections(f.read_text(encoding="utf-8", errors="replace"))
            tot = 0
            for n in range(1, 6):
                if n in secs:
                    tot += len(re.findall(r"\S+", re.split(r"^## ", secs[n], flags=re.M)[0]))
            per_file[str(f.relative_to(bookdir))] = tot
        for r in cov["rows"]:
            if r["kind"] != "content" or r["status"] != "covered": continue
            # Prompt Phase 2 Step 2: chapter introductions/summaries live in chapter-summary.md,
            # not topic files, so parity (a topic-file measure) does not apply to them.
            if r["topic_files"] and all(tf.endswith("chapter-summary.md") for tf in r["topic_files"]): continue
            tot = sum(per_file.get(tf, 0) for tf in r["topic_files"])
            ratio = tot / r["source_words"] if r["source_words"] else 0
            print(f"parity ch{r['chapter']} {r['section']}: {tot}/{r['source_words']} = {ratio:.2f}")
            if ratio < 1.0:
                fails.append(f"parity <1.0 for ch{r['chapter']} {r['section']}: {ratio:.2f}")
    # web topics length
    webdir = base / "web"
    if webdir.exists() and not book_filter:
        for f in sorted(webdir.glob("topic-*.md")):
            secs = sections(f.read_text(encoding="utf-8", errors="replace"))
            tot = sum(len(re.findall(r"\S+", re.split(r"^## ", secs.get(n, ""), flags=re.M)[0])) for n in range(1, 6))
            if tot < 1200: fails.append(f"{f.relative_to(ROOT)}: web topic sections1-5 {tot} words (<1200)")
    print(f"topic files checked: {len(files)}")
    for x in fails: print("FAIL:", x)
    print("depth_check:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
