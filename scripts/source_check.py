#!/usr/bin/env python3
"""R15: every practice problem / quiz / interview item has source_topics pointing
to existing topic files, and at least one key term from its title appears there.
Scope: practice-*.md under knowledge-base/ (problems are '### Problem ...' with a
'source_topics: [...]' line), course/**/quiz.json, course/**/interview.md.
If there are no items yet, passes vacuously but reports 0.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def topic_index():
    idx = {}
    for f in (ROOT / "knowledge-base").rglob("topic-*.md"):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^id:\s*(\S+)", txt, re.M)
        idx[m.group(1) if m else f.stem] = txt.lower()
    return idx

def main():
    idx = topic_index(); fails, items = [], 0
    for f in sorted((ROOT / "knowledge-base").rglob("practice-*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for chunk in re.split(r"^### Problem", txt, flags=re.M)[1:]:
            items += 1
            m = re.search(r"source_topics:\s*\[([^\]]*)\]", chunk)
            if not m or not m.group(1).strip():
                fails.append(f"{f.relative_to(ROOT)}: problem without source_topics"); continue
            srcs = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
            for s in srcs:
                if s not in idx: fails.append(f"{f.relative_to(ROOT)}: source_topic '{s}' has no topic file")
    for f in sorted((ROOT / "course").rglob("quiz.json")) if (ROOT / "course").exists() else []:
        try: data = json.loads(f.read_text())
        except Exception as e: fails.append(f"{f}: invalid JSON {e}"); continue
        qs = data if isinstance(data, list) else data.get("questions", [])
        for q in qs:
            items += 1
            if not q.get("source_topics"): fails.append(f"{f}: quiz item without source_topics")
            else:
                for s in q["source_topics"]:
                    if s not in idx: fails.append(f"{f}: quiz source '{s}' missing")
    print(f"source-checked items: {items}, topic files indexed: {len(idx)}, failures: {len(fails)}")
    for x in fails: print("FAIL:", x)
    print("source_check:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
