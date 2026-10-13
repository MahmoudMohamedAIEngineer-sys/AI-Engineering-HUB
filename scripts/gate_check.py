#!/usr/bin/env python3
"""Gate runner (R18). Usage: gate_check.py <N>. Writes proofs/gate-<N>.txt.
Exits non-zero on any failure. Gate definitions (fixed, Section 4):
 0: structure+state+scripts run, every book has toc.json, skills-installed.md complete
 1: skills-map.json valid, every topic id/tag/reason, sources.md >=8 sources
 1B: reading-list files exist, every book_needed topic in a reading path (checked via reading-list.json)
 2: per-chapter: use --chapter <n> --book <id>: coverage for that chapter complete + depth+code for its files
 2B: coverage_check --book passes for every book
 3: coverage --skills + depth + source pass
Later gates (4-9) are added when their phases start; calling them now fails honestly.
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout + r.stderr)

def main():
    n = sys.argv[1] if len(sys.argv) > 1 else "?"
    extra = sys.argv[2:]
    out, fails = [], []
    def check(label, args):
        rc, o = run(args); out.append(f"$ {' '.join(args)}\n{o}")
        if rc != 0: fails.append(label)
    if n == "0":
        for p in ["PROGRESS.md", "progress.json", "DECISIONS.md", "BLOCKERS.md", ".gitignore",
                  "skills-map/sources.md", "design-base/skills-installed.md"]:
            ok = (ROOT / p).exists(); out.append(f"exists {p}: {ok}")
            if not ok: fails.append(p)
        for s in ["build_toc.py","coverage_check.py","depth_check.py","code_check.py","source_check.py","pdf_check.py","inventory.py","gate_check.py"]:
            ok = (ROOT / "scripts" / s).exists(); out.append(f"script {s}: {ok}")
            if not ok: fails.append(s)
        check("inventory runs", ["scripts/inventory.py"])
        for b in [d.name for d in (ROOT/"books").iterdir() if d.is_dir()]:
            ok = (ROOT/"knowledge-base"/b/"toc.json").exists(); out.append(f"toc {b}: {ok}")
            if not ok: fails.append(f"toc {b}")
        txt = (ROOT/"design-base/skills-installed.md").read_text() if (ROOT/"design-base/skills-installed.md").exists() else ""
        for kw in ["installed", "unavailable", "substituted"]:
            if kw not in txt: fails.append(f"skills-installed missing '{kw}'"); out.append(f"skills-installed lacks term: {kw}")
    elif n == "1":
        sm = ROOT/"skills-map/skills-map.json"
        try:
            d = json.loads(sm.read_text()); topics=[]
            def walk(sk):
                topics.extend(sk.get("topics",[]))
                for s in sk.get("skills",[]): walk(s)
            for a in d.get("areas",[]):
                for s in a.get("skills",[]): walk(s)
            bad = [t for t in topics if not t.get("id") or not t.get("tag") or not t.get("reason")]
            out.append(f"topics={len(topics)} invalid={len(bad)}")
            if bad: fails.append("invalid topics")
        except Exception as e:
            out.append(f"skills-map invalid: {e}"); fails.append("skills-map")
        src = (ROOT/"skills-map/sources.md").read_text() if (ROOT/"skills-map/sources.md").exists() else ""
        nsrc = src.count("http")
        out.append(f"sources.md urls~{nsrc}")
        if nsrc < 8: fails.append("sources <8")
    elif n in ("1B", "1b"):
        rl = ROOT/"skills-map/reading-list.json"
        if not rl.exists(): fails.append("reading-list.json missing")
        else:
            d = json.loads(rl.read_text())
            need = set(d.get("book_needed_topics", [])); paths = d.get("reading_paths", [])
            covered = set()
            for p in paths: covered.update(p.get("covers_topics", []))
            missing = need - covered
            out.append(f"book_needed_topics={len(need)} covered_by_paths={len(need & covered)} missing={len(missing)}")
            if missing: fails.append(f"topics without path: {sorted(missing)[:10]}")
            for p in paths:
                if not p.get("books"): fails.append(f"path {p.get('skill')} has no books")
                for b in p.get("books", []):
                    if b.get("verified") not in (True, "unverified"): fails.append(f"unverified-flag missing: {b.get('title')}")
    elif n == "2":
        # gate_check.py 2 --book <id> --chapter <n>
        opts = dict(zip(extra[::2], extra[1::2]))
        book, ch = opts.get("--book"), int(opts.get("--chapter", "0"))
        cov = json.loads((ROOT/"knowledge-base"/book/"coverage.json").read_text())
        rows = [r for r in cov["rows"] if r["chapter"] == ch and r["kind"] in ("content", "example")]
        todo = [r for r in rows if r["status"] == "todo"]
        out.append(f"chapter {ch}: headings {len(rows)}, todo {len(todo)}")
        if todo: fails.append("chapter headings todo")
        check("depth", ["scripts/depth_check.py", "--book", book])
        check("code", ["scripts/code_check.py"])
    elif n in ("2B", "2b"):
        for b in [d.name for d in (ROOT/"books").iterdir() if d.is_dir()]:
            check(f"coverage {b}", ["scripts/coverage_check.py", "--book", b])
    elif n == "3":
        check("skills coverage", ["scripts/coverage_check.py", "--skills"])
        check("depth", ["scripts/depth_check.py"])
        check("source", ["scripts/source_check.py"])
    else:
        out.append(f"Gate {n} not yet implemented in this phase order (Phases 4-9 not started). FAIL by design (R20).")
        fails.append(f"gate {n} not implemented")
    result = "PASS" if not fails else f"FAIL: {fails}"
    out.append(f"GATE {n}: {result}")
    text = "\n".join(out)
    (ROOT/"proofs").mkdir(exist_ok=True)
    (ROOT/f"proofs/gate-{n}.txt").write_text(text)
    print(text)
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
