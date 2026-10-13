#!/usr/bin/env python3
"""Run every ```python code block in topic/practice files under knowledge-base/.
Convention: a block may be followed by ```text block containing expected output;
if present, stdout must match it exactly (stripped). Blocks marked
'# expect-error' are skipped. Pure-stdlib only (runs with system python3).
"""
import re, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def blocks(txt):
    pattern = re.compile(r"```python\n(.*?)```(?:\s*```text\n(.*?)```)?", re.S)
    return pattern.findall(txt)

def main():
    fails, total = [], 0
    for f in sorted((ROOT / "knowledge-base").rglob("*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for i, (code, expected) in enumerate(blocks(txt)):
            if "# expect-error" in code: continue
            total += 1
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                tf.write(code); path = tf.name
            r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                fails.append(f"{f.relative_to(ROOT)} block{i}: exit {r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
            elif expected.strip() and r.stdout.strip() != expected.strip():
                fails.append(f"{f.relative_to(ROOT)} block{i}: output mismatch\n got: {r.stdout.strip()!r}\n exp: {expected.strip()!r}")
    print(f"code blocks run: {total}, failures: {len(fails)}")
    for x in fails: print("FAIL:", x)
    print("code_check:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
