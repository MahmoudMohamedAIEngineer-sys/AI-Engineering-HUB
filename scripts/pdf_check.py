#!/usr/bin/env python3
"""Phase 8 PDFs: file >=20KB, >=3 pages (counts '/Type /Page' occurrences as a
lower bound), no raw LaTeX commands in extracted strings. Runs only when
exports/ contains PDFs; before Phase 8 it reports none and passes vacuously.
Full text-ratio check runs in Phase 8 once site pages exist (see DECISIONS).
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def main():
    pdfs = sorted((ROOT / "exports").glob("*.pdf")) if (ROOT / "exports").exists() else []
    fails = []
    for p in pdfs:
        data = p.read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
        print(f"{p.name}: {len(data)} bytes, ~{pages} pages")
        if len(data) < 20000: fails.append(f"{p.name}: {len(data)} bytes (<20000) - outline-sized PDF")
        if pages < 3: fails.append(f"{p.name}: ~{pages} pages (<3)")
        if rb"\frac" in data or rb"\begin{" in data: fails.append(f"{p.name}: raw LaTeX present")
    print(f"pdfs checked: {len(pdfs)}, failures: {len(fails)}")
    for x in fails: print("FAIL:", x)
    print("pdf_check:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
