#!/usr/bin/env python3
"""Run the executable checks that back the lessons, and record what actually happened.

Each `content/checks/*.py` module exposes functions named `check_*`. Every function returns a
string describing the evidence it produced, or raises (an assertion failure, most often). This
tool runs them, captures the output verbatim, and writes `knowledge/lessons/checks-report.json`.

Design rules
  * A check that raises is a failure, and its message is recorded. Nothing is summarised away.
  * The report states which interpreter and which library versions were used, because "the
    numbers looked right on the machine that wrote the prose" is not evidence.
  * The tool never edits a verification record. Verification records are authored, and they cite
    this report by hash; a change to the checks therefore invalidates an old record visibly.

Run with the development virtualenv: `.venv/bin/python tools/run_content_checks.py`
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import platform
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
CHECKS_DIR = REPO / "content" / "checks"
OUT_PATH = REPO / "knowledge" / "lessons" / "checks-report.json"


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"content_check_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "sympy", "yaml"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "not installed"
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def main() -> int:
    if not CHECKS_DIR.exists():
        print("no content/checks directory found", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    failed = 0

    for path in sorted(CHECKS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = load_module(path)
        except Exception:
            failed += 1
            results.append(
                {
                    "module": path.name,
                    "check": "(module import)",
                    "status": "failed",
                    "evidence": traceback.format_exc(),
                }
            )
            continue

        for name in sorted(dir(module)):
            if not name.startswith("check_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            captured = io.StringIO()
            try:
                with redirect_stdout(captured):
                    evidence = function()
                status = "passed"
            except Exception:
                status = "failed"
                failed += 1
                evidence = traceback.format_exc()
            results.append(
                {
                    "module": path.name,
                    "check": name,
                    "status": status,
                    "evidence": (evidence if isinstance(evidence, str) else str(evidence)).strip(),
                    "stdout": captured.getvalue().strip() or None,
                }
            )

    report = {
        "generated_by": "tools/run_content_checks.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "environment": library_versions(),
        "checks_directory_hash": hashlib.sha256(
            b"".join(sorted(path.read_bytes() for path in CHECKS_DIR.glob("*.py")))
        ).hexdigest(),
        "summary": {
            "checks": len(results),
            "passed": len(results) - failed,
            "failed": failed,
        },
        "results": results,
        "what_this_establishes": (
            "That each check function ran to completion and produced the reported evidence. "
            "It does not establish source fidelity or pedagogical quality, which are separate "
            "checks in the lesson's verification record."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"checks: {report['summary']['checks']}  passed: {report['summary']['passed']}  failed: {failed}")
    for result in results:
        marker = "PASS" if result["status"] == "passed" else "FAIL"
        print(f"  [{marker}] {result['module']}::{result['check']}")

    if failed:
        print(f"\n{failed} check(s) failed. See {OUT_PATH.relative_to(REPO)} for the tracebacks.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
