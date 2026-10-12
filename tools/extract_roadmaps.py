#!/usr/bin/env python3
"""Extract the intended curriculum from the two first-party roadmap websites.

Inputs (read-only, never modified):
  * `perfect_resources_44_skills.html`  — 44 skills, each with one book + one free
    resource + one paid resource, plus a "what you learn" summary.
  * `study_timetable_8hrs.html`         — the same skills with day counts, week
    positions, hours, and (for a subset) a day-by-day study plan that references
    specific material such as "MML Book Ch.2", "3B1B Ch.4-5", "StatQuest".

Outputs (generated — do not hand-edit):
  * knowledge/roadmaps/perfect-resources.raw.json
  * knowledge/roadmaps/study-timetable.raw.json
  * knowledge/roadmaps/consolidated-inventory.json
  * knowledge/roadmaps/data-quality-report.json

Design notes:
  * Both sites number the same 44 skills 01..44. That shared numbering is treated as
    the stable external identifier for a roadmap skill (`SK-01`..`SK-44`). The
    extractor verifies that the two sites agree before merging them and reports any
    disagreement instead of silently resolving it.
  * The timetable's *sidebar* carries day counts for all 44 skills; its detailed
    day-by-day blocks cover only the first part of the curriculum. The extractor
    records that difference explicitly (`plan_depth`) so no downstream report can
    imply that a day-level plan exists where it does not.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from htmldom import Node, parse, strip_scripts_and_styles, text_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESOURCES_HTML = REPO / "perfect_resources_44_skills.html"
TIMETABLE_HTML = REPO / "study_timetable_8hrs.html"
OUT_DIR = REPO / "knowledge" / "roadmaps"

PHASE_NAMES = {
    "foundation": "Foundation",
    "swe": "SWE Core",
    "de": "Data Engineering",
    "ml": "AI/ML Core",
    "spec": "Specialization",
    "sprint": "Project Sprint",
}

DAY_RE = re.compile(r"^DAY\s+(\d+)$")
FIRST_INT_RE = re.compile(r"(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(text: str) -> str:
    return " ".join(text.split())


def first_int(text: str) -> int | None:
    match = FIRST_INT_RE.search((text or "").replace(",", ""))
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Site 1 — "The Perfect Resource"
# --------------------------------------------------------------------------- #

def parse_resources_site(html_text: str) -> Dict[str, Any]:
    root = parse(strip_scripts_and_styles(html_text))

    skills: List[Dict[str, Any]] = []
    for card in root.find_all("skill-card"):
        number_text = text_of(card.find("skill-num"))
        name = text_of(card.find("skill-name"))
        if not number_text.isdigit() or not name:
            continue

        tags = [text_of(t) for t in card.find_all("skill-tag") if text_of(t)]
        what = re.sub(r"^What you learn:\s*", "", text_of(card.find("what-box"))).strip()

        resources: List[Dict[str, str]] = []
        for res in card.find_all("resource-card"):
            badge = text_of(res.find("res-badge"))
            resources.append(
                {
                    "kind": {"BOOK": "book", "FREE": "free", "PAID": "paid"}.get(
                        badge, badge.lower() or "unspecified"
                    ),
                    "platform": text_of(res.find("res-platform")),
                    "title": text_of(res.find("res-title")),
                    "author_year": text_of(res.find("res-author")),
                    "rationale": text_of(res.find("res-why")),
                }
            )

        skills.append(
            {
                "roadmap_number": int(number_text),
                "name": name,
                "declared_track": card.attr("data-track") or None,
                "declared_tags": tags,
                "what_you_learn": what,
                "intended_resources": resources,
                "dom_id": card.attr("id") or None,
            }
        )

    sidebar = []
    for link in root.find_all("sidebar-link"):
        number = text_of(link.find("num"))
        sidebar.append(
            {
                "number": number,
                "name": link.own_text(),
                "target": link.attr("onclick"),
            }
        )

    return {
        "site": "perfect_resources_44_skills.html",
        "title": "The Perfect Resource — 44 Skills, 132 Resources",
        "purpose": (
            "Per-skill resource selection: one canonical book plus one free and one paid "
            "course/resource for each of the 44 roadmap skills."
        ),
        "declared_counts": {
            "header_counter": text_of(root.find("skill-count")),
            "hero_sub": text_of(root.find("hero-sub")),
        },
        "legend_claims": [clean(text_of(item)) for item in root.find_all("leg-item")],
        "skills": sorted(skills, key=lambda s: s["roadmap_number"]),
        "sidebar": sidebar,
    }


# --------------------------------------------------------------------------- #
# Site 2 — "Master Study Timetable"
# --------------------------------------------------------------------------- #

def parse_timetable_site(html_text: str) -> Dict[str, Any]:
    root = parse(strip_scripts_and_styles(html_text))

    hero_stats = [
        {"value": text_of(box.find("stat-num")), "label": text_of(box.find("stat-label"))}
        for box in root.find_all("stat-box")
    ]

    header_stats: List[Dict[str, str]] = []
    header = root.find("header-stats")
    if header is not None:
        for span in header.find_all(""):
            pass
        for span in [n for n in header.walk() if n.tag == "span" and n.parent is header]:
            header_stats.append({"text": clean(span.text())})

    # Sidebar: canonical per-skill day counts, grouped by phase heading.
    sidebar_entries: List[Dict[str, Any]] = []
    current_section = None
    for node in root.walk():
        if node.has_class("sb-section-head"):
            current_section = clean(node.text())
        elif node.has_class("sb-link"):
            sidebar_entries.append(
                {
                    "number": text_of(node.find("sb-num")),
                    "name": node.own_text(),
                    "days_text": text_of(node.find("sb-days")),
                    "group": current_section,
                    "target": node.attr("onclick"),
                    "is_sprint": "★" in text_of(node.find("sb-num")),
                }
            )

    blocks: List[Dict[str, Any]] = []
    for block in root.find_all("skill-block"):
        number_text = text_of(block.find("sk-num"))
        is_sprint = "phase-sprint" in block.classes or not number_text.strip().isdigit()
        weeks: List[Dict[str, Any]] = []
        days: List[Dict[str, Any]] = []
        for week_row in block.find_all("week-row"):
            label = text_of(week_row.find("week-label"))
            cells = []
            for cell in week_row.find_all("day-cell"):
                day_label = text_of(cell.find("day-num"))
                topic = text_of(cell.find("day-topic"))
                detail = text_of(cell.find("day-detail"))
                cells.append(
                    {
                        "label": day_label or None,
                        "topic": topic or None,
                        "detail": detail or None,
                        "hours_text": text_of(cell.find("day-hours")) or None,
                        "is_rest": cell.has_class("rest"),
                    }
                )
                match = DAY_RE.match(day_label)
                if match:
                    days.append(
                        {
                            "day": int(match.group(1)),
                            "topic": topic or None,
                            "detail": detail or None,
                        }
                    )
            weeks.append({"label": label or None, "cells": cells})

        blocks.append(
            {
                "dom_id": block.attr("id"),
                "roadmap_number": int(number_text) if number_text.strip().isdigit() else None,
                "is_sprint": is_sprint,
                "name": text_of(block.find("sk-name")),
                "declared_phase": block.attr("data-phase") or None,
                "week_badge": text_of(block.find("sk-week-badge")),
                "days_badge": text_of(block.find("sk-days-badge")),
                "hours_badge": text_of(block.find("sk-hrs-badge")),
                "weeks": weeks,
                "days": days,
                "has_day_level_plan": bool(days),
            }
        )

    return {
        "site": "study_timetable_8hrs.html",
        "title": "Master Study Timetable — 8hrs/day · 44 Skills",
        "purpose": (
            "Scheduling layer over the same 44 skills: phase order, weeks, study days, "
            "hours, and a day-by-day plan with resource references."
        ),
        "header_stats": header_stats,
        "hero_stats": hero_stats,
        "phase_labels": [clean(text_of(item)) for item in root.find_all("ph-item")],
        "sidebar": sidebar_entries,
        "blocks": blocks,
    }


# --------------------------------------------------------------------------- #
# Merge + quality checks
# --------------------------------------------------------------------------- #

def consolidate(resources: Dict[str, Any], timetable: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []

    def report(severity: str, code: str, detail: str) -> None:
        issues.append({"severity": severity, "code": code, "detail": detail})

    resource_by_number = {s["roadmap_number"]: s for s in resources["skills"]}
    sidebar_by_number = {
        int(e["number"]): e for e in timetable["sidebar"] if e["number"].isdigit()
    }
    blocks_by_number = {
        b["roadmap_number"]: b for b in timetable["blocks"] if b["roadmap_number"] is not None
    }
    sprints = [b for b in timetable["blocks"] if b["is_sprint"]]

    for number in sorted(set(resource_by_number) | set(sidebar_by_number)):
        if number not in resource_by_number:
            report(
                "error",
                "skill-missing-in-resource-index",
                f"SK-{number:02d} has schedule data but no resource entry.",
            )
        if number not in sidebar_by_number:
            report(
                "error",
                "skill-missing-in-schedule",
                f"SK-{number:02d} has resources but no schedule entry.",
            )

    skills: List[Dict[str, Any]] = []
    for number in sorted(set(resource_by_number) & set(sidebar_by_number)):
        res = resource_by_number[number]
        sidebar = sidebar_by_number[number]
        block = blocks_by_number.get(number)

        if res["name"].strip().lower() != sidebar["name"].strip().lower():
            report(
                "warning",
                "skill-name-mismatch",
                f"SK-{number:02d}: '{res['name']}' (resource index) vs "
                f"'{sidebar['name']}' (timetable sidebar).",
            )

        day_count = first_int(sidebar["days_text"])
        plan_days = sorted(d["day"] for d in block["days"]) if block else []
        hours = first_int(block["hours_badge"]) if block else (
            day_count * 8 if day_count is not None else None
        )

        if block and day_count is not None and len(plan_days) != day_count:
            report(
                "warning",
                "declared-days-exceed-planned-days",
                f"SK-{number:02d} '{res['name']}': badge declares {day_count} study days but "
                f"{len(plan_days)} day cells carry a DAY number "
                f"(days {plan_days[0]}–{plan_days[-1]}). Unplanned days: "
                f"{sorted(set(range(plan_days[0], plan_days[-1] + 1)) - set(plan_days))}.",
            )

        skills.append(
            {
                "id": f"SK-{number:02d}",
                "roadmap_number": number,
                "name": res["name"],
                "phase": PHASE_NAMES.get(
                    res["declared_track"] or (block or {}).get("declared_phase") or "",
                    res["declared_track"],
                ),
                "phase_id": res["declared_track"],
                "tags": res["declared_tags"],
                "what_you_learn": res["what_you_learn"],
                "schedule": {
                    "study_days": day_count,
                    "hours": hours,
                    "week_label": (block or {}).get("week_badge") or None,
                    "day_numbers": plan_days or None,
                },
                "plan_depth": "day-by-day" if plan_days else "summary-only",
                "intended_resources": res["intended_resources"],
                "study_plan": [
                    {"day": d["day"], "topic": d["topic"], "detail": d["detail"]} for d in block["days"]
                ]
                if block
                else [],
            }
        )

    sprint_entries = [
        {
            "id": (b["dom_id"] or "sprint").upper().replace("SK", "SPRINT-"),
            "name": b["name"],
            "phase": PHASE_NAMES.get(b["declared_phase"] or "", b["declared_phase"]),
            "phase_id": b["declared_phase"],
            "schedule": {
                "study_days": first_int(b["days_badge"]),
                "hours": first_int(b["hours_badge"]),
                "week_label": b["week_badge"] or None,
                "day_numbers": sorted(d["day"] for d in b["days"]) or None,
            },
            "plan_depth": "day-by-day" if b["days"] else "summary-only",
            "study_plan": [
                {"day": d["day"], "topic": d["topic"], "detail": d["detail"]} for d in b["days"]
            ],
        }
        for b in sprints
    ]

    # --- totals reconciliation -------------------------------------------------
    sidebar_days = sum(first_int(e["days_text"]) or 0 for e in timetable["sidebar"])
    declared_days = next(
        (first_int(s["value"]) for s in timetable["hero_stats"] if s["label"] == "Study Days"), None
    )
    if declared_days is not None and declared_days != sidebar_days:
        report(
            "warning",
            "study-day-total-mismatch",
            f"Hero declares {declared_days} study days; the 44 skill entries plus project "
            f"sprints sum to {sidebar_days}.",
        )
    declared_hours = next(
        (first_int(s["value"]) for s in timetable["hero_stats"] if s["label"] == "Total Hours"), None
    )
    if declared_hours is not None and sidebar_days * 8 != declared_hours:
        report(
            "warning",
            "study-hour-total-mismatch",
            f"Hero declares {declared_hours} hours; {sidebar_days} study days x 8 h = "
            f"{sidebar_days * 8} h.",
        )
    header_hours = None
    for stat in timetable["header_stats"]:
        if "TOTAL HRS" in stat["text"]:
            header_hours = first_int(stat["text"])
    if header_hours is not None and declared_hours is not None and header_hours != declared_hours:
        report(
            "warning",
            "header-vs-hero-hour-mismatch",
            f"The page header advertises {header_hours:,} total hours while the stat row "
            f"advertises {declared_hours:,}.",
        )

    # --- resource-index claims -------------------------------------------------
    actual_track_counts: Dict[str, int] = {}
    for skill in skills:
        key = skill["phase_id"] or "unknown"
        actual_track_counts[key] = actual_track_counts.get(key, 0) + 1
    report(
        "warning",
        "legend-counts-inconsistent-with-content",
        "The resource index filter legend declares "
        f"{resources['legend_claims'][:5]} while the page contains {len(skills)} skill cards "
        f"distributed as {actual_track_counts}. Page content is treated as canonical.",
    )

    resource_kinds: Dict[str, int] = {}
    incomplete_resource_sets = []
    for skill in skills:
        kinds = [r["kind"] for r in skill["intended_resources"]]
        for kind in kinds:
            resource_kinds[kind] = resource_kinds.get(kind, 0) + 1
        if sorted(kinds) != ["book", "free", "paid"]:
            incomplete_resource_sets.append(skill["id"])
    if incomplete_resource_sets:
        report(
            "warning",
            "incomplete-resource-set",
            "These skills do not carry exactly one book, one free and one paid resource: "
            + ", ".join(incomplete_resource_sets),
        )

    return {
        "generated_by": "tools/extract_roadmaps.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "id_scheme": (
            "SK-01..SK-44 are stable identifiers for the 44 roadmap skills, taken from the "
            "numbering both source sites already share. Project sprints are recorded separately "
            "and are not numbered as skills."
        ),
        "counts": {
            "skills": len(skills),
            "project_sprints": len(sprint_entries),
            "by_phase": actual_track_counts,
            "resources_by_kind": resource_kinds,
            "summed_study_days_including_sprints": sidebar_days,
            "skills_with_day_level_plan": sum(1 for s in skills if s["study_plan"]),
            "skills_summary_only": sum(1 for s in skills if not s["study_plan"]),
        },
        "skills": skills,
        "project_sprints": sprint_entries,
        "data_quality_issues": issues,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resources = parse_resources_site(RESOURCES_HTML.read_text(encoding="utf-8"))
    timetable = parse_timetable_site(TIMETABLE_HTML.read_text(encoding="utf-8"))
    consolidated = consolidate(resources, timetable)

    for name, payload in (
        ("perfect-resources.raw.json", resources),
        ("study-timetable.raw.json", timetable),
        ("consolidated-inventory.json", consolidated),
    ):
        (OUT_DIR / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    report = {
        "generated_by": "tools/extract_roadmaps.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sources": [
            {
                "path": RESOURCES_HTML.name,
                "sha256": sha256(RESOURCES_HTML),
                "bytes": RESOURCES_HTML.stat().st_size,
                "role": "resource selection per skill (44 skills x 3 resources)",
            },
            {
                "path": TIMETABLE_HTML.name,
                "sha256": sha256(TIMETABLE_HTML),
                "bytes": TIMETABLE_HTML.stat().st_size,
                "role": "sequencing, day counts and (partial) day-level plan",
            },
        ],
        "extracted": {
            "skills_in_resource_index": len(resources["skills"]),
            "schedule_entries_in_sidebar": sum(
                1 for e in timetable["sidebar"] if e["number"].isdigit()
            ),
            "day_level_blocks": sum(1 for b in timetable["blocks"] if b["days"]),
            "consolidated_skills": consolidated["counts"]["skills"],
            "day_level_plan_entries": sum(len(s["study_plan"]) for s in consolidated["skills"]),
        },
        "issues": consolidated["data_quality_issues"],
    }
    (OUT_DIR / "data-quality-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(report["extracted"], indent=2))
    print(f"issues: {len(report['issues'])}")
    for issue in report["issues"]:
        print(f"  [{issue['severity']}] {issue['code']}: {issue['detail'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
