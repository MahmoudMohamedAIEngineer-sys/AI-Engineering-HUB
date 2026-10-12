#!/usr/bin/env python3
"""Derive the coverage report from the knowledge base.

Inputs (read):
  knowledge/curriculum/domains.json
  knowledge/curriculum/competencies.json
  knowledge/curriculum/skill-competency-map.json
  knowledge/sources/mml/section-concept-map.json
  knowledge/sources/mml/structure.json
  knowledge/sources/mml/inventory.json
  knowledge/lessons/registry.json

Outputs (generated):
  knowledge/curriculum/coverage.json
  knowledge/curriculum/coverage-report.md

The point of this script is to make unsupported claims visible. It computes what the
artefacts actually support, compares that with the status someone wrote down by hand, and
reports any disagreement rather than quietly trusting either side.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "knowledge"


def load(relative: str) -> Any:
    return json.loads((KB / relative).read_text(encoding="utf-8"))


def main() -> int:
    domains = load("curriculum/domains.json")["domains"]
    competencies = load("curriculum/competencies.json")["competencies"]
    skill_map = load("curriculum/skill-competency-map.json")
    section_map = load("sources/mml/section-concept-map.json")
    structure = load("sources/mml/structure.json")
    inventory = load("sources/mml/inventory.json")
    registry = load("lessons/registry.json")

    sections_in_source: Set[str] = {
        section["id"] for chapter in structure["chapters"] for section in chapter["sections"]
    }
    competencies_by_id = {c["id"]: c for c in competencies}

    # Which competencies does each source section claim to support?
    sections_by_competency: Dict[str, List[str]] = {}
    for section in section_map["sections"]:
        for competency in section["competencies"]:
            sections_by_competency.setdefault(competency, []).append(section["id"])

    # Which skills claim each competency?
    skills_by_competency: Dict[str, List[str]] = {}
    for skill in skill_map["skills"]:
        for competency in skill["competencies"]:
            skills_by_competency.setdefault(competency, []).append(skill["id"])

    # Which lessons exist per competency?
    lessons_by_competency: Dict[str, List[str]] = {}
    for lesson in registry["lessons"]:
        for competency in lesson["competencies"]:
            lessons_by_competency.setdefault(competency, []).append(lesson["id"])

    entries: List[Dict[str, Any]] = []
    for competency in competencies:
        cid = competency["id"]
        coverage = competency["source_coverage"]
        declared_sections = list(coverage["mml_sections"])
        mapped_sections = sorted(sections_by_competency.get(cid, []))
        missing_from_map = sorted(set(declared_sections) - set(mapped_sections))
        missing_from_competency = sorted(set(mapped_sections) - set(declared_sections))
        unknown_sections = sorted(s for s in declared_sections if s not in sections_in_source)

        lessons = lessons_by_competency.get(cid, [])
        verified_lessons = [
            lesson["id"]
            for lesson in registry["lessons"]
            if lesson["id"] in lessons and lesson.get("publication_state") == "verified"
        ]

        if coverage["status"] == "none":
            computed = "no_source_supplied"
        elif coverage["status"] == "partial" and not lessons:
            computed = "partially_sourced_no_lessons"
        elif coverage["status"] == "sourced" and not lessons:
            computed = "sourced_no_lessons"
        elif lessons and len(verified_lessons) == len(lessons):
            computed = "verified"
        elif lessons:
            computed = "lessons_exist_not_all_verified"
        else:
            computed = "undefined"

        entries.append(
            {
                "competency_id": cid,
                "name": competency["name"],
                "domain": competency["domain"],
                "target_level": competency["target_level"],
                "declared_source_status": coverage["status"],
                "mml_sections_declared": declared_sections,
                "mml_sections_mapped": mapped_sections,
                "mml_sections_unknown": unknown_sections,
                "declared_but_not_mapped": missing_from_map,
                "mapped_but_not_declared": missing_from_competency,
                "roadmap_skills": competency["roadmap_skills"],
                "lessons": lessons,
                "verified_lessons": verified_lessons,
                "computed_coverage": computed,
                "supplemental_needed": coverage["supplemental_needed"],
            }
        )

    # --- integrity issues across the artefacts -------------------------------
    issues: List[Dict[str, str]] = []
    for entry in entries:
        if entry["mml_sections_unknown"]:
            issues.append(
                {
                    "code": "competency-cites-unknown-section",
                    "detail": f"{entry['competency_id']} cites sections that do not exist in the source index: "
                    + ", ".join(entry["mml_sections_unknown"]),
                }
            )
        if entry["declared_but_not_mapped"]:
            issues.append(
                {
                    "code": "section-map-missing-competency-link",
                    "detail": f"{entry['competency_id']} declares sections {', '.join(entry['declared_but_not_mapped'])} "
                    "but section-concept-map.json does not link them back.",
                }
            )
        if entry["mapped_but_not_declared"]:
            issues.append(
                {
                    "code": "competency-missing-declared-section",
                    "detail": f"{entry['competency_id']} is linked from sections {', '.join(entry['mapped_but_not_declared'])} "
                    "in the section map but does not declare them.",
                }
            )
        if entry["computed_coverage"] == "verified" and not entry["verified_lessons"]:
            issues.append(
                {
                    "code": "coverage-claims-verified-without-verification-records",
                    "detail": f"{entry['competency_id']} reports verified coverage but has no verified lesson.",
                }
            )

    # Section titles live in the generated source index, not duplicated in the map.
    section_titles = {
        section["id"]: section["title"]
        for chapter in structure["chapters"]
        for section in chapter["sections"]
    }
    unmapped_sections = [
        {
            "id": section["id"],
            "role": section["role"],
            "title": section_titles.get(section["id"], "(title not in source index)"),
        }
        for section in section_map["sections"]
        if not section["competencies"]
    ]

    # --- roadmap-side gaps ---------------------------------------------------
    skills_without_sources = [
        {
            "id": skill["id"],
            "name": skill["name"],
            "status": skill["status"],
            "source_need": skill["source_need"],
        }
        for skill in skill_map["skills"]
        if skill["status"] in {"awaiting_sources", "identified", "gap_under_review"}
    ]
    skills_with_cross_skill_prerequisites_met_later = []
    order = {skill["id"]: index for index, skill in enumerate(skill_map["skills"])}
    for index, skill in enumerate(skill_map["skills"]):
        for required in skill["requires_skills"]:
            if required in order and order[required] > index:
                skills_with_cross_skill_prerequisites_met_later.append(
                    {
                        "skill": skill["id"],
                        "requires": required,
                        "detail": (
                            f"{skill['id']} ({skill['name']}) requires {required} "
                            f"({skill_map['skills'][order[required]]['name']}), which the curriculum order places later."
                        ),
                    }
                )

    summary = {
        "generated_by": "tools/build_coverage.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "totals": {
            "domains": len(domains),
            "competencies": len(competencies),
            "roadmap_skills": len(skill_map["skills"]),
            "source_sections_mapped": len(section_map["sections"]),
            "lessons": len(registry["lessons"]),
            "verified_lessons": len(
                [l for l in registry["lessons"] if l.get("publication_state") == "verified"]
            ),
        },
        "competency_coverage_counts": {},
        "by_domain": {},
    }

    for entry in entries:
        key = entry["computed_coverage"]
        summary["competency_coverage_counts"][key] = (
            summary["competency_coverage_counts"].get(key, 0) + 1
        )
        domain_entry = summary["by_domain"].setdefault(
            entry["domain"], {"competencies": 0, "sourced": 0, "partial": 0, "none": 0}
        )
        domain_entry["competencies"] += 1
        domain_entry[entry["declared_source_status"]] = (
            domain_entry.get(entry["declared_source_status"], 0) + 1
        )

    report = {
        "summary": summary,
        "inputs": {
            relative: hashlib.sha256((KB / relative).read_bytes()).hexdigest()
            for relative in (
                "curriculum/domains.json",
                "curriculum/competencies.json",
                "curriculum/skill-competency-map.json",
                "sources/mml/section-concept-map.json",
                "sources/mml/structure.json",
                "sources/mml/inventory.json",
                "lessons/registry.json",
            )
        },
        "competencies": entries,
        "issues": issues,
        "unmapped_source_sections": unmapped_sections,
        "skills_awaiting_sources": skills_without_sources,
        "prerequisite_order_conflicts": skills_with_cross_skill_prerequisites_met_later,
        "source_exercise_inventory": inventory["exercises"],
    }

    (KB / "curriculum" / "coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # --- human-readable report ----------------------------------------------
    lines: List[str] = []
    add = lines.append
    add("# Curriculum coverage report")
    add("")
    add(f"Generated by `tools/build_coverage.py` on {summary['generated_at']}. "
        "Every number below is derived from the knowledge base; none of it is hand-written.")
    add("")
    add("## What exists")
    add("")
    add("| Artefact | Count |")
    add("| --- | --- |")
    add(f"| Curriculum domains | {summary['totals']['domains']} |")
    add(f"| Competencies defined | {summary['totals']['competencies']} |")
    add(f"| Roadmap skills mapped | {summary['totals']['roadmap_skills']} |")
    add(f"| Source sections mapped to competencies | {summary['totals']['source_sections_mapped']} |")
    add(f"| Lessons written | {summary['totals']['lessons']} |")
    add(f"| Lessons verified | {summary['totals']['verified_lessons']} |")
    add("")
    add("## Coverage by computed state")
    add("")
    add("| Computed state | Competencies | Meaning |")
    add("| --- | --- | --- |")
    meanings = {
        "sourced_no_lessons": "The supplied book covers it; no lesson written yet.",
        "partially_sourced_no_lessons": "The supplied book contributes but does not complete it; no lesson written yet.",
        "no_source_supplied": "No source in the repository covers it. Awaiting material.",
        "lessons_exist_not_all_verified": "Lessons exist but are not all verified.",
        "verified": "Lessons exist and every one of them is verified.",
        "undefined": "State not derivable from the present data; investigate.",
    }
    for key in sorted(summary["competency_coverage_counts"]):
        add(f"| `{key}` | {summary['competency_coverage_counts'][key]} | {meanings.get(key, '')} |")
    add("")
    add("## By domain")
    add("")
    add("| Domain | Competencies | Sourced | Partial | No source |")
    add("| --- | --- | --- | --- | --- |")
    for domain_id in sorted(summary["by_domain"]):
        data = summary["by_domain"][domain_id]
        add(
            f"| {domain_id} | {data['competencies']} | {data.get('sourced', 0)} | "
            f"{data.get('partial', 0)} | {data.get('none', 0)} |"
        )
    add("")
    add("## Competencies with no supplied source")
    add("")
    add("These are recorded as gaps, not as completed work. Each needs either a supplied book or a")
    add("supplemental source before a lesson can be written against it.")
    add("")
    gaps = [e for e in entries if e["declared_source_status"] == "none"]
    for entry in gaps:
        add(f"- **{entry['competency_id']} {entry['name']}** ({entry['domain']}, target: {entry['target_level']}) — roadmap skills: "
            f"{', '.join(entry['roadmap_skills']) or 'none named'}")
    add("")
    add("## Competencies covered by the supplied book")
    add("")
    add("| Competency | Declared | Sections | Lessons |")
    add("| --- | --- | --- | --- |")
    for entry in entries:
        if entry["declared_source_status"] == "none":
            continue
        add(
            f"| {entry['competency_id']} | {entry['declared_source_status']} | "
            f"{', '.join(entry['mml_sections_declared']) or '—'} | "
            f"{len(entry['lessons'])} |"
        )
    add("")
    add("## Roadmap skills awaiting source material")
    add("")
    for skill in skills_without_sources:
        add(f"- **{skill['id']} {skill['name']}** ({skill['status']}) — needs: {skill['source_need']}")
    add("")
    add("## Prerequisite order conflicts")
    add("")
    if skills_with_cross_skill_prerequisites_met_later:
        for conflict in skills_with_cross_skill_prerequisites_met_later:
            add(f"- {conflict['detail']}")
    else:
        add("None detected among the order recorded in `skill-competency-map.json`.")
    add("")
    add("## Integrity issues")
    add("")
    if issues:
        for issue in issues:
            add(f"- `{issue['code']}`: {issue['detail']}")
    else:
        add("No inconsistencies detected between competencies, the section map and the source index.")
    add("")
    add("## Source sections with no competency link")
    add("")
    if unmapped_sections:
        add("These sections are in the source index but deliberately link to no competency "
            "(bibliography, study guidance, or framing):")
        add("")
        for section in unmapped_sections:
            add(f"- {section['id']} {section['title']} ({section['role']})")
    else:
        add("Every section in the source index links to at least one competency.")
    add("")
    add("## Supplied exercise sets")
    add("")
    add("| Chapter | Printed page | Items located | Chapters without a set |")
    add("| --- | --- | --- | --- |")
    exercises = inventory["exercises"]
    for chapter, data in sorted(exercises["per_chapter"].items(), key=lambda kv: int(kv[0])):
        add(f"| {chapter} | {data['printed_page']} | {data['item_count']} | — |")
    add(f"\nChapters without a supplied exercise set: {', '.join(exercises['chapters_without_supplied_exercise_sets'])}.")
    add("")

    (KB / "curriculum" / "coverage-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["totals"], indent=2))
    print("coverage states:", json.dumps(summary["competency_coverage_counts"], indent=2))
    print("integrity issues:", len(issues))
    for issue in issues:
        print(f"  - {issue['code']}: {issue['detail'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
