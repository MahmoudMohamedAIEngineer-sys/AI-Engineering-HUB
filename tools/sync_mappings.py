#!/usr/bin/env python3
"""Reconcile the two directions of the curriculum mappings.

The knowledge base records the source mapping twice, deliberately:

  * `knowledge/curriculum/competencies.json`  -> `source_coverage.mml_sections`
    (what a competency claims the supplied book covers)
  * `knowledge/sources/mml/section-concept-map.json` -> `competencies`
    (what each book section claims to teach)

Two authored views of one relationship will drift. `tools/build_coverage.py` reports any
disagreement instead of hiding it, and this script resolves the disagreements it can:

  * a section linked from the map but missing from the competency's list is added there,
    because the section map is the fine-grained view that lessons are grounded in;
  * a competency declared by a section's owner but missing from the map is added there.

The same problem exists for the roadmap-skill dimension:

  * `competencies.json` -> `roadmap_skills`   (which skills a competency serves)
  * `skill-competency-map.json` -> `competencies`  (which competencies a skill develops)

Both directions are unioned here as well, so that a competency can never be claimed to serve
a skill that does not list it.

The union is written back to both files, and every change is printed. Anything that is not
a simple set disagreement is left alone for a human to decide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "knowledge"
COMPETENCIES = KB / "curriculum" / "competencies.json"
SECTION_MAP = KB / "sources" / "mml" / "section-concept-map.json"


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    competencies_doc = json.loads(COMPETENCIES.read_text(encoding="utf-8"))
    section_doc = json.loads(SECTION_MAP.read_text(encoding="utf-8"))

    map_links: Dict[str, Set[str]] = {}
    for section in section_doc["sections"]:
        map_links[section["id"]] = set(section["competencies"])

    declared: Dict[str, Set[str]] = {
        c["id"]: set(c["source_coverage"]["mml_sections"]) for c in competencies_doc["competencies"]
    }

    changes: List[str] = []

    # 1. sections claimed by a competency but not by the section itself
    for section in section_doc["sections"]:
        for competency_id in sorted(declared.get(section["id"], set())):
            pass  # placeholder to keep the loop shape obvious; handled below
    for competency_id, sections in declared.items():
        for section_id in sorted(sections):
            if section_id in map_links and competency_id not in map_links[section_id]:
                map_links[section_id].add(competency_id)
                changes.append(
                    f"section {section_id}: linked to {competency_id} "
                    "(declared by the competency, missing from the section map)"
                )

    # 2. competency entries must carry every section that links to them
    sections_by_competency: Dict[str, Set[str]] = {}
    for section_id, competency_ids in map_links.items():
        for competency_id in competency_ids:
            sections_by_competency.setdefault(competency_id, set()).add(section_id)

    for competency in competencies_doc["competencies"]:
        competency_id = competency["id"]
        target = sections_by_competency.get(competency_id, set())
        current = set(competency["source_coverage"]["mml_sections"])
        added = sorted(target - current, key=lambda s: [int(p) for p in s.split(".")])
        if added:
            competency["source_coverage"]["mml_sections"] = sorted(
                current | target, key=lambda s: [int(p) for p in s.split(".")]
            )
            changes.append(f"{competency_id}: now declares sections {', '.join(added)}")
        if not target and current:
            changes.append(
                f"{competency_id}: declares sections {', '.join(sorted(current))} that no section "
                "in the map links back to; left unchanged for review"
            )

    for section in section_doc["sections"]:
        section["competencies"] = sorted(map_links[section["id"]])

    section_doc["reconciliation"] = {
        "tool": "tools/sync_source_coverage.py",
        "rule": (
            "The section map is the fine-grained view and the union of the two directions is "
            "treated as correct; this file records that the union was applied rather than "
            "asserting that a human reviewed each link."
        ),
        "review_required": True,
    }

    # --- skill <-> competency direction -------------------------------------
    skill_doc = json.loads((KB / "curriculum" / "skill-competency-map.json").read_text(encoding="utf-8"))
    skills_in_order = skill_doc["skills"]
    skill_links: Dict[str, Set[str]] = {s["id"]: set(s["competencies"]) for s in skills_in_order}

    competency_links: Dict[str, Set[str]] = {
        c["id"]: set(c["roadmap_skills"]) for c in competencies_doc["competencies"]
    }

    for competency_id, skills in competency_links.items():
        for skill_id in sorted(skills):
            if skill_id in skill_links and competency_id not in skill_links[skill_id]:
                skill_links[skill_id].add(competency_id)
                changes.append(
                    f"skill {skill_id}: linked to {competency_id} "
                    "(declared by the competency, missing from the skill map)"
                )

    competencies_by_skill: Dict[str, Set[str]] = {}
    for skill_id, competency_ids in skill_links.items():
        for competency_id in competency_ids:
            competencies_by_skill.setdefault(competency_id, set()).add(skill_id)

    for competency in competencies_doc["competencies"]:
        competency_id = competency["id"]
        target = competencies_by_skill.get(competency_id, set())
        current = set(competency["roadmap_skills"])
        added = sorted(target - current)
        if added:
            competency["roadmap_skills"] = sorted(current | target)
            changes.append(f"{competency_id}: now serves roadmap skills {', '.join(added)}")

    for skill in skills_in_order:
        skill["competencies"] = sorted(skill_links[skill["id"]])

    skill_doc["reconciliation"] = {
        "tool": "tools/sync_mappings.py",
        "rule": "Skill <-> competency links are unioned in both directions; see the same note in section-concept-map.json.",
        "review_required": True,
    }

    dump(COMPETENCIES, competencies_doc)
    dump(SECTION_MAP, section_doc)
    dump(KB / "curriculum" / "skill-competency-map.json", skill_doc)

    print(f"changes applied: {len(changes)}")
    for change in changes:
        print("  -", change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
