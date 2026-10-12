#!/usr/bin/env python3
"""Validate the knowledge base.

This is the check that keeps the curriculum from quietly becoming inconsistent: dangling
identifiers, prerequisite cycles, unknown status values, source sections that do not exist,
and lessons that claim verification without a verification record.

Exit code 0 means every check that can be automated passed. It does not mean the content is
correct — source fidelity and technical correctness are separate checks that need their own
methods (see knowledge/sources/mml/pdf-verification.json for an example of what they look
like when they are done properly).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "knowledge"


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks_run: List[str] = []

    def check(self, name: str, condition: bool, message: str, severity: str = "error") -> None:
        self.checks_run.append(name)
        if condition:
            return
        (self.errors if severity == "error" else self.warnings).append(f"[{name}] {message}")


def load(relative: str) -> Any:
    return json.loads((KB / relative).read_text(encoding="utf-8"))


def find_cycles(edges: Dict[str, Set[str]]) -> List[List[str]]:
    """Return prerequisite cycles (each as the path that closes the loop)."""
    cycles: List[List[str]] = []
    visited: Set[str] = set()
    stack: List[str] = []
    on_stack: Set[str] = set()

    def visit(node: str) -> None:
        visited.add(node)
        stack.append(node)
        on_stack.add(node)
        for neighbour in sorted(edges.get(node, set())):
            if neighbour in on_stack:
                start = stack.index(neighbour)
                cycles.append(stack[start:] + [neighbour])
            elif neighbour not in visited:
                visit(neighbour)
        stack.pop()
        on_stack.discard(node)

    for node in sorted(edges):
        if node not in visited:
            visit(node)
    return cycles


def main() -> int:
    report = Report()

    states = load("states.json")
    valid_topic_status = set(states["topic_status"]["values"])
    valid_publication = set(states["publication_state"]["values"])
    valid_verification = set(states["verification_status"]["values"])

    domains = load("curriculum/domains.json")["domains"]
    competencies = load("curriculum/competencies.json")["competencies"]
    skill_map = load("curriculum/skill-competency-map.json")
    section_map = load("sources/mml/section-concept-map.json")
    structure = load("sources/mml/structure.json")
    inventory = load("sources/mml/inventory.json")
    manifest = load("sources/mml/manifest.json")
    registry = load("lessons/registry.json")
    references = load("references/external.json")["references"]
    roadmap_inventory = load("roadmaps/consolidated-inventory.json")

    domain_ids = {d["id"] for d in domains}
    competency_ids = {c["id"] for c in competencies}
    source_section_ids = {
        section["id"] for chapter in structure["chapters"] for section in chapter["sections"]
    }
    roadmap_skill_ids = {s["id"] for s in roadmap_inventory["skills"]}
    audit_finding_ids = {f["id"] for f in skill_map.get("audit_findings", [])}
    reference_ids = {r["id"] for r in references}

    # --- identifiers are unique ---------------------------------------------
    for label, items, key in (
        ("domains", domains, "id"),
        ("competencies", competencies, "id"),
        ("roadmap skills", skill_map["skills"], "id"),
        ("section map entries", section_map["sections"], "id"),
        ("references", references, "id"),
        ("audit findings", skill_map.get("audit_findings", []), "id"),
    ):
        ids = [item[key] for item in items]
        report.check(
            f"unique-ids:{label}",
            len(ids) == len(set(ids)),
            f"duplicate identifiers in {label}: {sorted({i for i in ids if ids.count(i) > 1})}",
        )

    # --- referential integrity ----------------------------------------------
    for competency in competencies:
        cid = competency["id"]
        report.check(
            "competency-domain-exists",
            competency["domain"] in domain_ids,
            f"{cid} names domain {competency['domain']} which is not defined in domains.json",
        )
        for skill_id in competency["roadmap_skills"]:
            report.check(
                "competency-roadmap-skill-exists",
                skill_id in roadmap_skill_ids,
                f"{cid} names roadmap skill {skill_id} which is not in the extracted inventory",
            )
        for section_id in competency["source_coverage"]["mml_sections"]:
            report.check(
                "competency-source-section-exists",
                section_id in source_section_ids,
                f"{cid} cites source section {section_id} which is not in the source index",
            )
        for prerequisite in competency["prerequisites"]:
            report.check(
                "competency-prerequisite-exists",
                prerequisite["id"] in competency_ids,
                f"{cid} requires {prerequisite['id']} which is not a defined competency",
            )
            report.check(
                "competency-prerequisite-justified",
                bool(prerequisite.get("because", "").strip()),
                f"{cid} lists prerequisite {prerequisite['id']} without a reason",
            )
        report.check(
            "competency-outcomes-are-observable",
            len(competency["outcomes"]) >= 2,
            f"{cid} has fewer than two observable outcomes",
            severity="warning",
        )
        report.check(
            "competency-level-valid",
            competency["target_level"]
            in {level["id"] for level in states["proficiency_levels"]},
            f"{cid} has target_level {competency['target_level']} which is not a defined level",
        )

    # --- prerequisite graph --------------------------------------------------
    edges = {
        c["id"]: {p["id"] for p in c["prerequisites"]}
        for c in competencies
    }
    for cid, prerequisites in edges.items():
        report.check(
            "no-self-prerequisite", cid not in prerequisites, f"{cid} lists itself as a prerequisite"
        )
    cycles = find_cycles(edges)
    report.check(
        "prerequisite-graph-is-acyclic",
        not cycles,
        "prerequisite cycles found: " + "; ".join(" -> ".join(cycle) for cycle in cycles),
    )

    # --- skill map -----------------------------------------------------------
    inventory_skill_ids = {s["id"] for s in roadmap_inventory["skills"]}
    mapped_skill_ids = {s["id"] for s in skill_map["skills"]}
    report.check(
        "every-roadmap-skill-is-mapped",
        mapped_skill_ids == inventory_skill_ids,
        "skills missing from the competency map: "
        + ", ".join(sorted(inventory_skill_ids - mapped_skill_ids))
        + "; skills mapped but not in the roadmap: "
        + ", ".join(sorted(mapped_skill_ids - inventory_skill_ids)),
    )
    for skill in skill_map["skills"]:
        report.check(
            "skill-status-valid",
            skill["status"] in valid_topic_status,
            f"{skill['id']} has status {skill['status']} which is not a defined topic status",
        )
        for competency_id in skill["competencies"]:
            report.check(
                "skill-competency-exists",
                competency_id in competency_ids,
                f"{skill['id']} maps to {competency_id} which is not a defined competency",
            )
        for required in skill["requires_skills"]:
            report.check(
                "skill-prerequisite-exists",
                required in inventory_skill_ids,
                f"{skill['id']} requires {required} which is not a roadmap skill",
            )
        for required in skill["requires_competencies"]:
            report.check(
                "skill-competency-prerequisite-exists",
                required in competency_ids,
                f"{skill['id']} requires competency {required} which is not defined",
            )
        report.check(
            "skill-has-source-need-stated",
            bool(skill.get("source_need", "").strip()),
            f"{skill['id']} has no source requirement recorded",
            severity="warning",
        )

    # --- competencies reachable from the roadmap ----------------------------
    mapped_competencies = {c for s in skill_map["skills"] for c in s["competencies"]}
    orphan_competencies = sorted(competency_ids - mapped_competencies)
    report.check(
        "competencies-traceable-to-roadmap-or-audit",
        not orphan_competencies,
        "competencies not reachable from any roadmap skill (they must be audit additions, "
        "which is allowed, but should be listed): " + ", ".join(orphan_competencies),
        severity="warning",
    )

    # --- section map ---------------------------------------------------------
    for section in section_map["sections"]:
        report.check(
            "section-map-section-exists",
            section["id"] in source_section_ids,
            f"section map entry {section['id']} is not in the source index",
        )
        for competency_id in section["competencies"]:
            report.check(
                "section-map-competency-exists",
                competency_id in competency_ids,
                f"section {section['id']} links to {competency_id} which is not defined",
            )

    # both directions agree
    declared = {
        c["id"]: set(c["source_coverage"]["mml_sections"]) for c in competencies
    }
    reverse: Dict[str, Set[str]] = {}
    for section in section_map["sections"]:
        for competency_id in section["competencies"]:
            reverse.setdefault(competency_id, set()).add(section["id"])
    for competency_id, sections in declared.items():
        report.check(
            "section-map-and-competency-agree",
            sections == reverse.get(competency_id, set()),
            f"{competency_id}: competency declares {sorted(sections)} but the section map links "
            f"{sorted(reverse.get(competency_id, set()))}. Run tools/sync_source_coverage.py.",
            severity="warning",
        )

    # --- exercise coverage claims -------------------------------------------
    exercise_chapters = set(inventory["exercises"]["chapters_with_supplied_exercise_sets"])
    mapped_exercise_chapters = {entry["chapter"] for entry in section_map.get("exercise_sets", [])}
    report.check(
        "exercise-sets-mapped-consistently",
        exercise_chapters == mapped_exercise_chapters,
        f"exercise sets found in the source: {sorted(exercise_chapters)}; recorded in the "
        f"section map: {sorted(mapped_exercise_chapters)}",
    )
    for entry in section_map.get("exercise_sets", []):
        source_entry = inventory["exercises"]["per_chapter"].get(entry["chapter"], {})
        report.check(
            "exercise-set-item-count-matches-index",
            source_entry.get("item_count") == entry["items_located_in_extraction"],
            f"chapter {entry['chapter']}: section map records "
            f"{entry['items_located_in_extraction']} located items but the source index located "
            f"{source_entry.get('item_count')}",
        )
        pdf_verification_path = KB / "sources" / "mml" / "pdf-verification.json"
        if pdf_verification_path.exists():
            verified = json.loads(pdf_verification_path.read_text(encoding="utf-8"))
            verified_entry = verified.get("exercise_sets", {}).get("per_chapter", {}).get(
                entry["chapter"], {}
            )
            report.check(
                "exercise-set-pdf-count-matches-verification",
                len(verified_entry.get("pdf_item_ids", [])) == entry["items_in_source_pdf"],
                f"chapter {entry['chapter']}: section map records {entry['items_in_source_pdf']} items "
                f"in the PDF but the PDF check found {len(verified_entry.get('pdf_item_ids', []))}",
            )
        first = source_entry.get("printed_page")
        report.check(
            "exercise-set-printed-page-matches-index",
            first == entry.get("printed_page"),
            f"chapter {entry['chapter']}: printed page differs between the section map "
            f"({entry.get('printed_page')}) and the source index ({first})",
        )

    # --- source manifest -----------------------------------------------------
    for file_entry in manifest["files"]:
        path = REPO / file_entry["path"]
        report.check(
            "manifest-file-exists", path.exists(), f"source file {file_entry['path']} is missing"
        )
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report.check(
                "manifest-hash-matches",
                digest == file_entry["sha256"],
                f"{file_entry['path']} has changed since it was indexed; re-run tools/index_mml.py "
                "and review the dependent lessons",
            )

    # --- references ----------------------------------------------------------
    for reference in references:
        report.check(
            "reference-declares-retrieval-status",
            "retrieved" in reference,
            f"{reference['id']} does not record whether it was retrieved",
        )
        for finding in reference.get("supports_findings", []):
            report.check(
                "reference-finding-exists",
                finding in audit_finding_ids,
                f"{reference['id']} supports finding {finding} which does not exist",
            )

    # --- lessons, exercises, verification records ---------------------------
    # These checks read the generated registry, so they validate what the interface will actually
    # display. A contradiction between a lesson file and its verification record is an error, not a
    # formatting problem: it would let the site claim a state the evidence does not support.
    lesson_ids = {lesson["id"] for lesson in registry["lessons"]}
    report.check(
        "unique-ids:lessons",
        len(lesson_ids) == len(registry["lessons"]),
        "duplicate lesson identifiers",
    )
    records_by_lesson = {record["lesson_id"]: record for record in registry["verification_records"]}
    competencies_with_lessons = {
        competency_id for lesson in registry["lessons"] for competency_id in lesson["competencies"]
    }

    for lesson in registry["lessons"]:
        lid = lesson["id"]
        report.check(
            "lesson-publication-state-valid",
            lesson["publication_state"] in valid_publication,
            f"{lid} has publication_state {lesson['publication_state']}",
        )
        for competency_id in lesson["competencies"]:
            report.check(
                "lesson-competency-exists",
                competency_id in competency_ids,
                f"{lid} claims competency {competency_id} which is not defined",
            )
        for source in lesson.get("sources", []):
            report.check(
                "lesson-source-work-known",
                source["work_id"] == manifest["work_id"],
                f"{lid} cites work {source['work_id']} which is not a registered source",
            )
            for section_id in source.get("sections", []):
                report.check(
                    "lesson-source-section-exists",
                    section_id in source_section_ids,
                    f"{lid} cites source section {section_id} which does not exist",
                )
        for anchor in (lesson.get("source_anchors") or {}).get("sections", []):
            report.check(
                "lesson-source-anchor-exists",
                anchor["id"] in source_section_ids,
                f"{lid} anchors to section {anchor['id']} which does not exist in the source index",
            )
            report.check(
                "lesson-source-anchor-page-matches-index",
                any(
                    section["id"] == anchor["id"] and section["printed_page"] == anchor["printed_page"]
                    for chapter in structure["chapters"]
                    for section in chapter["sections"]
                ),
                f"{lid} records printed page {anchor['printed_page']} for section {anchor['id']}, "
                "which disagrees with the source index",
            )

        # The lesson file on disk must be the one that was indexed, and it must exist.
        content_path = KB.parent / lesson["content_path"]
        report.check(
            "lesson-content-file-exists",
            content_path.exists(),
            f"{lid} points at {lesson['content_path']} which does not exist",
        )
        if content_path.exists():
            digest = hashlib.sha256(content_path.read_bytes()).hexdigest()
            report.check(
                "lesson-content-unchanged-since-index",
                digest == lesson["content_sha256"],
                f"{lid}: {lesson['content_path']} changed after the registry was generated; "
                "re-run tools/build_content_index.py",
            )

        # Verification: presence, direction, and the state the interface will show.
        declared = lesson.get("publication_state")
        record = records_by_lesson.get(lid)
        record_id = lesson.get("verification_record_id")
        if record_id:
            report.check(
                "lesson-verification-record-exists",
                any(r["id"] == record_id for r in registry["verification_records"]),
                f"{lid} names verification record {record_id} which does not exist",
            )
        if declared in {"verified", "published"}:
            report.check(
                "verified-lesson-has-verification-record",
                record is not None and record["outcome"] == "verified",
                f"{lid} is marked {declared} but its verification record says "
                f"{record['outcome'] if record else 'none'}",
            )
        if record:
            incomplete = [
                check["name"]
                for check in record["checks"]
                if check["status"] == "not_run"
                and check["name"]
                in {
                    "source_fidelity",
                    "technical_correctness",
                    "mathematical_validity",
                    "assessment_integrity",
                    "publication_integrity",
                }
            ]
            report.check(
                "verified-record-runs-core-checks",
                record["outcome"] != "verified" or not incomplete,
                f"{record['id']} claims verified but these checks were not run: {', '.join(incomplete)}",
            )
            report.check(
                "verification-record-outcome-vocabulary",
                record["outcome"] in {"verified", "needs_revision"},
                f"{record['id']} has outcome {record['outcome']}",
            )
            if record["outcome"] == "needs_revision":
                report.check(
                    "needs-revision-record-explains-itself",
                    bool(record.get("outcome_explanation", "").strip())
                    and bool(record.get("unresolved_issues")),
                    f"{record['id']} is needs_revision without an explanation and unresolved issues",
                )

    for exercise in registry["exercises"]:
        report.check(
            "exercise-origin-declared",
            exercise["origin"] in {"source", "adapted", "original", "external"},
            f"{exercise['id']} has origin {exercise['origin']}",
        )
        report.check(
            "exercise-belongs-to-known-lesson",
            exercise.get("lesson_id") in lesson_ids,
            f"{exercise['id']} claims lesson {exercise.get('lesson_id')} which does not exist",
        )
        for competency_id in exercise.get("competencies", []):
            report.check(
                "exercise-competency-exists",
                competency_id in competency_ids,
                f"{exercise['id']} claims competency {competency_id} which is not defined",
            )
        report.check(
            "exercise-declares-expected-outcome",
            bool(exercise.get("expected_outcome", "").strip()),
            f"{exercise['id']} does not say what a correct answer must contain",
        )
        report.check(
            "exercise-declares-difficulty",
            exercise.get("difficulty") in
            {"recognition", "guided", "independent", "multi_step", "transfer", "open_ended"},
            f"{exercise['id']} has difficulty {exercise.get('difficulty')}",
        )
        report.check(
            "exercise-has-solution-or-rubric",
            bool(exercise.get("solution_path")) or bool(exercise.get("rubric")),
            f"{exercise['id']} has neither a solution nor a rubric",
        )
        if exercise.get("solution_path"):
            solution_file = KB.parent / exercise["solution_path"].split("#")[0]
            report.check(
                "exercise-solution-file-exists",
                solution_file.exists(),
                f"{exercise['id']} points at {exercise['solution_path']} which does not exist",
            )
        if exercise.get("solution_verified_by"):
            report.check(
                "exercise-original-declares-no-source",
                exercise["origin"] != "source" or bool(exercise.get("origin_detail")),
                f"{exercise['id']} is marked as coming from the source without saying where",
            )
        if exercise["origin"] in {"source", "adapted"}:
            report.check(
                "sourced-exercise-cites-source",
                bool(exercise.get("origin_detail", {}).get("section")),
                f"{exercise['id']} is marked {exercise['origin']} without citing a source section",
            )

    for record in registry["verification_records"]:
        report.check(
            "verification-record-for-known-lesson",
            record["lesson_id"] in lesson_ids,
            f"{record['id']} refers to lesson {record['lesson_id']} which does not exist",
        )
        for check in record["checks"]:
            report.check(
                "verification-status-valid",
                check["status"] in valid_verification,
                f"verification record {record['id']} uses status {check['status']}",
            )
            report.check(
                "verification-check-describes-method",
                bool(check.get("method", "").strip()),
                f"verification record {record['id']}: check {check['name']} does not say how it was run",
            )
            report.check(
                "verification-check-states-what-was-not-checked",
                check["status"] in {"passed", "partial"} or bool(check.get("not_checked", "").strip()),
                f"verification record {record['id']}: check {check['name']} does not state what it "
                "did not establish",
            )
        for relative in record.get("evidence_files", {}).values():
            report.check(
                "verification-evidence-file-exists",
                (KB.parent / relative).exists(),
                f"verification record {record['id']} cites evidence file {relative} which does not exist",
            )

    # --- placeholders: what the interface shows where no material exists ----
    placeholders_path = KB.parent / registry["placeholders"]["path"]
    report.check(
        "placeholders-file-exists",
        placeholders_path.exists(),
        f"{registry['placeholders']['path']} is missing; run tools/build_content_index.py",
    )
    if placeholders_path.exists():
        placeholders = json.loads(placeholders_path.read_text(encoding="utf-8"))["placeholders"]
        placeholder_ids = [entry["competency_id"] for entry in placeholders]
        report.check(
            "placeholder-competencies-exist",
            all(pid in competency_ids for pid in placeholder_ids),
            "placeholder file names competencies that are not in the framework",
        )
        unsourced = {
            competency["id"]
            for competency in competencies
            if competency["source_coverage"]["status"] != "sourced"
            and competency["id"] not in competencies_with_lessons
        }
        report.check(
            "placeholders-cover-every-unsourced-competency-without-a-lesson",
            set(placeholder_ids) == unsourced,
            "the placeholder set and the set of unsourced competencies disagree: "
            f"missing {sorted(unsourced - set(placeholder_ids))}, "
            f"extra {sorted(set(placeholder_ids) - unsourced)}",
        )
        report.check(
            "placeholders-has-no-duplicates",
            len(placeholder_ids) == len(set(placeholder_ids)),
            "duplicate competency identifiers in the placeholder file",
        )
        for entry in placeholders:
            report.check(
                "placeholder-declares-what-is-missing",
                bool(entry["supplemental_needed"]) or bool(entry["roadmap_skills"]),
                f"{entry['competency_id']} has no stated source need and no roadmap skill",
            )
            report.check(
                "placeholder-has-no-lesson",
                entry["competency_id"] not in competencies_with_lessons,
                f"{entry['competency_id']} has a lesson but is still listed as awaiting sources",
            )
            report.check(
                "placeholder-status-is-awaiting-sources",
                entry["status"] == "awaiting_sources",
                f"{entry['competency_id']} has status {entry['status']}",
            )

    # --- executable checks report -------------------------------------------
    checks_report_path = KB / "lessons" / "checks-report.json"
    report.check(
        "content-checks-report-exists",
        checks_report_path.exists(),
        "knowledge/lessons/checks-report.json is missing; run tools/run_content_checks.py",
    )
    if checks_report_path.exists():
        checks_report = json.loads(checks_report_path.read_text(encoding="utf-8"))
        for record in registry["verification_records"]:
            claims_technical = any(
                check["name"] == "technical_correctness" and check["status"] == "passed"
                for check in record["checks"]
            )
            report.check(
                "technical-correctness-claim-has-passing-checks",
                not claims_technical or checks_report["summary"]["failed"] == 0,
                f"{record['id']} claims technical_correctness passed, but the checks report records "
                f"{checks_report['summary']['failed']} failing checks",
            )
    report.check(
        "registry-reports-no-inconsistencies",
        not registry.get("problems"),
        f"the registry reports {len(registry.get('problems', []))} inconsistencies: "
        f"{registry.get('problems')}",
    )

    # --- the supplied roadmap pages are served verbatim ---------------------
    # Phase 4 requires the learner's own roadmap pages to stay reachable *and* unmodified. They are
    # served from public/ as byte-identical copies; a copy that drifts from the original is an error,
    # because the served page would then be a different document from the one the learner wrote.
    served_copies = {
        "perfect_resources_44_skills.html": "public/perfect_resources_44_skills.html",
        "study_timetable_8hrs.html": "public/study_timetable_8hrs.html",
    }
    for original_relative, served_relative in served_copies.items():
        original = KB.parent / original_relative
        served = KB.parent / served_relative
        report.check(
            "roadmap-page-still-present",
            original.exists(),
            f"{original_relative} is missing from the repository root",
        )
        report.check(
            "roadmap-page-served-verbatim",
            served.exists()
            and original.exists()
            and hashlib.sha256(served.read_bytes()).hexdigest()
            == hashlib.sha256(original.read_bytes()).hexdigest(),
            f"{served_relative} is not a byte-identical copy of {original_relative}; "
            "the served roadmap page has drifted from the file the learner supplied",
        )

    # --- coverage report freshness ------------------------------------------
    coverage_path = KB / "curriculum" / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        for relative, recorded in coverage.get("inputs", {}).items():
            current = hashlib.sha256((KB / relative).read_bytes()).hexdigest()
            report.check(
                "coverage-report-is-current",
                current == recorded,
                f"{relative} changed after the coverage report was generated; re-run "
                "tools/build_coverage.py",
                severity="warning",
            )
    else:
        report.check(
            "coverage-report-exists", False, "knowledge/curriculum/coverage.json is missing"
        )

    # --- output --------------------------------------------------------------
    print(f"checks run: {len(set(report.checks_run))} distinct check types, {len(report.checks_run)} assertions")
    if report.warnings:
        print(f"\nwarnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print("  -", warning)
    if report.errors:
        print(f"\nerrors ({len(report.errors)}):")
        for error in report.errors:
            print("  -", error)
        print("\nRESULT: FAILED")
        return 1
    print("\nRESULT: PASSED (automated checks only; content correctness is verified separately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
