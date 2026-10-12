/**
 * Server-side access to the knowledge base.
 *
 * The application does not hold curriculum data of its own: every page is a view over the
 * generated files under `knowledge/` and the authored files under `content/`. If a fact is not
 * in those files, the interface does not know it, and the loaders below fail loudly rather than
 * falling back to invented defaults.
 */

import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();

function readJson<T>(relativePath: string): T {
  const absolute = path.join(ROOT, relativePath);
  if (!fs.existsSync(absolute)) {
    throw new Error(
      `Missing knowledge file: ${relativePath}. Run 'make check' (or the individual tools) to generate it.`,
    );
  }
  return JSON.parse(fs.readFileSync(absolute, "utf8")) as T;
}

// --------------------------------------------------------------------------- types

export type VerificationCheck = {
  name: string;
  status: "not_run" | "passed" | "failed" | "not_applicable" | "blocked" | "partial";
  method: string;
  evidence: string;
  not_checked?: string;
};

export type VerificationRecord = {
  id: string;
  lesson_id: string;
  reviewed_on: string;
  reviewer: string;
  outcome: "verified" | "needs_revision";
  outcome_explanation: string;
  checks: VerificationCheck[];
  unresolved_issues: { issue: string; next_action: string; severity: string }[];
  evidence_files: Record<string, string>;
};

export type Exercise = {
  id: string;
  lesson_id: string;
  competencies: string[];
  origin: "source" | "adapted" | "original" | "external";
  origin_detail: Record<string, string>;
  difficulty: string;
  prompt: string;
  expected_outcome: string;
  solution_path?: string;
  solution_verified_by?: string;
  rubric?: string;
};

export type Lesson = {
  id: string;
  title: string;
  competencies: string[];
  covers_outcomes: string[];
  sources: { work_id: string; sections: string[]; pages?: string; role: string }[];
  source_anchors?: {
    sections: { id: string; file: string; line: number; printed_page: number }[];
    numbered_objects?: string[];
  };
  supplemental_references: string[];
  prerequisites: { competency?: string; lesson?: string; need: string }[];
  estimated_effort: { reading_minutes: number; exercise_minutes: number };
  interactive?: { component: string; data_path: string; serves: string };
  content_path: string;
  content_sha256: string;
  publication_state: string;
  verification_record_id: string | null;
  verification_record_outcome: string | null;
  last_verified: string | null;
  provenance: Record<string, string>;
};

export type Competency = {
  id: string;
  domain: string;
  name: string;
  why: string;
  outcomes: string[];
  prerequisites: { id: string; because: string }[];
  target_level: string;
  roadmap_skills: string[];
  source_coverage: { status: string; mml_sections: string[]; supplemental_needed: string[]; notes: string };
};

export type Domain = { id: string; name: string; purpose: string; roadmap_skills: string[]; notes: string };

export type Placeholder = {
  competency_id: string;
  title: string;
  domain: string;
  target_level: string;
  why_it_matters: string | null;
  declared_source_status: string;
  source_status_explanation: string;
  supplemental_needed: string[];
  roadmap_skills: { skill: string; name: string; phase: string; study_days: number; what_you_learn: string }[];
  roadmap_order: { first_roadmap_position: number | null; note: string };
  phase: string | null;
  what_would_unblock_it: string[];
  status: string;
  readiness_to_render: string;
  placeholder_means: string;
};

export type Section = {
  id: string;
  title: string;
  printed_page: number;
  anchor: { file: string; line: number } | null;
  status: string;
};

export type Chapter = {
  id: string;
  title: string;
  printed_page: number;
  part: string;
  source_file: string | null;
  sections: Section[];
  exercises_printed_page: number | null;
  subsection_headings_found: string[];
};

// --------------------------------------------------------------------------- loaders

export const getRegistry = () =>
  readJson<{
    generated_at: string;
    lessons: Lesson[];
    exercises: Exercise[];
    verification_records: VerificationRecord[];
    placeholders: { path: string; count: number; by_domain: Record<string, number> };
    current_state: Record<string, unknown>;
    problems: string[];
  }>("knowledge/lessons/registry.json");

export const getCompetencies = () =>
  readJson<{ version: string; last_reviewed: string; competencies: Competency[] }>(
    "knowledge/curriculum/competencies.json",
  );

export const getDomains = () => readJson<{ domains: Domain[] }>("knowledge/curriculum/domains.json");

export const getCoverage = () =>
  readJson<{
    summary: {
      generated_at: string;
      totals: Record<string, number>;
      competency_coverage_counts: Record<string, number>;
      by_domain: Record<string, { competencies: number; sourced: number; partial: number; none: number }>;
    };
    competencies: {
      competency_id: string;
      name: string;
      domain: string;
      declared_source_status: string;
      mml_sections_declared: string[];
      lessons: string[];
      computed_coverage: string;
      supplemental_needed: string[];
    }[];
    issues: { code: string; detail: string }[];
    skills_awaiting_sources: { id: string; name: string; status: string; source_need: string }[];
  }>("knowledge/curriculum/coverage.json");

export const getPlaceholders = () =>
  readJson<{ counts: { placeholders: number; by_domain: Record<string, number> }; placeholders: Placeholder[] }>(
    "content/curriculum/placeholders.json",
  );

export const getStructure = () =>
  readJson<{
    work_id: string;
    edition: string;
    chapters: Chapter[];
    exercises_index: Record<string, number>;
    totals: Record<string, number>;
    duplicate_headings: Record<string, { file: string; line: number }[]>;
    missing_sections: string[];
  }>("knowledge/sources/mml/structure.json");

export const getManifest = () =>
  readJson<{
    work_id: string;
    title: string;
    authors: string[];
    edition: string;
    publisher: string;
    canonical_web: string;
    note_on_identity: string;
    license_note: string;
    files: { path: string; sha256: string; bytes: number; lines: number }[];
    structure_snapshot: Record<string, number>;
    processing_status: string;
  }>("knowledge/sources/mml/manifest.json");

export const getDefects = () =>
  readJson<{
    summary: { errors: number; warnings: number; info: number };
    defects: { code: string; severity: string; detail: string }[];
  }>("knowledge/sources/mml/defects.json");

export const getInventory = () =>
  readJson<{
    numbered_objects: Record<string, { total_unique: number; per_chapter: Record<string, string[]> }>;
    images: { references: number; distinct_urls: number; hosts: Record<string, number>; note: string };
    exercises: {
      chapters_with_supplied_exercise_sets: string[];
      chapters_without_supplied_exercise_sets: string[];
      total_items_located: number;
      per_chapter: Record<string, { chapter: string; printed_page: number | null; item_count: number }>;
    };
  }>("knowledge/sources/mml/inventory.json");

export const getPdfVerification = () =>
  readJson<{
    status: string;
    page_offset: { printed_to_pdf_offset: number | null; consistent_across_all_exercise_sections: boolean };
    exercise_sets: {
      pdf_pages_with_exercise_headings: number[];
      chapters_without_sets: number[];
      whole_pdf_search_after_last_exercise_set: { conclusion: string };
    };
    not_checked: string[];
  }>("knowledge/sources/mml/pdf-verification.json");

export const getRoadmapInventory = () =>
  readJson<{
    counts: {
      skills: number;
      project_sprints: number;
      by_phase: Record<string, number>;
      skills_with_day_level_plan: number;
      skills_summary_only: number;
      summed_study_days_including_sprints: number;
    };
    skills: {
      id: string;
      name: string;
      phase: string;
      tags: string[];
      what_you_learn: string;
      schedule: { study_days: number | null; hours: number | null; week_label: string | null };
      plan_depth: string;
      intended_resources: { kind: string; platform: string; title: string; author_year: string; rationale: string }[];
    }[];
    project_sprints: { id: string; name: string; phase: string; schedule: Record<string, unknown> }[];
  }>("knowledge/roadmaps/consolidated-inventory.json");

export const getRoadmapQuality = () =>
  readJson<{
    sources: { path: string; role: string; sha256: string; bytes: number }[];
    extracted: Record<string, number>;
    issues: { severity: string; code: string; detail: string }[];
  }>("knowledge/roadmaps/data-quality-report.json");

export const getSkillMap = () =>
  readJson<{
    skills: {
      id: string;
      name: string;
      competencies: string[];
      requires_skills: string[];
      requires_competencies: string[];
      status: string;
      source_need: string;
      audit_note: string;
    }[];
    audit_findings: {
      id: string;
      severity: string;
      finding: string;
      evidence: string;
      recommendation: string;
      classification: string;
      status: string;
    }[];
  }>("knowledge/curriculum/skill-competency-map.json");

export const getReferences = () =>
  readJson<{
    references: {
      id: string;
      title: string;
      publisher: string;
      url: string;
      authority: string;
      retrieved: boolean;
      retrieved_on: string | null;
      used_for: string;
      supports_findings: string[];
      notes: string;
    }[];
  }>("knowledge/references/external.json");

export const getChecksReport = () =>
  readJson<{
    generated_at: string;
    environment: Record<string, string>;
    summary: { checks: number; passed: number; failed: number };
    results: { module: string; check: string; status: string; evidence: string }[];
  }>("knowledge/lessons/checks-report.json");

// --------------------------------------------------------------------------- lesson text

const FRONT_MATTER = /^---\s*\n[\s\S]*?\n---\s*\n/;

/** Lesson markdown without its YAML front matter (the front matter is data, not prose). */
export function readLessonMarkdown(contentPath: string): string {
  const absolute = path.join(ROOT, contentPath);
  if (!fs.existsSync(absolute)) {
    throw new Error(`Lesson file not found: ${contentPath}`);
  }
  return fs.readFileSync(absolute, "utf8").replace(FRONT_MATTER, "");
}

/**
 * The state the interface is allowed to display for a lesson.
 *
 * Derived from evidence rather than from what the lesson file claims about itself: a lesson with
 * no verification record is a draft, and a lesson whose record says `needs_revision` is not
 * verified no matter what its front matter says.
 */
export type DisplayState = "draft" | "needs_revision" | "verified";

export function displayStateFor(lesson: Lesson, record: VerificationRecord | undefined): DisplayState {
  if (!record) return "draft";
  if (record.outcome === "verified") return "verified";
  return "needs_revision";
}

export function getLessonBundle(lessonId: string) {
  const registry = getRegistry();
  const lesson = registry.lessons.find((item) => item.id === lessonId);
  if (!lesson) return null;
  const record = registry.verification_records.find((item) => item.id === lesson.verification_record_id);
  const exercises = registry.exercises.filter((item) => item.lesson_id === lesson.id);
  return {
    lesson,
    record,
    exercises,
    markdown: readLessonMarkdown(lesson.content_path),
    state: displayStateFor(lesson, record),
  };
}
