import Link from "next/link";
import { Badge, HonestyNote, Stat } from "@/lib/ui";
import {
  displayStateFor,
  getCoverage,
  getPlaceholders,
  getReferences,
  getRegistry,
  getRoadmapQuality,
  getStructure,
} from "@/lib/knowledge";

export default function HomePage() {
  const registry = getRegistry();
  const coverage = getCoverage();
  const placeholders = getPlaceholders();
  const structure = getStructure();
  const quality = getRoadmapQuality();
  const references = getReferences();

  const totals = coverage.summary.competency_coverage_counts;
  const lesson = registry.lessons[0];
  const record = lesson
    ? registry.verification_records.find((item) => item.id === lesson.verification_record_id)
    : undefined;

  return (
    <div className="shell">
      <p className="eyebrow">Working build · generated {registry.generated_at}</p>
      <h1>A source-grounded atlas of AI engineering</h1>
      <p className="lede">
        The repository this site renders holds two connected layers. A <strong>curriculum map</strong>{" "}
        says what has to be learned, why it matters, what it depends on, and — for most of it — which
        source is still missing. A <strong>content layer</strong> holds the lessons: derivations
        reconstructed from supplied books, exercises with declared origins, and verification records
        that state what was checked and what was not.
      </p>
      <p className="lede" style={{ marginTop: 14 }}>
        One lesson exists so far. It was written to prove the pipeline end to end — front matter,
        generated index, executable checks, verification record — on a topic whose source is fully
        available. Most of the map is deliberately empty and says so.
      </p>

      <div className="grid grid-2" style={{ marginTop: 34, alignItems: "start" }}>
        <div>
          <h2 style={{ marginTop: 0 }}>Where the material actually is</h2>
          <div className="grid grid-4">
            <Stat value={coverage.summary.totals.competencies} label="competencies in the framework" />
            <Stat value={totals.sourced} label="fully sourced" />
            <Stat value={totals.partial} label="partly sourced" />
            <Stat value={totals.none} label="no source yet" />
          </div>
          <p className="small muted" style={{ marginTop: 14 }}>
            Coverage is computed from the source index and the authored framework, not asserted.
            The counts are in <code>knowledge/curriculum/coverage.json</code>.
          </p>
        </div>

        <div>
          <h2 style={{ marginTop: 0 }}>What exists as learning material</h2>
          <div className="grid grid-4">
            <Stat value={registry.current_state.lessons_authored as number} label="lessons authored" />
            <Stat value={registry.current_state.exercises_authored as number} label="exercises" />
            <Stat value={registry.current_state.lessons_verified as number} label="lessons verified" />
            <Stat value={placeholders.counts.placeholders} label="topics awaiting sources" />
          </div>
          <p className="small muted" style={{ marginTop: 14 }}>
            A lesson counts as verified only when its verification record says so. The record for the
            pilot lesson currently says <em>{record?.outcome.replace("_", " ") ?? "no record"}</em>,
            which is why the count above reads {String(registry.current_state.lessons_verified)}.
          </p>
        </div>
      </div>

      <h2>Start here</h2>
      <div className="grid grid-3">
        <div className="panel">
          <p className="eyebrow">The one finished lesson</p>
          <h3 style={{ marginTop: 0 }}>
            <Link href={`/lessons/${lesson?.id}`} style={{ color: "inherit", textDecoration: "none" }}>
              {lesson?.title}
            </Link>
          </h3>
          <p className="small muted" style={{ margin: "0 0 10px" }}>
            {lesson?.sources.map((source) => `${source.work_id} ${source.sections.join(", ")}`).join(" · ")}
            {lesson?.sources[0]?.pages ? ` · printed pp. ${lesson.sources[0].pages}` : ""}
          </p>
          <p style={{ fontSize: 14.5 }}>
            Determinants as volume scaling, eigenvalues as invariant directions, and a full answer to
            when a matrix can be diagonalised. Seven exercises, one with a rubric, all with the
            verification that was run recorded alongside the answer.
          </p>
          <p>
            <Badge kind={displayStateFor(lesson!, record) === "verified" ? "verified" : "draft"}>
              {displayStateFor(lesson!, record) === "verified" ? "verified" : "checks pending"}
            </Badge>
          </p>
        </div>

        <div className="panel">
          <p className="eyebrow">The map</p>
          <h3 style={{ marginTop: 0 }}>
            <Link href="/curriculum" style={{ color: "inherit", textDecoration: "none" }}>
              Curriculum and coverage
            </Link>
          </h3>
          <p style={{ fontSize: 14.5 }}>
            {coverage.summary.totals.competencies} competencies in ten domains, each with the outcome
            it has to produce, its prerequisites, the roadmap skills that demand it, and its source
            status. {totals.none} have no source at all; {totals.partial} are partly covered.
          </p>
          <p className="small muted" style={{ marginBottom: 0 }}>
            The placeholders file carries the source need for every one of them:{" "}
            <code>{registry.placeholders.path}</code>.
          </p>
        </div>

        <div className="panel">
          <p className="eyebrow">The evidence</p>
          <h3 style={{ marginTop: 0 }}>
            <Link href="/method" style={{ color: "inherit", textDecoration: "none" }}>
              Method and verification
            </Link>
          </h3>
          <p style={{ fontSize: 14.5 }}>
            What “verified” means here, which checks are automated and which are not, the defect
            register for both supplied sources, and the list of places where this repository knows it
            is imperfect.
          </p>
          <p className="small muted" style={{ marginBottom: 0 }}>
            {structure.totals.sections} sections indexed · {quality.issues.length} recorded roadmap
            defects · {references.references.length} external references,{" "}
            {references.references.filter((reference) => reference.retrieved).length} retrieved.
          </p>
        </div>
      </div>

      <h2>Two layers, one direction of truth</h2>
      <div className="grid grid-2">
        <div className="panel panel-plain">
          <h3 style={{ marginTop: 0 }}>The curriculum map</h3>
          <p style={{ fontSize: 14.5 }}>
            Domains, competencies, prerequisites and source needs, plus the user&apos;s own roadmap
            translated into machine-readable form. It is allowed to contain unresolved items, and it
            currently contains many: {placeholders.counts.placeholders} competencies are waiting for
            a source, and the roadmap extraction recorded {quality.issues.length} defects in the
            supplied tables rather than silently repairing them.
          </p>
          <p style={{ fontSize: 14.5, marginBottom: 0 }}>
            <Link href="/roadmap">Read the roadmap inventory as extracted →</Link>
          </p>
        </div>
        <div className="panel panel-plain">
          <h3 style={{ marginTop: 0 }}>The content layer</h3>
          <p style={{ fontSize: 14.5 }}>
            Lessons, derivations, exercises and verification records, added topic by topic as sources
            arrive. Each lesson records the source sections it uses by printed page, which exercises
            are adapted from the source and which are original, and what its checks actually
            established.
          </p>
          <p style={{ fontSize: 14.5, marginBottom: 0 }}>
            <Link href="/lessons">See the lesson index →</Link>
          </p>
        </div>
      </div>

      <HonestyNote>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          <li>
            There is <strong>one lesson</strong>, and no human has yet compared it against the book it
            cites. Its record therefore says <em>checks pending</em>, not verified.
          </li>
          <li>
            <strong>Most of the curriculum is not written and cannot be written yet</strong>:{" "}
            {totals.none} competencies have no supplied source. Those entries are placeholders that
            state what is missing; they contain no lesson text at all.
          </li>
          <li>
            The roadmap tables this curriculum was seeded from contain real defects —{" "}
            {quality.issues.length} of them, listed in{" "}
            <code>knowledge/roadmaps/data-quality-report.json</code>. They were recorded, not smoothed
            over.
          </li>
        </ul>
      </HonestyNote>
    </div>
  );
}
