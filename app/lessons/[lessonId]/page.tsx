import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Badge, CheckRow, StateBadge } from "@/lib/ui";
import RankTruncationLoader from "@/components/demos/rank-truncation-loader";
import { getCompetencies, getLessonBundle, getRegistry, getChecksReport } from "@/lib/knowledge";

export function generateStaticParams() {
  return getRegistry().lessons.map((lesson) => ({ lessonId: lesson.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ lessonId: string }> }) {
  const { lessonId } = await params;
  const bundle = getLessonBundle(lessonId);
  return { title: bundle ? bundle.lesson.title : "Lesson" };
}

const ORIGIN_LABEL: Record<string, { kind: "source" | "verified" | "draft" | "gap" | "plain"; text: string }> = {
  source: { kind: "source", text: "from the source" },
  adapted: { kind: "draft", text: "adapted from the source" },
  original: { kind: "plain", text: "written for this platform" },
  external: { kind: "gap", text: "external" },
};

export default async function LessonPage({ params }: { params: Promise<{ lessonId: string }> }) {
  const { lessonId } = await params;
  const bundle = getLessonBundle(lessonId);
  if (!bundle) notFound();

  const { lesson, record, exercises, markdown, state } = bundle;
  const { competencies } = getCompetencies();
  const checks = getChecksReport();

  const checkedBy = (name: string) => record?.checks.find((check) => check.name === name);
  const sourceFidelity = checkedBy("source_fidelity");
  const pedagogical = checkedBy("pedagogical_completeness");

  return (
    <div className="shell" style={{ display: "grid", gap: 34, gridTemplateColumns: "minmax(0, 1fr)" }}>
      <header>
        <p className="eyebrow">
          {lesson.id} · {lesson.competencies.join(", ")}
        </p>
        <h1>{lesson.title}</h1>
        <p className="lede">{lesson.provenance.source_supported.split(".").slice(0, 2).join(". ")}.</p>
        <p style={{ marginTop: 16, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <StateBadge state={state} />
          {lesson.sources.map((source) => (
            <Badge key={source.work_id} kind="source">
              {source.work_id} {source.sections.join(", ")}
              {source.pages ? ` · pp. ${source.pages}` : ""}
            </Badge>
          ))}
          <Badge kind="plain">
            {lesson.estimated_effort.reading_minutes} min reading · {lesson.estimated_effort.exercise_minutes} min
            practice
          </Badge>
        </p>
      </header>

      <div className="grid grid-2">
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Source anchors
          </p>
          <p className="small muted" style={{ marginTop: 0 }}>
            What in the book this lesson rests on, by printed page. The book itself is not quoted here
            and its figures are not reproduced; you read those in your own copy.
          </p>
          <dl className="meta-grid">
            {lesson.source_anchors?.sections.map((section) => (
              <div key={section.id} style={{ display: "contents" }}>
                <dt>MML {section.id}</dt>
                <dd>
                  printed p. {section.printed_page} ·{" "}
                  <span className="muted small">
                    {section.file}, line {section.line}
                  </span>
                </dd>
              </div>
            ))}
            {lesson.source_anchors?.numbered_objects?.map((object) => (
              <div key={object} style={{ display: "contents" }}>
                <dt className="muted">cited</dt>
                <dd className="small">{object}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Prerequisites, with the specific fact each supplies
          </p>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14 }}>
            {lesson.prerequisites.map((prerequisite) => (
              <li key={prerequisite.competency} style={{ marginBottom: 8 }}>
                <Link href={`/curriculum/${prerequisite.competency}`}>
                  <strong>{prerequisite.competency}</strong>
                </Link>{" "}
                {competencies.find((competency) => competency.id === prerequisite.competency)?.name} —{" "}
                <span className="muted">{prerequisite.need}</span>
              </li>
            ))}
          </ul>
          <p className="small muted" style={{ marginBottom: 0 }}>
            Only competencies can be prerequisites here; the interface refuses to link to material
            that does not exist.
          </p>
        </div>
      </div>

      <article className="prose">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
          {markdown}
        </ReactMarkdown>
      </article>

      {lesson.interactive ? (
        <section aria-labelledby="interactive-heading">
          <h2 id="interactive-heading">The trade-off, movable</h2>
          <p className="muted" style={{ maxWidth: "68ch", fontSize: 14.5 }}>
            {lesson.interactive.serves}
          </p>
          <RankTruncationLoader />
          <p className="small muted" style={{ maxWidth: "68ch" }}>
            Data: <code>{lesson.interactive.data_path}</code>, built by{" "}
            <code>tools/build_demo_data.py</code>. The component performs the reconstruction in your
            browser; the numbers beside it are the recorded output of the lesson&apos;s checks, so the
            demonstration cannot drift away from the mathematics.
          </p>
        </section>
      ) : null}

      <section>
        <h2>Exercises</h2>
        <p className="muted" style={{ maxWidth: "68ch", fontSize: 14.5 }}>
          Origin is declared per exercise. &ldquo;Adapted&rdquo; means the question comes from the
          source but the instance used here was chosen for this platform, so the book&apos;s own
          exercise remains available for independent practice. Solutions are in{" "}
          {exercises[0]?.solution_path?.split("#")[0]}.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Exercise</th>
                <th>Focus</th>
                <th>Difficulty</th>
                <th>Origin</th>
                <th>Verified by</th>
              </tr>
            </thead>
            <tbody>
              {exercises.map((exercise, index) => (
                <tr key={exercise.id}>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <a href={exercise.solution_path?.split("#")[1] ? `#${exercise.solution_path.split("#")[1]}` : "#"}>
                      {index + 1}
                    </a>
                  </td>
                  <td>{exercise.prompt.split(".")[0]}.</td>
                  <td>{exercise.difficulty.replace(/_/g, " ")}</td>
                  <td>
                    <Badge kind={ORIGIN_LABEL[exercise.origin].kind}>{ORIGIN_LABEL[exercise.origin].text}</Badge>
                    {exercise.origin_detail?.printed_page ? (
                      <div className="small muted">source p. {exercise.origin_detail.printed_page}</div>
                    ) : null}
                  </td>
                  <td className="small muted">{exercise.solution_verified_by}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>Verification record</h2>
        <div className="grid grid-2">
          <div className="panel">
            <p className="eyebrow" style={{ marginBottom: 8 }}>
              {record ? record.id : "no record"}
            </p>
            <p style={{ marginTop: 0 }}>
              Outcome: <StateBadge state={state} /> <span className="small muted">on {record?.reviewed_on}</span>
            </p>
            <p className="small" style={{ color: "var(--ink-soft)" }}>
              {record?.outcome_explanation}
            </p>
            <p className="small muted" style={{ marginBottom: 0 }}>
              Reviewer: {record?.reviewer}.
            </p>
          </div>
          <div className="panel">
            <p className="eyebrow" style={{ marginBottom: 4 }}>
              Checks
            </p>
            <ul className="check-list">
              {record?.checks.map((check) => (
                <CheckRow key={check.name} label={check.name} status={check.status} />
              ))}
            </ul>
          </div>
        </div>

        <div style={{ marginTop: 18 }}>
          <details>
            <summary>What the executable checks actually did</summary>
            <p className="small muted">
              {checks.summary.checks} checks, {checks.summary.passed} passed, {checks.summary.failed} failed, run{" "}
              {checks.generated_at} with numpy {checks.environment.numpy} and sympy {checks.environment.sympy}.
              Output recorded in <code>knowledge/lessons/checks-report.json</code>.
            </p>
            <ul style={{ fontSize: 14 }}>
              {checks.results.map((result) => (
                <li key={result.check} style={{ marginBottom: 8 }}>
                  <code>{result.check}</code>
                  <div className="small muted">{result.evidence.split("\n").slice(0, 3).join(" ").slice(0, 420)}</div>
                </li>
              ))}
            </ul>
          </details>

          {sourceFidelity ? (
            <details>
              <summary>Checks that were not run — read this before trusting the lesson</summary>
              <p className="small">{sourceFidelity.method}</p>
              {pedagogical ? <p className="small">{pedagogical.method}</p> : null}
              <ul style={{ fontSize: 14 }}>
                {record?.unresolved_issues.map((issue) => (
                  <li key={issue.issue} style={{ marginBottom: 10 }}>
                    <Badge kind={issue.severity === "blocking" ? "gap" : "draft"}>{issue.severity}</Badge>{" "}
                    <strong>{issue.issue}</strong>
                    <div className="small muted">{issue.next_action}</div>
                  </li>
                ))}
              </ul>
              <p className="small muted" style={{ marginBottom: 0 }}>
                This is why the lesson is displayed as <em>checks pending</em> and is counted as zero
                verified lessons on the home page.
              </p>
            </details>
          ) : null}
        </div>
      </section>

      <section>
        <h2>Provenance</h2>
        <dl className="meta-grid" style={{ maxWidth: "80ch" }}>
          {Object.entries(lesson.provenance).map(([key, value]) => (
            <div key={key} style={{ display: "contents" }}>
              <dt>{key.replace(/_/g, " ")}</dt>
              <dd className="small">{value}</dd>
            </div>
          ))}
          <div style={{ display: "contents" }}>
            <dt>file</dt>
            <dd className="small">
              <code>{lesson.content_path}</code>
              <br />
              <span className="muted">sha256 {lesson.content_sha256.slice(0, 16)}…</span>
            </dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
