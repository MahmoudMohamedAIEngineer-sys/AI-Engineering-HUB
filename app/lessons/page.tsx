import Link from "next/link";
import { Badge, StateBadge } from "@/lib/ui";
import { displayStateFor, getCoverage, getRegistry } from "@/lib/knowledge";

export const metadata = { title: "Lessons" };

export default function LessonsPage() {
  const registry = getRegistry();
  const coverage = getCoverage();

  const competenciesWithLessons = new Set(registry.lessons.flatMap((lesson) => lesson.competencies));
  const awaiting = coverage.competencies
    .filter((competency) => !competenciesWithLessons.has(competency.competency_id))
    .sort((a, b) => a.competency_id.localeCompare(b.competency_id));

  return (
    <div className="shell">
      <p className="eyebrow">Content layer</p>
      <h1>Lessons</h1>
      <p className="lede">
        {registry.current_state.lessons_authored as number} lesson in the repository, at{" "}
        {registry.current_state.exercises_authored as number} exercises, plus{" "}
        {registry.current_state.topics_with_placeholders as number} topics that are waiting for a
        source. New material is added topic by topic as books arrive; nothing here is generated from a
        template, and there is no filler to make the list look fuller than it is.
      </p>

      <h2>Written</h2>
      <ul className="item-list">
        {registry.lessons.map((lesson) => {
          const record = registry.verification_records.find((item) => item.id === lesson.verification_record_id);
          const exerciseCount = registry.exercises.filter((exercise) => exercise.lesson_id === lesson.id).length;
          return (
            <li key={lesson.id}>
              <Link href={`/lessons/${lesson.id}`}>
                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                  <span className="item-title">{lesson.title}</span>
                  <StateBadge state={displayStateFor(lesson, record)} />
                </div>
                <div className="item-meta">
                  {lesson.id} · {exerciseCount} exercises ·{" "}
                  {lesson.sources
                    .map((source) => `${source.work_id} ${source.sections.join(", ")} (printed pp. ${source.pages})`)
                    .join("; ")}
                </div>
                <div className="item-meta" style={{ marginTop: 6 }}>
                  {lesson.provenance.source_supported.split(".")[0]}.
                </div>
              </Link>
            </li>
          );
        })}
      </ul>

      <h2>Not written, and why</h2>
      <p className="muted" style={{ maxWidth: "68ch", fontSize: 14.5 }}>
        These are competencies with no lesson. Each one needs a source that is not in the repository
        yet; the placeholder file records what it is waiting for. They are listed here so that the
        absence is part of the interface rather than something a reader has to discover.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Competency</th>
              <th>Topic</th>
              <th>Source status</th>
              <th>Waiting for</th>
            </tr>
          </thead>
          <tbody>
            {awaiting.map((competency) => (
              <tr key={competency.competency_id}>
                <td>
                  <Link href={`/curriculum/${competency.competency_id}`}>{competency.competency_id}</Link>
                </td>
                <td>{competency.name}</td>
                <td>
                  <Badge kind={competency.declared_source_status === "partial" ? "draft" : "gap"}>
                    {competency.declared_source_status === "partial" ? "partly sourced" : "no source"}
                  </Badge>
                </td>
                <td className="small muted">
                  {competency.supplemental_needed.length > 0
                    ? competency.supplemental_needed.join("; ")
                    : "a primary source covering this topic"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
