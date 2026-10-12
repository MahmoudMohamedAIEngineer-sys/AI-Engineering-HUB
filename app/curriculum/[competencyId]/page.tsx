import Link from "next/link";
import { notFound } from "next/navigation";
import { Badge, StateBadge } from "@/lib/ui";
import { getCompetencies, getCoverage, getPlaceholders, getRegistry, getSkillMap } from "@/lib/knowledge";

export function generateStaticParams() {
  return getCompetencies().competencies.map((competency) => ({ competencyId: competency.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ competencyId: string }> }) {
  const { competencyId } = await params;
  const competency = getCompetencies().competencies.find((item) => item.id === competencyId);
  return { title: competency ? `${competency.id} ${competency.name}` : "Competency" };
}

export default async function CompetencyPage({ params }: { params: Promise<{ competencyId: string }> }) {
  const { competencyId } = await params;
  const { competencies } = getCompetencies();
  const competency = competencies.find((item) => item.id === competencyId);
  if (!competency) notFound();

  const coverage = getCoverage().competencies.find((item) => item.competency_id === competencyId);
  const placeholder = getPlaceholders().placeholders.find((item) => item.competency_id === competencyId);
  const registry = getRegistry();
  const skillMap = getSkillMap();
  const lessons = registry.lessons.filter((lesson) => lesson.competencies.includes(competencyId));
  const dependents = competencies.filter((item) =>
    item.prerequisites.some((prerequisite) => prerequisite.id === competencyId),
  );

  return (
    <div className="shell" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr)", gap: 30 }}>
      <header>
        <p className="eyebrow">
          {competency.domain} · target level: {competency.target_level} · {competency.id}
        </p>
        <h1>{competency.name}</h1>
        <p className="lede">{competency.why}</p>
        <p style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
          {lessons.length > 0 ? (
            <Badge kind="source">{lessons.length} lesson written</Badge>
          ) : (
            <StateBadge state="awaiting_sources" />
          )}
          <Badge kind={coverage?.declared_source_status === "sourced" ? "verified" : coverage?.declared_source_status === "partial" ? "draft" : "gap"}>
            {coverage?.declared_source_status === "sourced"
              ? "source available"
              : coverage?.declared_source_status === "partial"
                ? "partly covered by the supplied book"
                : "no source in the repository"}
          </Badge>
        </p>
      </header>

      <section>
        <h2>What you must be able to do</h2>
        <ul style={{ maxWidth: "68ch" }}>
          {competency.outcomes.map((outcome) => (
            <li key={outcome} style={{ marginBottom: 6 }}>
              {outcome}
            </li>
          ))}
        </ul>
      </section>

      {lessons.length > 0 ? (
        <section>
          <h2>Lessons</h2>
          <ul className="item-list">
            {lessons.map((lesson) => {
              const record = registry.verification_records.find((item) => item.id === lesson.verification_record_id);
              return (
                <li key={lesson.id}>
                  <Link href={`/lessons/${lesson.id}`}>
                    <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                      <span className="item-title">{lesson.title}</span>
                      {record ? (
                        <StateBadge state={record.outcome === "verified" ? "verified" : "needs_revision"} />
                      ) : (
                        <StateBadge state="draft" />
                      )}
                    </div>
                    <div className="item-meta">
                      {lesson.source_anchors?.sections
                        .map((section) => `MML ${section.id} (printed p. ${section.printed_page})`)
                        .join(" · ")}
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>
      ) : (
        <section>
          <h2>No lesson yet</h2>
          <div className="placeholder">
            <p style={{ marginTop: 0 }}>
              <strong>Nothing has been written for this competency, and nothing will be until a source
              is supplied.</strong>{" "}
              {placeholder?.source_status_explanation ??
                "The repository holds no material that could support a lesson here."}
            </p>
            {placeholder ? (
              <>
                <p className="small" style={{ marginBottom: 6 }}>
                  <strong>What is missing:</strong>
                </p>
                <ul className="small" style={{ marginTop: 0 }}>
                  {placeholder.supplemental_needed.length > 0 ? (
                    placeholder.supplemental_needed.map((need) => <li key={need}>{need}</li>)
                  ) : (
                    <li>A primary source covering the outcomes above.</li>
                  )}
                </ul>
                <p className="small" style={{ marginBottom: 0 }}>
                  <strong>Where this sits in your own plan:</strong>{" "}
                  {placeholder.roadmap_skills.length > 0
                    ? placeholder.roadmap_skills
                        .map((skill) => `${skill.skill} ${skill.name} (${skill.study_days} study days)`)
                        .join("; ")
                    : "no roadmap skill maps to this competency"}
                  . {placeholder.roadmap_order.note}
                </p>
                <p className="small muted" style={{ marginBottom: 0, marginTop: 8 }}>
                  This card is generated from <code>content/curriculum/placeholders.json</code>. It
                  contains no teaching text by design: a placeholder must never look like a lesson.
                </p>
              </>
            ) : null}
          </div>
        </section>
      )}

      <section className="grid grid-2">
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Prerequisites
          </p>
          {competency.prerequisites.length === 0 ? (
            <p className="small muted" style={{ margin: 0 }}>
              None declared.
            </p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14 }}>
              {competency.prerequisites.map((prerequisite) => (
                <li key={prerequisite.id} style={{ marginBottom: 8 }}>
                  <Link href={`/curriculum/${prerequisite.id}`}>{prerequisite.id}</Link>{" "}
                  {competencies.find((item) => item.id === prerequisite.id)?.name ?? "(not in framework)"} —{" "}
                  <span className="muted">{prerequisite.because}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Other competencies that need this one
          </p>
          {dependents.length === 0 ? (
            <p className="small muted" style={{ margin: 0 }}>
              None declared. That is a gap in the framework, not a statement about importance.
            </p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14 }}>
              {dependents.map((dependent) => (
                <li key={dependent.id}>
                  <Link href={`/curriculum/${dependent.id}`}>{dependent.id}</Link>{" "}
                  <span className="muted">{dependent.name}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section>
        <h2>Where its material comes from</h2>
        <dl className="meta-grid" style={{ maxWidth: "80ch" }}>
          <dt>declared status</dt>
          <dd className="small">{coverage?.declared_source_status}</dd>
          <dt>indexed sections</dt>
          <dd className="small">
            {coverage && coverage.mml_sections_declared.length > 0 ? (
              coverage.mml_sections_declared.map((section, index) => (
                <span key={section}>
                  {index > 0 ? " · " : ""}
                  <Link href={`/source#${section.replace(".", "-")}`}>MML {section}</Link>
                </span>
              ))
            ) : (
              <span className="muted">none</span>
            )}
          </dd>
          <dt>supplemental needed</dt>
          <dd className="small">
            {competency.source_coverage.supplemental_needed.length > 0
              ? competency.source_coverage.supplemental_needed.join("; ")
              : "—"}
          </dd>
          <dt>notes</dt>
          <dd className="small">{competency.source_coverage.notes}</dd>
        </dl>
        {competency.roadmap_skills.length > 0 ? (
          <>
            <h3>Roadmap skills that demand it</h3>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Skill</th>
                    <th>Name</th>
                    <th>Status in the map</th>
                    <th>Source the skill asks for</th>
                  </tr>
                </thead>
                <tbody>
                  {competency.roadmap_skills.map((skillId) => {
                    const skill = skillMap.skills.find((item) => item.id === skillId);
                    return (
                      <tr key={skillId}>
                        <td>{skillId}</td>
                        <td>{skill?.name}</td>
                        <td>
                          <Badge kind={skill?.status === "sourced" ? "verified" : skill?.status === "partial" ? "draft" : "gap"}>
                            {skill?.status ?? "—"}
                          </Badge>
                        </td>
                        <td className="small muted">{skill?.source_need || "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}
