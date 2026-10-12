import Link from "next/link";
import { Badge, Stat } from "@/lib/ui";
import { getCompetencies, getCoverage, getDomains } from "@/lib/knowledge";

export const metadata = { title: "Curriculum" };

const STATUS_KIND = { sourced: "verified", partial: "draft", none: "gap" } as const;

export default function CurriculumPage() {
  const { domains } = getDomains();
  const coverage = getCoverage();
  const { competencies } = getCompetencies();
  const counts = coverage.summary.competency_coverage_counts;

  const byDomain = new Map(domains.map((domain) => [domain.id, { domain, rows: [] as typeof coverage.competencies }]));
  for (const row of coverage.competencies) {
    byDomain.get(row.domain)?.rows.push(row);
  }

  return (
    <div className="shell">
      <p className="eyebrow">Curriculum map · reviewed {getCompetencies().last_reviewed}</p>
      <h1>What has to be learned, and what it rests on</h1>
      <p className="lede">
        {coverage.summary.totals.competencies} competencies in {domains.length} domains. Each states
        the outcome it must produce, its prerequisites, why it matters for AI engineering, and where
        its material comes from. Coverage is computed from the source index and this framework — it is
        not a claim of completeness, and the roadmap that seeded it is not proof that any of it has
        been taught.
      </p>

      <div className="grid grid-4" style={{ marginTop: 30 }}>
        <Stat value={counts.sourced} label="fully covered by the supplied book" />
        <Stat value={counts.partial} label="partly covered" />
        <Stat value={counts.none} label="no source yet" />
        <Stat value={domains.length} label="domains" />
      </div>

      <h2>Domains</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Domain</th>
              <th>What it is for</th>
              <th>Competencies</th>
              <th>Coverage</th>
            </tr>
          </thead>
          <tbody>
            {domains.map((domain) => {
              const stats = coverage.summary.by_domain[domain.id] ?? { competencies: 0, sourced: 0, partial: 0, none: 0 };
              return (
                <tr key={domain.id}>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <strong>{domain.id}</strong>
                    <div className="small muted">{domain.name}</div>
                  </td>
                  <td style={{ fontSize: 14 }}>{domain.purpose}</td>
                  <td>
                    <Link href={`#${domain.id}`}>{stats.competencies}</Link>
                  </td>
                  <td style={{ whiteSpace: "nowrap", fontSize: 13.5 }}>
                    <Badge kind="verified">{stats.sourced}</Badge> <span className="muted">sourced</span>
                    <br />
                    <Badge kind="draft">{stats.partial}</Badge> <span className="muted">partial</span>
                    <br />
                    <Badge kind="gap">{stats.none}</Badge> <span className="muted">no source</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {domains.map((domain) => {
        const rows = byDomain.get(domain.id)?.rows ?? [];
        if (rows.length === 0) return null;
        return (
          <section key={domain.id} id={domain.id}>
            <h2>
              {domain.id} — {domain.name}
            </h2>
            <p className="muted" style={{ maxWidth: "68ch", fontSize: 14.5 }}>
              {domain.purpose}
            </p>
            <ul className="item-list">
              {rows.map((row) => {
                const competency = competencies.find((item) => item.id === row.competency_id);
                return (
                  <li key={row.competency_id}>
                    <Link href={`/curriculum/${row.competency_id}`}>
                      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                        <span className="item-title">
                          <span className="muted small">{row.competency_id}</span> {row.name}
                        </span>
                        <Badge kind={STATUS_KIND[row.declared_source_status as keyof typeof STATUS_KIND]}>
                          {row.declared_source_status === "sourced"
                            ? "source available"
                            : row.declared_source_status === "partial"
                              ? "partly sourced"
                              : "no source yet"}
                        </Badge>
                        {row.lessons.length > 0 ? <Badge kind="source">{row.lessons.length} lesson</Badge> : null}
                      </div>
                      <div className="item-meta">
                        {competency?.outcomes[0]}
                      </div>
                      <div className="item-meta">
                        {row.mml_sections_declared.length > 0
                          ? `MML ${row.mml_sections_declared.join(", ")}`
                          : "no indexed source sections"}
                        {competency && competency.roadmap_skills.length > 0
                          ? ` · roadmap: ${competency.roadmap_skills.join(", ")}`
                          : ""}
                        {` · target level: ${competency?.target_level}`}
                      </div>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}

      <h2>Integrity of the map itself</h2>
      <p className="muted" style={{ maxWidth: "68ch", fontSize: 14.5 }}>
        {coverage.issues.length} consistency issues between the framework and the source index were
        found and reported by the coverage tool; they are listed in{" "}
        <code>knowledge/curriculum/coverage.json</code> and checked on every build.
      </p>
    </div>
  );
}
