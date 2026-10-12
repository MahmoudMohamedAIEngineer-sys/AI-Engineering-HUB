import { getRoadmapInventory, getRoadmapQuality, getSkillMap } from "@/lib/knowledge";
import { Badge, Stat } from "@/lib/ui";

export const metadata = { title: "Roadmap" };

const SEVERITY_KIND = { high: "gap", medium: "draft", low: "plain" } as const;

export default function RoadmapPage() {
  const inventory = getRoadmapInventory();
  const quality = getRoadmapQuality();
  const skillMap = getSkillMap();

  const phases = Object.entries(inventory.counts.by_phase);
  const highSeverity = quality.issues.filter((issue) => issue.severity === "high");

  return (
    <div className="shell">
      <p className="eyebrow">Extracted from the supplied HTML tables</p>
      <h1>Your roadmap, as machine-readable data</h1>
      <p className="lede">
        The two roadmap pages you supplied are the seed of this curriculum. They were extracted into
        structured data rather than copied: {inventory.counts.skills} skills across {phases.length} phases,{" "}
        {inventory.counts.project_sprints} project sprints, and{" "}
        {inventory.counts.summed_study_days_including_sprints} study days. Everything the extraction found
        wrong with them is listed on this page, because a curriculum built on a defective plan inherits the
        defect.
      </p>

      <div className="grid grid-4" style={{ marginTop: 28 }}>
        <Stat value={inventory.counts.skills} label="skills" />
        <Stat value={inventory.counts.skills_with_day_level_plan} label="with day-by-day plans" />
        <Stat value={inventory.counts.skills_summary_only} label="summary only" />
        <Stat value={quality.issues.length} label="defects found in the source tables" />
      </div>

      <div className="grid grid-2" style={{ marginTop: 30 }}>
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Where the numbers came from
          </p>
          <dl className="meta-grid">
            {quality.sources.map((source) => (
              <div key={source.path} style={{ display: "contents" }}>
                <dt className="small">{source.role}</dt>
                <dd className="small">
                  <code>{source.path}</code>
                  <div className="muted">
                    {source.bytes.toLocaleString()} bytes · sha256 {source.sha256.slice(0, 12)}…
                  </div>
                </dd>
              </div>
            ))}
          </dl>
          <p className="small">
            The HTML files themselves are never modified. Extraction reads them and writes JSON under{" "}
            <code>knowledge/roadmaps/</code>.
          </p>
          <p className="small" style={{ marginBottom: 0 }}>
            <strong>Your pages, served verbatim:</strong>{" "}
            <a href="/perfect_resources_44_skills.html" target="_blank" rel="noreferrer">
              the 44-skills resource page
            </a>{" "}
            ·{" "}
            <a href="/study_timetable_8hrs.html" target="_blank" rel="noreferrer">
              the 8-hour study timetable
            </a>
            . These are byte-identical copies of the files you supplied, served so the originals stay
            reachable; the validator fails the build if either copy drifts from its original.
          </p>
        </div>

        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            What the extraction could not resolve
          </p>
          <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            <li>
              {highSeverity.length} of the {quality.issues.length} defects are severe enough to change what a
              learner would study.
            </li>
            <li>
              Coverage at skill level: {skillMap.skills.filter((skill) => skill.status === "sourced").length} sourced,{" "}
              {skillMap.skills.filter((skill) => skill.status === "partial").length} partial,{" "}
              {skillMap.skills.filter((skill) => skill.status === "none").length} with no source in the repository.
            </li>
            <li>
              {skillMap.audit_findings.length} audit findings ({skillMap.audit_findings.filter((f) => f.classification === "audit-addition").length} of
              them additions made by this audit, not problems in your plan).
            </li>
          </ul>
        </div>
      </div>

      <h2>Skills</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Skill</th>
              <th>Phase</th>
              <th>What it covers</th>
              <th>Plan</th>
              <th>Source status</th>
            </tr>
          </thead>
          <tbody>
            {inventory.skills.map((skill) => {
              const mapping = skillMap.skills.find((item) => item.id === skill.id);
              return (
                <tr key={skill.id}>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {skill.id}
                    <div className="small muted">{skill.name}</div>
                  </td>
                  <td className="small">{skill.phase}</td>
                  <td className="small">{skill.what_you_learn}</td>
                  <td className="small">
                    {skill.schedule.study_days ?? "—"} days
                    <div className="muted">{skill.plan_depth}</div>
                  </td>
                  <td>
                    <Badge kind={mapping?.status === "sourced" ? "verified" : mapping?.status === "partial" ? "draft" : "gap"}>
                      {mapping?.status ?? "—"}
                    </Badge>
                    <div className="small muted">{mapping?.source_need}</div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h2>Defect register</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Each defect names the two places in the source tables that disagree, so you can check it yourself
        in the original files. Nothing was repaired silently.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Severity</th>
              <th>Code</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {quality.issues.map((issue) => (
              <tr key={issue.code + issue.detail.slice(0, 30)}>
                <td>
                  <Badge kind={SEVERITY_KIND[issue.severity as keyof typeof SEVERITY_KIND] ?? "plain"}>
                    {issue.severity}
                  </Badge>
                </td>
                <td className="small">
                  <code>{issue.code}</code>
                </td>
                <td className="small">{issue.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Audit findings</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Findings from comparing your roadmap against the competency framework and the supplied sources.
        They are classified: <em>audit-addition</em> means the topic was added by this audit and is not a
        criticism of your plan.
      </p>
      {skillMap.audit_findings.map((finding) => (
        <details key={finding.id}>
          <summary>
            <Badge kind={finding.severity === "high" ? "gap" : finding.severity === "medium" ? "draft" : "plain"}>
              {finding.severity}
            </Badge>{" "}
            {finding.id} — {finding.finding}
          </summary>
          <p className="small">
            <strong>Evidence.</strong> {finding.evidence}
          </p>
          <p className="small">
            <strong>Recommendation.</strong> {finding.recommendation}
          </p>
          <p className="small muted" style={{ marginBottom: 0 }}>
            Classification: {finding.classification} · status: {finding.status}
          </p>
        </details>
      ))}
    </div>
  );
}
