import Link from "next/link";
import { Badge } from "@/lib/ui";
import { getChecksReport, getCoverage, getReferences, getRegistry } from "@/lib/knowledge";

export const metadata = { title: "Method" };

export default function MethodPage() {
  const registry = getRegistry();
  const checks = getChecksReport();
  const coverage = getCoverage();
  const references = getReferences();

  return (
    <div className="shell">
      <p className="eyebrow">How the material is made and checked</p>
      <h1>Method</h1>
      <p className="lede">
        This site exists to make knowledge navigable, so the rules it follows are part of the content
        rather than an appendix. What follows is the list of things this repository will not do, how a
        lesson is verified, and where it is currently failing to meet its own standard.
      </p>

      <h2>Rules that constrain the material</h2>
      <div className="grid grid-2">
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>What is never done</h3>
          <ul style={{ fontSize: 14.5, paddingLeft: 18 }}>
            <li>No fabricated citations, page numbers, quotations, historical claims or benchmarks.</li>
            <li>
              No markdown or HTML file supplied by the reader is modified. Those files are the canonical
              source, and the index under <code>knowledge/</code> is generated separately.
            </li>
            <li>
              No book text and no book figures are reproduced. Prose here is written for this platform and
              cites the source by section and printed page; the original figures are pointed at by number.
            </li>
            <li>Nothing is marked verified without a verification record that shows what was checked.</li>
            <li>
              No topic is filled in because a page looked empty. A competency with no source gets a
              placeholder that says what is missing.
            </li>
          </ul>
        </div>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>How a lesson gets written</h3>
          <ol style={{ fontSize: 14.5, paddingLeft: 18 }}>
            <li>The source section is located in the index; the printed page is recorded.</li>
            <li>
              The reasoning is reconstructed rather than paraphrased: motivation, assumptions, every step
              of each derivation, and the limits of the result.
            </li>
            <li>
              The source&apos;s own problem sets are searched first. Exercises adapted from the source keep
              their item number and page; exercises authored here are labelled as written for this
              platform.
            </li>
            <li>Every numerical claim is recomputed by a script under <code>content/checks/</code>.</li>
            <li>
              A verification record is written listing each check, its status, and the checks that were not
              run.
            </li>
            <li>The lesson is generated into the index, and the interface reads its state from there.</li>
          </ol>
        </div>
      </div>

      <h2>What “verified” means here</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Nine checks are recorded per lesson. Only some of them are automated, and the interface shows
        which: a check with no method is not counted as passed. The state of a lesson is taken from its
        record, never from wording inside the lesson file itself.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Check</th>
              <th>Automated?</th>
              <th>What it means</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["source fidelity", "no", "A reader compares the lesson against the book section it cites."],
              ["technical correctness", "yes", "Every numerical and algebraic claim recomputed by script."],
              ["mathematical validity", "partly", "Derivations re-derived; results cited from the source are not re-proved."],
              ["pedagogical completeness", "partly", "Each declared outcome is taught somewhere and assessed somewhere."],
              ["prerequisite integrity", "yes", "Prerequisites exist in the framework and have a route to them."],
              ["assessment integrity", "yes", "Every exercise has an outcome, an origin and a solution."],
              ["citation integrity", "yes", "Every cited identifier exists in the source index."],
              ["interactive correctness", "n/a", "Applies only where a lesson has an interactive component."],
              ["publication integrity", "no", "Rendering, links and small-screen behaviour checked after publication."],
            ].map(([name, automated, meaning]) => (
              <tr key={name}>
                <td style={{ whiteSpace: "nowrap" }}>{name}</td>
                <td>
                  <Badge kind={automated === "yes" ? "verified" : automated === "no" ? "draft" : "plain"}>{automated}</Badge>
                </td>
                <td className="small">{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Executable evidence</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        The checks for lesson {registry.lessons[0]?.id} were last run {checks.generated_at} against numpy{" "}
        {checks.environment.numpy} and sympy {checks.environment.sympy}: {checks.summary.checks} checks,{" "}
        {checks.summary.passed} passed, {checks.summary.failed} failed. Each result and its output is in{" "}
        <code>knowledge/lessons/checks-report.json</code>.
      </p>
      {checks.results.slice(0, 3).map((result) => (
        <details key={result.check}>
          <summary>
            <code>{result.check}</code> — what it measured
          </summary>
          <p className="small" style={{ whiteSpace: "pre-wrap" }}>
            {result.evidence.slice(0, 900)}
          </p>
        </details>
      ))}

      <h2>Where this repository is currently failing its own standard</h2>
      <ul style={{ maxWidth: "72ch", fontSize: 14.5 }}>
        <li>
          <strong>The one lesson is not verified.</strong> No human has read the book section against it.
          Its record therefore says checks pending, and the home page counts zero verified lessons.
        </li>
        <li>
          <strong>{coverage.summary.competency_coverage_counts.none} competencies have no source.</strong>{" "}
          They cannot be written until books are supplied. The{" "}
          <Link href="/lessons">lesson index</Link> lists every one of them rather than hiding the absence.
        </li>
        <li>
          <strong>The roadmap tables have {getRegistry().current_state.topics_with_placeholders as number} topics
          that are placeholders.</strong> Their priority is taken from the roadmap&apos;s own order, not from a
          judgement this platform made.
        </li>
        <li>
          <strong>Interactive components are absent.</strong> The interface is deliberately just reading and
          navigation until the content model proves itself; no demonstration is shipped that does not teach
          something the text cannot.
        </li>
      </ul>

      <h2>Notation</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Two notation systems meet here: the supplied book&apos;s and this platform&apos;s. The book&apos;s
        notation is kept inside the mathematics, simplified forms are used in prose and code, and every
        place the two differ is written down — including the symbols that mean more than one thing
        across the curriculum (<code>λ</code>, <code>σ</code>, <code>k</code>, <code>D</code>) and the
        rule each lesson follows when a collision could mislead. The full table is{" "}
        <code>docs/notation.md</code> in the repository.
      </p>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        In lesson sources, <code>$...$</code> is inline mathematics, <code>$$...$$</code> is display
        mathematics, and backticks are code or identifiers rather than mathematics. Prose mathematics
        is written as display equations or backticked symbols rather than sprinkling inline math
        through sentences; inline math is reserved for a symbol that genuinely appears inside a
        sentence where the alternative would be a parenthetical definition.
      </p>

      <h2>External references</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Where material comes from outside the supplied books, the reference is recorded with its authority
        level and whether it was actually retrieved. A reference that was not retrieved can support a
        claim about the source&apos;s existence, not about its content.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Reference</th>
              <th>Authority</th>
              <th>Retrieved</th>
              <th>Used for</th>
            </tr>
          </thead>
          <tbody>
            {references.references.map((reference) => (
              <tr key={reference.id}>
                <td className="small">
                  <a href={reference.url} target="_blank" rel="noreferrer noopener">
                    {reference.title}
                  </a>
                  <div className="muted">{reference.publisher}</div>
                </td>
                <td>
                  <Badge kind={reference.authority === "primary" ? "verified" : reference.authority === "institutional" ? "source" : "draft"}>
                    {reference.authority}
                  </Badge>
                </td>
                <td>
                  <Badge kind={reference.retrieved ? "verified" : "draft"}>
                    {reference.retrieved ? `yes · ${reference.retrieved_on}` : "no"}
                  </Badge>
                </td>
                <td className="small">{reference.used_for}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
