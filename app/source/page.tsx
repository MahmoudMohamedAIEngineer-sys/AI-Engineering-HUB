import { getDefects, getInventory, getManifest, getPdfVerification, getStructure } from "@/lib/knowledge";
import { Badge, Stat } from "@/lib/ui";

export const metadata = { title: "Source library" };

export default function SourcePage() {
  const manifest = getManifest();
  const structure = getStructure();
  const inventory = getInventory();
  const defects = getDefects();
  const pdf = getPdfVerification();

  return (
    <div className="shell">
      <p className="eyebrow">Source library · one work indexed</p>
      <h1>{manifest.title}</h1>
      <p className="lede">
        {manifest.authors.join(", ")} · {manifest.edition} · {manifest.publisher}
      </p>
      <p className="muted small" style={{ maxWidth: "70ch" }}>
        {manifest.note_on_identity}
      </p>

      <div className="grid grid-4" style={{ marginTop: 26 }}>
        <Stat value={structure.totals.sections} label="table-of-contents sections located" />
        <Stat value={structure.totals.subsection_headings} label="subsection headings" />
        <Stat value={inventory.numbered_objects.equations.total_unique} label="numbered equations" />
        <Stat value={inventory.numbered_objects.figures.total_unique} label="figure references" />
      </div>

      <div className="grid grid-2" style={{ marginTop: 30 }}>
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            How the source is represented
          </p>
          <dl className="meta-grid">
            <dt>canonical form</dt>
            <dd className="small">
              Markdown, supplied by the reader. The PDF is used only to verify the index independently;
              it is never treated as the content source.
            </dd>
            <dt>files</dt>
            <dd className="small">
              {manifest.files.map((file) => (
                <div key={file.path}>
                  <code>{file.path}</code>
                  <span className="muted"> · {file.lines.toLocaleString()} lines · sha256 {file.sha256.slice(0, 12)}…</span>
                </div>
              ))}
            </dd>
            <dt>images</dt>
            <dd className="small">
              {inventory.images.references} figure references, {inventory.images.distinct_urls} distinct URLs, all
              external. {inventory.images.note}
            </dd>
            <dt>license</dt>
            <dd className="small">{manifest.license_note}</dd>
          </dl>
        </div>

        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 8 }}>
            Independent verification against the PDF
          </p>
          <p className="small" style={{ marginTop: 0 }}>
            Printed page to PDF page offset: <strong>{pdf.page_offset.printed_to_pdf_offset}</strong>, consistent
            across every exercise section. Chapters containing exercise sets appear on PDF pages{" "}
            {pdf.exercise_sets.pdf_pages_with_exercise_headings.join(", ")}.
          </p>
          <p className="small">
            Chapters {pdf.exercise_sets.chapters_without_sets.join(", ")} contain no exercise sets.{" "}
            <strong>{pdf.exercise_sets.whole_pdf_search_after_last_exercise_set.conclusion}</strong>
          </p>
          <p className="small muted" style={{ marginBottom: 0 }}>
            Not checked in this pass: {pdf.not_checked.join("; ")}.
          </p>
        </div>
      </div>

      <h2>Defect register</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        {defects.summary.errors} errors, {defects.summary.warnings} warnings, {defects.summary.info} informational
        findings recorded while indexing the source. They are kept rather than fixed, because a cleaned index
        would hide what the extraction actually found.
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
            {defects.defects.map((defect) => (
              <tr key={defect.code + defect.detail.slice(0, 24)}>
                <td>
                  <Badge kind={defect.severity === "error" ? "gap" : defect.severity === "warning" ? "draft" : "plain"}>
                    {defect.severity}
                  </Badge>
                </td>
                <td className="small">
                  <code>{defect.code}</code>
                </td>
                <td className="small">{defect.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Structure map</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        Every chapter and section, with the printed page it starts on and the line it was found at in
        the markdown. Sections are what lessons cite, so this table is the join between the library and
        the curriculum.
      </p>
      {structure.chapters.map((chapter) => (
        <section key={chapter.id} id={`chapter-${chapter.id}`}>
          <h3 style={{ marginBottom: 4 }}>
            Chapter {chapter.id} — {chapter.title}
            <span className="muted small">
              {" "}
              · printed p. {chapter.printed_page} · {chapter.sections.length} sections
            </span>
          </h3>
          <p className="small muted" style={{ marginTop: 2 }}>
            {chapter.exercises_printed_page
              ? `Exercise set on printed p. ${chapter.exercises_printed_page}.`
              : "No exercise set in the supplied source for this chapter."}
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Section</th>
                  <th>Title</th>
                  <th>Printed page</th>
                  <th>Located at</th>
                </tr>
              </thead>
              <tbody>
                {chapter.sections.map((section) => (
                  <tr key={section.id} id={section.id.replace(".", "-")}>
                    <td style={{ whiteSpace: "nowrap" }}>{section.id}</td>
                    <td>{section.title}</td>
                    <td>{section.printed_page}</td>
                    <td className="small muted">
                      {section.anchor ? (
                        <>
                          <code>{section.anchor.file.split("/").pop()}</code> line {section.anchor.line}
                        </>
                      ) : (
                        <Badge kind="gap">not located</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {structure.missing_sections.length > 0 ? (
        <p className="small">
          <Badge kind="gap">incomplete</Badge> Sections in the table of contents that could not be located in
          the markdown: {structure.missing_sections.join(", ")}. This is recorded rather than repaired, and it
          limits which topics can be sourced from this work.
        </p>
      ) : null}

      <h2>Exercise sets in the source</h2>
      <p className="muted" style={{ maxWidth: "70ch", fontSize: 14.5 }}>
        The platform searches the source&apos;s own problem sets before authoring anything, so it needs to know
        which chapters have them. {inventory.exercises.total_items_located} items were located in{" "}
        {inventory.exercises.chapters_with_supplied_exercise_sets.length} chapters;{" "}
        {inventory.exercises.chapters_without_supplied_exercise_sets.length} chapters have none. Where the
        wording of an item did not survive extraction, the item is recorded as missing rather than
        reconstructed.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Chapter</th>
              <th>Items located</th>
              <th>Items in the PDF</th>
              <th>Printed page of the set</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(inventory.exercises.per_chapter).map(([chapterId, data]) => (
              <tr key={chapterId}>
                <td>{chapterId}</td>
                <td>{data.item_count}</td>
                <td>{structure.exercises_index[chapterId] ?? "—"}</td>
                <td>{data.printed_page ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
