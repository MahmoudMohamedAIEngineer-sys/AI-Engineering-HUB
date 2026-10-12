/** Small shared presentational pieces. Server components: no client JavaScript unless required. */

import Link from "next/link";
import type { ReactNode } from "react";

export function Badge({
  kind = "plain",
  children,
}: {
  kind?: "source" | "verified" | "draft" | "gap" | "plain";
  children: ReactNode;
}) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

/** The state a lesson is in, said in words a reader can act on. */
export function StateBadge({ state }: { state: "draft" | "needs_revision" | "verified" | "awaiting_sources" | null }) {
  if (state === "verified") return <Badge kind="verified">verified</Badge>;
  if (state === "needs_revision") return <Badge kind="draft">checks pending</Badge>;
  if (state === "awaiting_sources") return <Badge kind="gap">no source yet</Badge>;
  if (state === "draft") return <Badge kind="draft">draft</Badge>;
  return <Badge kind="plain">—</Badge>;
}

export function Stat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className="stat">
      <span className="stat-num">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

export function Panel({ children, plain = false }: { children: ReactNode; plain?: boolean }) {
  return <div className={plain ? "panel panel-plain" : "panel"}>{children}</div>;
}

export function SectionCard({
  eyebrow,
  title,
  href,
  meta,
  children,
}: {
  eyebrow?: string;
  title: string;
  href?: string;
  meta?: ReactNode;
  children?: ReactNode;
}) {
  const heading = href ? (
    <Link href={href} style={{ color: "inherit", textDecoration: "none" }}>
      {title}
    </Link>
  ) : (
    title
  );
  return (
    <article className="panel">
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      <h3 style={{ marginTop: 0, marginBottom: 6 }}>{heading}</h3>
      {meta ? <p className="small muted" style={{ margin: "0 0 8px" }}>{meta}</p> : null}
      {children}
    </article>
  );
}

/**
 * A statement of what a page establishes and what it does not.
 *
 * The platform's rule is that unverified material must be visibly labelled, so this component is
 * used wherever a reader could otherwise mistake a draft for a checked work.
 */
export function HonestyNote({ children }: { children: ReactNode }) {
  return (
    <div className="panel" style={{ marginTop: 28, fontSize: 14 }}>
      <p className="eyebrow" style={{ marginBottom: 6 }}>
        What this page establishes, and what it does not
      </p>
      <div style={{ color: "var(--ink-soft)" }}>{children}</div>
    </div>
  );
}

export function CheckRow({ label, status }: { label: string; status: string }) {
  const kind =
    status === "passed" ? "verified" : status === "not_run" ? "draft" : status === "not_applicable" ? "plain" : "gap";
  return (
    <li>
      <span className="muted small">{label.replace(/_/g, " ")}</span>
      <span>
        <Badge kind={kind as "verified" | "draft" | "plain" | "gap"}>{status.replace(/_/g, " ")}</Badge>
      </span>
    </li>
  );
}
