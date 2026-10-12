# ADR 0002 — Application stack for the learning interface

**Status:** proposed (awaiting confirmation before phase 4 implements it) · **Date:** 2026-10-12

## Context

The interface must render long-form prose, mathematics, code, interactive demonstrations and
data-driven navigation (curriculum, sources, references, progress). It must work on a small
screen, respect reduced motion, and not be slowed down by the interactive parts on pages that do
not use them. The knowledge base already exists as JSON, and lesson prose will exist as files
with typed front matter (ADR 0001).

Nothing is installed yet; this decision is being taken now so that the interface phase does not
start by rewriting itself.

## Decision (proposed)

- **Next.js (App Router) with React and TypeScript, statically generated where possible.** Every
  curriculum, source and reference page is data over local files, so it can be pre-rendered; only
  interactive demonstrations need client-side code. Static generation also keeps the reading
  experience fast on a content-heavy site.
- **Content model in TypeScript with schema validation** (Zod, or an equivalent) shared between
  the build step and the app, so a lesson whose front matter contradicts the registry fails the
  build rather than rendering a wrong badge.
- **KaTeX for mathematics**, chosen over MathJax for render speed on content-heavy pages. The
  project's scope is well-defined notation, not arbitrary macro packages.
- **Shiki for code highlighting** at build time, so no highlighting runtime is shipped.
- **Interactive demonstrations are island components**, loaded only on the pages that use them,
  with a static fallback (a figure plus the same numbers in text) and reduced-motion support from
  the start.
- **React Three Fiber and similar libraries are not adopted as a default.** They are considered
  only where a specific concept needs three-dimensional manipulation (for example a linear
  transformation acting on a plane). A page that does not need it must not pay for it.
- **No database, no server, no authentication in this phase.** Progress can be stored locally
  until there is a real requirement for accounts; adding auth would create a security surface with
  no user benefit yet.

## Consequences

- The interface is a build-time view over the knowledge base; the curriculum can be inspected
  without running the app.
- Static generation keeps the reading experience fast, at the cost of a build step that must
  validate content.
- The stack is mainstream enough that maintenance is not a research project, and Node 22 is
  already present.

## Alternatives considered

| Alternative | Why not (yet) |
| --- | --- |
| Plain HTML pages like the two roadmap sites | Those pages work because their data is small and static. This curriculum has hundreds of competencies, sections and lessons with cross-references; hand-maintained HTML would drift immediately |
| Astro | A strong fit for content sites and a legitimate alternative. Rejected for now only because the interactive islands are a first-class requirement here and React's ecosystem for the specific visualisations (interactive linear algebra, optimisation landscapes) is richer |
| Vite + React SPA without a framework | Loses per-page HTML to pre-render for a reading-heavy site |
| Notion/an existing platform as the host | The project's value is in its verification and traceability model, which would have to be abandoned in a general-purpose tool |
| Docusaurus | Built for documentation of software, not for a curriculum with prerequisites, coverage and assessment state |

**Open question for the user:** whether to proceed with this stack or to express a preference.
This ADR is deliberately marked *proposed* rather than *accepted*.
