# Decisions

## DEC-001: Site framework (2026-10-13, Phase 0)
- Type: reversible
- Goal / constraints: render long math-heavy explanations readably; interactives later; static export; simple to maintain solo.
- Options compared: A Next.js/React, B Astro static, C plain HTML+JS generated from Markdown
- Scores: | Criterion(weight) | A | B | C |
  | Content depth/accuracy (35) | 4 (KaTeX+MDX mature) | 4 | 3 |
  | Fidelity/completeness/ordering (20) | 4 | 4 | 3 |
  | Verifiability (15) | 4 (Playwright/Lighthouse standard) | 4 | 3 |
  | Comprehension/engagement (10) | 5 | 4 | 2 |
  | Simplicity/maintainability (10) | 3 | 5 | 4 |
  | Use of provided resources (5) | 4 | 3 | 2 |
  | Speed/cost (5) | 3 | 5 | 5 |
- Winner: B Astro (totals: A=395, B=410, C=300 weighted approx — B wins on simplicity+speed at equal content fidelity)
- Why: Content is Markdown-first; Astro ships zero-JS by default with islands for the few interactives, keeping reading fast. Next.js adds server complexity with no content benefit.
- Strongest argument against: React ecosystem (R3F etc.) is richer in Next — mitigated: Astro supports React islands if ever needed.
- Revisit if: interactives need shared app state across pages.
- Supersedes: none

## DEC-002: Hosting target (2026-10-13, Phase 0)
- Type: reversible
- Goal / constraints: free/cheap static hosting, preview URLs, easy rollback.
- Options: A GitHub Pages, B Cloudflare Pages, C Vercel
- Scores: Content 35: A4 B4 C4; Fidelity 20: A4 B4 C4; Verifiability 15: A3 B4 C5; Engagement 10: A3 B4 C4; Simplicity 10: A5 B4 C4; Resources 5: A3 B3 C3; Speed/cost 5: A5 B5 C3
- Winner: B Cloudflare Pages (free tier, fast CDN, preview deploys)
- Why: Static Astro output deploys anywhere; Cloudflare Pages gives previews without cost.
- Strongest argument against: GitHub Pages is simpler with this GitHub repo — accepted as fallback; build output is portable.
- Revisit if: Cloudflare account unavailable (D8: needs user account; site phase only).
- Supersedes: none

## DEC-003: Repo name (2026-10-13, Phase 0)
- Type: reversible
- Goal: keep existing cloned repo. Options: A keep AI-Engineering-HUB, B rename ai-engineering-hub, C new repo.
- Winner: A keep existing name/path. Why: renaming risks breaking the tracked session branch; name already matches the project.
- Strongest argument against: case differs from spec's lowercase — cosmetic only.
- Revisit if: publishing publicly. Supersedes: none

## DEC-004: Design direction (2026-10-13, Phase 0)
- Type: reversible
- Goal / constraints: maximize readability of long math content (R14: design serves content); dark-on-light reading, restrained motion.
- Options: A editorial serif reading theme, B dark developer theme (like user's 44-skills HTML), C colorful dashboard
- Winner: A editorial reading theme (light, high-contrast serif body + sans UI, one accent), tokens defined in Phase 5.
- Why: Long-form study reading is the product; dashboard/dark themes fatigue over hours.
- Strongest argument against: dark theme matches user's provided HTML files — noted; a dark toggle can be added in Phase 5 without changing direction.
- Revisit if: Phase 5 user testing shows strain. Supersedes: none

## DEC-005: Verification tooling language/runtime (2026-10-13, Phase 0)
- Type: reversible
- Goal: scripts run anywhere with zero install. Options: A Python stdlib only, B Python+numpy, C Node.
- Winner: A for all gate scripts and topic code blocks (pure stdlib: math/fractions). NumPy checks may be added as supplementary proofs with .venv but gates never depend on them.
- Why: sandbox python3 is externally-managed (no pip system install); stdlib guarantees code_check reproducibility for the learner too.
- Strongest argument against: NumPy is idiomatic for ML code — mitigated: Phase 3+ programming topics will teach NumPy explicitly; math topics verify the underlying arithmetic by hand-checkable code, which is pedagogically stronger.
- Revisit if: topic code needs real datasets. Supersedes: none

## DEC-006: Subsection merging in Chapter 2 pilot (2026-10-13, Phase 2)
- Type: reversible
- Goal: obey Step 2 granularity without fragmenting tiny sections.
- Rule applied: headings <300 source words, and unnumbered helper headings (e.g. "Calculating the Inverse"), are `merged` into their parent numbered section's topic file, recorded in coverage.json topic_files. Numbered sections >=300 words get their own topic file.
- Why: master prompt explicitly permits merging <~300-word sections (log here). Supersedes: none

## DEC-007: Target learner level & code language (2026-10-13, Phase 1B)
- Type: reversible
- Goal: calibrate depth. Decision: learner = aspiring professional AI engineer with high-school math (the book's stated audience, confirmed at mml-book.com), aiming at employable depth. Code language: Python (top skill in AI postings per Stanford AI Index cited in sources.md #7; cross-checked #5,#6).
- Supersedes: none
