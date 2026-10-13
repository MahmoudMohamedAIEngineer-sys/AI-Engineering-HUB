# Practice: Two Ways to Read This Book

Source topics for all problems: `ml.foundations.how-to-read-this-book`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [ml.foundations.how-to-read-this-book]
State the bottom-up and top-down strategies and one advantage and one disadvantage of each.

**Solution.** Bottom-up: foundations first; advantage = always rely on known concepts; disadvantage = unmotivated definitions are forgotten. Top-down: start from a practical need; advantage = purpose is always visible; disadvantage = knowledge rests on not-yet-understood words. [BOOK: mathematics-for-machine-learning, ch.1, §1.2]

### Problem 2 (recall)
source_topics: [ml.foundations.how-to-read-this-book]
How is the book structured to support both strategies? Name the two parts and the bridging chapter.

**Solution.** Part I (Ch.2-7) foundations, Part II (Ch.9-12) four ML pillars, Chapter 8 bridges by restating data/models/learning mathematically; pointers run forward and backward between parts. [BOOK: mathematics-for-machine-learning, ch.1, §1.2]

### Problem 3 (computation)
source_topics: [ml.foundations.how-to-read-this-book]
Using the dependency dict in the topic's code (Ch.10 prerequisites via Ch.8: [8], Ch.8: [2..7]), list the full set of chapters that must precede Chapter 10 in a bottom-up traversal.

**Solution.** Ch.8 requires 2,3,4,5,6,7; Ch.10 requires 8; so bottom-up set before 10 = {2,3,4,5,6,7,8}. (Part II chapters 9,11,12 are not prerequisites.) [DERIVED from topic code]

### Problem 4 (computation)
source_topics: [ml.foundations.how-to-read-this-book]
A learner reads only Part II chapters in order 9,10,11,12. Count how many of the four pillar chapters they read before ever opening Chapter 8, and state the risk named by the authors.

**Solution.** 4 of 4 before Chapter 8. Risk: the top-down fragility — using words/results whose foundations were never built, so knowledge can collapse when assumptions are examined. [BOOK: mathematics-for-machine-learning, ch.1, §1.2]

### Problem 5 (derivation)
source_topics: [ml.foundations.how-to-read-this-book]
Argue why 'most readers mix both strategies' follows from the two disadvantage statements, rather than being an extra assumption.

**Solution.** Pure bottom-up loses motivation before payoff; pure top-down accumulates undefined terms. Mixing assigns each study unit the direction whose disadvantage does not bite: foundations with a near payoff bottom-up, blocked terms top-down from a live problem. So mixing is the strategy that avoids both stated failure modes. [DERIVED]

### Problem 6 (application (ML))
source_topics: [ml.foundations.how-to-read-this-book]
You must ship a PCA feature-reduction step in two weeks and have never studied eigenvectors. Sketch a top-down plan with explicit drill-down targets.

**Solution.** Run the PCA recipe; drill: data matrix/centering (Ch.2), covariance X^T X (topic-inverse-and-transpose), eigenvectors (Ch.4 topics when written), projections/why top eigenvector (Ch.3 topics). Each blockage becomes a bottom-up mini-study, then return to the pipeline. [DERIVED]

### Problem 7 (application (ML))
source_topics: [ml.foundations.how-to-read-this-book]
Your bottom-up study stalled at abstract vector spaces. Apply this section to recover, in two concrete steps.

**Solution.** (1) Attach a destination: pick the Chapter 9 regression derivation that uses subspaces and read its statement. (2) Study only the subspace facts that derivation cites, then return. This converts unmotivated definitions into needed tools, the book's modular intent. [DERIVED]

### Problem 8 (interview)
source_topics: [ml.foundations.how-to-read-this-book]
Interview-style: 'How do you learn a new ML technique?' Give a model answer and a weak-answer example.

**Solution.** Model: 'Top-down from the system goal until a term blocks me, then bottom-up through that term's prerequisites, verify on a tiny numeric example, reconnect to the system.' Weak: 'I watch tutorials until it clicks' — no prerequisites, no verification. source_topics: [ml.foundations.how-to-read-this-book]
