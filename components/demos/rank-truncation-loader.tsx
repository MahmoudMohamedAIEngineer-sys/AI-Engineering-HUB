"use client";

import dynamic from "next/dynamic";
import { useState } from "react";

/**
 * Defers loading the demonstration until it is asked for.
 *
 * The demonstration carries a 21 kB data file and a canvas renderer. A reader who never opens it
 * should not download either, which is the rule this platform follows for interactive material:
 * it is an island, loaded on demand, and it must earn its place by teaching something the text
 * cannot.
 */
const RankTruncation = dynamic(() => import("./rank-truncation"), {
  ssr: false,
  loading: () => (
    <p className="small muted" style={{ margin: 0 }}>
      Loading the demonstration…
    </p>
  ),
});

export default function RankTruncationLoader({ label }: { label?: string }) {
  const [shown, setShown] = useState(false);

  if (!shown) {
    return (
      <div className="panel" style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 320px" }}>
          <p className="eyebrow" style={{ marginBottom: 4 }}>
            Interactive
          </p>
          <p style={{ margin: 0, fontSize: 14.5 }}>
            {label ??
              "Keep only the k largest singular values and watch the matrix come back: what survives, what it costs to store, and the exact error Eckart-Young predicts."}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShown(true)}
          style={{
            font: "inherit",
            padding: "8px 16px",
            border: "1px solid var(--ink)",
            background: "var(--ink)",
            color: "var(--paper)",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          Open the demonstration
        </button>
      </div>
    );
  }
  return <RankTruncation />;
}
