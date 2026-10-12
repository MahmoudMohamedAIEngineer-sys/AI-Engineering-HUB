"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import data from "@/content/demos/rank-truncation.json";

/**
 * Rank-truncation demonstration.
 *
 * Everything numerical comes from the file built by `tools/build_demo_data.py`: the matrix, its
 * full SVD, and the recorded error and storage arithmetic. The component only performs the
 * reconstruction A_k = U_k Σ_k V_kᵀ and draws it. It deliberately does **not** recompute the
 * singular value decomposition in the browser, so the theorem being demonstrated and the numbers
 * shown cannot drift apart.
 *
 * The matrix is synthetic and built from explicit formulas (see the builder script); it is a
 * teaching object, not a dataset, and the interface says so.
 */

const SIZE = data.matrix.length;

function reconstruct(k: number): number[][] {
  // U_k Σ_k V_kᵀ, computed row by row in the browser.
  const out: number[][] = [];
  for (let row = 0; row < SIZE; row += 1) {
    const line: number[] = new Array(SIZE).fill(0);
    for (let component = 0; component < k; component += 1) {
      const weight = data.u[row][component] * data.singular_values[component];
      const vt = data.vt[component];
      for (let column = 0; column < SIZE; column += 1) {
        line[column] += weight * vt[column];
      }
    }
    out.push(line);
  }
  return out;
}

function draw(canvas: HTMLCanvasElement, values: number[][]) {
  const scale = Math.max(2, Math.floor(canvas.clientWidth / SIZE));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = SIZE * scale * ratio;
  canvas.height = SIZE * scale * ratio;
  canvas.style.height = `${SIZE * scale}px`;
  const context = canvas.getContext("2d");
  if (!context) return;
  // Values are clamped for display only: a truncated reconstruction can leave [0, 1], and the
  // error that causes is exactly what the error figures beside the images report.
  for (let row = 0; row < SIZE; row += 1) {
    for (let column = 0; column < SIZE; column += 1) {
      const value = Math.min(1, Math.max(0, values[row][column]));
      const level = Math.round(255 * (1 - value));
      context.fillStyle = `rgb(${level}, ${level}, ${level})`;
      context.fillRect(column * scale * ratio, row * scale * ratio, scale * ratio, scale * ratio);
    }
  }
}

function ImagePanel({
  values,
  title,
  caption,
}: {
  values: number[][];
  title: string;
  caption: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) draw(ref.current, values);
  }, [values]);
  return (
    <figure style={{ margin: 0 }}>
      <figcaption>
        <span className="eyebrow" style={{ display: "block", marginBottom: 4 }}>
          {title}
        </span>
        <canvas
          ref={ref}
          width={SIZE * 8}
          height={SIZE * 8}
          style={{ width: "100%", maxWidth: 288, imageRendering: "pixelated", border: "1px solid var(--rule)" }}
          role="img"
          aria-label={caption}
        />
        <p className="small muted" style={{ marginTop: 6, maxWidth: 288 }}>
          {caption}
        </p>
      </figcaption>
    </figure>
  );
}

export default function RankTruncation() {
  const [k, setK] = useState(3);
  const original = useMemo(() => data.matrix, []);
  const approximation = useMemo(() => reconstruct(k), [k]);
  const row = data.storage[k];
  const honestNote = data.verification.note;

  return (
    <div>
      <div style={{ display: "flex", gap: 28, flexWrap: "wrap", alignItems: "flex-start" }}>
        <ImagePanel
          values={original}
          title="The matrix"
          caption={`A ${SIZE}x${SIZE} matrix built from three formulas: a smooth bump, a diagonal ridge and fine texture. Full rank in exact arithmetic.`}
        />
        <ImagePanel
          values={approximation}
          title={`Rank-${k} reconstruction`}
          caption={
            k === 0
              ? "Keeping no components leaves the zero matrix, which is rank 0."
              : `Keeping the ${k} largest singular value${k === 1 ? "" : "s"}. Grey levels are clamped to [0, 1] for display only.`
          }
        />
      </div>

      <div style={{ marginTop: 18 }}>
        <label htmlFor="rank-slider" style={{ display: "block", fontSize: 14, marginBottom: 6 }}>
          <strong>Keep k = {k}</strong> of {SIZE} components
        </label>
        <input
          id="rank-slider"
          type="range"
          min={0}
          max={SIZE}
          step={1}
          value={k}
          onChange={(event) => setK(Number(event.target.value))}
          style={{ width: "100%", maxWidth: 420 }}
          aria-describedby="rank-readout"
        />
        <p id="rank-readout" className="small" style={{ marginTop: 10, maxWidth: "68ch" }}>
          Frobenius error <code>‖A − A_k‖_F = {row.frobenius_error.toFixed(3)}</code> (
          {(row.relative_frobenius_error * 100).toFixed(2)}% of <code>‖A‖_F</code>) · spectral error{" "}
          <code>‖A − A_k‖₂ = {row.spectral_error_equals_next_singular_value.toFixed(3)}</code>
          {k < SIZE ? (
            <>
              , which is <strong>σ_{k + 1} = {data.singular_values[k].toFixed(3)}</strong>, exactly as the
              Eckart-Young theorem says
            </>
          ) : (
            ", which is zero because nothing has been discarded"
          )}
          . Stored entries <code>{row.entries}</code> against <code>{row.full_entries}</code> for the full
          matrix ({(row.ratio * 100).toFixed(1)}%). Energy retained{" "}
          {k === 0 ? "0%" : `${(data.energy.cumulative[k - 1] * 100).toFixed(2)}%`}.
        </p>
      </div>

      <details style={{ marginTop: 14 }}>
        <summary>What this demonstration does and does not show</summary>
        <p className="small">
          The images and the error figures are computed from the SVD of the matrix shown, which was
          computed once by the build script and recorded in{" "}
          <code>content/demos/rank-truncation.json</code>. The spectral error displayed is the recorded
          next singular value, not a fresh measurement, so the identity it illustrates cannot be
          faked by a rounding difference: it is the theorem itself.
        </p>
        <p className="small">
          Storage is counted as <code>k(m + n + 1)</code> entries for the factors against{" "}
          <code>mn</code> for the full matrix. Real compressed storage uses coarser numbers per entry,
          headers and metadata, so the ratio shown is an upper bound on the saving, not a measurement
          of a file size.
        </p>
        <p className="small muted">{data.what_this_is}</p>
        <p className="small muted" style={{ marginBottom: 0 }}>
          {honestNote}
        </p>
      </details>
    </div>
  );
}
