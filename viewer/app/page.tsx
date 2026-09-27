"use client";

import { useEffect, useState } from "react";
import Hero from "../components/Hero";
import Sections, { UncoveredStrip } from "../components/Sections";
import type { Report } from "../lib/types";

export default function Page() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("data/report.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setReport)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main className="page">
      <header className="topbar">
        <span className="brand">Blindspot</span>
        {report && (
          <span className="meta mono">
            run {report.run.run_id} · pipeline {report.run.pipeline} · {report.run.dataset} ({report.run.frames} frames)
          </span>
        )}
      </header>
      {error && <div className="card" style={{ borderColor: "var(--fail)" }}><span className="mono">E_REPORT</span> could not load report.json: {error}</div>}
      {!report && !error && <p className="caption mono">loading report…</p>}
      {report && (
        <>
          <UncoveredStrip report={report} />
          {report.map2d
            ? <Hero report={report} />
            : <div className="card"><span className="not-measured">2-D map not measured yet</span> — run <code>uv run python bench/hook_grid.py</code></div>}
          <Sections report={report} />
        </>
      )}
    </main>
  );
}
