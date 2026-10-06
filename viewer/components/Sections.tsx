// Everything below the Hero: uncovered regions, curves, savings, evidence,
// measurements, limitations and the two reproduction paths.

import { fmt } from "../lib/map.mjs";
import type { Measured, Report } from "../lib/types";

export function UncoveredStrip({ report }: { report: Report }) {
  return (
    <div className="uncovered-strip" role="region" aria-label="Uncovered regions">
      <strong>UNCOVERED</strong>
      <span>your validation set covers only part of the conditions probed:</span>
      {report.uncovered_regions.map((r) => (
        <span key={r.axis} className="badge">{r.axis} {r.coverage_percent.toFixed(1)}%</span>
      ))}
    </div>
  );
}

function Curve({ c, threshold }: { c: Report["curves"][number]; threshold: number }) {
  const W = 320, H = 160, L = 36, B = 28;
  const xs = c.points.map((p) => p.value);
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const ymax = Math.max(...c.points.map((p) => p.map50), threshold) * 1.1;
  const X = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - 8);
  const Y = (v: number) => 8 + (1 - v / ymax) * (H - B - 8);
  const path = [...c.points].sort((a, b) => a.value - b.value)
    .map((p, k) => `${k ? "L" : "M"}${X(p.value).toFixed(1)},${Y(p.map50).toFixed(1)}`).join(" ");
  return (
    <figure className="card" style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`mAP@50 versus ${c.axis}`}>
        <rect x={L} y={Y(threshold)} width={W - L - 8} height={H - B - Y(threshold)} fill="var(--fail-dim)" opacity={0.5} />
        <line x1={L} x2={W - 8} y1={Y(threshold)} y2={Y(threshold)} stroke="var(--fail)" strokeWidth={1} />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
        <text x={L} y={H - 8} fill="var(--text-1)" fontSize={11} className="mono">{fmt(lo, c.unit, 3)}</text>
        <text x={W - 8} y={H - 8} fill="var(--text-1)" fontSize={11} textAnchor="end" className="mono">{fmt(hi, c.unit, 3)}</text>
      </svg>
      <figcaption className="caption mono">{c.axis}</figcaption>
    </figure>
  );
}

function measured(m: Measured) {
  return m === "not measured"
    ? <span className="not-measured">not measured</span>
    : <span className="mono">{fmt(m.value, m.unit)} <span className="caption">({m.source})</span></span>;
}

export default function Sections({ report }: { report: Report }) {
  const levels = report.efficiency ? Object.values(report.efficiency.by_grid_level) : [];
  return (
    <>
      <hr className="divider" />
      <section className="panel">
        <h2>Where this pipeline fails, one axis at a time</h2>
        <div className="grid-4">
          {report.curves.map((c) => <Curve key={c.axis} c={c} threshold={report.run.threshold_map50} />)}
        </div>
        <table style={{ marginTop: "var(--s-16)" }}>
          <thead><tr><th>axis</th><th>boundary</th><th>probes (verified bisection)</th><th>reproduce</th></tr></thead>
          <tbody>
            {report.findings.map((f) => (
              <tr key={f.axis}>
                <td className="num">{f.axis}</td>
                <td className="num">{f.lower !== null && f.upper !== null ? `${fmt(f.lower, f.unit)} – ${fmt(f.upper, f.unit)}` : f.status}</td>
                <td className="num">{f.probes_used ?? <span className="not-measured">not measured</span>}</td>
                <td className="num caption">{f.reproduce.command}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <hr className="divider" />
      <section className="panel">
        <h2>Probe savings against an exhaustive grid</h2>
        <p className="caption">{report.efficiency?.note}</p>
        <table>
          <thead><tr><th>grid points</th><th>savings, verified bisection</th><th>savings, cheap bisection</th></tr></thead>
          <tbody>
            {levels.map((l) => (
              <tr key={l.grid_steps}>
                <td className="num">{l.grid_steps}</td>
                <td className="num">{l.mean_savings_verified !== null ? `${l.mean_savings_verified.toFixed(3)}×` : <span className="not-measured">not measured</span>}</td>
                <td className="num">{l.mean_savings_cheap !== null ? `${l.mean_savings_cheap.toFixed(3)}×` : <span className="not-measured">not measured</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <hr className="divider" />
      <section className="panel">
        <h2>Evidence frames</h2>
        <p className="caption">Illustrative. Chosen by a stated rule: redistributable sources only, people not
          the subject, preferring the most confident wrong box the degradation created, otherwise the
          largest loss of correct boxes. The measured claim is the set-level mAP@50 above.</p>
        <div className="grid-2">
          {report.evidence_frames.map((e) => (
            <figure key={e.axis} style={{ margin: 0 }}>
              <img className="evidence" src={`data/frames/${e.png}`} alt={`${e.axis} evidence frame`} />
              <figcaption className="caption">
                <span className="mono">{e.axis} = {fmt(e.condition_value, e.unit)}</span> ·{" "}
                {e.selection_mode === "confident_false_positive"
                  ? <>most confident wrong box <span className="mono">{e.frame.max_fp_score.toFixed(2)}</span></>
                  : <>failure mode here: missed detections, no confident wrong box</>}
                {" "}· correct boxes{" "}
                <span className="mono">{e.frame.baseline_tp} → {e.frame.fail_tp}</span> ·{" "}
                <a href={e.attribution.source}>source</a>, {e.attribution.license}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <hr className="divider" />
      <div className="grid-2">
        <section className="panel">
          <h2>Uncovered regions</h2>
          <ul className="list">
            {report.uncovered_regions.flatMap((r) => r.uncovered.map((u) => (
              <li key={`${r.axis}-${u.from}`} className="mono">
                {r.axis} {fmt(u.from, u.unit)} – {fmt(u.to, u.unit)} — never tested
              </li>
            )))}
          </ul>
          <p className="caption">Inferred from image statistics of your undegraded frames, not from capture metadata.</p>
        </section>
        <section className="panel">
          <h2>Known limitations</h2>
          <ul className="list">
            <li>sim-to-real gap, low light (third-party NOD photos, illuminance estimated): the synthetic boundary is too pessimistic by at least {measured(report.measurements.sim2real_gap)}</li>
            <li>COOL vs stock OpenCV 5 on the same Graviton4: relative speed {measured(report.measurements.cool_vs_stock)}</li>
            <li>Graviton4 vs x86, stock OpenCV 5: relative speed {measured(report.measurements.graviton_vs_x86)}</li>
            {report.limitations.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </section>
      </div>

      <hr className="divider" />
      <section className="panel">
        <h2>Reproduce</h2>
        <div className="grid-2">
          <div><div className="label">A. full (AWS)</div>
            <pre>{`cd infra && npm ci\nnpx cdk bootstrap --qualifier bspot \\\n    --toolkit-stack-name CDKToolkit-blindspot\nnpx cdk deploy Blindspot\ncd .. && uv run --group cloud blindspot cloud-run \\\n    --dataset val/road100 --budget 0.40`}</pre></div>
          <div><div className="label">B. local (no AWS)</div>
            <pre>{`docker build -t blindspot:local .\ndocker run --rm --network none \\\n    -v "$PWD/val:/val:ro" blindspot:local \\\n    run --dataset /val/road100 --probes 8 \\\n    --axis motion_blur.exposure_ms --grid-steps 17`}</pre></div>
        </div>
        <p className="caption">Full steps in the repository README. Screens narrower than 768 px are not supported.</p>
      </section>
    </>
  );
}
