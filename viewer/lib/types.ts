// Mirrors src/blindspot/report.py. The Python schema is the authority; this is
// only what the viewer reads.
export type Interval = { from: number; to: number; unit: string };
export type Uncovered = { axis: string; unit: string; coverage_percent: number; uncovered: Interval[]; inferred_from?: string };
export type Detection = { box: number[]; label: string; score: number; matched: boolean };
export type Truth = { box: number[]; label: string };
export type Cell = {
  i: number; j: number; exposure_ms: number; illuminance_lux: number; psf_length_px: number;
  map50: number; failed: boolean; counts: Record<string, number>; image: string | null;
  detections?: { predictions: Detection[]; truths: Truth[] };
};
export type Map2D = {
  x: { axis: string; unit: string; values: number[]; spacing: string };
  y: { axis: string; unit: string; values: number[]; spacing: string };
  coupling: string;
  camera_model: Record<string, number | boolean>;
  cells: Cell[][];
  seed: number;
  n_failed: number;
  n_cells: number;
  command: string;
  demo_frame: { image_id: string; attribution: { source: string; license: string; license_url: string }; selection: string; faces: string };
};
export type Measured = { value: number; unit: string; source: string } | "not measured";
export type Report = {
  uncovered_regions: Uncovered[];
  run: { run_id: string; pipeline: string; dataset: string; frames: number; objects: number; seed: number;
         baseline_map50: number; threshold_map50: number; criterion: string };
  findings: { axis: string; unit: string; status: string; lower: number | null; upper: number | null;
              probes_used: number | null;
              reproduce: { axis: string; value: number; unit: string; seed: number; source_frame: string; command: string } }[];
  map2d: Map2D | null;
  curves: { axis: string; unit: string; points: { value: number; map50: number }[] }[];
  efficiency: { by_grid_level: Record<string, { grid_steps: number; mean_savings_verified: number | null; mean_savings_cheap: number | null }>; note: string } | null;
  evidence_frames: { axis: string; unit: string; condition_value: number; png: string; selection_mode: string;
                     frame: { image_id: string; max_fp_score: number; baseline_tp: number; fail_tp: number };
                     attribution: { source: string; license: string; license_url: string }; reproduce: string }[];
  measurements: { sim2real_gap: Measured; cool_vs_stock: Measured; graviton_vs_x86: Measured };
  limitations: string[];
};
