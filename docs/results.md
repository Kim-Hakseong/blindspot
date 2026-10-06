# Results

Every figure here is cited to a file under `bench/out/` and verified by
`uv run python tools/check_doc_numbers.py`. Nothing is quoted that has no
generating command.

## Setup

| | |
|---|---|
| Validation set | `road100` — 100 frames, 984 labelled objects, COCO val2017 under CC BY 2.0 / CC BY-SA 2.0 |
| Pipeline under test | YOLOX-S (Apache-2.0, OpenCV model zoo) via `cv::dnn` |
| Undegraded baseline | mAP@50 = 0.6111 <!--bench:efficiency.baseline.mAP50--> |
| Failure criterion | mAP@50 below 0.3666 <!--bench:efficiency.criterion.threshold_map50--> , i.e. 60% of the pipeline's own baseline |
| Seed | 20260906 |

The baseline is consistent with YOLOX-S's published COCO performance, which is
the main external check available on the metric implementation.

```bash
uv run python bench/grid_baseline.py --axis motion_blur.exposure_ms --steps 20
uv run python bench/boundary_efficiency.py --levels 5 9 17 33
```

## Where this pipeline fails

Boundaries located by exhaustive grid at 33 steps. Each is an **interval**, not
a point: its width is the residual uncertainty the probe budget bought.

| Axis | Boundary | Physical meaning |
|---|---|---|
| Motion blur (exposure) | 12.50 <!--bench:efficiency.axes[0].levels[3].grid.lower--> – 13.75 <!--bench:efficiency.axes[0].levels[3].grid.upper--> ms | At 60 deg/s and 900 px focal length, a PSF around 12 px long |
| Illuminance | 12.98 <!--bench:efficiency.axes[1].levels[3].grid.lower--> – 25.47 <!--bench:efficiency.axes[1].levels[3].grid.upper--> lux | Deep dusk / poorly lit interior |
| Fog (extinction β) | 0.06 <!--bench:efficiency.axes[2].levels[3].grid.lower--> – 0.0638 <!--bench:efficiency.axes[2].levels[3].grid.upper--> 1/m | Meteorological visibility around 62 m |
| JPEG quality | 7.97 <!--bench:efficiency.axes[3].levels[3].grid.lower--> – 10.94 <!--bench:efficiency.axes[3].levels[3].grid.upper--> q | Aggressive recompression |

These are properties of *this pipeline on this validation set*, not of YOLOX in
general. That is the point of the tool: the numbers are meant to be regenerated
for your pipeline and your data.

## Probe savings against an exhaustive grid

Savings is reported as a curve because the two strategies scale differently. To
halve the uncertainty in a boundary's position, a grid must double its probe
count while bisection needs one more probe. A single savings figure quoted
without its precision target is an arbitrary point on this curve.

| Grid | Probes (grid) | Probes (verified bisection) | Savings |
|---|---|---|---|
| 5 | 5 | 5 | 1.00× <!--bench:efficiency.summary.by_grid_level.5.mean_savings_verified--> |
| 9 | 9 | 6 | 1.50× <!--bench:efficiency.summary.by_grid_level.9.mean_savings_verified--> |
| 17 | 17 | 7 | 2.43× <!--bench:efficiency.summary.by_grid_level.17.mean_savings_verified--> |
| 33 | 33 | 8 | 4.12× <!--bench:efficiency.summary.by_grid_level.33.mean_savings_verified--> |

Cheap bisection, which skips the monotonicity check, reaches
4.71× <!--bench:efficiency.summary.by_grid_level.33.mean_savings_cheap--> at the
same precision. It is the blinder method and both are reported.

**At coarse precision active search saves nothing.** At 5 grid points it costs
the same 5 probes for no benefit. It only pays when a precise boundary is
wanted, which is the honest statement of when this technique is worth using.

### What was actually measured

The ratio 33/8 is arithmetic and would be the same on any axis. The empirical
content is this: on **all four axes**, verified bisection returned the *same
boundary interval as the exhaustive grid, with zero error*, using 8 probes
instead of 33.

| Axis | Boundary error vs grid | Savings valid |
|---|---|---|
| `motion_blur.exposure_ms` | 0 <!--bench:efficiency.axes[0].levels[3].bisection_verified.boundary_error--> | yes |
| `low_light.illuminance_lux` | 0 <!--bench:efficiency.axes[1].levels[3].bisection_verified.boundary_error--> | yes |
| `fog.beta_per_m` | 0 <!--bench:efficiency.axes[2].levels[3].bisection_verified.boundary_error--> | yes |
| `jpeg.quality` | 0 <!--bench:efficiency.axes[3].levels[3].bisection_verified.boundary_error--> | yes |

A search that used fewer probes but found a *different* boundary would have
saved nothing, so savings is only reported as valid when the located boundary
lands within one grid cell of the grid's.

## Two axes at once: the Blindspot Map

A real camera has one shutter, and its exposure time sets both how far the
image smears and how much light the sensor collects. The map therefore ties
the two together (`low_light.exposure_ms` = `motion_blur.exposure_ms`) and
probes a 17 x 17 grid of exposure x illuminance, every cell on all 100 frames.
Camera model: 60 deg/s pan, fixed gain, no auto-exposure.

```bash
uv run python bench/hook_grid.py --steps 17
```

273 <!--bench:hook_grid.n_failed--> of 289 <!--bench:hook_grid.n_cells-->
conditions fail. What passes is a closed window: short exposure and bright
light. At 114–173 lux it is two-sided in exposure -- too short starves the
sensor, too long smears the image -- and below about 75 lux it closes.

**Cross-check with the 1-D sweep.** At 400 lux the blur edge falls between
10.75 ms (mAP 0.3800 <!--bench:hook_grid.cells[0][4].map50-->, pass) and
13.19 ms (mAP 0.3221 <!--bench:hook_grid.cells[0][5].map50-->, fail),
overlapping the independent 1-D boundary of 12.50–13.75 ms.

**The axes interact.** At 10.75 ms and 49.5 lux each axis on its own is
inside its 1-D envelope -- blur below 12.5 ms, light above 25.5 lux (and the
1-D light sweep used an even shorter 10 ms shutter) -- yet the combined
condition fails, at mAP 0.2950 <!--bench:hook_grid.cells[5][4].map50-->.
One-axis envelopes overstate the safe region.

### 2-D boundary sampling

Level-set estimation (Gotovos et al., 2013) against the exhaustive map.
Probes are deterministic, so looking a cell up in the map is exactly what
re-running that probe would return, and each lookup counts as one probe.

```bash
uv run python bench/levelset_efficiency.py
```

| | |
|---|---|
| Probes | 22 <!--bench:levelset_efficiency.primary.probes_used--> of 289 |
| Misclassified cells | 0 <!--bench:levelset_efficiency.primary.misclassified_cells--> |
| Savings | 13.14× <!--bench:levelset_efficiency.primary.savings--> |

**Condition on this number:** 94% of this map fails, so the passing region is
small and the rest is uniform -- a favourable shape for this method. On
synthetic surfaces with a larger window and a diagonal edge the same code
needed 18 and 13 probes, which suggests the result is not only an artefact of
this map, but the real-data figure is stated for this map. Sensitivity to the
confidence parameter is in `bench/out/levelset_efficiency.json`.

## A second pipeline, same frames

The same 1-D search on NanoDet-Plus-m (a different detector family), road100:

| Axis | YOLOX-S | NanoDet-Plus-m |
|---|---|---|
| Undegraded mAP@50 | 0.6111 | 0.4219 <!--bench:efficiency_road100_nanodet_plus_m.baseline.mAP50--> |
| Motion blur | 12.50–13.75 ms | 12.50 <!--bench:efficiency_road100_nanodet_plus_m.axes[0].levels[3].grid.lower-->–13.75 ms |
| Illuminance | 12.98–25.47 lux | 25.47 <!--bench:efficiency_road100_nanodet_plus_m.axes[1].levels[3].grid.lower-->–37.95 <!--bench:efficiency_road100_nanodet_plus_m.axes[1].levels[3].grid.upper--> lux |
| Fog β | 0.060–0.064 /m | 0.030 <!--bench:efficiency_road100_nanodet_plus_m.axes[2].levels[3].grid.lower-->–0.03375 <!--bench:efficiency_road100_nanodet_plus_m.axes[2].levels[3].grid.upper--> /m |
| JPEG q | 7.97–10.94 | 5.00 <!--bench:efficiency_road100_nanodet_plus_m.axes[3].levels[3].grid.lower-->–7.97 <!--bench:efficiency_road100_nanodet_plus_m.axes[3].levels[3].grid.upper--> |

Same blur edge; NanoDet fails in darker conditions sooner, in fog at half the
extinction coefficient, and tolerates harsher JPEG compression. These are
different operating envelopes on identical frames, which is the case for
measuring a pipeline rather than assuming a model family's robustness.
Verified bisection again matched the grid's interval with zero error on all
four axes.

## Across datasets and pipelines

The same comparison on three validation sets (road100, indoor100, and the
73-frame retail100) and two pipelines (YOLOX-S, NanoDet-Plus-m):

```bash
uv run python bench/boundary_efficiency.py --dataset val/<set> --pipeline <name> --levels 5 9 17 33
uv run python bench/efficiency_matrix.py
```

| | |
|---|---|
| Combinations | 6 <!--bench:efficiency_matrix.summary.combinations--> |
| Axis searches | 24 <!--bench:efficiency_matrix.summary.axis_runs--> |
| Boundaries located | 24 <!--bench:efficiency_matrix.summary.located--> |
| Worst boundary error vs grid | 0 <!--bench:efficiency_matrix.summary.max_boundary_error--> |
| Lowest verified savings at 33 grid points | 4.125× <!--bench:efficiency_matrix.summary.min_savings_verified--> |

As before, the savings ratio is arithmetic (33 grid probes against 8); the
measured result is that verified bisection returned the grid's own interval on
every one of the 24 searches. None of the 24 responses was non-monotone at the
scan spacing, so the verified mode's refusal path was never exercised on real
data -- it is covered by tests only.

The envelopes themselves move with the scene as well as the model: YOLOX-S
fails in fog at 0.0525 <!--bench:efficiency_matrix.rows[10].boundary[0]--> /m
indoors against 0.06 /m on roads, and NanoDet's blur edge drops to
8.75 <!--bench:efficiency_matrix.rows[4].boundary[0]--> ms indoors. Full
per-axis table: `bench/out/efficiency_matrix.json`.

## On AWS

Deployed with `cdk deploy` to a dedicated account profile; every run below went
through Step Functions, AWS Batch on Fargate (Graviton, arm64) and the DynamoDB
ledgers. Costs are **measured** by `tools/cost_report.py` from each ECS task's
billed time, not estimated.

**Full run** -- road100, 100 frames, four axes, $0.40 contract
(`bench/out/cloud_runs/20261005-132343-667165-*.json`):

| | |
|---|---|
| Wall time | 14 min 41 s |
| Probes (Fargate tasks) | 33 <!--bench:cloud_runs/20261005-132343-667165-cost.fargate_tasks--> |
| Measured cost | $0.0749 <!--bench:cloud_runs/20261005-132343-667165-cost.total_usd--> |
| Motion-blur boundary | 12.50 <!--bench:cloud_runs/20261005-132343-667165-envelope.findings[0].lower-->–13.75 <!--bench:cloud_runs/20261005-132343-667165-envelope.findings[0].upper--> ms |
| Illuminance boundary | 12.98 <!--bench:cloud_runs/20261005-132343-667165-envelope.findings[1].lower-->–25.47 <!--bench:cloud_runs/20261005-132343-667165-envelope.findings[1].upper--> lux |

All four boundaries and per-axis probe counts are identical to the local
benchmark above: the cloud planner replays the local search, and the
degradations are deterministic, so Graviton/Linux and macOS agree exactly.

**Budget halt and approval** -- a deliberately starved $0.02 contract
(`bench/out/cloud_runs/20261005-134003-56d2f0-*.json`). The run stopped after
four probes in `AWAITING_APPROVAL`; `blindspot approve`, with a named approver
and a reason, resumed it under contract version 2 and it completed. Both the
halt reasons and the approval are in the decision ledger. The approval in this
test was issued by the project's automation under the operator's name; it shows
the mechanism, not a person's decision. Measured cost across
both halves: $0.0464 <!--bench:cloud_runs/20261005-134003-56d2f0-cost.total_usd-->.

**Reproduced from a fresh clone** -- after a full teardown, the README's
path A was followed literally from a new `git clone`
(`bench/out/cloud_runs/20261005-165654-d03e9b-*.json`): 14 min 50 s,
33 <!--bench:cloud_runs/20261005-165654-d03e9b-cost.fargate_tasks--> tasks,
$0.0752 <!--bench:cloud_runs/20261005-165654-d03e9b-cost.total_usd-->, and the
same four boundaries.

**x86 vs Graviton, same work** -- `bench/fargate_stages.py` runs the fixed
64-probe batch (10 frames each, YOLOX-S) as `blindspot bench-stages` on both
Batch queues, same image, 2 vCPU / 4 GiB tasks, three runs per side
(`bench/out/cool/`). Times are total work per frame (the mean): stage times
are skewed by a few expensive conditions, and the median frame understates
the work -- on one comparison below it even reverses the sign.

| | Graviton (arm64) | x86 |
|---|---|---|
| CPU | Neoverse V2 (Graviton4), all 3 runs | 1 × Cascade Lake 8259CL, 2 × Sapphire Rapids 8488C |
| Degrade | 19.1 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.per_stage_mean_ms.degrade--> | 21.4 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.per_stage_mean_ms.degrade--> |
| Measure | 15.7 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.per_stage_mean_ms.measure--> | 20.9 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.per_stage_mean_ms.measure--> |
| Infer (`cv::dnn`) | 473.0 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.per_stage_mean_ms.infer--> | 401.1 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.per_stage_mean_ms.infer--> |
| Frame (mean) | 507.8 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.mean_ms_per_frame--> | 443.4 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.mean_ms_per_frame--> |
| Cost per 1,000 frames | $0.01114 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.usd_per_1000_frames--> | $0.01216 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.usd_per_1000_frames--> |

x86 is faster per frame and Graviton is cheaper per frame on Fargate. The x86
side is not one CPU: Fargate placed the three tasks on two generations,
584.8 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[0].mean_ms_per_frame--> ms per frame on Cascade Lake against
397.8 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[1].mean_ms_per_frame--> and 443.4 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[2].mean_ms_per_frame--> ms on Sapphire Rapids,
so "x86 on Fargate" is a distribution, not a number.
59 <!--bench:cool/fargate_x86_vs_arm64.map50_identical_probes--> of 64 <!--bench:cool/fargate_x86_vs_arm64.map50_probes_compared--> probes gave
identical mAP across architectures; each side's three runs were bit-identical.

**COOL as a third arm (EC2)** -- `bench/ec2_cool.py` launches the instances
from code, runs the same batch three times per arm, and every instance
terminates itself; none was left running. Arm 3 is the Cloud Optimized
OpenCV build (`5.1.0-dev`) on the same instance as arm 2. Arms 2 and 3 ran on
c8g.large (2 vCPU) and on the vendor-recommended m8g.4xlarge (16 vCPU)
(`bench/out/cool/ec2_three_way.json`, `bench/out/cool/m8g-4xlarge/`). Mean
time per frame:

| | x86 stock (c7i.large) | Graviton4 stock (c8g.large) | Graviton4 COOL (c8g.large) | Graviton4 stock (m8g.4xlarge) | Graviton4 COOL (m8g.4xlarge) |
|---|---|---|---|---|---|
| Measure | 16.6 <!--bench:cool/ec2_three_way.arms.x86_stock.per_stage_mean_ms.measure--> ms | 14.8 <!--bench:cool/ec2_three_way.arms.graviton_stock.per_stage_mean_ms.measure--> ms | 14.7 <!--bench:cool/ec2_three_way.arms.graviton_cool.per_stage_mean_ms.measure--> ms | 12.8 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_stock.per_stage_mean_ms.measure--> ms | 12.8 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_cool.per_stage_mean_ms.measure--> ms |
| Infer (`cv::dnn`) | 332.6 <!--bench:cool/ec2_three_way.arms.x86_stock.per_stage_mean_ms.infer--> ms | 468.6 <!--bench:cool/ec2_three_way.arms.graviton_stock.per_stage_mean_ms.infer--> ms | 513.8 <!--bench:cool/ec2_three_way.arms.graviton_cool.per_stage_mean_ms.infer--> ms | 61.4 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_stock.per_stage_mean_ms.infer--> ms | 67.9 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_cool.per_stage_mean_ms.infer--> ms |
| Frame (mean) | 368.2 <!--bench:cool/ec2_three_way.arms.x86_stock.mean_ms_per_frame--> ms | 502.3 <!--bench:cool/ec2_three_way.arms.graviton_stock.mean_ms_per_frame--> ms | 546.2 <!--bench:cool/ec2_three_way.arms.graviton_cool.mean_ms_per_frame--> ms | 93.1 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_stock.mean_ms_per_frame--> ms | 97.8 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_cool.mean_ms_per_frame--> ms |
| Cost per 1,000 frames | $0.00913 <!--bench:cool/ec2_three_way.arms.x86_stock.usd_per_1000_frames--> | $0.01113 <!--bench:cool/ec2_three_way.arms.graviton_stock.usd_per_1000_frames--> | $0.01362 <!--bench:cool/ec2_three_way.arms.graviton_cool.usd_per_1000_frames--> | $0.01857 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_stock.usd_per_1000_frames--> | $0.02060 <!--bench:cool/m8g-4xlarge/ec2_three_way.arms.graviton_cool.usd_per_1000_frames--> |

The two effects are reported separately. **Chip effect** (x86 vs Graviton4,
both stock, c8g.large): Graviton4 ran at 0.733 <!--bench:cool/ec2_three_way.chip_effect.speedup--> of x86's
speed and, at EC2 prices, cost more per frame -- unlike on Fargate. **COOL
effect** (stock vs COOL on the same machine): 0.920 <!--bench:cool/ec2_three_way.cool_effect.speedup--> on
c8g.large and 0.952 <!--bench:cool/m8g-4xlarge/ec2_three_way.cool_effect.speedup--> on m8g.4xlarge; inference
0.912 <!--bench:cool/ec2_three_way.cool_effect.stage_speedup.infer-->x and 0.905 <!--bench:cool/m8g-4xlarge/ec2_three_way.cool_effect.stage_speedup.infer-->x, image
measurement 1.006 <!--bench:cool/ec2_three_way.cool_effect.stage_speedup.measure-->x and 1.001 <!--bench:cool/m8g-4xlarge/ec2_three_way.cool_effect.stage_speedup.measure-->x.
By the median frame alone COOL would have looked marginally faster on
m8g.4xlarge (1.007 <!--bench:cool/m8g-4xlarge/ec2_three_way.cool_effect.median_frame_ratio-->); total work says it was slower.
COOL arms' cost includes its list software fee (zero during the trial). mAP
agreed between stock and COOL on 63 <!--bench:cool/m8g-4xlarge/ec2_three_way.cool_effect.map50_identical_probes--> of 64 probes.
Only our measurements and COOL's version string are published.

## Sim-to-real (third-party real photos, estimated illuminance)

No first-party capture was made. Real photos: NOD (Night Object Detection), I. Morawski, Y.-A. Chen, Y.-S. Lin, W. H. Hsu, BMVC 2021, <https://github.com/igor-morawski/NOD>; images licensed `CC BY-NC-SA 2.0` (as declared in NOD's annotation files), used as a non-commercial research benchmark. Only aggregate metrics are published; no NOD image or derived image appears in this repository, the viewer, the video or the Devpost gallery.

`bench/sim2real.py` runs YOLOX-S on 286 <!--bench:sim2real.selection.measured--> NOD night photos
(668 <!--bench:sim2real.objects--> labelled cars; test and validation splits; images whose
largest object is a person excluded) and compares each bin of estimated
illuminance with Blindspot's synthetic prediction: the low-light degradation
applied to road100 at the bin's median estimated lux and the median exposure
time the cameras recorded, scored against the same threshold as the boundary
(0.367 <!--bench:sim2real.threshold_map50-->).

**What is measured and what is estimated.** Exposure time, aperture and ISO
are values the cameras recorded (EXIF). Illuminance is **estimated, not
measured**: the incident-light exposure equation, scaled by each photo's mean
linear brightness relative to mid-grey (the photos are deliberately dark, so
the unscaled equation overstates the light).

| Estimated lux (range) | Images | Recorded exposure, median (ms) | Real mAP@50 | Synthetic mAP@50 | Real / synthetic |
|---|---|---|---|---|---|
| 0.0574 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].lux_min-->–1.94 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].lux_max--> | 58 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].images--> | 12.5 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].exposure_ms_median_recorded--> | 0.400 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].real_map50--> | 0.00021 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[0].synthetic_map50--> | pass / fail |
| 1.99 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].lux_min-->–4.91 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].lux_max--> | 57 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].images--> | 12.5 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].exposure_ms_median_recorded--> | 0.576 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].real_map50--> | 0.120 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[1].synthetic_map50--> | pass / fail |
| 5.01 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].lux_min-->–10.2 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].lux_max--> | 57 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].images--> | 12.5 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].exposure_ms_median_recorded--> | 0.639 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].real_map50--> | 0.271 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[2].synthetic_map50--> | pass / fail |
| 10.2 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].lux_min-->–18.2 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].lux_max--> | 57 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].images--> | 10.0 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].exposure_ms_median_recorded--> | 0.591 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].real_map50--> | 0.337 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[3].synthetic_map50--> | pass / fail |
| 18.4 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].lux_min-->–193 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].lux_max--> | 57 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].images--> | 8.0 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].exposure_ms_median_recorded--> | 0.656 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].real_map50--> | 0.426 <!--bench:sim2real.estimators.exif_brightness_corrected.bins[4].synthetic_map50--> | pass / pass |

The synthetic model is **too pessimistic**: it predicts failure in
4 <!--bench:sim2real.gap.bins_synthetic_fail_real_pass--> of 5 bins where the real photos pass, and the real
photos never fail, even in the darkest bin (median 1.08 <!--bench:sim2real.gap.darkest_bin_lux_median--> lux,
mAP 0.400 <!--bench:sim2real.gap.darkest_bin_real_map50-->). The real failure point lies below the data, so the
gap is a bound: the synthetic boundary's lower edge (12.98 <!--bench:sim2real.synthetic_boundary_lux[0]--> lux)
overstates the failure illuminance by at least
12.0 <!--bench:sim2real.gap.synthetic_boundary_overstates_failure_illuminance_by_at_least-->×.

Two other estimators are reported and not used for the conclusion. The
unscaled EXIF equation puts every photo at 3.75 <!--bench:sim2real.estimators.exif.bins[0].lux_min--> lux or more,
where real and synthetic trivially agree. The image-statistics estimator
(SNR inverted through the synthetic sweep) saturates at the axis maximum for
most photos, because in-camera noise reduction makes real night photos look
cleaner than the synthetic sensor -- it is not usable on real camera output,
and Blindspot's coverage report, which uses it, inherits that limitation.

**Reading.** The tool measured its own blind spot, and the error is on the conservative side: it over-warns rather than misses failures. The model is deliberately **not** calibrated against these photos, because calibrating and validating on the same images would be fitting to the answer. Next step: model the camera's image-signal-processor noise reduction, then validate on a separate real set.

Limits: one axis (low light), one detector, different scenes on the real and
synthetic sides, and a threshold taken from the road-scene baseline.

## Known limitations

1. **Synthetic degradation is not real degradation; on low light it errs pessimistic.**
   Measured on third-party night photos (NOD) with estimated
   illuminance, the synthetic low-light boundary overstates the failure
   illuminance by at least 12.0 <!--bench:sim2real.gap.synthetic_boundary_overstates_failure_illuminance_by_at_least-->×
   -- it over-warns rather than misses failures. It is deliberately not
   calibrated on those photos; modelling in-camera noise reduction is next.
   Motion blur, fog and JPEG have no real-world comparison yet.

2. **Bisection assumes monotonicity.** Verified mode scans before refining and
   reports `NOT_MONOTONE` rather than guessing, but a failure band narrower
   than the scan spacing can still be missed. Cheap mode misses interior bands
   entirely — this is pinned by a test, not merely documented.

3. **Two axes at most.** Blur and light are mapped jointly, and the map shows
   one-axis envelopes overstate the safe region. Other pairs, and three or
   more axes at once, are not explored.

4. **Fog depth is approximated.** Depth is a linear ramp increasing towards the
   top of frame, the standard ground-plane camera assumption. Scenes violating
   it get mis-graded fog.

5. **Coverage is inferred from image statistics, not capture metadata.** A
   scene that is intrinsically low-contrast reads as foggier than it was shot,
   and on real night photos the SNR estimate saturates because in-camera noise
   reduction makes them look clean (measured in the sim-to-real section).
   EXIF or a capture log would be better and is not yet used for coverage.

6. **`mtf50_cy_px` is a proxy**, not an ISO 12233 measurement. It is comparable
   between degradations of one scene, not across scenes.

7. **No H.264 CRF axis.** OpenCV 5's `VideoWriter` exposes no rate control in
   this build, so an honest CRF unit is unreachable through OpenCV alone. See
   [`degradation-axes.md`](degradation-axes.md) for the evidence.

8. **Three pipelines, two families.** YOLOX-S, YOLOX-Nano and NanoDet-Plus
   pass the same adapter contract, but two share a family, and all are
   COCO-trained detectors. Segmentation, tracking and non-COCO pipelines are
   untested.
