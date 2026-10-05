## Inspiration

My day job is hardware-in-the-loop testing: we never trust a flight controller until we have pushed simulated physical conditions past the point where it breaks. Vision pipelines rarely get that treatment. A detector is usually validated on a fixed test set, and nobody can say *at what exposure time* or *at how many lux* it stops working. I wanted the same discipline for cameras: physical inputs, a measured failure boundary, and an honest list of the conditions that were never tested.

## What it does

Blindspot takes a detection pipeline and a validation set, degrades the images in **physical units** (exposure in ms, illuminance in lux, fog extinction in 1/m, JPEG quality), and searches for the exact condition where detection fails. It reports:

- the **failure boundary** on each axis and across pairs of axes,
- the **uncovered regions**: the share of the condition space your validation set never exercised (this is the first field in every report),
- **evidence frames**: the wrong detections that the degradation caused, each with a command that reproduces the same number.

At 15.6 <!--bench:hook_grid.cells[2][6].exposure_ms--> ms exposure and 173 <!--bench:hook_grid.cells[2][6].illuminance_lux--> lux, YOLOX-S reports a blurred bus as "car 0.60 <!--bench:hook_cells/x06_y02.predictions[0].score-->". The command printed under that frame reproduces the same mAP exactly.

## How I built it

- **OpenCV 5** for every degradation kernel: motion blur from exposure and angular velocity, defocus from aperture and subject distance, lens distortion via `remap`, rolling shutter, fog from an extinction coefficient, low light with Poisson–Gaussian sensor noise, and JPEG recompression. Detectors run through `cv::dnn` (YOLOX-S and NanoDet, ONNX).
- **Determinism first.** Every kernel is a pure function of `(image, params, seed)`. Same seed, byte-identical output, verified across separate processes.
- **Boundary search instead of a grid.** A verified bisection scans first and refines second, so it cannot miss a failure band in the middle of the range. A 2-D sampler maps pairs of axes.
- **No LLM in the judgment path.** Metrics, boundary decisions and cost limits never import an LLM client; a test checks the import graph.
- **AWS, deployed with one `cdk deploy`.** Step Functions runs the search in rounds; each round fans out to AWS Batch on Fargate, where the same worker image runs on Graviton (arm64) or x86. Every probe writes a record to DynamoDB, and the planner is a pure function of that ledger that replays the local search, so a cloud run visits exactly the same probes. A full four-axis run on 100 frames took 14 min 41 s on Graviton, used 33 <!--bench:cloud_runs/20261005-132343-667165-cost.fargate_tasks--> Fargate tasks and cost $0.0749 <!--bench:cloud_runs/20261005-132343-667165-cost.total_usd--> measured from billed task time, against a $0.40 <!--bench:cloud_runs/20261005-132343-667165-definition.definition.budget_usd--> per-run contract; all four boundaries were identical to the local run. Torn down and redeployed from a fresh clone by following only the README, it reproduced the same four boundaries for $0.0752 <!--bench:cloud_runs/20261005-165654-d03e9b-cost.total_usd-->. A deliberately starved contract stopped the run after four probes; it continued only after `blindspot approve` recorded an approver, an amount and a reason as a new contract version. In that test the approval command was issued by my automation, not typed by a person.
- **Report viewer: https://d18du1w0ii5yhw.cloudfront.net/** — a static Next.js export in a private S3 bucket, served by CloudFront through origin access control. No login, no API: the page and its report JSON are plain files, so the report stays readable even when the control plane is torn down.
- [TBD — COOL on Graviton4 comparison after W5]
- [TBD — MCP agent after W6: proposes probes, never decides pass/fail]

**Built with** (read from the deployed stacks by `tools/aws_services.py`): OpenCV 5, Python, ONNX, Next.js, AWS CDK, AWS Batch, Amazon ECS on AWS Fargate (Graviton and x86), AWS Step Functions, AWS Lambda, Amazon DynamoDB, Amazon S3, Amazon CloudFront, Amazon ECR, Amazon VPC, Amazon CloudWatch, AWS X-Ray, AWS IAM.

## Results (measured, `bench/out/*.json`)

- Baseline mAP@50 on the clean set: **0.611 <!--bench:efficiency.baseline.mAP50-->** (YOLOX-S, 984 <!--bench:efficiency.dataset.objects--> objects).
- Every axis has a sharp boundary: motion blur 12.50 <!--bench:efficiency.axes[0].levels[3].grid.lower-->–13.75 <!--bench:efficiency.axes[0].levels[3].grid.upper--> ms, low light 12.98 <!--bench:efficiency.axes[1].levels[3].grid.lower-->–25.47 <!--bench:efficiency.axes[1].levels[3].grid.upper--> lux, fog 0.0600 <!--bench:efficiency.axes[2].levels[3].grid.lower-->–0.0638 <!--bench:efficiency.axes[2].levels[3].grid.upper--> 1/m (about 62 m visibility), JPEG quality 7.97 <!--bench:efficiency.axes[3].levels[3].grid.lower-->–10.94 <!--bench:efficiency.axes[3].levels[3].grid.upper-->.
- **One-axis limits overstate the safe region.** At 10.75 <!--bench:hook_grid.cells[5][4].exposure_ms--> ms and 49.5 <!--bench:hook_grid.cells[5][4].illuminance_lux--> lux each axis is inside its own 1-D limit, but together mAP falls to 0.295 <!--bench:hook_grid.cells[5][4].map50-->.
- **Limits depend on the model.** On the same frames, NanoDet fails in fog at about half the density YOLOX-S tolerates.
- **Search efficiency depends on precision.** At 1.25 <!--bench:efficiency.axes[0].levels[3].precision--> ms precision the bisection needs 8 <!--bench:efficiency.axes[0].levels[3].bisection_verified.probes_used--> probes instead of 33 <!--bench:efficiency.axes[0].levels[3].grid.probes_used--> (4.12× <!--bench:efficiency.summary.by_grid_level.33.mean_savings_verified-->); at coarse precision it saves nothing (1.00× <!--bench:efficiency.summary.by_grid_level.5.mean_savings_verified-->). Across 24 <!--bench:efficiency_matrix.summary.axis_runs--> searches (3 datasets × 2 models × 4 axes) every boundary matched the full grid exactly. In 2-D, 22 <!--bench:levelset_efficiency.primary.probes_used--> probes classified a 289 <!--bench:hook_grid.n_cells-->-cell map with no errors; that map is 94% failures (273 <!--bench:hook_grid.n_failed--> of 289), which favors the method.
- **x86 vs Graviton, same image and same 64-probe batch** (2 vCPU Fargate tasks, three runs per side). x86 was faster: 432.1 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.per_frame_total_median_ms--> ms per frame against 497.4 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.per_frame_total_median_ms--> ms on Graviton4. Priced per frame, Graviton was cheaper: $0.01091 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.usd_per_1000_frames--> against $0.01185 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.usd_per_1000_frames--> per 1,000 frames. The OpenCV stages themselves ran faster on Graviton (image measurement 14.8 <!--bench:cool/fargate_x86_vs_arm64.arms.arm64.per_stage_median_ms.measure--> vs 19.6 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.per_stage_median_ms.measure--> ms); the gap is DNN inference, which dominates frame time. Fargate x86 is not one CPU: the three x86 tasks landed on two generations, 571.4 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[0].per_frame_total_median_ms--> ms on Cascade Lake against 387.2 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[1].per_frame_total_median_ms--> and 432.1 <!--bench:cool/fargate_x86_vs_arm64.arms.x86.runs[2].per_frame_total_median_ms--> ms on Sapphire Rapids. 59 <!--bench:cool/fargate_x86_vs_arm64.map50_identical_probes--> of 64 <!--bench:cool/fargate_x86_vs_arm64.map50_probes_compared--> probes gave identical mAP; the rest differed by at most 0.000246 <!--bench:cool/fargate_x86_vs_arm64.map50_max_abs_difference-->.
- [TBD — sim-to-real gap: currently **not measured**]

## Challenges

- **Severity direction.** Low light and low JPEG quality get *worse* as the number goes *down*. My first sweep reported "no boundary" on those axes. It was my bug, not a robust detector. Search now runs in a normalized severity coordinate so the direction cannot be wrong.
- **Pure bisection misses interior failure bands**, because both endpoints pass. The verified mode scans before it refines; the cheap mode's blind spot is pinned by a test.
- **I refused to fake an axis.** The OpenCV 5 wheel exposes no rate control for H.264, so CRF 1, 23 and 45 produced identical files. Labeling another knob "CRF" would be a fake unit, so H.264 is not implemented and the evidence is documented.
- **Evidence-frame honesty.** My first picker showed a wrong box that existed even without degradation. It now counts only errors the degradation caused.

## What I learned

A speedup without a stated precision is a number without a meaning. Writing the unfavorable results next to the favorable ones (1.00× <!--bench:efficiency.summary.by_grid_level.5.mean_savings_verified--> next to 4.12× <!--bench:efficiency.summary.by_grid_level.33.mean_savings_verified-->) made the project more convincing, not less.

## Known limitations

- Sim-to-real gap not yet measured [update after W6].
- H.264 recompression not implemented (no rate control in the OpenCV 5 wheel).
- Results are not bit-identical across CPUs. On real x86 and Graviton Fargate tasks, 18 <!--bench:cloud_runs/cross_arch.identical_map50_full_precision--> of 33 <!--bench:cloud_runs/cross_arch.probes_in_common--> probes matched to full precision and the rest differed by at most 0.000142 <!--bench:cloud_runs/cross_arch.max_abs_map50_difference--> mAP. Every pass/fail decision, and so every boundary, agreed; the closest probe was 0.0058 <!--bench:cloud_runs/cross_arch.smallest_margin_to_threshold--> from the threshold, but a probe nearer than the gap could flip.
- Two detectors tested; independence of the method from the model is shown on two, not proven in general.