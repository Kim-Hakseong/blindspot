# Video script (draft, under 5 minutes)

Every figure on screen comes from `bench/out/*.json` and is cited here the same
way as in the writeup. Everything shown runs live or is a recording of a real
run named by its run id; nothing is simulated, sped up without saying so, or
precomputed and presented as live.

| Time | Picture | Narration (key points) |
|---|---|---|
| 0:00–0:30 | Report viewer, Blindspot Map; drag the exposure slider until a wrong label appears | The hook: at 15.6 <!--bench:hook_grid.cells[2][6].exposure_ms--> ms and 173 <!--bench:hook_grid.cells[2][6].illuminance_lux--> lux the detector calls a bus a "car". The command under the frame reproduces it. |
| 0:30–0:50 | Face to camera | Who I am; why I test vision pipelines the way hardware is tested. |
| 0:50–1:40 | `docs/architecture-1.png`, then the Step Functions console for a real run | OpenCV 5 degradations in physical units, boundary search, AWS Batch on Graviton and x86, the DynamoDB ledger, one `cdk deploy`. |
| 1:40–2:30 | Results: boundaries, one-axis vs two-axis map, probe savings next to the no-savings case | The unfavourable numbers are shown next to the favourable ones. |
| 2:30–3:20 | `bs-decisions` for run `20261005-205745-0109b4` | **The agent case.** The agent proposed a probe split and justified it with "This allocation reduces the largest gaps in the validation set." That is wrong: probes cannot change a validation set. It cost nothing, because the agent only proposes; the gate checked that the contract could pay, the agent's own model cost was charged first, and the run stopped for a person. Show the human approval record (contract v2) and the run finishing with the same boundaries. |
| 3:20–4:10 | x86 vs Graviton table; COOL arm when measured | Same work on each chip; report what was faster and what was cheaper, including where Graviton lost. |
| 4:10–4:45 | Known limitations | Sim-to-real not yet measured (or the measured gap); H.264 not implemented; results not bit-identical across CPUs. |
| 4:45–5:00 | README top block | One command to deploy, one to run. |

Rules for recording: no faces of bystanders (blur or omit), no AWS account
identifiers on screen, COOL named only as what was measured, no logos.
