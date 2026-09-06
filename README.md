# Blindspot

**Find where your vision pipeline starts lying.**

---

## Reproduce path A — full AWS deployment

```bash
git clone https://github.com/Kim-Hakseong/blindspot.git
cd blindspot
uv sync --frozen                          # pinned via uv.lock
uv run cdk deploy --all --require-approval never
uv run blindspot run --repo . --dataset ./val/road100 --budget 0.40
```

## Reproduce path B — local, no AWS credentials

```bash
docker run --rm -v "$PWD/val:/val" \
  ghcr.io/kim-hakseong/blindspot:local \
  run --dataset /val --probes 8
```

Both paths print a report URL and write `report.json`. Path B runs a reduced
probe count on one machine and needs no cloud account of any kind.

### Run the test suite

```bash
uv sync --frozen
uv run pytest -q                                    # everything
uv run pytest tests/test_determinism.py -v          # byte-identical replay
uv run pytest tests/test_no_llm_in_judgment.py -v   # no LLM on the judgment path
```

---

## What this is

A vision pipeline reports `mAP@50 = 0.87`. Under what exposure time? At what
illuminance? At what JPEG quality? The number is almost never published with
the conditions under which it holds, so the first time a team learns the
operating limit of their detector is weeks after deployment, from a customer.

Blindspot takes a pipeline and a small labelled validation set — 100 images is
enough — and searches for the exact capture conditions where that pipeline
starts being wrong. It reports the boundary in physical units, with the seed
and command that regenerate every failing frame.

It is not a detector. It is a tool that interrogates detectors.

### The output

- **Uncovered regions first.** The report opens with the parts of condition
  space your validation set never exercised — not with the conditions that
  passed.
- **Boundaries in physical units.** PSF length in px derived from exposure time
  and angular velocity; circle of confusion in px; illuminance in lux; fog
  extinction coefficient β in 1/m; JPEG quality; H.264 CRF. No "intensity 0–1"
  sliders and no "low/medium/high".
- **Every failure is reproducible.** Each boundary carries
  `{axis, value, unit, seed, source_frame, command}`. A degradation is a pure
  function of `(image, params, seed)`, so a claimed failure regenerates
  byte-for-byte or it does not go in the report.

### How OpenCV 5 is used

Substantively, on the critical path — not as an image loader:

| Stage | OpenCV 5 usage |
|---|---|
| Degradation synthesis | PSF convolution (`filter2D`), lens distortion and rolling-shutter warp (`remap`), atmospheric scattering, Poisson-Gaussian sensor noise, codec re-encoding (`imencode`) |
| Degradation measurement | MTF50 estimation, Laplacian variance, SNR, contrast — how degraded each frame *objectively* is, measured rather than assumed |
| Pipeline execution | The pipeline under test runs through `cv::dnn` |
| Evidence | Prediction/ground-truth overlays on failing frames |

### What decides pass or fail

Deterministic code. Metrics, boundary location, and cost accounting cannot
import an LLM client, and `tests/test_no_llm_in_judgment.py` enforces that by
inspecting the import graph. An agent proposes which axis to spend remaining
budget on; it does not adjudicate anything.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — architecture diagram (text source + render)
- [`docs/technical-report.md`](docs/technical-report.md) — problem, users, evaluation, limitations
- [`docs/engineering-rules.md`](docs/engineering-rules.md) — the rules this codebase is held to
- [`DATASETS.md`](DATASETS.md) — dataset provenance and licences

## Licence

MIT — see [`LICENSE`](LICENSE).
