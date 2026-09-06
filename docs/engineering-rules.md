# Engineering rules

The constraints this codebase is held to, and the tests that enforce them. They
exist because the product is a measurement tool: if the measurements are not
trustworthy, there is no product.

## Determinism

A degradation is a pure function of `(image, params, seed)`.

- No global RNG. `np.random.default_rng(seed)` and nothing else.
- The same `(image, params, seed)` produces byte-identical output, across
  processes and independent of `PYTHONHASHSEED`.
- Kernels never mutate their input array.
- A kernel declares whether it is stochastic, and the suite verifies the claim
  in both directions: a stochastic kernel that ignores its seed fails, and a
  deterministic kernel whose output varies with the seed fails.

Enforced by `tests/test_determinism.py`, driven off the kernel registry so a
newly registered kernel is covered on registration. **If this file fails, no
other work proceeds** — every downstream number would be unfalsifiable.

## Physical units

Degradation strength is exposed only in units a capture engineer can look up.

Permitted: exposure time (ms), angular velocity (deg/s), PSF length (px),
circle of confusion (px), illuminance (lux), MTF50 (cy/px), extinction
coefficient (1/m), aperture (f-number), JPEG quality, H.264 CRF.

Forbidden: `intensity`, `strength`, `severity`, `level`, `amount`, `scale`, and
any 0–1 or low/medium/high knob. `tests/test_units_roundtrip.py` rejects these
by name and by unit.

Every axis must also *do* what its unit claims: sweeping it has to move a
declared objective measurement in a declared direction, checked on the
endpoints and by rank correlation across the sweep. An axis that does not
measurably bite cannot carry a boundary.

## No model on the judgment path

`metrics/`, `boundary/`, `cost/`, `degrade/` and `measure/` decide what the
report claims. They may not reach an LLM client, directly or transitively.

`tests/test_no_llm_in_judgment.py` walks the transitive import graph of those
packages and fails on any forbidden root. The test also contains a
self-check that builds a deliberately offending module and asserts the
detector catches it — a guard that cannot fail is not a guarantee.

An agent may propose which axis to spend remaining budget on. It does not
decide pass/fail, where a boundary is, or what anything cost.

## Numbers

- Every figure in the documentation carries a citation to the benchmark output
  that produced it, verified by `tools/check_doc_numbers.py`. A number with no
  generating command does not go in a document, a report, or a video.
- Benchmarks are the only source of figures. A constant hardcoded in prose is a
  defect.
- When a measurement contradicts an earlier estimate, the measurement wins and
  every document is updated in the same change.
- An unfavourable measurement is published as measured, with its conditions
  stated next to it. It is not re-run under friendlier conditions and it is not
  quietly dropped.
- A quantity that was not measured is labelled "not measured". It is never
  estimated into existence.

## Reproducibility

- Dependencies are pinned exactly — `uv.lock`, `package-lock.json`, and image
  digests. No range specifiers.
- Infrastructure is CDK code. A resource created by hand in a console is a
  defect: it is moved into CDK and the hand-made one deleted.
- Every run carries a `run_id` and every probe writes one ledger record. Logs
  alone are not a ledger.
- Two reproduction paths are maintained: full cloud deployment, and a local
  single-command mode that needs no cloud credentials.

## Cost

- The budget contract is fixed when a run starts and cannot change during it.
- Reaching the cap **stops** the run and persists partial results. It does not
  overspend and it does not silently truncate.
- A request beyond the contract halts in `AWAITING_APPROVAL` rather than
  proceeding.

## Testing

- A failing test is written before the behaviour it describes.
- New degradation kernels, metrics, and boundary criteria each arrive with
  determinism and unit coverage.
- One commit is one reversible decision. Work that may be cut later is
  committed separately so it can be reverted whole.

## Responsible use

- Dataset provenance and licences are recorded in `DATASETS.md`.
- Frames containing identifiable faces are blurred or excluded from demos and
  published material.
- The tool's central limitation — synthetic degradation is not real degradation
  — is stated in the report and measured rather than argued about.
