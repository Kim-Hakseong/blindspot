# Architecture

The diagram source is committed here as text, not only as an image, so it can
be diffed and corrected. The rendered version is generated from this source.

## System

```mermaid
flowchart TB
    dev["Developer<br/>blindspot run --dataset ./val --budget 0.40"]

    subgraph control["AWS App Runner — FastAPI control plane"]
        api["POST /runs → run_id, budget contract fixed<br/>GET /runs/{id}/envelope → boundary JSON"]
    end

    subgraph orch["AWS Step Functions — probe loop"]
        plan["PlanProbes<br/>deterministic scheduler (bisection)"]
        agent{{"MCP tool call → Bedrock<br/>proposes axis priority only"}}
        map["Map state — parallel fan-out"]
        evald["EvaluateBoundary<br/>converged? budget spent?"]
    end

    subgraph worker["AWS Batch on Fargate — probe worker (arm64 + x86-64)"]
        direction TB
        w1["1. load source frame from S3"]
        w2["2. OpenCV 5 degradation<br/>(image, physical params, seed)"]
        w3["3. OpenCV 5 objective measurement<br/>MTF50 · Laplacian var · SNR · blockiness"]
        w4["4. run pipeline under test — cv::dnn"]
        w5["5. metrics — mAP@50, IoU, counts"]
        w6["6. write probe record + overlay"]
        w1 --> w2 --> w3 --> w4 --> w5 --> w6
    end

    subgraph store["State and artefacts"]
        ddb[("DynamoDB<br/>bs-runs · bs-probes · bs-decisions")]
        s3[("S3<br/>frames · overlays · report JSON")]
        cw["CloudWatch<br/>metrics · logs"]
    end

    viewer["CloudFront + S3<br/>report viewer — public, no credentials"]
    ci["GitHub Actions<br/>blindspot check → PR fails on boundary regression"]

    dev --> api --> plan
    plan -.proposal only.-> agent
    agent -.-> plan
    plan --> map --> worker
    worker --> ddb
    worker --> s3
    worker --> cw
    worker --> evald
    evald -->|not converged| plan
    evald -->|budget exceeded| hold["AWAITING_APPROVAL<br/>partial results kept"]
    s3 --> viewer
    api --> ci

    classDef judgment fill:#1f6feb,stroke:#0d419d,color:#fff
    classDef llm fill:#8957e5,stroke:#6639ba,color:#fff
    class w2,w3,w5,plan,evald judgment
    class agent llm
```

Blue nodes are the judgment path: degradation, measurement, metrics, boundary
decisions and budget accounting. They are deterministic and cannot import an
LLM client. The single purple node is the only place a model is consulted, and
it may only propose which axis to spend remaining budget on — the scheduler
decides whether to accept, and records the rejection when it does not.

## Probe lifecycle

```mermaid
sequenceDiagram
    participant S as Scheduler (deterministic)
    participant B as Budget contract
    participant W as Probe worker (OpenCV 5)
    participant L as Ledger (DynamoDB)

    S->>B: request(1 probe)
    alt within contract
        B-->>S: RUNNING, approved
        S->>W: (frames, degradation, physical params, seed)
        W->>W: degrade → measure → cv::dnn → mAP@50
        W->>L: probe record + reproduction command
        W-->>S: mAP@50
        S->>S: failed? = mAP@50 < 0.60 × baseline
    else exceeds contract
        B-->>S: AWAITING_APPROVAL, refused
        S->>L: record refusal with reason
        S->>S: halt, keep partial results
    end
```

## Current implementation status

| Component | Status |
|---|---|
| OpenCV 5 degradation kernels (7 axes) and composite conditions | Implemented, tested |
| Objective degradation measurement | Implemented, tested |
| `cv::dnn` pipelines: YOLOX-S, YOLOX-Nano, NanoDet-Plus | Implemented, tested |
| mAP@50 / IoU metrics, one shared matching rule | Implemented, tested |
| 1-D boundary search (bisection, verified mode) | Implemented, measured |
| 2-D boundary search (level-set estimation) | Implemented, tested on synthetic surfaces |
| Coverage / uncovered regions (population response) | Implemented, tested |
| Budget contract, round planner, planner Lambda adapter | Implemented, tested |
| Probe worker, local JSONL and DynamoDB ledgers | Implemented, tested |
| Worker image, arm64 and x86-64 | Built locally; arm64 reproduces native results exactly |
| CDK stack (VPC, S3, DynamoDB, Batch x2, Step Functions, Lambda, Budgets) | Synthesised and tested; **not deployed** |
| Report schema and static viewer | Implemented, rendered locally; **not hosted** |
| COOL on Graviton comparison | Not built |
| MCP agent + decision ledger (agent side) | Not built |
| sim-to-real gap | **Not measured** |

Rows marked not deployed, not hosted, not built or not measured are not
described anywhere in this repository as though they exist.

## Design decisions worth stating

**The boundary is an interval, not a point.** Its width is the residual
uncertainty the probe budget bought. A point estimate would claim precision
nobody paid for.

**Bisection is checked, not trusted.** It assumes monotonicity, which detectors
do not guarantee. The verified mode scans before it refines, because a failure
band strictly inside the swept range leaves both endpoints passing and pure
bisection would report "no boundary" after two probes.

**Degradation is a pure function of `(image, params, seed)`.** This is what
lets a report cite a failing condition by recipe instead of shipping pixels,
and what lets a reader regenerate it byte-for-byte.

**The ledger is the only state.** The planner is a pure function of the run
definition and the probe ledger, and it replays the local search rather than
reimplementing it, so a cloud run and a local run visit the same probes.

**One shutter, two effects.** In the 2-D map a single exposure time sets both
the motion-blur length and the light the sensor collects, so the map never
probes a camera that cannot exist.

**The budget cannot be raised mid-run.** A limit the running code can lift is
not a limit, and an agent that can widen its own budget has none.
