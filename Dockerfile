# Blindspot probe worker and local mode. One image, built for linux/arm64
# (Graviton) and linux/amd64 (x86 control). Base images are pinned by digest.
FROM ghcr.io/astral-sh/uv:0.11.12@sha256:3a59a3cdd5f7c217faa36e32dbc7fddbb0412889c2a0a5229f6d790e5a019dd7 AS uv

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, from the lockfile only, for layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group cloud --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --group cloud

# Pipelines under test and the face anonymiser. All Apache-2.0 / MIT; see
# DATASETS.md and runner/registry.py for sources.
COPY models/yolox_s.onnx models/yolox_nano.onnx models/nanodet_plus_m.onnx \
     models/face_detection_yunet.onnx ./models/

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "--group", "cloud", "blindspot"]
CMD ["--help"]
