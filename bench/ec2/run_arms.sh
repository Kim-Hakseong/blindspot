#!/bin/bash
# Runs ON a benchmark EC2 instance, from a clone of this repository at a pinned
# commit (infra/cool_bench_stack.py writes the user-data that calls it).
#
#   run_arms.sh <role> <bucket> <prefix> <manifest_digest>
#
# role x86      -> arm 1: stock OpenCV 5 wheel on c7i (x86)
# role graviton -> arm 2: stock OpenCV 5 wheel on c8g, and
#                  arm 3: the COOL build on the same instance,
#                  interleaved stock/COOL so drift over time hits both equally.
# Each report goes to s3://<bucket>/<prefix>/ as soon as it exists, so a run cut
# short by the 60-minute limit still leaves its finished arms behind.
set -euxo pipefail
ROLE=$1 BUCKET=$2 PREFIX=$3 DIGEST=$4
REPEATS=${REPEATS:-3}
export AWS_DEFAULT_REGION=us-east-1
OUT="s3://$BUCKET/$PREFIX"

uv sync --frozen --no-dev --group cloud
sh tools/fetch_models.sh
uv run --no-sync python tools/fetch_dataset.py --name road100
got=$(sha256sum val/road100/manifest.json | cut -c1-12)
[ "$got" = "$DIGEST" ] || { echo "dataset manifest $got != expected $DIGEST"; exit 1; }

stock() {  # arm label
  .venv/bin/blindspot bench-stages --label "$1" --dataset val/road100 --frames 10 --out "$OUT"
}

if [ "$ROLE" = x86 ]; then
  for r in $(seq 1 "$REPEATS"); do stock "ec2-x86-stock-r$r"; done
  exit 0
fi

# --- COOL: our code and pinned dependencies, with only cv2 taken from COOL ---
# The vendor's documented "system Python" setup: COOL's libraries on
# LD_LIBRARY_PATH and its Python package directory on the import path. (Its
# venvs set these only when activated; calling their python directly does not.)
COOL_LIB=/opt/cool/cpp_sdk/lib
COOL_SITE=/opt/cool/python_3.12/site-packages
uv export --frozen --no-dev --group cloud --no-hashes --no-emit-project \
  | grep -v '^opencv-python-headless' > /tmp/req-no-opencv.txt
uv venv /opt/bs-cool --python 3.12
uv pip install --python /opt/bs-cool/bin/python -r /tmp/req-no-opencv.txt
uv pip install --python /opt/bs-cool/bin/python --no-deps .
# Appended after our own site-packages: our pinned numpy wins, cv2 exists only in COOL's.
echo "$COOL_SITE" > /opt/bs-cool/lib/python3.12/site-packages/zz_cool_cv2.pth
METHOD=pth
if ! LD_LIBRARY_PATH="$COOL_LIB" /opt/bs-cool/bin/python - "$COOL_SITE" <<'EOF'
import os, sys, cv2
print("cool cv2", cv2.__version__, "from", os.path.realpath(cv2.__file__))
assert os.path.realpath(cv2.__file__).startswith(os.path.realpath(sys.argv[1])), cv2.__file__
EOF
then
  # Fallback: COOL's own activated venv with our package added; COOL's numpy
  # is then used too, and the report's fingerprint records which numpy ran.
  if ( source /opt/cool/venvs/python_3.12/bin/activate && python -c "import cv2" ); then
    ( source /opt/cool/venvs/python_3.12/bin/activate
      grep -v '^numpy' /tmp/req-no-opencv.txt > /tmp/req-no-opencv-numpy.txt
      uv pip install --python "$(command -v python)" -r /tmp/req-no-opencv-numpy.txt
      uv pip install --python "$(command -v python)" --no-deps . )
    METHOD=cool-venv
  else
    # Without COOL, still measure arm 2 so the chip comparison is not lost.
    echo "cool_cv2_method=none (COOL cv2 did not import; errors above)"
    for r in $(seq 1 "$REPEATS"); do stock "ec2-graviton-stock-r$r"; done
    exit 3
  fi
fi
echo "cool_cv2_method=$METHOD"

cool() {
  if [ "$METHOD" = pth ]; then
    LD_LIBRARY_PATH="$COOL_LIB" /opt/bs-cool/bin/blindspot bench-stages --label "$1" \
      --dataset val/road100 --frames 10 --out "$OUT"
  else
    ( source /opt/cool/venvs/python_3.12/bin/activate
      blindspot bench-stages --label "$1" --dataset val/road100 --frames 10 --out "$OUT" )
  fi
}
for r in $(seq 1 "$REPEATS"); do
  stock "ec2-graviton-stock-r$r"
  cool "ec2-graviton-cool-r$r"
done
