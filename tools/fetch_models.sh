#!/bin/sh
# Fetch the pipelines under test and the face anonymiser, and verify each by
# SHA-256. Models are not committed; all are Apache-2.0 or MIT (see
# src/blindspot/runner/registry.py and DATASETS.md).
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
fetch() {
  name=$1 url=$2 sum=$3
  if [ ! -f "models/$name" ]; then curl -fsSL -o "models/$name" "$url"; fi
  echo "$sum  models/$name" | shasum -a 256 -c -
}
ZOO=https://github.com/opencv/opencv_zoo/raw/main/models
fetch yolox_s.onnx "$ZOO/object_detection_yolox/object_detection_yolox_2022nov.onnx" \
  c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
fetch nanodet_plus_m.onnx "$ZOO/object_detection_nanodet/object_detection_nanodet_2022nov.onnx" \
  4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186
fetch yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx \
  c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
fetch face_detection_yunet.onnx "$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
