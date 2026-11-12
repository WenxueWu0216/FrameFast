#!/usr/bin/env bash
# One command to install everything and run the full FrameFast ladder on a CUDA machine
# (Colab, Lambda, any Linux box with an NVIDIA GPU).
#
#   bash scripts/run_on_gpu.sh                 # default ladder (configs/gpu_default.toml)
#   CONFIG=configs/gpu_quick.toml bash scripts/run_on_gpu.sh
#   FRAMEFAST_TRT=1 bash scripts/run_on_gpu.sh # also install torch-tensorrt for rung 5
#
# Outputs: docs/results.md, docs/profile_top_ops.md, runs/<config>/results.json, runs/profile/trace.json
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/gpu_default.toml}"
PY="${PY:-python3}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU visible (nvidia-smi failed). This script is for GPU machines;" >&2
  echo "       on CPU use 'make bench-cpu-smoke'." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

echo "== installing dependencies"
$PY -m pip install -q -U pip
$PY -m pip install -q -e ".[dev,gpu]"
if [[ "${FRAMEFAST_TRT:-0}" == "1" ]]; then
  # torch-tensorrt wheels pin an exact torch version; pip may replace the installed torch.
  $PY -m pip install -q torch-tensorrt || echo "WARNING: torch-tensorrt install failed; TRT rungs will be skipped"
fi
$PY - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name())
PY

echo "== unit tests (including GPU kernel tests)"
$PY -m pytest -q

echo "== profiling the baseline transformer"
$PY -m bench.profile --config "$CONFIG" --out docs/profile_top_ops.md --trace runs/profile/trace.json

echo "== running the optimization ladder ($CONFIG)"
$PY -m bench.run_bench --config "$CONFIG"

echo
echo "Done. Results: docs/results.md   Profile: docs/profile_top_ops.md"
