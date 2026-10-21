# FrameFast — see README.md for details.
PY      ?= python
CONFIG  ?= configs/gpu_default.toml
SMOKE   := configs/cpu_smoke.toml

.PHONY: help install install-gpu install-trt lint format test test-gpu bench bench-quick bench-full \
        bench-cpu-smoke profile results-placeholder clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install:  ## dev install (CPU is fine)
	$(PY) -m pip install -e ".[dev]"

install-gpu:  ## install with GPU extras (triton, torchao, lpips, video export)
	$(PY) -m pip install -e ".[dev,gpu]"

install-trt:  ## add torch-tensorrt (must match your torch version; may reinstall torch)
	$(PY) -m pip install torch-tensorrt

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## auto-format
	ruff check --fix .
	ruff format .

test:  ## unit tests (GPU tests auto-skip without CUDA; Triton runs in the interpreter on CPU)
	$(PY) -m pytest -q

test-gpu:  ## only the GPU-marked tests
	$(PY) -m pytest -q -m gpu

bench:  ## full ladder on GPU -> docs/results.md (CONFIG=... to override)
	$(PY) -m bench.run_bench --config $(CONFIG)

bench-quick:  ## short GPU sanity ladder -> runs/gpu_quick/results.md
	$(PY) -m bench.run_bench --config configs/gpu_quick.toml

bench-full:  ## 50-step, 10-run, 3x2 quality-set ladder -> docs/results.md
	$(PY) -m bench.run_bench --config configs/gpu_full.toml

bench-cpu-smoke:  ## tiny CPU config: verifies the harness + profiler end to end (CI)
	TRITON_INTERPRET=1 $(PY) -m bench.run_bench --config $(SMOKE)
	$(PY) -m bench.profile --config $(SMOKE) --out build/smoke/profile.md --trace build/smoke/trace.json
	@grep -q "NOT YET MEASURED ON GPU" build/smoke/results.md
	@grep -q "| ok |" build/smoke/results.md
	@echo "smoke OK: build/smoke/results.md, build/smoke/profile.md"

profile:  ## torch.profiler top-ops table -> docs/profile_top_ops.md
	$(PY) -m bench.profile --config $(CONFIG) --out docs/profile_top_ops.md --trace runs/profile/trace.json

results-placeholder:  ## regenerate the 'NOT YET MEASURED ON GPU' docs/results.md
	$(PY) -m bench.run_bench --config $(CONFIG) --placeholder

clean:
	rm -rf build runs .pytest_cache .ruff_cache
