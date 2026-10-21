"""FrameFast: an inference-optimization ladder for open-source video diffusion models.

The package is organised around three ideas:

* :mod:`framefast.pipeline` builds the pipeline under test (the real CogVideoX-2B on a GPU, or a
  tiny random-weight CogVideoX for CPU tests and harness smoke runs).
* :mod:`framefast.variants` is a registry of optimizations ("rungs" of the ladder). Every variant
  is applied to a *fresh* pipeline so each one is measured in isolation against the fp16 baseline.
* :mod:`framefast.timing` / :mod:`framefast.quality` implement the measurements (CUDA events,
  per-stage breakdown, PSNR/SSIM/LPIPS).
"""

__version__ = "0.1.0"
