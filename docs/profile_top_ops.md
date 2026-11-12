# FrameFast profile: top ops

> **NOT YET MEASURED ON GPU.**
>
> This file is produced by `make profile` (`python -m bench.profile`) on a CUDA machine. It will
> contain the module-level breakdown of transformer GPU time, the op-category breakdown and the
> top aten ops by self device time for the transformer and the VAE decode. The CPU smoke
> version (`make bench-cpu-smoke` -> `build/smoke/profile.md`) only shows that the profiler runs.
