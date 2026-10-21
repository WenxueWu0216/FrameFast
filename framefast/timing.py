"""Low-overhead, per-stage timing of a diffusers video pipeline call.

On CUDA every boundary is a ``torch.cuda.Event(enable_timing=True)`` recorded on the current
stream; nothing synchronizes until the call has finished, so the measurement does not perturb
the asynchronous launch queue (a host-side ``time.perf_counter`` around an async CUDA call would
only measure launch time). On CPU the same interface falls back to ``perf_counter``.

Stage boundaries are obtained without modifying the model:

* **text_encode** — wraps ``pipe.encode_prompt``.
* **denoise** — from the end of text encoding to the ``callback_on_step_end`` of the last step
  (includes latent preparation, the transformer, CFG combination and the scheduler step).
  Per-step times come from consecutive step-end callbacks.
* **vae_decode** — wraps ``pipe.vae.decode``.
* **total** — the whole ``pipe(...)`` call; ``other = total - (text + denoise + vae)`` covers
  post-processing (uint8/float conversion) and Python overhead.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch


class _Mark:
    """A timestamp that is either a CUDA event or a host clock reading."""

    __slots__ = ("event", "t")

    def __init__(self, cuda: bool) -> None:
        self.event: torch.cuda.Event | None = None
        self.t: float = 0.0
        if cuda:
            self.event = torch.cuda.Event(enable_timing=True)
            self.event.record()
        else:
            self.t = time.perf_counter()

    def ms_until(self, other: _Mark) -> float:
        """Milliseconds from ``self`` to ``other`` (requires prior synchronization on CUDA)."""
        if self.event is not None and other.event is not None:
            return float(self.event.elapsed_time(other.event))
        return (other.t - self.t) * 1e3


@dataclass
class StageTimes:
    """Per-stage wall time of a single pipeline call, in milliseconds."""

    total_ms: float
    text_encode_ms: float
    denoise_ms: float
    vae_decode_ms: float
    step_ms: list[float] = field(default_factory=list)

    @property
    def other_ms(self) -> float:
        """Time not attributed to a named stage (post-processing, Python overhead)."""
        return self.total_ms - self.text_encode_ms - self.denoise_ms - self.vae_decode_ms

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "total_ms": self.total_ms,
            "text_encode_ms": self.text_encode_ms,
            "denoise_ms": self.denoise_ms,
            "vae_decode_ms": self.vae_decode_ms,
            "other_ms": self.other_ms,
            "step_ms": list(self.step_ms),
        }


class StageTimer:
    """Instrument a CogVideoX-style pipeline for per-stage timing.

    Usage::

        timer = StageTimer(pipe, cuda=True)
        with timer.attach():
            out = timer.call(prompt=..., num_inference_steps=20)
        times = timer.result()
    """

    def __init__(self, pipe: Any, cuda: bool) -> None:
        self.pipe = pipe
        self.cuda = cuda
        self._marks: dict[str, _Mark] = {}
        self._steps: list[_Mark] = []
        # ``pipe.encode_prompt`` is invoked once per call; ``vae.decode`` may be invoked more than
        # once if a variant decodes in slices, so we keep the first start and the last end.
        self._vae_start: _Mark | None = None
        self._vae_end: _Mark | None = None

    # -- instrumentation -------------------------------------------------------------------
    def _mark(self) -> _Mark:
        return _Mark(self.cuda)

    @contextlib.contextmanager
    def attach(self) -> Iterator[StageTimer]:
        """Temporarily wrap ``encode_prompt`` and ``vae.decode`` with timing marks."""
        pipe = self.pipe
        orig_encode = pipe.encode_prompt
        vae = pipe.vae
        orig_decode = vae.decode
        had_encode_attr = "encode_prompt" in vars(pipe)
        had_decode_attr = "decode" in vars(vae)

        @functools.wraps(orig_encode)
        def encode_prompt(*args: Any, **kwargs: Any) -> Any:
            self._marks["text_start"] = self._mark()
            out = orig_encode(*args, **kwargs)
            self._marks["text_end"] = self._mark()
            return out

        @functools.wraps(orig_decode)
        def decode(*args: Any, **kwargs: Any) -> Any:
            if self._vae_start is None:
                self._vae_start = self._mark()
            out = orig_decode(*args, **kwargs)
            self._vae_end = self._mark()
            return out

        pipe.encode_prompt = encode_prompt
        vae.decode = decode
        try:
            yield self
        finally:
            if had_encode_attr:
                pipe.encode_prompt = orig_encode
            else:
                del pipe.encode_prompt
            if had_decode_attr:
                vae.decode = orig_decode
            else:
                del vae.decode

    def _on_step_end(self, pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict:
        self._steps.append(self._mark())
        return {}

    def call(self, **kwargs: Any) -> Any:
        """Run ``pipe(**kwargs)`` with timing marks (must be inside :meth:`attach`)."""
        self._marks.clear()
        self._steps.clear()
        self._vae_start = self._vae_end = None
        if "callback_on_step_end" in kwargs:
            raise ValueError("StageTimer owns callback_on_step_end")
        self._marks["start"] = self._mark()
        out = self.pipe(callback_on_step_end=self._on_step_end, **kwargs)
        self._marks["end"] = self._mark()
        return out

    def result(self) -> StageTimes:
        """Synchronize and convert the recorded marks into :class:`StageTimes`."""
        if self.cuda:
            torch.cuda.synchronize()
        m = self._marks
        for key in ("start", "end", "text_start", "text_end"):
            if key not in m:
                raise RuntimeError(f"timing mark {key!r} missing; was the call instrumented?")
        if not self._steps or self._vae_start is None or self._vae_end is None:
            raise RuntimeError("no denoising steps / VAE decode were observed")
        steps: list[float] = []
        prev = m["text_end"]
        for mk in self._steps:
            steps.append(prev.ms_until(mk))
            prev = mk
        return StageTimes(
            total_ms=m["start"].ms_until(m["end"]),
            text_encode_ms=m["text_start"].ms_until(m["text_end"]),
            denoise_ms=m["text_end"].ms_until(self._steps[-1]),
            vae_decode_ms=self._vae_start.ms_until(self._vae_end),
            step_ms=steps,
        )


def wall_clock(fn: Callable[[], Any], cuda: bool) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, seconds)``, synchronizing CUDA on both sides.

    Used for cold-start (first call) measurement, where compilation dominates and host time
    is the meaningful quantity.
    """
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if cuda:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0
