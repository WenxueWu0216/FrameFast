"""Shared fixtures. On machines without CUDA, Triton kernels run in interpreter mode."""

from __future__ import annotations

import os

import torch

if not torch.cuda.is_available():
    # Must be set before any @triton.jit function is defined so the kernel is created as an
    # interpreted function that executes on CPU tensors.
    os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest

from framefast.pipeline import build_tiny_pipeline

TINY_CALL = dict(
    prompt="a golden retriever running through a sunlit meadow",
    num_frames=9,
    height=32,
    width=32,
    num_inference_steps=2,
    guidance_scale=6.0,
    max_sequence_length=16,
    output_type="pt",
)


@pytest.fixture
def tiny_pipe():
    """Fresh tiny random-weight CogVideoX pipeline (fp32, CPU)."""
    return build_tiny_pipeline(max_sequence_length=16)


@pytest.fixture
def tiny_call():
    """Keyword arguments for a fast tiny-pipeline call (returns a new dict each time)."""
    return dict(TINY_CALL, generator=torch.Generator().manual_seed(0))


def has_cuda() -> bool:
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
