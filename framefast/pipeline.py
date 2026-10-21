"""Construction of the pipeline under test.

Two sources are supported:

* A real Hugging Face checkpoint (default ``THUDM/CogVideoX-2b``) loaded in fp16.
* ``"tiny-random"``: a CogVideoX pipeline with the *same module classes* but tiny, randomly
  initialised weights and an offline word-level tokenizer. It needs no network and runs in a
  few seconds on CPU, so CI can exercise every code path of the harness and of the
  optimizations (module patching, quantization, kernel substitution) without a GPU.
"""

from __future__ import annotations

from typing import Any

import torch

TINY_MODEL_ID = "tiny-random"

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype string to a :class:`torch.dtype`."""
    try:
        return _DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPES)}") from exc


def _tiny_tokenizer(max_length: int) -> Any:
    """Offline word-level tokenizer exposing the HF ``PreTrainedTokenizerFast`` interface."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "a", "the", "of", "on", "in", "at", "and", "through", "against", "slowly", "slow",
        "dog", "golden", "retriever", "running", "sunlit", "meadow", "motion", "cinematic",
        "lighting", "aerial", "drone", "shot", "waves", "crashing", "black", "volcanic", "rocks",
        "sunset", "steaming", "cup", "coffee", "wooden", "table", "camera", "orbiting",
        "shallow", "depth", "field", "cat", "city", "night",
    ]  # fmt: skip
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=max_length,
    )


def build_tiny_pipeline(max_sequence_length: int = 16, seed: int = 0) -> Any:
    """Build a tiny random-weight :class:`diffusers.CogVideoXPipeline` (fp32, CPU).

    The shapes follow the diffusers CogVideoX unit-test configuration: 2 transformer layers,
    hidden size 32 (4 heads x 8), a 4-level VAE with 8 channels, and a 2-layer T5 encoder.
    Weights are seeded so outputs are reproducible across processes.
    """
    from diffusers import (
        AutoencoderKLCogVideoX,
        CogVideoXDDIMScheduler,
        CogVideoXPipeline,
        CogVideoXTransformer3DModel,
    )
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(seed)
    transformer = CogVideoXTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=32,
        num_layers=2,
        sample_width=8,
        sample_height=8,
        sample_frames=9,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=max_sequence_length,
    )
    torch.manual_seed(seed + 1)
    vae = AutoencoderKLCogVideoX(
        in_channels=3,
        out_channels=3,
        down_block_types=("CogVideoXDownBlock3D",) * 4,
        up_block_types=("CogVideoXUpBlock3D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=2,
        temporal_compression_ratio=4,
    )
    torch.manual_seed(seed + 2)
    text_encoder = T5EncoderModel(
        T5Config(
            vocab_size=64,
            d_model=32,
            d_kv=8,
            d_ff=37,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    scheduler = CogVideoXDDIMScheduler()
    pipe = CogVideoXPipeline(
        tokenizer=_tiny_tokenizer(max_sequence_length),
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pipeline(model_id: str, dtype: torch.dtype, max_sequence_length: int = 226) -> Any:
    """Load the pipeline on CPU (placement onto the device is done by the variant runner).

    Args:
        model_id: HF hub id (e.g. ``THUDM/CogVideoX-2b``) or :data:`TINY_MODEL_ID`.
        dtype: Weight dtype. The tiny model ignores this and stays fp32 unless asked otherwise.
        max_sequence_length: Only used by the tiny model (text length of the toy tokenizer).
    """
    if model_id == TINY_MODEL_ID:
        pipe = build_tiny_pipeline(max_sequence_length=max_sequence_length)
        return pipe.to(dtype=dtype) if dtype != torch.float32 else pipe
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe
