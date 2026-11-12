"""CPU checks for the TensorRT path: input capture, positional-embedding pinning, torch.export
of the transformer and the VAE decoder, and the pipeline shim. TensorRT itself needs a GPU."""

import pytest
import torch

from framefast.trt import (
    TransformerExportWrapper,
    TRTTransformerShim,
    VAEDecodeWrapper,
    capture_transformer_inputs,
    export_inputs,
    pin_positional_embeddings,
    probe_export,
)
from tests.conftest import requires_cuda


def test_capture_and_pin(tiny_pipe, tiny_call):
    cap = capture_transformer_inputs(tiny_pipe, tiny_call)
    hs = cap["hidden_states"]
    assert hs.shape == (2, 3, 4, 4, 4)  # CFG doubles the batch; 9 frames -> 3 latent frames
    tr = tiny_pipe.transformer.eval()
    args, has_rot = export_inputs(cap)
    wrapper = TransformerExportWrapper(tr, has_rot)
    with torch.no_grad():
        before = wrapper(*args)
        assert pin_positional_embeddings(tr, tuple(hs.shape)) is True
        assert pin_positional_embeddings(tr, tuple(hs.shape)) is False  # idempotent
        after = wrapper(*args)
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)


def test_transformer_export_requires_pinning(tiny_pipe, tiny_call):
    cap = capture_transformer_inputs(tiny_pipe, tiny_call)
    tr = tiny_pipe.transformer.eval()
    args, has_rot = export_inputs(cap)
    probe, _ = probe_export("transformer", TransformerExportWrapper(tr, has_rot), args)
    # Non-native resolution: diffusers builds the pos-embedding with NumPy inside forward.
    assert not probe.ok and "FAILED" in probe.summary()
    pin_positional_embeddings(tr, tuple(cap["hidden_states"].shape))
    probe, ep = probe_export("transformer", TransformerExportWrapper(tr, has_rot), args)
    assert probe.ok, probe.error
    with torch.no_grad():
        torch.testing.assert_close(ep.module()(*args), TransformerExportWrapper(tr, has_rot)(*args))


def test_vae_decode_export_probe(tiny_pipe):
    vae = tiny_pipe.vae.eval()
    z = torch.randn(1, 4, 3, 4, 4)
    probe, ep = probe_export("vae.decode", VAEDecodeWrapper(vae), (z,))
    assert probe.ok, probe.error
    with torch.no_grad():
        torch.testing.assert_close(ep.module()(z), vae.decode(z).sample)


def test_shim_runs_pipeline_with_exported_module(tiny_pipe, tiny_call):
    """Stand-in for the TRT engine: the exported program must reproduce the eager pipeline."""
    with torch.no_grad():
        ref = tiny_pipe(**tiny_call).frames
    tr = tiny_pipe.transformer.eval()
    cap = capture_transformer_inputs(tiny_pipe, dict(tiny_call, generator=torch.Generator().manual_seed(0)))
    pin_positional_embeddings(tr, tuple(cap["hidden_states"].shape))
    args, has_rot = export_inputs(cap)
    _, ep = probe_export("transformer", TransformerExportWrapper(tr, has_rot), args)
    tiny_pipe.transformer = TRTTransformerShim(
        ep.module(), tr.config, tr.dtype, tr.device, tuple(a.shape for a in args), has_rot
    )
    with torch.no_grad():
        out = tiny_pipe(**dict(tiny_call, generator=torch.Generator().manual_seed(0))).frames
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match="static shapes"):
        tiny_pipe(**dict(tiny_call, prompt=[tiny_call["prompt"]] * 2))


@requires_cuda
@pytest.mark.gpu
def test_trt_transformer_end_to_end_cuda(tiny_pipe, tiny_call):
    pytest.importorskip("torch_tensorrt")
    from framefast.trt import compile_transformer_trt

    pipe = tiny_pipe.to("cuda", torch.float16)
    call = dict(tiny_call, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = pipe(**call).frames
        compile_transformer_trt(pipe, dict(call, generator=torch.Generator().manual_seed(0)), torch.float16)
        out = pipe(**dict(call, generator=torch.Generator().manual_seed(0))).frames
    assert (out.float() - ref.float()).abs().mean() < 5e-2
