import pytest

from framefast.timing import StageTimer


def test_stage_timer_on_tiny_pipeline(tiny_pipe, tiny_call):
    timer = StageTimer(tiny_pipe, cuda=False)
    with timer.attach():
        out = timer.call(**tiny_call)
    t = timer.result()
    assert out.frames.shape == (1, 9, 3, 32, 32)
    assert len(t.step_ms) == tiny_call["num_inference_steps"]
    for v in (t.total_ms, t.text_encode_ms, t.denoise_ms, t.vae_decode_ms):
        assert v > 0
    assert t.text_encode_ms + t.denoise_ms + t.vae_decode_ms <= t.total_ms + 1e-6
    assert sum(t.step_ms) == pytest.approx(t.denoise_ms)
    # instrumentation is removed afterwards
    assert "encode_prompt" not in vars(tiny_pipe) and "decode" not in vars(tiny_pipe.vae)


def test_stage_timer_rejects_user_callback(tiny_pipe, tiny_call):
    timer = StageTimer(tiny_pipe, cuda=False)
    with timer.attach(), pytest.raises(ValueError):
        timer.call(callback_on_step_end=lambda *a: {}, **tiny_call)


def test_tiny_pipeline_is_deterministic(tiny_pipe, tiny_call):
    import torch

    a = tiny_pipe(**tiny_call).frames
    b = tiny_pipe(**dict(tiny_call, generator=torch.Generator().manual_seed(0))).frames
    assert torch.equal(a, b)
