import pytest
import numpy as np

torch = pytest.importorskip("torch")

from ctag_repro.direct import DirectPrediction, build_patch_surrogate
from ctag_repro.direct_training import LiveDirectSynth, direct_losses, train_direct
from ctag_repro.distill import AtomicShardStore, get_profile


def test_direct_objective_is_finite_and_backpropagates():
    logits = torch.randn(2, 3, 78, requires_grad=True)
    predicted = torch.sigmoid(logits)
    teachers = torch.rand(2, 3, 78)
    text = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    surrogate = build_patch_surrogate()
    loss, metrics = direct_losses(predicted, teachers, text, surrogate)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"semantic", "head0", "coverage", "diversity", "total"}
    assert logits.grad is not None


def test_tiny_training_run_saves_resumable_safe_bundle(tmp_path):
    pytest.importorskip("safetensors")
    rng = np.random.default_rng(4)
    random_store = AtomicShardStore(tmp_path / "random", "random-patches")
    random_store.append(
        "random-0",
        {
            "parameters": rng.random((16, 78), dtype=np.float32),
            "audio_embeddings": rng.normal(size=(16, 512)).astype(np.float16),
        },
    )
    teacher_store = AtomicShardStore(tmp_path / "teachers", "prompt-teachers")
    teacher_store.append(
        "prompt-0",
        {
            "prompts": np.asarray(["train horn"]),
            "text_embeddings": rng.normal(size=(1, 512)).astype(np.float16),
            "parameters": rng.random((1, 8, 78), dtype=np.float32),
            "scores": rng.random((1, 8), dtype=np.float32),
        },
    )
    result = train_direct(
        tmp_path,
        tmp_path / "training",
        get_profile("smoke"),
        device="cpu",
    )
    assert result["surrogate_step"] == 2
    assert result["generator_step"] == 2
    assert (tmp_path / "training" / "model" / "model.safetensors").is_file()
    resumed = train_direct(
        tmp_path,
        tmp_path / "training",
        get_profile("smoke"),
        device="cpu",
    )
    assert resumed["generator_step"] == 2


def test_live_session_direct_mode_never_calls_audio_reranking(tmp_path):
    class Encoder:
        def embed_audio(self, *_args):
            raise AssertionError("strict inference must not invoke CLAP audio")

    class Predictor:
        text_encoder = Encoder()

        def predict(self, prompts):
            variants = np.full((8, 78), 0.25, dtype=np.float32)
            return [
                DirectPrediction(
                    prompts[0], variants[0], variants, {"direct_model": 0.001}, {}
                )
            ]

    class Synth:
        def render(self, *_args):
            raise AssertionError("strict inference must not render variants")

        def render_one(self, parameters):
            return np.zeros(8, dtype=np.float32), {"parameters": parameters.tolist()}

    session = object.__new__(LiveDirectSynth)
    session.predictor = Predictor()
    session.synth = Synth()
    session.config = type("Config", (), {"sample_rate": 8})()
    result = session.generate("train horn", tmp_path, selection="direct")
    assert result["selected_head"] == 0
    assert result["similarity"] is None
    assert (tmp_path / "train-horn-71704480.wav").is_file()
