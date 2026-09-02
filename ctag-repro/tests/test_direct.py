import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from ctag_repro.direct import (
    DirectModelConfig,
    DirectPatchPredictor,
    PARAMETER_CONTRACT_HASH,
    PARAMETER_LAYOUT,
    build_direct_model,
    load_model_bundle,
    save_model_bundle,
)
from ctag_repro.direct_training import export_direct


class FakeTextEncoder:
    def embed_text(self, values):
        result = np.ones((len(list(values)), 512), dtype=np.float32)
        return result / np.linalg.norm(result, axis=1, keepdims=True)


def test_direct_model_has_eight_bounded_78_parameter_heads():
    model = build_direct_model(DirectModelConfig())
    result = model(torch.randn(3, 512))
    assert result.shape == (3, 8, 78)
    assert torch.all((result >= 0) & (result <= 1))


def test_safe_bundle_round_trip_and_contract_guard(tmp_path: Path):
    config = DirectModelConfig(hidden_dims=(16, 8))
    model = build_direct_model(config)
    save_model_bundle(tmp_path, model, config, {"purpose": "test"})
    loaded, loaded_config, metadata = load_model_bundle(tmp_path)
    assert loaded_config == config
    assert metadata == {"purpose": "test"}
    sample = torch.randn(2, 512)
    torch.testing.assert_close(model(sample), loaded(sample))

    manifest = json.loads((tmp_path / "model.json").read_text())
    manifest["parameter_contract_hash"] = "wrong"
    (tmp_path / "model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parameter ordering"):
        load_model_bundle(tmp_path)


def test_predictor_selects_head_zero_without_rendering():
    config = DirectModelConfig(hidden_dims=(8,), heads=3)
    predictor = DirectPatchPredictor(
        build_direct_model(config), config, FakeTextEncoder(), "cpu"
    )
    result = predictor.predict(["train horn"])[0]
    assert result.variants.shape == (3, 78)
    np.testing.assert_array_equal(result.parameters, result.variants[0])
    assert result.metadata["selection"] == "head0"
    assert len(PARAMETER_CONTRACT_HASH) == 64
    assert sum(size for _, size in PARAMETER_LAYOUT) == 78


def test_fp32_and_int8_onnx_exports(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = DirectModelConfig(hidden_dims=(16, 8), heads=2)
    save_model_bundle(tmp_path / "model", build_direct_model(config), config)
    report = export_direct(tmp_path / "model", tmp_path / "export", quantize=True)
    assert report["fp32_max_abs_error"] <= 1e-5
    assert Path(report["int8"]).is_file()
