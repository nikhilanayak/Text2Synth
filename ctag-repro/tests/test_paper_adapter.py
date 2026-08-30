import numpy as np

from ctag_repro.paper_backend import CLAPEncoder


class _FakeTorch:
    class _Inference:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    def inference_mode(self):
        return self._Inference()


class _FakeCLAP:
    def __init__(self):
        self.seen = None

    def get_text_embedding(self, values, use_tensor):
        self.seen = values
        return np.arange(len(values) * 4, dtype=np.float32).reshape(len(values), 4)


def test_single_prompt_is_duplicated_only_at_legacy_clap_boundary():
    encoder = object.__new__(CLAPEncoder)
    encoder._torch = _FakeTorch()
    encoder.model = _FakeCLAP()
    result = encoder.embed_text(["train horn"])
    assert encoder.model.seen == ["train horn", "train horn"]
    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result[0], [0, 1, 2, 3])
