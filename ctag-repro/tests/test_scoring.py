import numpy as np
import pytest

from ctag_repro.scoring import negative_cosine_fitness, normalize_embeddings


def test_normalize_embeddings():
    result = normalize_embeddings(np.array([[3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(result, [[0.6, 0.8]])


def test_negative_cosine_is_minimization_objective():
    audio = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    text = np.array([1.0, 0.0])
    # Positive cosine becomes negative fitness; a negative cosine produces a
    # positive value and is neutralized exactly like upstream CTAG.
    np.testing.assert_allclose(negative_cosine_fitness(audio, text), [-1, 0, 0])


def test_objective_does_not_renormalize_clap_outputs():
    audio = np.array([[0.5, 0.0]], dtype=np.float32)
    text = np.array([0.5, 0.0], dtype=np.float32)
    np.testing.assert_allclose(negative_cosine_fitness(audio, text), [-0.25])


def test_embedding_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        negative_cosine_fitness(np.zeros((2, 3)), np.zeros(4))
