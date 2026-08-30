"""CTAG objective functions."""

from __future__ import annotations

import numpy as np


def normalize_embeddings(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def negative_cosine_fitness(
    audio_embeddings: np.ndarray, text_embedding: np.ndarray
) -> np.ndarray:
    """Compute the paper's minimization objective: negative CLAP similarity."""

    # LAION-CLAP already returns L2-normalized embeddings. Do not normalize a
    # second time here: the authors' objective is the direct matrix product,
    # and preserving it avoids changing last-bit candidate rankings.
    audio = np.asarray(audio_embeddings, dtype=np.float32)
    text = np.asarray(np.atleast_2d(text_embedding), dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError(f"expected a 2-D audio embedding matrix, got {audio.shape}")
    if text.shape[0] != 1:
        raise ValueError("exactly one text embedding is required per search")
    if audio.shape[1] != text.shape[1]:
        raise ValueError("audio and text embedding dimensions differ")
    if not np.isfinite(audio).all() or not np.isfinite(text).all():
        raise ValueError("embeddings must be finite")
    fitness = -(audio @ text.T).reshape(-1)
    # Preserve the reference implementation's guard. Valid negative cosine
    # fitness is in [-1, 0]; invalid/positive values become neutral fitness.
    return np.where((fitness < -1.0) | (fitness > 0.0), 0.0, fitness).astype(
        np.float32
    )
