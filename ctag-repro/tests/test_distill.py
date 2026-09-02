from pathlib import Path

import numpy as np

from ctag_repro.distill import (
    AtomicShardStore,
    full_training_prompts,
    get_profile,
    mine_retrieval_teachers,
    prompt_split,
)


def test_atomic_store_is_idempotent_and_round_trips(tmp_path: Path):
    store = AtomicShardStore(tmp_path, "test")
    values = {"parameters": np.arange(12, dtype=np.float32).reshape(3, 4)}
    assert store.append("unit-a", values)
    assert not store.append("unit-a", values)
    assert store.count == 3
    np.testing.assert_array_equal(store.load()["parameters"], values["parameters"])
    resumed = AtomicShardStore(tmp_path, "test")
    assert resumed.completed_units == {"unit-a"}


def test_retrieval_returns_descending_cosine_matches():
    embeddings = np.eye(4, dtype=np.float32)
    parameters = np.arange(8, dtype=np.float32).reshape(4, 2)
    selected, scores = mine_retrieval_teachers(
        embeddings, parameters, np.array([0.8, 0.1, 0.2, 0.3]), count=3
    )
    np.testing.assert_array_equal(selected, parameters[[0, 3, 2]])
    assert np.all(scores[:-1] >= scores[1:])


def test_training_profiles_are_decision_complete():
    assert get_profile("smoke").generator_steps == 2
    assert get_profile("balanced").random_patches == 250_000
    assert get_profile("quality").iterations == 300


def test_full_vocabulary_is_cached_and_evaluation_prompts_are_excluded(tmp_path: Path):
    project = tmp_path / "project"
    data = project / "data"
    data.mkdir(parents=True)
    (data / "regression-prompts.txt").write_text("held out\n")
    (data / "direct-eval-prompts.txt").write_text("novel alarm\n")
    for filename in ("audioset-sounds.txt", "esc50-sounds.txt", "esc10-sounds.txt"):
        (data / filename).write_text("paper sound\nheld out\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "audioset-class-labels.csv").write_text(
        "index,mid,display_name\n0,/m/a,Speech\n1,/m/b,novel alarm\n"
    )
    assert full_training_prompts(project, cache) == ["Speech", "paper sound"]
    assert prompt_split("Speech") == prompt_split("Speech")
