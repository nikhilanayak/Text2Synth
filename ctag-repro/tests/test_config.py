from pathlib import Path

import pytest

from ctag_repro.config import RunConfig


def test_paper_defaults_match_release():
    config = RunConfig.paper()
    assert config.population_size == 50
    assert config.iterations == 300
    assert config.sample_rate == 48_000
    assert config.control_rate == 480
    assert config.duration_seconds == 2.0
    assert config.num_samples == 96_000
    assert config.seed == 42


def test_smoke_only_reduces_search_budget():
    config = RunConfig.smoke()
    assert config.population_size == 2
    assert config.iterations == 2
    assert config.num_samples == 96_000


def test_sample_count_matches_synthax_ceiling():
    assert RunConfig(duration_seconds=0.00001).num_samples == 1


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    expected = RunConfig.smoke(seed=7)
    expected.write_yaml(path)
    assert RunConfig.from_yaml(path) == expected


@pytest.mark.parametrize(
    "field,value", [("population_size", 1), ("iterations", 0), ("log_every", 0)]
)
def test_invalid_search_config(field, value):
    with pytest.raises(ValueError):
        RunConfig(**{field: value})
