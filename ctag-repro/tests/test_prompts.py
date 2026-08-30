from pathlib import Path

import pytest

from ctag_repro.prompts import load_prompts


def test_semicolon_prompt_string():
    assert load_prompts(" spray ; train horn ") == ["spray", "train horn"]


def test_file_without_trailing_newline_keeps_last_prompt(tmp_path: Path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("spray\ntrain horn")
    assert load_prompts(prompt_file) == ["spray", "train horn"]


def test_empty_prompts_rejected():
    with pytest.raises(ValueError):
        load_prompts(" ; ")


def test_very_long_prompt_is_not_treated_as_a_path():
    prompt = "x" * 10_000
    assert load_prompts(prompt) == [prompt]
