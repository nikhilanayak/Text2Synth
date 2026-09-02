import json
from pathlib import Path


def test_training_notebook_is_clean_and_contains_complete_workflow():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "DirectPatch_Train.ipynb").read_text())
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for command in (
        "ctag doctor --require-gpu",
        "ctag train-live",
        "ctag evaluate-direct",
        "LiveDirectSynth",
        "ctag export-direct",
    ):
        assert command in code
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
