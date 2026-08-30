"""Prompt parsing shared by CLI and library entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union


def load_prompts(source: Union[str, Path, Iterable[str]]) -> List[str]:
    """Load prompts from a file, semicolon-separated string, or iterable.

    Unlike the released helper, this intentionally retains the last line when
    a prompt file does not end with a newline.
    """

    is_file = isinstance(source, Path)
    if isinstance(source, str):
        try:
            is_file = Path(source).is_file()
        except OSError:
            # Very long free-form prompts can exceed filesystem name limits;
            # they are prompt text, not paths.
            is_file = False
    if is_file:
        raw = Path(source).read_text().splitlines()
    elif isinstance(source, str):
        raw = source.split(";")
    else:
        raw = list(source)
    prompts = [str(item).strip() for item in raw if str(item).strip()]
    if not prompts:
        raise ValueError("at least one non-empty prompt is required")
    return prompts
