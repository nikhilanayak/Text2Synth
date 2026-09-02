"""CTAG: creative text-to-audio generation via synthesizer programming."""

from .config import RunConfig
from .pipeline import CTAGPipeline, RunResult

__all__ = ["CTAGPipeline", "RunConfig", "RunResult"]
__version__ = "0.4.0"
