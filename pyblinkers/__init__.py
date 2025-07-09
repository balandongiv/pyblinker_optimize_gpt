"""Primary package initialization for pyblinkers."""

from .fit_blink import FitBlinks
from .extract_blink_properties import BlinkProperties
try:
    from .pipeline import extract_features
except Exception:  # pragma: no cover - optional dependency
    extract_features = None

__all__ = [
    "FitBlinks",
    "BlinkProperties",
    "extract_features",
]
