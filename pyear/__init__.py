"""pyear package."""

try:
    from .pipeline import extract_features
    __all__ = ["extract_features"]
except Exception:  # pragma: no cover - optional dependency
    __all__ = []
