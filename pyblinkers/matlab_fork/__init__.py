"""MATLAB-style helper functions ported to Python."""

from .matlab_forking import (
    corrMatlab,
    get_intersection,
    polyfitMatlab,
    polyvalMatlab,
    weighted_corr,
)
from .line_intersection_matlab import lines_intersection_matlabx
from .mad_matlab import mad_matlab

__all__ = [
    "corrMatlab",
    "get_intersection",
    "polyfitMatlab",
    "polyvalMatlab",
    "weighted_corr",
    "lines_intersection_matlabx",
    "mad_matlab",
]

