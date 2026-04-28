"""icpipe — Initial-condition analysis utilities for cosmological simulations.

Top-level exports:
- ICField               — the main HDF5-based field/spectrum class
- PowerSpectrumResult   — dataclass returned by ICField.power(...)
- LinearTheory          — linear-theory P_δ, P_v, ξ, ψ helpers (see icpipe.theory)

Submodules:
- icpipe.field          — ICField class
- icpipe.theory         — LinearTheory class
- icpipe.windows        — CIC window function
- icpipe.binning        — radial binning helpers
- icpipe.io             — HDF5 reading and ASCII pk_* / pv_* writers
"""

from . import binning, io, theory, windows
from .field import ICField, PowerSpectrumResult
from .theory import LinearTheory

__version__ = "0.1.0"
__all__ = [
    "ICField", "PowerSpectrumResult",
    "LinearTheory",
    "binning", "io", "theory", "windows",
]
