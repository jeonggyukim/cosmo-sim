"""icpipe — Initial-condition analysis utilities for cosmological simulations.

Top-level exports are added incrementally as the modules are implemented:
- icpipe.field.ICField  — HDF5 → CIC grids → spectra/correlations
- icpipe.theory.LinearTheory  — linear-theory P_δ, P_v, ξ, ψ from CLASS
- icpipe.windows  — CIC window and deconvolution
- icpipe.binning  — radial-bin utilities
- icpipe.io  — ASCII I/O for pk_*.txt, pv_*.txt, etc.
"""

__version__ = "0.1.0"
