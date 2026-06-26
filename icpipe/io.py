"""I/O helpers: read SWIFT IC HDF5, write ASCII spectra in pk_*/pv_* format."""

from __future__ import annotations

from typing import Any, Sequence

import h5py
import numpy as np


def read_ic_hdf5(path: str, want_velocities: bool = False) -> dict[str, Any]:
    """Read a SWIFT IC HDF5 file and return a dict with positions, optionally
    velocities, and the relevant header attributes.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    want_velocities : bool
        If True, also load PartType1/Velocities (km/s, peculiar units).

    Returns
    -------
    dict with keys:
        coords     : (N, 3) float64, particle positions in Mpc.
        velocities : (N, 3) float64, peculiar velocity in km/s (if want_velocities).
        boxsize    : float, box side length in Mpc.
        redshift   : float, IC redshift.
        N          : int, particle count.
    """
    with h5py.File(path, "r") as f:
        boxsize  = float(np.asarray(f["Header"].attrs["BoxSize"]).reshape(-1)[0])
        redshift = float(np.asarray(f["Header"].attrs["Redshift"]).reshape(-1)[0])
        coords   = f["PartType1/Coordinates"][:]
        out = dict(coords=coords, boxsize=boxsize,
                   redshift=redshift, N=len(coords))
        if want_velocities:
            out["velocities"] = f["PartType1/Velocities"][:]
    return out


def write_pk(path: str, *, k, P_raw, P_ss, P_nodeconv, sigma_P, nmodes,
             fold_m=None, header_lines: Sequence[str] = ()) -> None:
    """Write a P(k) table in the format used by ``compute_pk.py``.

    Columns: k[h/Mpc] P_raw P_shot_sub P_nodeconv sigma_P nmodes fold_m

    All P columns are in (Mpc/h)^3.
    """
    if fold_m is None:
        fold_m = np.ones(len(k), dtype=int)
    cols = "Columns: k[h/Mpc]  P_raw[(Mpc/h)^3]  P_shot_sub[(Mpc/h)^3]  P_nodeconv[(Mpc/h)^3]  sigma_P  nmodes  fold_m"
    header = "\n".join(list(header_lines) + [cols])
    arr = np.column_stack([k, P_raw, P_ss, P_nodeconv, sigma_P, nmodes, fold_m])
    fmt = ["%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%d", "%d"]
    np.savetxt(path, arr, header=header, fmt=fmt)


def write_pv(path: str, *, k, Pv_raw, Pv_nodeconv, sigma_Pv, nmodes,
             header_lines: Sequence[str] = ()) -> None:
    """Write a velocity P_v(k) table.

    Columns: k[h/Mpc] Pv_raw[(km/s)^2 (Mpc/h)^3] Pv_nodeconv sigma_Pv nmodes

    Pv_raw is the deconvolved (CIC-window-corrected) vector velocity power
    spectrum P_v(k) = sum_i |v_i(k)|^2 averaged in the bin. Pv_nodeconv is
    the same quantity before deconvolution. No shot-noise subtraction is
    applied (CIC velocity grid is a deterministic ratio momentum/density).
    """
    cols = "Columns: k[h/Mpc]  Pv_raw[(km/s)^2 (Mpc/h)^3]  Pv_nodeconv[(km/s)^2 (Mpc/h)^3]  sigma_Pv  nmodes"
    header = "\n".join(list(header_lines) + [cols])
    arr = np.column_stack([k, Pv_raw, Pv_nodeconv, sigma_Pv, nmodes])
    fmt = ["%.6e", "%.6e", "%.6e", "%.6e", "%d"]
    np.savetxt(path, arr, header=header, fmt=fmt)


def read_pk(path: str) -> dict[str, np.ndarray]:
    """Read a pk_*.txt table written by ``write_pk`` or ``compute_pk.py``."""
    arr = np.loadtxt(path, comments="#")
    return dict(k=arr[:, 0], P_raw=arr[:, 1], P_ss=arr[:, 2],
                P_nodeconv=arr[:, 3], sigma_P=arr[:, 4],
                nmodes=arr[:, 5].astype(int),
                fold_m=arr[:, 6].astype(int) if arr.shape[1] > 6 else None)


def read_pv(path: str) -> dict[str, np.ndarray]:
    """Read a pv_*.txt table written by ``write_pv``."""
    arr = np.loadtxt(path, comments="#")
    return dict(k=arr[:, 0], Pv_raw=arr[:, 1], Pv_nodeconv=arr[:, 2],
                sigma_Pv=arr[:, 3], nmodes=arr[:, 4].astype(int))


# ---------------------------------------------------------------------------
# Raw IC / white-noise readers (inspection helpers; not used by ICField).
# ---------------------------------------------------------------------------

def read_wnoise(path: str, dtype=np.float32) -> np.ndarray:
    """Read a MUSIC2 ``wnoise_NNNN.bin`` white-noise dump.

    File layout: three uint32 dimensions (nx, ny, nz), followed by
    nx*ny*nz scalars of ``dtype`` (MUSIC2 default is float32).

    Returns
    -------
    numpy.ndarray of shape (nx, ny, nz).
    """
    with open(path, "rb") as f:
        nx, ny, nz = np.frombuffer(f.read(12), dtype=np.uint32)
        data = np.frombuffer(
            f.read(int(nx) * int(ny) * int(nz) * np.dtype(dtype).itemsize),
            dtype=dtype,
        )
    return data.reshape(nx, ny, nz)


def read_swift_ics(path: str) -> tuple[dict, dict]:
    """Read a SWIFT-format IC HDF5 file and return ``(header, parts)``.

    ``header`` is a dict of the ``/Header`` attributes.  ``parts`` maps
    ``PartType`` index → dict of dataset_name → ndarray, covering every
    ``PartType*`` group present.  Compared to :func:`read_ic_hdf5`, this
    is a general inspection reader (all particle types, all datasets);
    :func:`read_ic_hdf5` is the focused loader used by ``ICField``.
    """
    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        parts: dict[int, dict[str, np.ndarray]] = {}
        for key in f.keys():
            if not key.startswith("PartType"):
                continue
            ptype = int(key.replace("PartType", ""))
            grp = f[key]
            parts[ptype] = {k: grp[k][()] for k in grp.keys()}
    return header, parts


def print_swift_ics_summary(path: str) -> None:
    """Pretty-print header attributes and per-PartType dataset summaries."""
    header, parts = read_swift_ics(path)
    print("=== Header ===")
    for k, v in sorted(header.items()):
        print(f"  {k}: {v}")
    print("\n=== Particle groups ===")
    for ptype, data in sorted(parts.items()):
        npart = next(iter(data.values())).shape[0]
        print(f"\n  PartType{ptype}  ({npart} particles)")
        for field, arr in sorted(data.items()):
            print(f"    {field:20s}  shape={arr.shape}  dtype={arr.dtype}")
            if arr.ndim == 1:
                print(f"      min={arr.min():.6g}  max={arr.max():.6g}")
            else:
                print(f"      min={arr.min():.6g}  max={arr.max():.6g}  (per component)")
