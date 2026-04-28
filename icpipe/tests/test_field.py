"""Tests for icpipe.field.

These test the CIC + FFT pipeline against analytic expectations on a
synthetic IC field constructed in memory (no real HDF5 needed).
"""

import os

import h5py
import numpy as np
import pytest

from icpipe import ICField


@pytest.fixture(scope="module")
def fake_ic_hdf5(tmp_path_factory):
    """Build a tiny SWIFT-format IC HDF5 file for testing.

    A uniform-lattice particle distribution: particles on a regular grid,
    so the density field is exactly uniform → P_delta(k) = 0 for all k≠0.
    Velocities are all zero → P_v(k) = 0 for all k.
    """
    npart_side = 8
    L = 1.0  # Mpc
    coords = np.array(
        [(i, j, k) for i in range(npart_side)
                   for j in range(npart_side)
                   for k in range(npart_side)],
        dtype=np.float64,
    ) * (L / npart_side)
    velocities = np.zeros_like(coords)

    path = str(tmp_path_factory.mktemp("ic") / "test_uniform.hdf5")
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = L
        h.attrs["Redshift"] = 100.0
        pt = f.create_group("PartType1")
        pt.create_dataset("Coordinates", data=coords)
        pt.create_dataset("Velocities", data=velocities)
    return path


def test_load_metadata(fake_ic_hdf5):
    f = ICField(fake_ic_hdf5, h=1.0)
    assert f.N == 8 ** 3
    assert np.isclose(f.boxsize_mpc, 1.0)
    assert np.isclose(f.redshift, 100.0)
    assert f.ngrid == 8
    assert f.velocities is not None


def test_uniform_field_zero_power(fake_ic_hdf5):
    """A perfectly-lattice IC has delta = 0 → P_delta(k) tiny (machine zero
    plus small CIC artefacts at corners)."""
    f = ICField(fake_ic_hdf5, h=1.0, interlace=False)
    res = f.power("delta", nkbins=4, shot_noise=False)
    # P_raw is the deconvolved spectrum; should be ~machine zero.
    assert (np.abs(res.P_raw) < 1e-20).all()


def test_zero_velocity_zero_pv(fake_ic_hdf5):
    """All-zero velocities give P_v(k) = 0 exactly."""
    f = ICField(fake_ic_hdf5, h=1.0, interlace=False)
    res = f.power("velocity", nkbins=4)
    assert (np.abs(res.P_raw) < 1e-20).all()


def test_repr_runs(fake_ic_hdf5):
    f = ICField(fake_ic_hdf5, h=1.0)
    s = repr(f)
    assert "ICField" in s
    assert "ngrid" in s
