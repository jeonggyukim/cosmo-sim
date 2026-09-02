"""Locations used by the pencil-subvolume scripts.

The scripts live in the repository; the runs they read and write do not, because a
single sweep is tens of MB and a kept delta(q) field is 6 MB per seed. Every path
is resolved here so nothing else carries a machine-specific string.

Environment variables, all optional:

    MONOFONIC_TESTS   root holding data/ and the reference run
                      (default: ~/Documents/monofonic-tests)
    MONOFONIC_REF     reference run whose config and CLASS table the sweep reuses
                      (default: $MONOFONIC_TESTS/n64_deltaq_z200_L700)
    MONOFONIC_BIN     monofonIC built from the `lagrangian-density` branch of
                      github.com/jeonggyukim/monofonIC

Figures are written to $MONOFONIC_TESTS, data under $MONOFONIC_TESTS/data.
"""
import os

ROOT = os.path.expanduser(os.environ.get("MONOFONIC_TESTS", "~/Documents/monofonic-tests"))
DATA = os.path.join(ROOT, "data")
FIGS = ROOT
REF = os.path.expanduser(os.environ.get("MONOFONIC_REF",
                                        os.path.join(ROOT, "n64_deltaq_z200_L700")))
REF_CONF = os.path.join(REF, "deltaq_n64_L700.conf")
REF_POWERSPEC = os.path.join(REF, "deltaq_n64_L700_input_powerspec.txt")
REF_FIELD = os.path.join(REF, "delta_q_n64_L700.hdf5")
BIN = os.path.expanduser(os.environ.get(
    "MONOFONIC_BIN", "~/Library/CloudStorage/Dropbox/Projects/"
                     "monofonIC-lagrangian-density/build/monofonIC"))

# The 2LPT particle IC generated from the same seed, used only by the validation
# scripts that cross-check delta(q) against particles.
IC_DMGAS = os.path.join(ROOT, "n64_2lpt_dmgas_z200_L700", "ics_dmgas_n64_L700.hdf5")
IC_DM = os.path.join(ROOT, "n64_2lpt_dm_z200_L500", "ics_dm_n64.hdf5")


def require(*paths, binary=False):
    """Fail with a legible message rather than deep inside a run."""
    missing = [p for p in paths if not os.path.exists(p)]
    if binary and not os.path.exists(BIN):
        missing.append(BIN)
    if missing:
        raise SystemExit(
            "cannot find:\n  " + "\n  ".join(missing) +
            "\n\nSet MONOFONIC_TESTS (currently %s)" % ROOT +
            (", MONOFONIC_BIN" if binary else "") + " and re-run.")
