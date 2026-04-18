import h5py
import sys

# PartType indices (MUSIC swift plugin conventions)
PARTTYPE_GAS = 0
PARTTYPE_HIGHRES = 1   # high-res DM
PARTTYPE_COARSE = 2   # coarse DM (default)


def read_ics_swift(fname):
    with h5py.File(fname, 'r') as f:
        # --- Header ---
        hdr = dict(f['Header'].attrs)

        # --- Particles ---
        parts = {}
        for key in f.keys():
            if not key.startswith('PartType'):
                continue
            ptype = int(key.replace('PartType', ''))
            grp = f[key]
            parts[ptype] = {k: grp[k][()] for k in grp.keys()}

    return hdr, parts


def print_summary(fname):
    hdr, parts = read_ics_swift(fname)

    print("=== Header ===")
    for k, v in sorted(hdr.items()):
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


if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else 'ics_swift.hdf5'
    print_summary(fname)
