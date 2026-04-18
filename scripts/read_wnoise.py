import numpy as np
import sys


def read_wnoise(fname, dtype=np.float32):
    with open(fname, 'rb') as f:
        nx, ny, nz = np.frombuffer(f.read(12), dtype=np.uint32)
        data = np.frombuffer(f.read(nx * ny * nz * np.dtype(dtype).itemsize), dtype=dtype)
    return data.reshape(nx, ny, nz)


if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else 'wnoise_0008.bin'
    w = read_wnoise(fname)
    print(f"Shape : {w.shape}")
    print(f"dtype : {w.dtype}")
    print(f"min/max: {w.min():.4f} / {w.max():.4f}")
    print(f"mean/std: {w.mean():.4f} / {w.std():.4f}")
