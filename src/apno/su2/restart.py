"""SU2's binary restart file, as written by SU2 8.5.0 (layout verified byte by byte, 2026-10-01).

Header: five little-endian int32 = (535532, n_fields, n_points, 0, 0); then n_fields names, each
33 bytes NUL-padded; then n_points x n_fields float64, point by point. This build writes nothing
after the data, so the file size must match exactly; anything else is refused rather than guessed.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 535532
NAME_BYTES = 33


@dataclass(frozen=True)
class Restart:
    names: tuple
    data: np.ndarray

    @property
    def npoints(self):
        return self.data.shape[0]

    def field(self, name):
        return self.data[:, self.names.index(name)]


def read_restart(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path}: size {len(raw)} is smaller than the header")
    magic, nf, npnt, _, _ = struct.unpack("<5i", raw[:20])
    if magic != MAGIC:
        raise ValueError(f"{path}: not an SU2 binary restart (magic {magic}, expected {MAGIC})")
    offset = 20 + NAME_BYTES * nf
    expected = offset + 8 * nf * npnt
    if len(raw) != expected:
        raise ValueError(f"{path}: size {len(raw)} != expected {expected}")
    names = tuple(
        raw[20 + NAME_BYTES * k : 20 + NAME_BYTES * (k + 1)].split(b"\0")[0].decode()
        for k in range(nf)
    )
    data = np.frombuffer(raw, dtype="<f8", offset=offset).reshape(npnt, nf).copy()
    return Restart(names, data)


def write_restart(path, names, data):
    data = np.ascontiguousarray(data, dtype="<f8")
    head = struct.pack("<5i", MAGIC, len(names), data.shape[0], 0, 0)
    head += b"".join(n.encode().ljust(NAME_BYTES, b"\0") for n in names)
    Path(path).write_bytes(head + data.tobytes())


def on_grid(restart, X, Y, atol=1e-10):
    """Every field as an (NI, NJ) array, after checking each node's coordinates against the grid."""
    ni, nj = X.shape
    if restart.npoints != ni * nj:
        raise ValueError(f"restart has {restart.npoints} points, grid has {ni * nj}")
    out = {n: restart.field(n).reshape(nj, ni).T.copy() for n in restart.names}
    err = max(np.abs(out["x"] - X).max(), np.abs(out["y"] - Y).max())
    if err > atol:
        raise ValueError(f"restart nodes do not match the grid (max coordinate error {err:.3e})")
    return out
