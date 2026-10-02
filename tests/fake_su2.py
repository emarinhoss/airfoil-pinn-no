"""A stand-in for SU2_CFD for the driver tests. FAKE_SU2_MODE = ok | crash | nan.

It reads the mesh's nodes in file order and writes a free-stream restart in that order, plus a
history with a 6-order residual drop and constant forces, so the driver's real readers are
exercised.
"""

import math
import os
import re
import struct
import sys
from pathlib import Path


def main():
    cfg = Path(sys.argv[1]).read_text()
    mode = os.environ.get("FAKE_SU2_MODE", "ok")
    print('|   \\__ \\ |_| |/ /   Release 8.5.0 "Harrier"   (fake)')
    if mode == "crash":
        return 1
    mach = float(re.search(r"^MACH_NUMBER=\s*(\S+)", cfg, re.M).group(1))
    alpha = math.radians(float(re.search(r"^AOA=\s*(\S+)", cfg, re.M).group(1)))
    lines = Path("mesh.su2").read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("NPOIN="))
    n = int(lines[k].split("=")[1])
    pts = [tuple(map(float, ln.split()[:2])) for ln in lines[k + 1 : k + 1 + n]]
    g, r_gas, p, t = 1.4, 287.058, 101325.0, 288.15
    rho = p / (r_gas * t)
    u = mach * math.sqrt(g * r_gas * t)
    mx, my = rho * u * math.cos(alpha), rho * u * math.sin(alpha)
    e = p / (g - 1) + 0.5 * rho * u * u
    names = ["x", "y", "Density", "Momentum_x", "Momentum_y", "Energy"]
    head = struct.pack("<5i", 535532, len(names), n, 0, 0)
    head += b"".join(nm.encode().ljust(33, b"\0") for nm in names)
    body = b"".join(struct.pack("<6d", x, y, rho, mx, my, e) for x, y in pts)
    Path("restart.dat").write_bytes(head + body)
    rows = ['"Inner_Iter",    "rms[Rho]"    ,       "CD"       ,       "CL"       ,      "CMz"']
    for it in range(1001):
        cl = "nan" if (mode == "nan" and it > 900) else "0.25"
        rows.append(f"{it}, {-1.0 - 6.0 * it / 1000}, 0.0001, {cl}, 0.01")
    Path("history.csv").write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
