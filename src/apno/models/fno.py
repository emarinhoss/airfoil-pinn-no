"""A Fourier neural operator in the O-grid's index space (spec 6.3).

i (around the airfoil) is periodic, so the FFT along it is exact, and a learned i-mode means the
same physical wavenumber at 128 and 256 nodes. j (outward) is not periodic: the field is zero-padded
by radial_pad(NJ) = (NJ - 1)/6 rows before the FFT and cropped after, 8 at 128x49 and 16 at
256x97, so the padded domain spans the same eta at every resolution. The transfer between them is
only O(h), not exact: the padded field jumps to zero at the wrap and wraps back onto the wall row,
and that jump aliases differently at each resolution. Stage S2 measures the transfer error rather
than assuming it is zero.

The rule has changed twice (spec 6.3). From 2026-10-01 to 2026-10-26 it was
radial_pad_period_matched, round(3(NJ - 1)/16) - 1 rows, 8 and 17, which keeps the radial DFT
period at 19/16 in eta, so a learned radial mode is the same physical mode at both resolutions on
a wrap-periodic field (tests/test_fno.py). On the trained networks, 16 rows at 256x97 transfer
better than 17, by the validation transfer and near field of every checkpoint and by S4's test
near field (docs/findings.md, 2026-10-25 and 2026-10-26), so the rule went back to (NJ - 1)/6.
Training at 128x49 pads 8 rows under both rules, so models trained under either are the same
models; only their 256x97 evaluation differs. The old rule is kept for the probes whose records
were made with it, and a model evaluates with another rule through its `pad_rule` attribute.
Truncating the modes smooths jumps: smeared shocks and overshoot are expected and measured, not
fixed.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes_i, modes_j):
        super().__init__()
        self.mi, self.mj = modes_i, modes_j
        scale = 1.0 / (cin * cout)
        self.w_pos = nn.Parameter(scale * torch.randn(cin, cout, modes_i, modes_j,
                                                      dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(cin, cout, modes_i, modes_j,
                                                      dtype=torch.cfloat))

    def forward(self, x):
        b, _, ni, nj = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w_pos.shape[1], ni, nj // 2 + 1, dtype=xf.dtype,
                          device=x.device)
        w_pos, w_neg = self.w_pos.to(xf.dtype), self.w_neg.to(xf.dtype)
        out[:, :, : self.mi, : self.mj] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, : self.mi, : self.mj], w_pos)
        out[:, :, -self.mi :, : self.mj] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.mi :, : self.mj], w_neg)
        return torch.fft.irfft2(out, s=(ni, nj))


def radial_pad(nj):
    """Rows of zero padding at radial size NJ: (NJ - 1)/6, 8 at 128x49 and 16 at 256x97."""
    return (nj - 1) // 6


def radial_pad_period_matched(nj):
    """The rule until 2026-10-26: the radial DFT period is (NJ - 1) * 19/16 samples at every NJ,
    8 rows at 128x49 and 17 at 256x97."""
    return round((nj - 1) * 3 / 16) - 1


class FNO2d(nn.Module):
    def __init__(self, in_ch=6, out_ch=4, width=32, modes_i=24, modes_j=12, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes_i, modes_j)
                                      for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))
        self.pad_rule = radial_pad  # rows at radial size NJ; not a weight, not in the state dict

    def forward(self, x):
        nj = x.shape[-1]
        pad = self.pad_rule(nj)
        h = F.pad(self.lift(x), (0, pad))
        for k, (s, w) in enumerate(zip(self.spectral, self.pointwise)):
            h = s(h) + w(h)
            if k < len(self.spectral) - 1:
                h = F.gelu(h)
        return self.proj(h[..., :nj])
