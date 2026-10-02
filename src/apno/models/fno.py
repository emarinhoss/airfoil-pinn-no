"""A Fourier neural operator in the O-grid's index space (spec 6.3).

i (around the airfoil) is periodic, so the FFT along it is exact. j (outward) is not: the field is
zero-padded by (NJ - 1) * pad_fraction rows before the FFT and cropped after, so the padded domain
spans the same eta at every resolution. That is what lets one model run at 128x49 and 256x97.
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


class FNO2d(nn.Module):
    def __init__(self, in_ch=6, out_ch=4, width=32, modes_i=24, modes_j=12, layers=4,
                 pad_fraction=1 / 6):
        super().__init__()
        self.pad_fraction = pad_fraction
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes_i, modes_j)
                                      for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, x):
        nj = x.shape[-1]
        pad = round((nj - 1) * self.pad_fraction)
        h = F.pad(self.lift(x), (0, pad))
        for k, (s, w) in enumerate(zip(self.spectral, self.pointwise)):
            h = s(h) + w(h)
            if k < len(self.spectral) - 1:
                h = F.gelu(h)
        return self.proj(h[..., :nj])
