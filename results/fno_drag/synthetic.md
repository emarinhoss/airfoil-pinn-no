# Transfer against the row count at 256x97, without training

Relative L2 difference between the 128x49 output (8 rows) and the 256x97 output at its shared nodes. Vertex: where straight branches through the two points either side of the minimum meet. Written by `python scripts/probes/probe_fno_drag.py --synthetic`.

| | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | vertex |
|---|---|---|---|---|---|---|---|---|---|
| one spectral layer: constant 1 | 0.1180 | 0.0805 | 0.0427 | 0.0144 | 0.0412 | 0.0789 | 0.1166 | 0.1534 | 16.16 |
| one spectral layer: 1 + eta | 0.1276 | 0.0873 | 0.0469 | 0.0164 | 0.0431 | 0.0828 | 0.1226 | 0.1614 | 16.18 |
| one spectral layer: exp(-3 eta) | 0.1910 | 0.1318 | 0.0740 | 0.0315 | 0.0624 | 0.1178 | 0.1745 | 0.2300 | 16.25 |
| one spectral layer: cos, smooth across the wrap | 0.3223 | 0.2416 | 0.1609 | 0.0803 | 0.0000 | 0.0797 | 0.1586 | 0.2365 | 16.99 |
| untrained FNO2d: spectral weights x20 | 0.0006 | 0.0004 | 0.0002 | 0.0001 | 0.0002 | 0.0003 | 0.0005 | 0.0007 | 16.28 |
| untrained FNO2d: spectral weights x50 | 0.0016 | 0.0011 | 0.0006 | 0.0003 | 0.0005 | 0.0009 | 0.0014 | 0.0018 | 16.28 |
| untrained FNO2d: spectral weights x150 | 0.0073 | 0.0050 | 0.0029 | 0.0013 | 0.0023 | 0.0043 | 0.0063 | 0.0084 | 16.28 |
