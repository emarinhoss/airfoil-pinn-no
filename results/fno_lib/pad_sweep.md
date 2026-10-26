# FNO padding sweep at 256x97

Command: `python scripts/probes/probe_fno_pad_sweep.py` (rows 0 to 34). Validation cases only. Each checkpoint is evaluated at 256x97 with each number of radial padding rows, its 128x49 prediction at its own padding (8 rows). Group means over seeds.

## Best row count per group

| group | checkpoints | own rows at 256x97 | val L2 at 128x49 | val L2 at 256x97, own rows | best rows (val L2) | val L2 there | transfer, own rows | best rows (transfer) | transfer there |
|---|---|---|---|---|---|---|---|---|---|
| S2 fno-n896 | 3 | 17 | 0.0124 | 0.0411 | 16 | 0.0411 | 0.0360 | 16 | 0.0209 |
| abl-MPIL-n896 | 5 | 16 | 0.0117 | 0.0420 | 16 | 0.0420 | 0.0281 | 16 | 0.0281 |
| abl-none-n896 | 5 | 17 | 0.0126 | 0.0421 | 16 | 0.0410 | 0.0375 | 16 | 0.0215 |
| neuralop-n896-nopatience | 5 | 16 | 0.0120 | 0.0412 | 16 | 0.0412 | 0.0255 | 16 | 0.0255 |

## Validation L2 at 256x97, group means

| rows | S2 fno-n896 | abl-MPIL-n896 | abl-none-n896 | neuralop-n896-nopatience |
|---|---|---|---|---|
| 0 | 0.7197 | 0.9196 | 0.7196 | 0.9457 |
| 8 | 0.3984 | 0.4920 | 0.3989 | 0.4928 |
| 12 | 0.2294 | 0.2808 | 0.2308 | 0.2758 |
| 14 | 0.1329 | 0.1620 | 0.1335 | 0.1584 |
| 15 | 0.0846 | 0.0998 | 0.0845 | 0.0976 |
| 16 | 0.0411 | 0.0420 | 0.0410 | 0.0412 |
| 17 | 0.0411 | 0.0476 | 0.0421 | 0.0475 |
| 18 | 0.0859 | 0.1027 | 0.0874 | 0.1024 |
| 20 | 0.1635 | 0.2030 | 0.1649 | 0.1987 |
| 24 | 0.2946 | 0.3723 | 0.2962 | 0.3620 |
| 32 | 0.4932 | 0.6114 | 0.4960 | 0.5981 |
| 34 | 0.5319 | 0.6540 | 0.5350 | 0.6413 |

## Transfer (256x97 at every second node against 128x49), group means

| rows | S2 fno-n896 | abl-MPIL-n896 | abl-none-n896 | neuralop-n896-nopatience |
|---|---|---|---|---|
| 0 | 0.7144 | 0.9077 | 0.7146 | 0.9397 |
| 8 | 0.3948 | 0.4875 | 0.3951 | 0.4906 |
| 12 | 0.2249 | 0.2760 | 0.2266 | 0.2717 |
| 14 | 0.1271 | 0.1566 | 0.1278 | 0.1531 |
| 15 | 0.0731 | 0.0912 | 0.0726 | 0.0887 |
| 16 | 0.0209 | 0.0281 | 0.0215 | 0.0255 |
| 17 | 0.0360 | 0.0450 | 0.0375 | 0.0446 |
| 18 | 0.0896 | 0.1063 | 0.0916 | 0.1065 |
| 20 | 0.1658 | 0.2029 | 0.1677 | 0.1995 |
| 24 | 0.2911 | 0.3642 | 0.2935 | 0.3553 |
| 32 | 0.4880 | 0.5993 | 0.4916 | 0.5873 |
| 34 | 0.5271 | 0.6420 | 0.5308 | 0.6306 |
