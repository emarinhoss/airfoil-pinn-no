# The FNO's drag offset at 256x97 against the radial padding rows

Validation cases. Drag offset: CD at 256x97 minus the same model's CD at 128x49, in counts (no truth involved). Written by `python scripts/probes/probe_fno_drag.py`.

## fno-n896-s0

126 validation cases, of which 76 subcritical. CD at 128x49 minus SU2 at 256x97, median: -0.9 counts.

| rows | drag offset, mean (sub / super) | median | near field (sub / super) | transfer |
|---|---|---|---|---|
| 12 | -169.2 / -113.9 | -148.2 | 0.1620 / 0.1360 | 0.2226 |
| 13 | -134.5 / -95.9 | -120.2 | 0.1210 / 0.1034 | 0.1728 |
| 14 | -96.1 / -72.6 | -85.2 | 0.0810 / 0.0727 | 0.1260 |
| 15 | -57.0 / -46.1 | -51.5 | 0.0424 / 0.0454 | 0.0723 |
| 16 | -19.0 / -18.4 | -17.0 | 0.0124 / 0.0307 | 0.0207 |
| 17 | +17.2 / +9.3 | +15.6 | 0.0375 / 0.0478 | 0.0348 |
| 18 | +50.1 / +34.6 | +45.8 | 0.0731 / 0.0758 | 0.0880 |
| 19 | +77.4 / +55.5 | +71.3 | 0.1084 / 0.1058 | 0.1301 |
| 20 | +100.0 / +72.4 | +91.7 | 0.1427 / 0.1363 | 0.1634 |
| 21 | +119.0 / +86.0 | +108.5 | 0.1761 / 0.1667 | 0.1958 |
| mean of 16 and 17 | -0.9 / -4.5 | -1.2 | 0.0206 / 0.0364 | 0.0149 |
| 128x49 (own grid) | 0 | 0 | 0.0056 / 0.0260 | — |

Straight line through the offset over rows 14-19, per case (median [interquartile range]):

- subcritical (76): slope +33.9 [+29.9, +38.0] counts per row; crosses zero at 16.61 [16.57, 16.68] rows
- supercritical (50): slope +24.1 [+18.1, +33.3] counts per row; crosses zero at 16.70 [16.57, 16.97] rows
- all (126): slope +31.9 [+26.2, +37.4] counts per row; crosses zero at 16.63 [16.57, 16.74] rows
- largest residual from the line: 5.10 [3.72, 6.07] counts

Drag of the wall-Cp change from 16 to 17 rows, by region, mean over cases (total +32.8 counts): leading edge +34.8, upper -3.1, lower +1.6, trailing edge -0.5

## fno-n896-s1

126 validation cases, of which 76 subcritical. CD at 128x49 minus SU2 at 256x97, median: -1.0 counts.

| rows | drag offset, mean (sub / super) | median | near field (sub / super) | transfer |
|---|---|---|---|---|
| 12 | -159.5 / -111.2 | -146.7 | 0.1616 / 0.1418 | 0.2276 |
| 13 | -132.0 / -92.4 | -120.8 | 0.1205 / 0.1078 | 0.1779 |
| 14 | -96.7 / -69.9 | -88.8 | 0.0805 / 0.0752 | 0.1287 |
| 15 | -57.6 / -44.8 | -52.5 | 0.0419 / 0.0457 | 0.0731 |
| 16 | -18.1 / -19.2 | -16.9 | 0.0118 / 0.0288 | 0.0206 |
| 17 | +18.8 / +5.3 | +16.8 | 0.0375 / 0.0474 | 0.0355 |
| 18 | +51.9 / +26.8 | +46.4 | 0.0730 / 0.0770 | 0.0884 |
| 19 | +79.4 / +44.2 | +70.9 | 0.1080 / 0.1084 | 0.1309 |
| 20 | +102.0 / +58.2 | +90.1 | 0.1422 / 0.1401 | 0.1649 |
| 21 | +120.9 / +69.8 | +106.9 | 0.1753 / 0.1717 | 0.1972 |
| mean of 16 and 17 | +0.3 / -7.0 | -0.2 | 0.0206 / 0.0350 | 0.0135 |
| 128x49 (own grid) | 0 | 0 | 0.0058 / 0.0269 | — |

Straight line through the offset over rows 14-19, per case (median [interquartile range]):

- subcritical (76): slope +34.4 [+30.5, +38.6] counts per row; crosses zero at 16.62 [16.57, 16.64] rows
- supercritical (50): slope +22.5 [+12.7, +32.8] counts per row; crosses zero at 16.81 [16.63, 17.41] rows
- all (126): slope +32.0 [+26.8, +37.1] counts per row; crosses zero at 16.63 [16.59, 16.73] rows
- largest residual from the line: 5.60 [4.13, 6.87] counts

Drag of the wall-Cp change from 16 to 17 rows, by region, mean over cases (total +32.0 counts): leading edge +34.7, upper -4.4, lower +2.1, trailing edge -0.5

## fno-n896-s2

126 validation cases, of which 76 subcritical. CD at 128x49 minus SU2 at 256x97, median: -0.9 counts.

| rows | drag offset, mean (sub / super) | median | near field (sub / super) | transfer |
|---|---|---|---|---|
| 12 | -161.9 / -134.8 | -148.4 | 0.1650 / 0.1461 | 0.2247 |
| 13 | -132.5 / -111.4 | -121.9 | 0.1233 / 0.1087 | 0.1737 |
| 14 | -96.4 / -82.5 | -88.7 | 0.0826 / 0.0735 | 0.1267 |
| 15 | -57.1 / -50.7 | -51.8 | 0.0433 / 0.0433 | 0.0738 |
| 16 | -18.0 / -18.3 | -16.0 | 0.0135 / 0.0302 | 0.0215 |
| 17 | +18.6 / +12.3 | +17.5 | 0.0392 / 0.0515 | 0.0379 |
| 18 | +52.0 / +39.4 | +48.9 | 0.0759 / 0.0817 | 0.0924 |
| 19 | +79.6 / +61.4 | +76.0 | 0.1123 / 0.1131 | 0.1350 |
| 20 | +102.5 / +79.5 | +96.6 | 0.1478 / 0.1446 | 0.1691 |
| 21 | +121.7 / +94.3 | +113.2 | 0.1823 / 0.1756 | 0.2026 |
| mean of 16 and 17 | +0.3 / -3.0 | +0.3 | 0.0221 / 0.0381 | 0.0183 |
| 128x49 (own grid) | 0 | 0 | 0.0055 / 0.0241 | — |

Straight line through the offset over rows 14-19, per case (median [interquartile range]):

- subcritical (76): slope +34.1 [+30.9, +38.0] counts per row; crosses zero at 16.58 [16.53, 16.66] rows
- supercritical (50): slope +28.7 [+22.0, +36.2] counts per row; crosses zero at 16.70 [16.60, 16.80] rows
- all (126): slope +32.8 [+28.5, +37.7] counts per row; crosses zero at 16.61 [16.53, 16.71] rows
- largest residual from the line: 5.52 [4.44, 6.89] counts

Drag of the wall-Cp change from 16 to 17 rows, by region, mean over cases (total +34.2 counts): leading edge +34.3, upper -1.4, lower +1.7, trailing edge -0.4

