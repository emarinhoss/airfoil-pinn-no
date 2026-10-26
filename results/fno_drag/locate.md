# Where the 16-to-17-row change acts on the FNO's wall pressure

Validation cases. delta = wall Cp at 17 rows minus at 16, per case. Written by `python scripts/probes/probe_fno_drag_locate.py`.

## fno-n896-s0

Leading- and trailing-edge regions (x/c < 0.05 or > 0.95): 29 % of the wall nodes and 75 % of the wall's total |delta| (median over cases).

Median R^2 of delta = a + b * candidate, per case:

| candidate | subcritical | supercritical | all |
|---|---|---|---|
| cp_wall | 0.15 | 0.06 | 0.11 |
| cp_wall_minus_far | 0.15 | 0.06 | 0.11 |
| cp_off_wall | 0.98 | 0.76 | 0.97 |
| cp_along_wall | 0.01 | 0.25 | 0.03 |
| dx_k | 0.16 | 0.12 | 0.15 |
| all five | 0.99 | 0.91 | 0.98 |

Mean |change| from 16 to 17 rows by row j (0 = wall), channel p:

| column | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leading edge | 1.0e-02 | 6.9e-03 | 6.6e-03 | 6.7e-03 | 6.5e-03 | 3.9e-03 | 9.5e-04 | 1.2e-04 | 6.8e-05 | 3.8e-05 | 2.8e-04 |
| mid-chord lower | 7.2e-04 | 7.0e-04 | 6.5e-04 | 4.9e-04 | 7.6e-04 | 5.5e-04 | 5.6e-04 | 3.1e-04 | 1.2e-04 | 4.9e-05 | 1.3e-04 |
| mid-chord upper | 2.5e-03 | 2.4e-03 | 2.4e-03 | 2.4e-03 | 2.4e-03 | 2.4e-03 | 3.3e-03 | 8.4e-04 | 1.2e-04 | 8.4e-05 | 7.0e-04 |
| trailing edge | 3.0e-02 | 7.9e-02 | 1.8e-02 | 3.7e-03 | 2.2e-03 | 1.2e-03 | 4.6e-04 | 7.1e-05 | 7.7e-05 | 4.5e-05 | 3.7e-04 |

## fno-n896-s1

Leading- and trailing-edge regions (x/c < 0.05 or > 0.95): 29 % of the wall nodes and 77 % of the wall's total |delta| (median over cases).

Median R^2 of delta = a + b * candidate, per case:

| candidate | subcritical | supercritical | all |
|---|---|---|---|
| cp_wall | 0.16 | 0.06 | 0.13 |
| cp_wall_minus_far | 0.16 | 0.06 | 0.13 |
| cp_off_wall | 0.98 | 0.68 | 0.98 |
| cp_along_wall | 0.02 | 0.19 | 0.04 |
| dx_k | 0.16 | 0.10 | 0.14 |
| all five | 0.99 | 0.85 | 0.98 |

Mean |change| from 16 to 17 rows by row j (0 = wall), channel p:

| column | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leading edge | 1.0e-02 | 6.8e-03 | 6.7e-03 | 6.9e-03 | 6.3e-03 | 3.8e-03 | 1.2e-03 | 1.3e-04 | 7.8e-05 | 4.0e-05 | 3.1e-04 |
| mid-chord lower | 6.2e-04 | 5.1e-04 | 4.3e-04 | 4.2e-04 | 3.6e-04 | 3.2e-04 | 3.7e-04 | 2.9e-04 | 1.0e-04 | 4.7e-05 | 1.2e-04 |
| mid-chord upper | 2.8e-03 | 2.9e-03 | 2.7e-03 | 2.3e-03 | 1.9e-03 | 2.2e-03 | 3.1e-03 | 1.1e-03 | 1.3e-04 | 2.9e-05 | 6.7e-04 |
| trailing edge | 4.8e-02 | 6.8e-02 | 1.8e-02 | 3.0e-03 | 2.0e-03 | 1.5e-03 | 6.0e-04 | 9.3e-05 | 6.5e-05 | 2.4e-05 | 1.5e-04 |

## fno-n896-s2

Leading- and trailing-edge regions (x/c < 0.05 or > 0.95): 29 % of the wall nodes and 73 % of the wall's total |delta| (median over cases).

Median R^2 of delta = a + b * candidate, per case:

| candidate | subcritical | supercritical | all |
|---|---|---|---|
| cp_wall | 0.12 | 0.07 | 0.09 |
| cp_wall_minus_far | 0.12 | 0.07 | 0.09 |
| cp_off_wall | 0.97 | 0.89 | 0.96 |
| cp_along_wall | 0.01 | 0.07 | 0.03 |
| dx_k | 0.18 | 0.15 | 0.17 |
| all five | 0.99 | 0.94 | 0.98 |

Mean |change| from 16 to 17 rows by row j (0 = wall), channel p:

| column | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leading edge | 9.6e-03 | 6.5e-03 | 6.4e-03 | 7.1e-03 | 6.5e-03 | 4.3e-03 | 1.2e-03 | 1.8e-04 | 4.8e-05 | 6.1e-05 | 2.6e-04 |
| mid-chord lower | 7.1e-04 | 6.2e-04 | 5.3e-04 | 4.2e-04 | 5.7e-04 | 5.1e-04 | 4.6e-04 | 2.8e-04 | 1.4e-04 | 6.9e-05 | 1.1e-04 |
| mid-chord upper | 2.4e-03 | 2.6e-03 | 2.6e-03 | 2.4e-03 | 2.5e-03 | 3.1e-03 | 3.5e-03 | 1.1e-03 | 1.2e-04 | 3.8e-05 | 6.6e-04 |
| trailing edge | 3.4e-02 | 8.2e-02 | 1.6e-02 | 3.5e-03 | 2.5e-03 | 1.4e-03 | 6.6e-04 | 9.9e-05 | 4.8e-05 | 7.4e-05 | 1.4e-04 |

## untrained

Leading- and trailing-edge regions (x/c < 0.05 or > 0.95): 29 % of the wall nodes and 29 % of the wall's total |delta| (median over cases).

Median R^2 of delta = a + b * candidate, per case:

| candidate | subcritical | supercritical | all |
|---|---|---|---|
| cp_wall | 0.30 | 0.32 | 0.31 |
| cp_wall_minus_far | 0.46 | 0.42 | 0.44 |
| cp_off_wall | 0.99 | 0.99 | 0.99 |
| cp_along_wall | 0.01 | 0.01 | 0.01 |
| dx_k | 0.46 | 0.49 | 0.47 |
| all five | 1.00 | 1.00 | 1.00 |

Mean |change| from 16 to 17 rows by row j (0 = wall), channel p:

| column | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leading edge | 4.1e-06 | 4.4e-06 | 4.5e-06 | 4.0e-06 | 3.0e-07 | 4.4e-07 | 3.3e-06 | 3.2e-06 | 1.4e-06 | 2.5e-07 | 5.3e-07 |
| mid-chord lower | 3.7e-06 | 4.0e-06 | 4.0e-06 | 2.8e-06 | 1.3e-06 | 2.0e-06 | 2.0e-06 | 5.1e-06 | 1.4e-07 | 6.4e-07 | 1.8e-07 |
| mid-chord upper | 1.8e-06 | 1.1e-06 | 1.0e-07 | 2.6e-06 | 5.5e-06 | 1.1e-07 | 2.5e-06 | 5.6e-07 | 2.5e-06 | 2.0e-07 | 8.7e-08 |
| trailing edge | 1.1e-06 | 1.1e-06 | 1.0e-06 | 2.8e-07 | 2.2e-06 | 4.6e-06 | 1.4e-06 | 3.6e-06 | 1.5e-06 | 4.5e-07 | 1.1e-06 |

