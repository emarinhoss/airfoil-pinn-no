# FNO transfer ablation

Factors: M channel MLP, P symmetric padding, I spectral initialisation and bias placement, L two-layer lifting and projection (scripts/probes/fno_ablation.py). abl-none is FNO2d, abl-MPIL NeuralOperator's FNO.

## Configurations (mean over seeds)

| configuration | seeds | validation transfer | best validation loss | x floor, 256x97, in range, subcritical | x floor, 256x97, in range, supercritical |
|---|---|---|---|---|---|
| abl-none | 0, 1, 2, 3, 4 |  |  | 5.356 | 2.565 |
| abl-M | 0, 1, 2, 3, 4 |  |  | 5.757 | 2.924 |
| abl-P | 0, 1, 2, 3, 4 |  |  | 2.183 | 1.681 |
| abl-I | 0, 1, 2, 3, 4 |  |  | 4.949 | 2.84 |
| abl-L | 0, 1, 2, 3, 4 |  |  | 4.8 | 2.772 |
| abl-MPIL | 0, 1, 2, 3, 4 |  |  | 2.52 | 1.892 |

## Effects

Add-one: abl-X against abl-none. Remove-one: abl-MPIL without X against abl-MPIL. Effect is the difference in means (configuration with the factor minus without); p is a two-sided exhaustive permutation test over the seeds.

| factor | design | with | without | n | validation transfer effect | p | best validation loss effect | p | x floor, 256x97, in range, subcritical effect | p | x floor, 256x97, in range, supercritical effect | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | add-one | abl-M | abl-none | 5 + 5 |  |  |  |  | +0.4006 | 0.310 | +0.3592 | 0.167 |
| P | add-one | abl-P | abl-none | 5 + 5 |  |  |  |  | -3.173 | 0.008 | -0.884 | 0.008 |
| I | add-one | abl-I | abl-none | 5 + 5 |  |  |  |  | -0.4075 | 0.008 | +0.2749 | 0.381 |
| L | add-one | abl-L | abl-none | 5 + 5 |  |  |  |  | -0.5559 | 0.175 | +0.2069 | 0.270 |

Not yet run: abl-PIL, abl-MIL, abl-MPL, abl-MPI.
