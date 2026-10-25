# The library FNOs, scored as S4 scores its contestants

Command: `python scripts/probes/score_fno_libraries.py` over `runs/predictions_lib`, with S2 references fno-n896-s0, fno-n896-s1, fno-n896-s2 (variant `s2`). A library's variant is the library's name.

## Runs

best_val is loss A on the validation cases at 128x49; transfer is the relative L2 difference between the 256x97 prediction sampled at every second node and the 128x49 one. Wall times are on the device named: GPU wall time on different hardware is not one cost basis.

| tag | source | device | n_cases | n_params | epochs_run | best_epoch | best_val | val_rel_l2_256x97 | transfer_val_rel_l2 | wall_s | s_per_epoch | query_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl-I-n896-s0 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 287 | 0.01265 | 0.04726 | 0.04671 | 287.8 | 0.9593 | 0.001036 |
| abl-I-n896-s1 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 293 | 0.01346 | 0.04003 | 0.03485 | 297.4 | 0.9913 | 0.001154 |
| abl-I-n896-s2 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 294 | 0.01261 | 0.03895 | 0.03382 | 298 | 0.9933 | 0.001176 |
| abl-I-n896-s3 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 292 | 0.01307 | 0.04136 | 0.0371 | 295.4 | 0.9848 | 0.001041 |
| abl-I-n896-s4 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01387 | 0.05038 | 0.04928 | 290.7 | 0.969 | 0.001031 |
| abl-L-n896-s0 | ablate 2.0.0 | cuda | 881 | 3558772 | 300 | 279 | 0.01364 | 0.04627 | 0.04305 | 298.7 | 0.9955 | 0.001084 |
| abl-L-n896-s1 | ablate 2.0.0 | cuda | 881 | 3558772 | 300 | 293 | 0.01867 | 0.04972 | 0.04353 | 297.7 | 0.9924 | 0.001089 |
| abl-L-n896-s2 | ablate 2.0.0 | cuda | 881 | 3558772 | 300 | 294 | 0.01277 | 0.0445 | 0.04243 | 308.4 | 1.028 | 0.00109 |
| abl-L-n896-s3 | ablate 2.0.0 | cuda | 881 | 3558772 | 300 | 293 | 0.01447 | 0.05211 | 0.04836 | 297.9 | 0.9931 | 0.001099 |
| abl-L-n896-s4 | ablate 2.0.0 | cuda | 881 | 3558772 | 300 | 292 | 0.01298 | 0.04804 | 0.04738 | 295.1 | 0.9837 | 0.001083 |
| abl-M-n896-s0 | ablate 2.0.0 | cuda | 881 | 3565172 | 300 | 290 | 0.01267 | 0.04833 | 0.04627 | 376.4 | 1.255 | 0.001349 |
| abl-M-n896-s1 | ablate 2.0.0 | cuda | 881 | 3565172 | 300 | 288 | 0.01271 | 0.05158 | 0.04908 | 375.8 | 1.253 | 0.001344 |
| abl-M-n896-s2 | ablate 2.0.0 | cuda | 881 | 3565172 | 300 | 292 | 0.01253 | 0.04686 | 0.04497 | 378.4 | 1.261 | 0.001348 |
| abl-M-n896-s3 | ablate 2.0.0 | cuda | 881 | 3565172 | 300 | 286 | 0.01294 | 0.05391 | 0.05256 | 391 | 1.303 | 0.001347 |
| abl-M-n896-s4 | ablate 2.0.0 | cuda | 881 | 3565172 | 300 | 289 | 0.01266 | 0.05787 | 0.05889 | 380.5 | 1.268 | 0.001343 |
| abl-MPIL-n896-s0 | ablate 2.0.0 | cuda | 881 | 3568468 | 300 | 294 | 0.01109 | 0.04322 | 0.02987 | 401.1 | 1.337 | 0.002143 |
| abl-MPIL-n896-s1 | ablate 2.0.0 | cuda | 881 | 3568468 | 300 | 294 | 0.01194 | 0.04197 | 0.0298 | 399.1 | 1.33 | 0.001731 |
| abl-MPIL-n896-s2 | ablate 2.0.0 | cuda | 881 | 3568468 | 300 | 291 | 0.01157 | 0.04345 | 0.03013 | 389.5 | 1.298 | 0.001731 |
| abl-MPIL-n896-s3 | ablate 2.0.0 | cuda | 881 | 3568468 | 300 | 288 | 0.01259 | 0.0417 | 0.02559 | 401.3 | 1.338 | 0.001724 |
| abl-MPIL-n896-s4 | ablate 2.0.0 | cuda | 881 | 3568468 | 300 | 293 | 0.01137 | 0.03976 | 0.0249 | 406.2 | 1.354 | 0.001747 |
| abl-P-n896-s0 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 284 | 0.01255 | 0.04143 | 0.02058 | 296.7 | 0.9891 | 0.00137 |
| abl-P-n896-s1 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 293 | 0.01285 | 0.0395 | 0.02046 | 283.8 | 0.9459 | 0.001489 |
| abl-P-n896-s2 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 294 | 0.01185 | 0.04223 | 0.02159 | 295 | 0.9834 | 0.001499 |
| abl-P-n896-s3 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 288 | 0.0126 | 0.04025 | 0.02029 | 283 | 0.9432 | 0.00136 |
| abl-P-n896-s4 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01329 | 0.04162 | 0.02455 | 286 | 0.9532 | 0.001368 |
| abl-none-n896-s0 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 284 | 0.01256 | 0.04064 | 0.03475 | 288.2 | 0.9607 | 0.00188 |
| abl-none-n896-s1 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 285 | 0.01285 | 0.04073 | 0.03549 | 289.1 | 0.9638 | 0.00117 |
| abl-none-n896-s2 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 294 | 0.0119 | 0.04248 | 0.03803 | 284.4 | 0.948 | 0.001036 |
| abl-none-n896-s3 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01264 | 0.03976 | 0.03418 | 283.1 | 0.9438 | 0.001152 |
| abl-none-n896-s4 | ablate 2.0.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01317 | 0.04687 | 0.04511 | 283.9 | 0.9462 | 0.001197 |
| apno-n128-s0 | apno 0.1.0 | cuda | 126 | 3555476 | 300 | 288 | 0.05065 | 0.07419 | 0.04758 | 73.82 | 0.2461 | 0.0009709 |
| apno-n256-s0 | apno 0.1.0 | cuda | 252 | 3555476 | 300 | 284 | 0.03322 | 0.06082 | 0.04744 | 127.8 | 0.4258 | 0.0009648 |
| apno-n512-s0 | apno 0.1.0 | cuda | 502 | 3555476 | 300 | 287 | 0.01781 | 0.04768 | 0.04312 | 238.3 | 0.7943 | 0.0009766 |
| apno-n64-s0 | apno 0.1.0 | cuda | 63 | 3555476 | 300 | 297 | 0.08493 | 0.1015 | 0.04425 | 50.15 | 0.1672 | 0.0009833 |
| apno-n896-s0 | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 284 | 0.01256 | 0.0405 | 0.03456 | 409.9 | 1.366 | 0.001097 |
| apno-n896-s1 | apno 0.1.0 | cuda | 881 | 3555476 | 191 | 160 | 0.02035 | 0.04464 | 0.03547 | 260.8 | 1.365 | 0.001097 |
| apno-n896-s1-nopatience | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 293 | 0.01264 | 0.04059 | 0.03551 | 408.4 | 1.361 | 0.001129 |
| apno-n896-s2 | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 294 | 0.01169 | 0.04238 | 0.03808 | 408.6 | 1.362 | 0.0009714 |
| apno-n896-s3-nopatience | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01265 | 0.03961 | 0.03389 | 407.2 | 1.357 | 0.001612 |
| apno-n896-s4-nopatience | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 288 | 0.01324 | 0.04681 | 0.04511 | 408.6 | 1.362 | 0.000977 |
| neuralop-n896-s0 | neuralop 2.0.0 | cuda | 881 | 3568468 | 183 | 152 | 0.0245 | 0.05083 | 0.02594 | 243.3 | 1.33 | 0.002119 |
| neuralop-n896-s0-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 294 | 0.01225 | 0.04222 | 0.0271 | 398.1 | 1.327 | 0.002333 |
| neuralop-n896-s1-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 299 | 0.01219 | 0.04062 | 0.0271 | 400.5 | 1.335 | 0.002691 |
| neuralop-n896-s2-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 289 | 0.01161 | 0.04051 | 0.02562 | 396.9 | 1.323 | 0.001731 |
| neuralop-n896-s3-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 299 | 0.01248 | 0.04121 | 0.02546 | 406.6 | 1.355 | 0.001733 |
| neuralop-n896-s4-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 290 | 0.01162 | 0.0413 | 0.02215 | 400 | 1.333 | 0.001737 |
| physicsnemo-n896-s0 | physicsnemo 2.2.2 | cuda | 881 | 7095452 | 300 | 282 | 0.02432 | 0.05583 | 0.04681 | 424.4 | 1.415 | 0.00119 |
| fno-n896-s0 | S2 FNO2d | mps | 881 |  | 300 | 278 | 0.0126 |  | 0.03477 | 2160 | 7.2 | 0.0536 |
| fno-n896-s1 | S2 FNO2d | mps | 881 |  | 300 | 293 | 0.01274 |  | 0.03548 | 2242 | 7.474 | 0.01032 |
| fno-n896-s2 | S2 FNO2d | mps | 881 |  | 300 | 294 | 0.01178 |  | 0.03788 | 2237 | 7.457 | 0.01077 |

## Near field against the floor

x_floor is the contestant's mean near-field relative L2 over the floor's (SU2 at 256x97) on the same split and regime. The floor exists at 256x97 only, so 128x49 rows are divided by it too.

| method | variant | grid | n_train | seed | split | regime | n_ok | near_field_rel_l2_mean | floor_mean | x_floor |
|---|---|---|---|---|---|---|---|---|---|---|
| fno | abl-I | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2676 | 0.06511 | 4.111 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01096 | 0.007953 | 1.378 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.0396 | 0.02413 | 1.641 |
| fno | abl-I | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2592 | 0.06511 | 3.981 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01117 | 0.007953 | 1.404 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.04032 | 0.02413 | 1.671 |
| fno | abl-I | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2651 | 0.06511 | 4.072 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01103 | 0.007953 | 1.387 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03884 | 0.02413 | 1.61 |
| fno | abl-I | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2625 | 0.06511 | 4.033 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01089 | 0.007953 | 1.369 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03957 | 0.02413 | 1.64 |
| fno | abl-I | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2752 | 0.06511 | 4.227 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01119 | 0.007953 | 1.407 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.0413 | 0.02413 | 1.711 |
| fno | abl-I | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2535 | 0.06511 | 3.894 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.03867 | 0.007953 | 4.862 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.07287 | 0.02413 | 3.02 |
| fno | abl-I | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2626 | 0.06511 | 4.034 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.04076 | 0.007953 | 5.125 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06182 | 0.02413 | 2.562 |
| fno | abl-I | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2596 | 0.06511 | 3.987 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.04014 | 0.007953 | 5.047 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.05743 | 0.02413 | 2.38 |
| fno | abl-I | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2591 | 0.06511 | 3.98 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.03834 | 0.007953 | 4.821 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.06007 | 0.02413 | 2.489 |
| fno | abl-I | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2571 | 0.06511 | 3.949 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.03889 | 0.007953 | 4.89 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.09047 | 0.02413 | 3.749 |
| fno | abl-L | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2734 | 0.06511 | 4.199 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01139 | 0.007953 | 1.432 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.04131 | 0.02413 | 1.712 |
| fno | abl-L | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2882 | 0.06511 | 4.426 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01131 | 0.007953 | 1.422 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.05464 | 0.02413 | 2.264 |
| fno | abl-L | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2662 | 0.06511 | 4.089 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01104 | 0.007953 | 1.388 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.04078 | 0.02413 | 1.69 |
| fno | abl-L | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2712 | 0.06511 | 4.166 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01123 | 0.007953 | 1.412 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.0441 | 0.02413 | 1.827 |
| fno | abl-L | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2719 | 0.06511 | 4.176 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01088 | 0.007953 | 1.368 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.04248 | 0.02413 | 1.76 |
| fno | abl-L | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2485 | 0.06511 | 3.816 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.03873 | 0.007953 | 4.87 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.06458 | 0.02413 | 2.676 |
| fno | abl-L | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.3021 | 0.06511 | 4.64 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.03418 | 0.007953 | 4.297 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06996 | 0.02413 | 2.899 |
| fno | abl-L | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2524 | 0.06511 | 3.877 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.03629 | 0.007953 | 4.562 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.05637 | 0.02413 | 2.336 |
| fno | abl-L | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2579 | 0.06511 | 3.961 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.04898 | 0.007953 | 6.158 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.0707 | 0.02413 | 2.93 |
| fno | abl-L | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2563 | 0.06511 | 3.936 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.03271 | 0.007953 | 4.113 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.07283 | 0.02413 | 3.018 |
| fno | abl-M | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.3114 | 0.06511 | 4.783 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01064 | 0.007953 | 1.338 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03945 | 0.02413 | 1.635 |
| fno | abl-M | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2746 | 0.06511 | 4.218 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01092 | 0.007953 | 1.374 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.03934 | 0.02413 | 1.63 |
| fno | abl-M | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2805 | 0.06511 | 4.309 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.0105 | 0.007953 | 1.32 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03841 | 0.02413 | 1.592 |
| fno | abl-M | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2718 | 0.06511 | 4.175 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01062 | 0.007953 | 1.335 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03891 | 0.02413 | 1.612 |
| fno | abl-M | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.3014 | 0.06511 | 4.63 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01052 | 0.007953 | 1.323 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.03913 | 0.02413 | 1.621 |
| fno | abl-M | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.319 | 0.06511 | 4.9 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.04382 | 0.007953 | 5.509 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.06459 | 0.02413 | 2.676 |
| fno | abl-M | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2793 | 0.06511 | 4.29 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.05246 | 0.007953 | 6.596 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06567 | 0.02413 | 2.721 |
| fno | abl-M | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2726 | 0.06511 | 4.187 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.04299 | 0.007953 | 5.406 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.06464 | 0.02413 | 2.679 |
| fno | abl-M | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.268 | 0.06511 | 4.116 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.05048 | 0.007953 | 6.347 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.06718 | 0.02413 | 2.784 |
| fno | abl-M | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2903 | 0.06511 | 4.459 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.03918 | 0.007953 | 4.926 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.09075 | 0.02413 | 3.76 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2626 | 0.06511 | 4.033 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01139 | 0.007953 | 1.432 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.0394 | 0.02413 | 1.633 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2688 | 0.06511 | 4.129 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01196 | 0.007953 | 1.504 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.03838 | 0.02413 | 1.591 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2571 | 0.06511 | 3.949 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01205 | 0.007953 | 1.516 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03899 | 0.02413 | 1.616 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2584 | 0.06511 | 3.969 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01174 | 0.007953 | 1.477 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03861 | 0.02413 | 1.6 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2842 | 0.06511 | 4.366 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01111 | 0.007953 | 1.397 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.04061 | 0.02413 | 1.683 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2551 | 0.06511 | 3.918 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.01715 | 0.007953 | 2.157 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.05014 | 0.02413 | 2.078 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2582 | 0.06511 | 3.966 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.02144 | 0.007953 | 2.696 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.04582 | 0.02413 | 1.899 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.246 | 0.06511 | 3.779 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.02435 | 0.007953 | 3.062 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.04872 | 0.02413 | 2.019 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2498 | 0.06511 | 3.837 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.01958 | 0.007953 | 2.461 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.04168 | 0.02413 | 1.727 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2763 | 0.06511 | 4.244 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.0177 | 0.007953 | 2.226 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.04194 | 0.02413 | 1.738 |
| fno | abl-P | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2347 | 0.06511 | 3.605 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.0105 | 0.007953 | 1.321 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03763 | 0.02413 | 1.559 |
| fno | abl-P | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2504 | 0.06511 | 3.846 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01033 | 0.007953 | 1.299 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.03942 | 0.02413 | 1.633 |
| fno | abl-P | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2464 | 0.06511 | 3.784 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01011 | 0.007953 | 1.271 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03923 | 0.02413 | 1.625 |
| fno | abl-P | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2804 | 0.06511 | 4.307 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01031 | 0.007953 | 1.297 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.0375 | 0.02413 | 1.554 |
| fno | abl-P | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2651 | 0.06511 | 4.071 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01028 | 0.007953 | 1.293 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.03872 | 0.02413 | 1.605 |
| fno | abl-P | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2264 | 0.06511 | 3.477 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.01729 | 0.007953 | 2.173 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.03914 | 0.02413 | 1.622 |
| fno | abl-P | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2452 | 0.06511 | 3.767 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.01599 | 0.007953 | 2.01 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.04057 | 0.02413 | 1.681 |
| fno | abl-P | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2362 | 0.06511 | 3.628 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.01798 | 0.007953 | 2.261 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.04132 | 0.02413 | 1.712 |
| fno | abl-P | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2763 | 0.06511 | 4.243 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.01672 | 0.007953 | 2.102 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.03959 | 0.02413 | 1.64 |
| fno | abl-P | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2576 | 0.06511 | 3.957 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.01884 | 0.007953 | 2.369 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.04219 | 0.02413 | 1.748 |
| fno | abl-none | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.236 | 0.06511 | 3.625 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01049 | 0.007953 | 1.319 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03782 | 0.02413 | 1.567 |
| fno | abl-none | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2522 | 0.06511 | 3.873 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01025 | 0.007953 | 1.288 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.03931 | 0.02413 | 1.629 |
| fno | abl-none | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2444 | 0.06511 | 3.754 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01013 | 0.007953 | 1.274 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03897 | 0.02413 | 1.615 |
| fno | abl-none | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2826 | 0.06511 | 4.341 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01032 | 0.007953 | 1.297 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03753 | 0.02413 | 1.555 |
| fno | abl-none | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2635 | 0.06511 | 4.047 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01029 | 0.007953 | 1.294 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.03874 | 0.02413 | 1.605 |
| fno | abl-none | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2356 | 0.06511 | 3.619 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.04309 | 0.007953 | 5.418 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.05809 | 0.02413 | 2.407 |
| fno | abl-none | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2601 | 0.06511 | 3.994 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.04203 | 0.007953 | 5.285 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.0606 | 0.02413 | 2.511 |
| fno | abl-none | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2433 | 0.06511 | 3.737 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.04372 | 0.007953 | 5.497 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.06125 | 0.02413 | 2.538 |
| fno | abl-none | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2887 | 0.06511 | 4.435 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.04142 | 0.007953 | 5.209 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.05651 | 0.02413 | 2.341 |
| fno | abl-none | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2659 | 0.06511 | 4.084 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.04273 | 0.007953 | 5.372 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.07302 | 0.02413 | 3.026 |
| fno | apno | 128x49 | 128 | 0 | test_ext | supercritical | 14 | 0.3999 | 0.06511 | 6.142 |
| fno | apno | 128x49 | 128 | 0 | test_in | subcritical | 37 | 0.02184 | 0.007953 | 2.746 |
| fno | apno | 128x49 | 128 | 0 | test_in | supercritical | 24 | 0.1189 | 0.02413 | 4.927 |
| fno | apno | 128x49 | 256 | 0 | test_ext | supercritical | 14 | 0.325 | 0.06511 | 4.992 |
| fno | apno | 128x49 | 256 | 0 | test_in | subcritical | 37 | 0.01566 | 0.007953 | 1.969 |
| fno | apno | 128x49 | 256 | 0 | test_in | supercritical | 24 | 0.08212 | 0.02413 | 3.403 |
| fno | apno | 128x49 | 512 | 0 | test_ext | supercritical | 14 | 0.2865 | 0.06511 | 4.4 |
| fno | apno | 128x49 | 512 | 0 | test_in | subcritical | 37 | 0.01188 | 0.007953 | 1.493 |
| fno | apno | 128x49 | 512 | 0 | test_in | supercritical | 24 | 0.04844 | 0.02413 | 2.007 |
| fno | apno | 128x49 | 64 | 0 | test_ext | supercritical | 14 | 0.5531 | 0.06511 | 8.495 |
| fno | apno | 128x49 | 64 | 0 | test_in | subcritical | 37 | 0.04517 | 0.007953 | 5.679 |
| fno | apno | 128x49 | 64 | 0 | test_in | supercritical | 24 | 0.1591 | 0.02413 | 6.592 |
| fno | apno | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.233 | 0.06511 | 3.578 |
| fno | apno | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01053 | 0.007953 | 1.325 |
| fno | apno | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03755 | 0.02413 | 1.556 |
| fno | apno | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2568 | 0.06511 | 3.945 |
| fno | apno | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01907 | 0.007953 | 2.398 |
| fno | apno | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.04738 | 0.02413 | 1.963 |
| fno | apno | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2412 | 0.06511 | 3.704 |
| fno | apno | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01009 | 0.007953 | 1.269 |
| fno | apno | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03845 | 0.02413 | 1.593 |
| fno | apno | 256x97 | 128 | 0 | test_ext | supercritical | 14 | 0.4013 | 0.06511 | 6.163 |
| fno | apno | 256x97 | 128 | 0 | test_in | subcritical | 37 | 0.05286 | 0.007953 | 6.646 |
| fno | apno | 256x97 | 128 | 0 | test_in | supercritical | 24 | 0.1364 | 0.02413 | 5.654 |
| fno | apno | 256x97 | 256 | 0 | test_ext | supercritical | 14 | 0.3196 | 0.06511 | 4.908 |
| fno | apno | 256x97 | 256 | 0 | test_in | subcritical | 37 | 0.04827 | 0.007953 | 6.069 |
| fno | apno | 256x97 | 256 | 0 | test_in | supercritical | 24 | 0.1034 | 0.02413 | 4.284 |
| fno | apno | 256x97 | 512 | 0 | test_ext | supercritical | 14 | 0.2834 | 0.06511 | 4.353 |
| fno | apno | 256x97 | 512 | 0 | test_in | subcritical | 37 | 0.04682 | 0.007953 | 5.887 |
| fno | apno | 256x97 | 512 | 0 | test_in | supercritical | 24 | 0.0697 | 0.02413 | 2.888 |
| fno | apno | 256x97 | 64 | 0 | test_ext | supercritical | 14 | 0.5492 | 0.06511 | 8.435 |
| fno | apno | 256x97 | 64 | 0 | test_in | subcritical | 37 | 0.06738 | 0.007953 | 8.472 |
| fno | apno | 256x97 | 64 | 0 | test_in | supercritical | 24 | 0.1726 | 0.02413 | 7.153 |
| fno | apno | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2323 | 0.06511 | 3.568 |
| fno | apno | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.0428 | 0.007953 | 5.382 |
| fno | apno | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.05783 | 0.02413 | 2.396 |
| fno | apno | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2629 | 0.06511 | 4.039 |
| fno | apno | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.05067 | 0.007953 | 6.372 |
| fno | apno | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06433 | 0.02413 | 2.666 |
| fno | apno | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2396 | 0.06511 | 3.681 |
| fno | apno | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.04374 | 0.007953 | 5.5 |
| fno | apno | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.0614 | 0.02413 | 2.544 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2491 | 0.06511 | 3.826 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01032 | 0.007953 | 1.298 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.03977 | 0.02413 | 1.648 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2835 | 0.06511 | 4.355 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01024 | 0.007953 | 1.288 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03721 | 0.02413 | 1.542 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2648 | 0.06511 | 4.067 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01027 | 0.007953 | 1.291 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.0388 | 0.02413 | 1.608 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2563 | 0.06511 | 3.936 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.04217 | 0.007953 | 5.302 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06093 | 0.02413 | 2.525 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2894 | 0.06511 | 4.445 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.04137 | 0.007953 | 5.202 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.05614 | 0.02413 | 2.326 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2661 | 0.06511 | 4.088 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.04239 | 0.007953 | 5.33 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.07294 | 0.02413 | 3.023 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2733 | 0.06511 | 4.197 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01888 | 0.007953 | 2.374 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.05578 | 0.02413 | 2.311 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2667 | 0.06511 | 4.096 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.02853 | 0.007953 | 3.587 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.06584 | 0.02413 | 2.728 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2396 | 0.06511 | 3.681 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01171 | 0.007953 | 1.472 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03943 | 0.02413 | 1.634 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2551 | 0.06511 | 3.919 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01149 | 0.007953 | 1.445 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.04175 | 0.02413 | 1.73 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2307 | 0.06511 | 3.544 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01122 | 0.007953 | 1.411 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.03902 | 0.02413 | 1.617 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0.2887 | 0.06511 | 4.434 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0.01217 | 0.007953 | 1.53 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0.03987 | 0.02413 | 1.652 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0.2491 | 0.06511 | 3.826 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0.01104 | 0.007953 | 1.388 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0.03688 | 0.02413 | 1.528 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2338 | 0.06511 | 3.592 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.01796 | 0.007953 | 2.258 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.04445 | 0.02413 | 1.842 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2455 | 0.06511 | 3.771 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.01771 | 0.007953 | 2.226 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.04383 | 0.02413 | 1.816 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2243 | 0.06511 | 3.445 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.01848 | 0.007953 | 2.323 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.04304 | 0.02413 | 1.784 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0.2774 | 0.06511 | 4.261 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0.02458 | 0.007953 | 3.09 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0.04449 | 0.02413 | 1.843 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0.2356 | 0.06511 | 3.618 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0.01545 | 0.007953 | 1.942 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0.03613 | 0.02413 | 1.497 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.3019 | 0.06511 | 4.637 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01262 | 0.007953 | 1.587 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.06405 | 0.02413 | 2.654 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.3007 | 0.06511 | 4.619 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.04399 | 0.007953 | 5.531 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.08331 | 0.02413 | 3.452 |
| fno | s2 | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.232 | 0.06511 | 3.563 |
| fno | s2 | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01017 | 0.007953 | 1.278 |
| fno | s2 | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.0374 | 0.02413 | 1.55 |
| fno | s2 | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0.2482 | 0.06511 | 3.812 |
| fno | s2 | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0.01033 | 0.007953 | 1.299 |
| fno | s2 | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0.0398 | 0.02413 | 1.649 |
| fno | s2 | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0.2436 | 0.06511 | 3.742 |
| fno | s2 | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0.01007 | 0.007953 | 1.266 |
| fno | s2 | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0.0382 | 0.02413 | 1.583 |
| fno | s2 | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2318 | 0.06511 | 3.56 |
| fno | s2 | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.04233 | 0.007953 | 5.323 |
| fno | s2 | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.0575 | 0.02413 | 2.383 |
| fno | s2 | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2554 | 0.06511 | 3.923 |
| fno | s2 | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.04203 | 0.007953 | 5.285 |
| fno | s2 | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06079 | 0.02413 | 2.519 |
| fno | s2 | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0.2427 | 0.06511 | 3.728 |
| fno | s2 | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0.04373 | 0.007953 | 5.498 |
| fno | s2 | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0.0613 | 0.02413 | 2.54 |

## Forces, surface and near field

| method | variant | grid | n_train | seed | split | regime | n_ok | n_missing | n_non-finite | near_field_rel_l2_mean | cl_err_absmax | cd_err_counts_absmax | cp_max_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fno | abl-I | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2676 | 0.2288 | 180.8 | 1.162 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01096 | 0.008554 | 7.935 | 0.2435 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.0396 | 0.03299 | 46.82 | 0.39 |
| fno | abl-I | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2592 | 0.1265 | 108.5 | 1.145 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01117 | 0.008525 | 12.13 | 0.244 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04032 | 0.03355 | 41.02 | 0.4127 |
| fno | abl-I | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2651 | 0.1512 | 156.1 | 1.205 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01103 | 0.008865 | 9.585 | 0.245 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03884 | 0.02687 | 36.59 | 0.3992 |
| fno | abl-I | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2625 | 0.1954 | 167.4 | 1.159 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01089 | 0.008003 | 10.05 | 0.2445 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03957 | 0.02209 | 39.45 | 0.399 |
| fno | abl-I | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2752 | 0.1886 | 154.7 | 1.275 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01119 | 0.01008 | 8.405 | 0.2433 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.0413 | 0.02433 | 27.87 | 0.4004 |
| fno | abl-I | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2535 | 0.2168 | 178.4 | 1.143 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.03867 | 0.02206 | 31.17 | 0.2557 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.07287 | 0.06263 | 65.28 | 0.4323 |
| fno | abl-I | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2626 | 0.1173 | 116.1 | 1.209 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.04076 | 0.02369 | 28.87 | 0.2391 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06182 | 0.04925 | 43.44 | 0.5249 |
| fno | abl-I | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2596 | 0.1458 | 158.1 | 1.238 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.04014 | 0.02744 | 32.76 | 0.2223 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.05743 | 0.03084 | 47.88 | 0.4915 |
| fno | abl-I | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2591 | 0.1747 | 147.1 | 1.24 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.03834 | 0.02652 | 30.58 | 0.2259 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.06007 | 0.0407 | 37.19 | 0.5888 |
| fno | abl-I | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2571 | 0.1802 | 122.8 | 1.212 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.03889 | 0.03145 | 25.93 | 0.2094 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.09047 | 0.03395 | 44.68 | 0.4304 |
| fno | abl-L | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2734 | 0.2204 | 182.6 | 1.209 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01139 | 0.009205 | 7.437 | 0.2446 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.04131 | 0.03016 | 26.52 | 0.365 |
| fno | abl-L | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2882 | 0.1377 | 239.8 | 1.096 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01131 | 0.009897 | 9.923 | 0.2443 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.05464 | 0.03456 | 43.78 | 0.4584 |
| fno | abl-L | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2662 | 0.2341 | 264.5 | 1.124 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01104 | 0.009518 | 9.024 | 0.2443 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.04078 | 0.03271 | 41.06 | 0.3634 |
| fno | abl-L | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2712 | 0.2254 | 264.1 | 1.158 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01123 | 0.01075 | 10.49 | 0.2428 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.0441 | 0.01837 | 55.03 | 0.3876 |
| fno | abl-L | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2719 | 0.2294 | 231.5 | 1.223 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01088 | 0.008653 | 9.916 | 0.2442 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.04248 | 0.03116 | 45.13 | 0.3799 |
| fno | abl-L | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2485 | 0.2038 | 182.6 | 1.241 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.03873 | 0.0236 | 31.11 | 0.2524 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.06458 | 0.04187 | 29.64 | 0.4282 |
| fno | abl-L | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.3021 | 0.1408 | 253.5 | 1.19 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.03418 | 0.02073 | 68.63 | 0.2624 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06996 | 0.07113 | 73.87 | 0.5483 |
| fno | abl-L | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2524 | 0.2283 | 300.5 | 1.133 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.03629 | 0.01966 | 29.32 | 0.2866 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.05637 | 0.04208 | 43.13 | 0.3935 |
| fno | abl-L | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2579 | 0.233 | 276 | 1.146 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.04898 | 0.01969 | 29.67 | 0.2389 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.0707 | 0.03721 | 49.85 | 0.4619 |
| fno | abl-L | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2563 | 0.2191 | 237.9 | 1.221 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.03271 | 0.01417 | 34.16 | 0.2749 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.07283 | 0.06663 | 61.82 | 0.4041 |
| fno | abl-M | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.3114 | 0.2089 | 240.9 | 1.337 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01064 | 0.0115 | 6.176 | 0.2448 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03945 | 0.02378 | 32.5 | 0.3543 |
| fno | abl-M | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2746 | 0.1907 | 273.9 | 1.171 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01092 | 0.01056 | 7.317 | 0.2441 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.03934 | 0.03096 | 42.76 | 0.3586 |
| fno | abl-M | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2805 | 0.1779 | 321.4 | 1.271 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.0105 | 0.009248 | 9.088 | 0.245 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03841 | 0.02352 | 26.61 | 0.3472 |
| fno | abl-M | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2718 | 0.1573 | 281.1 | 1.342 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01062 | 0.009896 | 6.863 | 0.2454 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03891 | 0.02743 | 32.46 | 0.3429 |
| fno | abl-M | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.3014 | 0.2249 | 253.4 | 1.343 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01052 | 0.009358 | 14.26 | 0.2445 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.03913 | 0.02681 | 45.02 | 0.3308 |
| fno | abl-M | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.319 | 0.194 | 242.1 | 1.371 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04382 | 0.03462 | 34.05 | 0.2523 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.06459 | 0.0307 | 63.99 | 0.3769 |
| fno | abl-M | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2793 | 0.2035 | 274.6 | 1.137 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.05246 | 0.02259 | 37.1 | 0.2507 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06567 | 0.02211 | 76.7 | 0.4232 |
| fno | abl-M | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2726 | 0.1663 | 316.4 | 1.255 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.04299 | 0.03068 | 43.31 | 0.2514 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.06464 | 0.03427 | 46.28 | 0.3943 |
| fno | abl-M | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.268 | 0.1503 | 271.6 | 1.335 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.05048 | 0.0282 | 35.76 | 0.2466 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.06718 | 0.02305 | 48.3 | 0.499 |
| fno | abl-M | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2903 | 0.2431 | 263.9 | 1.21 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.03918 | 0.01299 | 32.83 | 0.2321 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.09075 | 0.01458 | 57.04 | 0.6863 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2626 | 0.1741 | 180.1 | 1.122 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01139 | 0.011 | 7.572 | 0.246 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.0394 | 0.0295 | 52.58 | 0.3512 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2688 | 0.1989 | 209.9 | 1.103 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01196 | 0.01254 | 7.135 | 0.2445 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.03838 | 0.0334 | 29.53 | 0.3303 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2571 | 0.1965 | 187 | 1.176 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01205 | 0.01418 | 10.77 | 0.2445 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03899 | 0.02466 | 40.24 | 0.3369 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2584 | 0.2119 | 162.3 | 1.063 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01174 | 0.01065 | 6.399 | 0.2456 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03861 | 0.02979 | 42.05 | 0.337 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2842 | 0.1754 | 211.1 | 1.15 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01111 | 0.009598 | 11.02 | 0.2448 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.04061 | 0.03147 | 48.79 | 0.3467 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2551 | 0.1785 | 190.5 | 1.128 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01715 | 0.01304 | 33.39 | 0.3785 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.05014 | 0.02125 | 51.54 | 0.4375 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2582 | 0.2004 | 246.1 | 1.14 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.02144 | 0.02088 | 25.93 | 0.41 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04582 | 0.02508 | 35.85 | 0.4404 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.246 | 0.1978 | 208.1 | 1.188 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.02435 | 0.02186 | 26.85 | 0.3958 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.04872 | 0.02936 | 41.48 | 0.4377 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2498 | 0.2152 | 176.4 | 1.127 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01958 | 0.01281 | 40.48 | 0.3589 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.04168 | 0.02958 | 63.84 | 0.4292 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2763 | 0.1738 | 236.2 | 1.207 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.0177 | 0.01288 | 20.4 | 0.4307 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.04194 | 0.03237 | 60.47 | 0.4424 |
| fno | abl-P | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2347 | 0.1795 | 168 | 1.232 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.0105 | 0.00983 | 7.939 | 0.2449 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03763 | 0.03838 | 55.38 | 0.396 |
| fno | abl-P | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2504 | 0.2417 | 182.5 | 1.217 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01033 | 0.00797 | 11.79 | 0.2443 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.03942 | 0.0294 | 32.5 | 0.3774 |
| fno | abl-P | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2464 | 0.1838 | 152.5 | 1.25 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01011 | 0.007835 | 10.29 | 0.2451 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03923 | 0.03368 | 42.71 | 0.3836 |
| fno | abl-P | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2804 | 0.1689 | 190.7 | 1.247 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01031 | 0.009936 | 7.577 | 0.244 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.0375 | 0.03578 | 36.62 | 0.4001 |
| fno | abl-P | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2651 | 0.2139 | 205 | 1.251 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01028 | 0.008322 | 13.81 | 0.244 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.03872 | 0.03309 | 56.36 | 0.4054 |
| fno | abl-P | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2264 | 0.1739 | 194 | 1.232 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01729 | 0.01879 | 42.29 | 0.3482 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03914 | 0.04674 | 72.25 | 0.4573 |
| fno | abl-P | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2452 | 0.239 | 194.6 | 1.234 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01599 | 0.01263 | 43.13 | 0.3793 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04057 | 0.03607 | 46.43 | 0.4591 |
| fno | abl-P | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2362 | 0.1744 | 167.5 | 1.264 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01798 | 0.0164 | 41.89 | 0.322 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.04132 | 0.04368 | 60.87 | 0.4408 |
| fno | abl-P | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2763 | 0.1656 | 213.7 | 1.26 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01672 | 0.01942 | 43.26 | 0.3551 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03959 | 0.03944 | 48.55 | 0.4587 |
| fno | abl-P | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2576 | 0.2123 | 218.4 | 1.256 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01884 | 0.01535 | 46.57 | 0.3683 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.04219 | 0.03739 | 73.58 | 0.4633 |
| fno | abl-none | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.236 | 0.1811 | 171.5 | 1.242 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01049 | 0.009688 | 7.561 | 0.2449 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03782 | 0.03814 | 55.08 | 0.3981 |
| fno | abl-none | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2522 | 0.2393 | 181.8 | 1.221 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01025 | 0.007689 | 12.33 | 0.2447 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.03931 | 0.02907 | 29.33 | 0.3772 |
| fno | abl-none | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2444 | 0.1896 | 155.2 | 1.25 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01013 | 0.007965 | 9.562 | 0.2451 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03897 | 0.03697 | 44.11 | 0.3834 |
| fno | abl-none | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2826 | 0.1726 | 210.5 | 1.253 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01032 | 0.009739 | 7.705 | 0.2439 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03753 | 0.03145 | 33.11 | 0.4003 |
| fno | abl-none | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2635 | 0.2147 | 205.4 | 1.247 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01029 | 0.008656 | 13.1 | 0.244 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.03874 | 0.03102 | 52.7 | 0.4092 |
| fno | abl-none | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2356 | 0.1768 | 176.1 | 1.302 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04309 | 0.0311 | 31.45 | 0.2435 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.05809 | 0.05198 | 52.64 | 0.5171 |
| fno | abl-none | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2601 | 0.2429 | 171.5 | 1.289 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.04203 | 0.0278 | 36.6 | 0.2366 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.0606 | 0.04533 | 34.88 | 0.535 |
| fno | abl-none | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2433 | 0.183 | 149.7 | 1.265 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.04372 | 0.02866 | 34.81 | 0.2383 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.06125 | 0.04939 | 44.68 | 0.4206 |
| fno | abl-none | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2887 | 0.168 | 211.2 | 1.306 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.04142 | 0.03479 | 22.96 | 0.2472 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.05651 | 0.03434 | 27.17 | 0.4723 |
| fno | abl-none | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2659 | 0.2129 | 190.3 | 1.257 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.04273 | 0.03108 | 28.39 | 0.2507 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.07302 | 0.03209 | 69.21 | 0.4738 |
| fno | apno | 128x49 | 128 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.3999 | 0.253 | 341.3 | 1.235 |
| fno | apno | 128x49 | 128 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.02184 | 0.02748 | 22.76 | 0.2442 |
| fno | apno | 128x49 | 128 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.1189 | 0.08295 | 107.8 | 0.684 |
| fno | apno | 128x49 | 256 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.325 | 0.2379 | 293.5 | 1.22 |
| fno | apno | 128x49 | 256 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01566 | 0.02077 | 13.92 | 0.2414 |
| fno | apno | 128x49 | 256 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.08212 | 0.06765 | 115.5 | 0.5871 |
| fno | apno | 128x49 | 512 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2865 | 0.1542 | 221.7 | 1.262 |
| fno | apno | 128x49 | 512 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01188 | 0.01504 | 5.202 | 0.244 |
| fno | apno | 128x49 | 512 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.04844 | 0.06434 | 102.7 | 0.4818 |
| fno | apno | 128x49 | 64 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.5531 | 0.3468 | 465 | 1.398 |
| fno | apno | 128x49 | 64 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04517 | 0.04215 | 78.95 | 0.2672 |
| fno | apno | 128x49 | 64 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.1591 | 0.1281 | 166.5 | 0.6749 |
| fno | apno | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.233 | 0.1729 | 181.8 | 1.235 |
| fno | apno | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01053 | 0.009377 | 7.631 | 0.2447 |
| fno | apno | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03755 | 0.03738 | 53.9 | 0.3995 |
| fno | apno | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2568 | 0.2419 | 184 | 1.185 |
| fno | apno | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01907 | 0.007675 | 21.82 | 0.2384 |
| fno | apno | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04738 | 0.02721 | 34.42 | 0.4235 |
| fno | apno | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2412 | 0.1798 | 164 | 1.25 |
| fno | apno | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01009 | 0.007955 | 10.99 | 0.2454 |
| fno | apno | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03845 | 0.03297 | 36.41 | 0.3815 |
| fno | apno | 256x97 | 128 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.4013 | 0.2518 | 319.6 | 1.252 |
| fno | apno | 256x97 | 128 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.05286 | 0.04629 | 44.04 | 0.3191 |
| fno | apno | 256x97 | 128 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.1364 | 0.08346 | 102.7 | 0.7706 |
| fno | apno | 256x97 | 256 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.3196 | 0.2227 | 292.5 | 1.247 |
| fno | apno | 256x97 | 256 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04827 | 0.04177 | 32.53 | 0.297 |
| fno | apno | 256x97 | 256 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.1034 | 0.07039 | 125.5 | 0.7304 |
| fno | apno | 256x97 | 512 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2834 | 0.1516 | 227.3 | 1.293 |
| fno | apno | 256x97 | 512 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04682 | 0.03751 | 26.88 | 0.2686 |
| fno | apno | 256x97 | 512 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.0697 | 0.05377 | 105.3 | 0.5925 |
| fno | apno | 256x97 | 64 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.5492 | 0.338 | 471.8 | 1.397 |
| fno | apno | 256x97 | 64 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.06738 | 0.05711 | 94.16 | 0.36 |
| fno | apno | 256x97 | 64 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.1726 | 0.1149 | 156.8 | 0.7431 |
| fno | apno | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2323 | 0.1707 | 183.1 | 1.294 |
| fno | apno | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.0428 | 0.03108 | 31.86 | 0.2437 |
| fno | apno | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.05783 | 0.04823 | 45.77 | 0.5094 |
| fno | apno | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2629 | 0.2417 | 171.2 | 1.246 |
| fno | apno | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.05067 | 0.02454 | 21.81 | 0.2363 |
| fno | apno | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06433 | 0.03628 | 38.34 | 0.5473 |
| fno | apno | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2396 | 0.1715 | 162.6 | 1.266 |
| fno | apno | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.04374 | 0.02856 | 34.18 | 0.2434 |
| fno | apno | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.0614 | 0.04381 | 42.98 | 0.421 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2491 | 0.243 | 184.4 | 1.22 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01032 | 0.007651 | 12.01 | 0.2442 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.03977 | 0.02831 | 35.49 | 0.3741 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2835 | 0.1676 | 194.4 | 1.238 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01024 | 0.009814 | 7.568 | 0.2441 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03721 | 0.03221 | 33.46 | 0.3984 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2648 | 0.2114 | 202.3 | 1.245 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01027 | 0.008797 | 14.08 | 0.2439 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.0388 | 0.03196 | 57.96 | 0.4097 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2563 | 0.2436 | 171.7 | 1.281 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.04217 | 0.02778 | 35.34 | 0.2343 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06093 | 0.04505 | 32.41 | 0.5294 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2894 | 0.1618 | 197.6 | 1.299 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.04137 | 0.0348 | 22.31 | 0.2486 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.05614 | 0.035 | 27.47 | 0.4735 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2661 | 0.2079 | 185.5 | 1.253 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.04239 | 0.03146 | 28.69 | 0.2496 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.07294 | 0.03276 | 72 | 0.4744 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2733 | 0.2089 | 242.3 | 1.18 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01888 | 0.01194 | 23.19 | 0.2514 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.05578 | 0.05646 | 78.92 | 0.416 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2667 | 0.2163 | 268.3 | 1.19 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.02853 | 0.01827 | 13.56 | 0.4288 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.06584 | 0.05642 | 86.96 | 0.5443 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2396 | 0.2019 | 217.1 | 1.174 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01171 | 0.01052 | 9.813 | 0.2455 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03943 | 0.02282 | 38.78 | 0.3416 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2551 | 0.193 | 255 | 1.139 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01149 | 0.01159 | 9.001 | 0.2458 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04175 | 0.03211 | 61.14 | 0.361 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2307 | 0.189 | 209.7 | 1.092 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01122 | 0.01092 | 6.903 | 0.2447 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.03902 | 0.02373 | 46.73 | 0.3426 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2887 | 0.2641 | 319.3 | 1.263 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.01217 | 0.01298 | 7.704 | 0.2441 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.03987 | 0.03336 | 52.14 | 0.3334 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2491 | 0.2169 | 186 | 1.197 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01104 | 0.008586 | 6.982 | 0.2447 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.03688 | 0.02109 | 33.66 | 0.3257 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2338 | 0.2126 | 243.6 | 1.187 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01796 | 0.01519 | 20.49 | 0.4171 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.04445 | 0.02348 | 47.5 | 0.4359 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2455 | 0.1885 | 271.2 | 1.164 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01771 | 0.01936 | 22.83 | 0.4191 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.04383 | 0.03258 | 67.39 | 0.4373 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2243 | 0.1951 | 236.3 | 1.142 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01848 | 0.01862 | 22.13 | 0.3949 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.04304 | 0.02062 | 59.92 | 0.4285 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0 | 0.2774 | 0.2713 | 341.1 | 1.281 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0 | 0.02458 | 0.01634 | 30.95 | 0.3784 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0 | 0.04449 | 0.03407 | 61.93 | 0.4132 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0 | 0.2356 | 0.2283 | 199.6 | 1.257 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0 | 0.01545 | 0.01256 | 21.07 | 0.3451 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0 | 0.03613 | 0.01937 | 39.84 | 0.3751 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.3019 | 0.2469 | 137.3 | 0.9838 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01262 | 0.0114 | 11.39 | 0.2471 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.06405 | 0.03037 | 45.71 | 0.4712 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.3007 | 0.2143 | 126.4 | 1.088 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04399 | 0.03276 | 40.68 | 0.2914 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.08331 | 0.05478 | 56.7 | 0.5619 |
| fno | s2 | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.232 | 0.1876 | 178.3 | 1.235 |
| fno | s2 | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01017 | 0.009472 | 7.486 | 0.2441 |
| fno | s2 | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.0374 | 0.03141 | 46.59 | 0.3933 |
| fno | s2 | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2482 | 0.2415 | 186 | 1.23 |
| fno | s2 | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.01033 | 0.007987 | 11.21 | 0.2441 |
| fno | s2 | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.0398 | 0.02774 | 32.54 | 0.3759 |
| fno | s2 | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2436 | 0.1804 | 168.8 | 1.25 |
| fno | s2 | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.01007 | 0.007973 | 10.62 | 0.2454 |
| fno | s2 | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.0382 | 0.0343 | 39.94 | 0.3783 |
| fno | s2 | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2318 | 0.1847 | 180.6 | 1.296 |
| fno | s2 | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.04233 | 0.03119 | 31.28 | 0.242 |
| fno | s2 | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.0575 | 0.04307 | 40.41 | 0.5039 |
| fno | s2 | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2554 | 0.2425 | 173.4 | 1.287 |
| fno | s2 | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.04203 | 0.02807 | 36.13 | 0.2364 |
| fno | s2 | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06079 | 0.04382 | 33.78 | 0.5309 |
| fno | s2 | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0 | 0.2427 | 0.1733 | 168.9 | 1.273 |
| fno | s2 | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0 | 0.04373 | 0.02851 | 34.49 | 0.2426 |
| fno | s2 | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0 | 0.0613 | 0.04512 | 35.94 | 0.4219 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.06511 | 0.02474 | 32.43 | 0.7352 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.007953 | 0.007339 | 6.893 | 0.3055 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.02413 | 0.01398 | 17.17 | 0.3029 |

## Shocks (supercritical cases)

Pooled over every (case, surface) where the truth has a shock; a miss is a predicted surface with no supersonic compression.

| method | variant | grid | n_train | seed | split | regime | n_supercritical | n_scored | n_true_shocks | n_missed | abs_x_err_median | abs_x_err_max | mach_ahead_err_median | mach_ahead_err_absmax | overshoot_median | overshoot_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fno | abl-I | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2614 | -0.0607 | 0.2859 | 0.9875 | 1.466 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.0729 | 0.3038 | 0.271 | 0.8801 |
| fno | abl-I | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03188 | 0.2124 | -0.07705 | 0.4089 | 1.072 | 1.546 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.01954 | 0.2954 | 0.295 | 1.104 |
| fno | abl-I | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03188 | 0.2124 | -0.04258 | 0.4301 | 1.068 | 1.553 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06924 | -0.04497 | 0.282 | 0.3067 | 1.079 |
| fno | abl-I | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03188 | 0.2858 | -0.1052 | 0.3695 | 1.014 | 1.476 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.03766 | 0.3175 | 0.2984 | 1.059 |
| fno | abl-I | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04338 | 0.2124 | -0.007788 | 0.5184 | 1.078 | 1.517 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.02003 | 0.2874 | 0.3283 | 0.9937 |
| fno | abl-I | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03045 | 0.2621 | -0.138 | 0.4419 | 1.012 | 1.455 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.007151 | 0.05773 | -0.1031 | 0.5249 | 0.4185 | 1.031 |
| fno | abl-I | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03494 | 0.2008 | -0.1407 | 0.4145 | 1.122 | 1.552 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.008691 | 0.04598 | -0.0776 | 0.431 | 0.4696 | 1.182 |
| fno | abl-I | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03143 | 0.2254 | -0.1624 | 0.3403 | 1.078 | 1.559 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.009018 | 0.04598 | -0.101 | 0.3628 | 0.4653 | 0.9931 |
| fno | abl-I | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03586 | 0.2743 | -0.08856 | 0.396 | 1.063 | 1.477 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01035 | 0.04598 | -0.05557 | 0.3675 | 0.5511 | 1.34 |
| fno | abl-I | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.2131 | -0.01591 | 0.3724 | 1.062 | 1.512 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.04828 | -0.08 | 0.4279 | 0.3667 | 0.829 |
| fno | abl-L | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2858 | -0.04535 | 0.4803 | 1.071 | 1.525 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03741 | 0.2976 | 0.2777 | 0.8341 |
| fno | abl-L | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02989 | 0.3022 | -0.118 | 0.844 | 0.8276 | 1.692 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.004785 | 0.06924 | -0.0663 | 0.3764 | 0.3631 | 1.117 |
| fno | abl-L | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02989 | 0.2858 | -0.08489 | 0.5281 | 0.9779 | 1.541 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.05239 | 0.2934 | 0.2778 | 1.038 |
| fno | abl-L | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03188 | 0.2614 | -0.07726 | 0.354 | 1.018 | 1.553 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.0431 | 0.3281 | 0.2988 | 0.8562 |
| fno | abl-L | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03923 | 0.2614 | -0.09198 | 0.2617 | 1.019 | 1.546 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03238 | 0.2871 | 0.3167 | 1.015 |
| fno | abl-L | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02659 | 0.2743 | -0.1335 | 0.4182 | 1.144 | 1.511 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06955 | -0.09907 | 0.4248 | 0.4076 | 0.7297 |
| fno | abl-L | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03516 | 0.291 | -0.1639 | 0.5397 | 0.9079 | 1.711 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01204 | 0.06955 | -0.08358 | 0.4321 | 0.4765 | 1.279 |
| fno | abl-L | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02317 | 0.2743 | -0.1781 | 0.5083 | 0.7989 | 1.54 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06955 | -0.1132 | 0.4077 | 0.37 | 0.6695 |
| fno | abl-L | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02746 | 0.5216 | -0.1902 | 0.4846 | 0.9306 | 1.545 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.009603 | 0.05773 | -0.108 | 0.5574 | 0.4394 | 0.7417 |
| fno | abl-L | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02862 | 0.2621 | -0.1594 | 0.3972 | 1.018 | 1.54 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.05773 | -0.07642 | 0.4739 | 0.3972 | 0.7927 |
| fno | abl-M | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2369 | -0.1203 | 0.4408 | 1.208 | 1.598 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.06199 | 0.4365 | 0.2706 | 0.7514 |
| fno | abl-M | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02779 | 0.2369 | -0.08641 | 0.3876 | 0.9129 | 1.541 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04174 | 0.2809 | 0.3196 | 0.9266 |
| fno | abl-M | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02989 | 0.2811 | -0.1083 | 0.5663 | 1.157 | 1.491 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.02887 | 0.3094 | 0.2825 | 0.9508 |
| fno | abl-M | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04789 | 0.3022 | -0.06298 | 0.3622 | 1.212 | 1.631 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.02933 | 0.2897 | 0.2711 | 0.9961 |
| fno | abl-M | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04789 | 0.2614 | -0.05853 | 0.4172 | 1.261 | 1.579 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.08153 | 0.309 | 0.2059 | 1.175 |
| fno | abl-M | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03494 | 0.2376 | -0.1695 | 0.3455 | 1.212 | 1.615 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.008691 | 0.05792 | -0.1145 | 0.3377 | 0.2805 | 1.007 |
| fno | abl-M | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02194 | 0.2376 | -0.1584 | 0.3976 | 0.8319 | 1.541 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.05773 | -0.0569 | 0.4143 | 0.3323 | 0.9089 |
| fno | abl-M | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03143 | 0.2804 | -0.2103 | 0.5301 | 1.161 | 1.463 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.06955 | -0.08104 | 0.3196 | 0.37 | 0.8013 |
| fno | abl-M | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04265 | 0.291 | -0.1317 | 0.4621 | 1.123 | 1.621 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.009956 | 0.06955 | -0.06412 | 0.3734 | 0.3673 | 0.983 |
| fno | abl-M | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.259 | -0.1025 | 0.4661 | 0.9831 | 1.595 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.01202 | 0.05773 | -0.07454 | 0.445 | 0.5871 | 1.135 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2614 | -0.05149 | 0.4138 | 0.9505 | 1.534 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04649 | 0.3128 | 0.2864 | 1.082 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.0243 | 0.2811 | -0.06304 | 0.4922 | 0.9229 | 1.582 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06924 | -0.04465 | 0.2989 | 0.2145 | 0.9438 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2369 | -0.09771 | 0.4479 | 1.08 | 1.499 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.065 | 0.2766 | 0.2665 | 1.093 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2369 | -0.06769 | 0.5018 | 0.7862 | 1.564 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.03793 | 0.3374 | 0.2576 | 0.992 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03923 | 0.2811 | -0.08931 | 0.4781 | 0.9916 | 1.497 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.05435 | 0.3082 | 0.2535 | 1.126 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02865 | 0.2499 | -0.1021 | 0.3948 | 0.8402 | 1.6 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06955 | -0.03901 | 0.23 | 0.3462 | 0.8582 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02023 | 0.2376 | -0.1457 | 0.4042 | 0.9257 | 1.595 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.05792 | -0.05683 | 0.2826 | 0.3155 | 0.8023 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03037 | 0.2254 | -0.1533 | 0.4351 | 1.075 | 1.495 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06815 | -0.02684 | 0.2376 | 0.3217 | 0.8569 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03045 | 0.2254 | -0.1745 | 0.434 | 0.8169 | 1.573 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04598 | -0.08227 | 0.4821 | 0.3971 | 0.7937 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03559 | 0.2698 | -0.123 | 0.4464 | 1.023 | 1.494 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.08145 | -0.06374 | 0.3297 | 0.336 | 1.102 |
| fno | abl-P | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2369 | -0.009993 | 0.4505 | 1.054 | 1.541 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.06924 | -0.02859 | 0.3884 | 0.2896 | 0.942 |
| fno | abl-P | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03302 | 0.2614 | -0.07514 | 0.4763 | 1.034 | 1.523 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03803 | 0.2918 | 0.2601 | 0.9598 |
| fno | abl-P | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.06992 | 0.4951 | 1.048 | 1.585 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04555 | 0.2854 | 0.2657 | 0.9264 |
| fno | abl-P | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04233 | 0.2369 | 0.03252 | 0.3342 | 1.023 | 1.568 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.03499 | 0.2761 | 0.2995 | 1.159 |
| fno | abl-P | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03552 | 0.2858 | -0.09815 | 0.3776 | 1.067 | 1.517 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03373 | 0.2794 | 0.3042 | 1.119 |
| fno | abl-P | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2254 | -0.07659 | 0.3892 | 1.072 | 1.555 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.00427 | 0.05773 | -0.07201 | 0.3569 | 0.3509 | 0.8672 |
| fno | abl-P | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.0337 | 0.2621 | -0.089 | 0.5241 | 1.107 | 1.522 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.05773 | -0.0441 | 0.3036 | 0.3797 | 0.952 |
| fno | abl-P | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.1886 | -0.1015 | 0.3994 | 1.054 | 1.609 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.05773 | -0.07966 | 0.4949 | 0.3564 | 0.9563 |
| fno | abl-P | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2131 | -0.09257 | 0.4599 | 1.055 | 1.563 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04598 | -0.05224 | 0.4136 | 0.3608 | 1.073 |
| fno | abl-P | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2743 | -0.1534 | 0.3438 | 1.074 | 1.512 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.009208 | 0.05773 | -0.07485 | 0.3895 | 0.373 | 0.9866 |
| fno | abl-none | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2369 | -0.01995 | 0.4587 | 1.052 | 1.541 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.06924 | -0.06555 | 0.4051 | 0.2974 | 0.9448 |
| fno | abl-none | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03302 | 0.2614 | -0.1086 | 0.4978 | 1.032 | 1.522 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03764 | 0.2912 | 0.2525 | 0.9456 |
| fno | abl-none | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.08636 | 0.4819 | 1.047 | 1.587 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.06585 | 0.2854 | 0.2932 | 0.9332 |
| fno | abl-none | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04233 | 0.2369 | 0.003882 | 0.3295 | 1.049 | 1.572 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.03502 | 0.2744 | 0.3088 | 1.134 |
| fno | abl-none | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03552 | 0.2858 | -0.1034 | 0.356 | 1.067 | 1.514 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03283 | 0.28 | 0.2983 | 1.096 |
| fno | abl-none | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.2254 | -0.1228 | 0.3742 | 1.137 | 1.574 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.05773 | -0.08822 | 0.4598 | 0.4305 | 1.08 |
| fno | abl-none | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.0337 | 0.2621 | -0.09619 | 0.3539 | 1.109 | 1.533 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01035 | 0.05773 | -0.08535 | 0.4045 | 0.515 | 1.041 |
| fno | abl-none | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2008 | -0.1327 | 0.3319 | 1.056 | 1.585 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.009002 | 0.05773 | -0.08473 | 0.4857 | 0.3646 | 0.8426 |
| fno | abl-none | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2254 | -0.1234 | 0.4136 | 1.149 | 1.571 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.00648 | 0.04598 | -0.07293 | 0.3722 | 0.443 | 1.174 |
| fno | abl-none | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2743 | -0.1452 | 0.4023 | 1.078 | 1.516 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.01088 | 0.05773 | -0.08823 | 0.4518 | 0.4034 | 0.9665 |
| fno | apno | 128x49 | 128 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.07275 | 0.519 | -0.1145 | 0.5221 | 0.9075 | 1.57 |
| fno | apno | 128x49 | 128 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.02421 | 0.1005 | -0.1029 | 0.4403 | 0.5611 | 1.419 |
| fno | apno | 128x49 | 256 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.06345 | 0.3138 | -0.1025 | 0.4457 | 0.9728 | 1.549 |
| fno | apno | 128x49 | 256 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.01822 | 0.09053 | -0.04737 | 0.4155 | 0.4382 | 1.283 |
| fno | apno | 128x49 | 512 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04789 | 0.1879 | -0.0915 | 0.4522 | 1.003 | 1.586 |
| fno | apno | 128x49 | 512 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.00957 | 0.09053 | -0.04202 | 0.4091 | 0.3453 | 1.211 |
| fno | apno | 128x49 | 64 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.227 | 0.5508 | 0.01168 | 4.181 | 1.302 | 1.783 |
| fno | apno | 128x49 | 64 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.03356 | 0.105 | -0.1595 | 0.415 | 0.612 | 1.413 |
| fno | apno | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.04881 | 0.4392 | 1.05 | 1.547 |
| fno | apno | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06924 | -0.02438 | 0.3981 | 0.2849 | 0.9735 |
| fno | apno | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03436 | 0.2614 | -0.1162 | 0.3657 | 1.01 | 1.51 |
| fno | apno | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.003383 | 0.06924 | -0.08233 | 0.4248 | 0.3706 | 1.076 |
| fno | apno | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.0405 | 0.4942 | 1.054 | 1.591 |
| fno | apno | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.06411 | 0.2859 | 0.2605 | 0.8928 |
| fno | apno | 256x97 | 128 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.08487 | 0.509 | -0.1705 | 0.4914 | 0.9367 | 1.602 |
| fno | apno | 256x97 | 128 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.03579 | 0.09975 | -0.1391 | 0.4013 | 0.6441 | 1.429 |
| fno | apno | 256x97 | 256 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.0562 | 0.2904 | -0.09176 | 0.4454 | 0.99 | 1.568 |
| fno | apno | 256x97 | 256 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.0202 | 0.09156 | -0.1243 | 0.4465 | 0.591 | 1.39 |
| fno | apno | 256x97 | 512 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04533 | 0.1764 | -0.1359 | 0.3739 | 1.018 | 1.579 |
| fno | apno | 256x97 | 512 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.01227 | 0.05792 | -0.0666 | 0.4129 | 0.5055 | 1.226 |
| fno | apno | 256x97 | 64 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.2198 | 0.5407 | 0.0003293 | 1.144 | 1.299 | 1.74 |
| fno | apno | 256x97 | 64 | 0 | test_in | supercritical | 24 | 24 | 27 | 3 | 0.03979 | 0.1057 | -0.1666 | 0.4915 | 0.6709 | 1.432 |
| fno | apno | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.2254 | -0.1153 | 0.4414 | 1.127 | 1.576 |
| fno | apno | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.05773 | -0.09319 | 0.458 | 0.4313 | 1.095 |
| fno | apno | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03854 | 0.2743 | -0.1715 | 0.4576 | 1.086 | 1.512 |
| fno | apno | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.01115 | 0.05773 | -0.1082 | 0.4699 | 0.5201 | 0.9519 |
| fno | apno | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03972 | 0.2008 | -0.1103 | 0.3185 | 1.06 | 1.588 |
| fno | apno | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.009002 | 0.05773 | -0.06985 | 0.5073 | 0.3826 | 0.8511 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03302 | 0.2614 | -0.0797 | 0.4957 | 1.065 | 1.521 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.01931 | 0.2902 | 0.2527 | 0.9327 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04233 | 0.2124 | -0.01696 | 0.5144 | 1.026 | 1.569 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.04338 | 0.2834 | 0.3007 | 1.114 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03552 | 0.2858 | -0.1037 | 0.3365 | 1.065 | 1.509 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.03316 | 0.2788 | 0.3145 | 1.181 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03494 | 0.2621 | -0.09358 | 0.3921 | 1.114 | 1.521 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01166 | 0.05773 | -0.08742 | 0.3704 | 0.5369 | 1.009 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04265 | 0.2131 | -0.1244 | 0.4707 | 1.086 | 1.569 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.007279 | 0.04598 | -0.07544 | 0.4533 | 0.4448 | 1.155 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03985 | 0.2743 | -0.1588 | 0.4066 | 1.075 | 1.509 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.01088 | 0.05773 | -0.09141 | 0.4667 | 0.4044 | 0.945 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04876 | 0.2614 | -0.1493 | 0.5134 | 1.05 | 1.537 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.05842 | -0.05398 | 0.2725 | 0.297 | 1.236 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04657 | 0.2499 | -0.1742 | 0.5427 | 0.9947 | 1.537 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.01034 | 0.06815 | -0.1028 | 0.3112 | 0.4707 | 1.179 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2614 | -0.1039 | 0.4113 | 0.963 | 1.531 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04971 | 0.2796 | 0.26 | 0.8703 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02989 | 0.2614 | -0.05665 | 0.499 | 0.9151 | 1.587 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.07658 | 0.2983 | 0.2658 | 1.058 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02989 | 0.2369 | -0.1307 | 0.4442 | 0.9345 | 1.539 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04352 | 0.2955 | 0.2413 | 1.121 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04789 | 0.2858 | -0.09782 | 0.3708 | 1.122 | 1.562 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06924 | -0.0559 | 0.2872 | 0.2251 | 0.9803 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2858 | -0.04347 | 0.4792 | 0.9973 | 1.547 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04576 | -0.07408 | 0.4907 | 0.2588 | 0.7704 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03516 | 0.2499 | -0.1407 | 0.3684 | 1.005 | 1.531 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06955 | -0.05443 | 0.2447 | 0.3536 | 0.7525 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02746 | 0.2499 | -0.1267 | 0.4843 | 0.9216 | 1.598 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.08145 | -0.07081 | 0.2511 | 0.2978 | 0.88 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.02746 | 0.2376 | -0.1663 | 0.4153 | 0.9434 | 1.589 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06955 | -0.08114 | 0.269 | 0.301 | 0.9761 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04894 | 0.2743 | -0.1314 | 0.3792 | 1.152 | 1.626 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06955 | -0.05054 | 0.2611 | 0.3352 | 0.8296 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03516 | 0.2621 | -0.07483 | 0.3927 | 1.121 | 1.607 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.04598 | -0.05797 | 0.4065 | 0.3101 | 0.6599 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04745 | 0.2614 | -0.04931 | 0.5441 | 0.7896 | 1.507 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.01173 | 0.105 | -0.07888 | 0.3547 | 0.4128 | 1.053 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.2621 | -0.1359 | 0.3882 | 0.8968 | 1.493 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.009727 | 0.1057 | -0.09184 | 0.4655 | 0.5353 | 1.032 |
| fno | s2 | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.06117 | 0.4267 | 1.051 | 1.551 |
| fno | s2 | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 1 | 0 | 0.06924 | -0.02246 | 0.384 | 0.2864 | 0.921 |
| fno | s2 | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03302 | 0.2614 | -0.08586 | 0.4989 | 1.053 | 1.518 |
| fno | s2 | 128x49 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.01888 | 0.2902 | 0.2524 | 0.949 |
| fno | s2 | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2124 | -0.07403 | 0.4898 | 1.056 | 1.591 |
| fno | s2 | 128x49 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.06401 | 0.2884 | 0.2725 | 0.8934 |
| fno | s2 | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04193 | 0.2131 | -0.1247 | 0.3879 | 1.127 | 1.581 |
| fno | s2 | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 2 | 0 | 0.05773 | -0.08888 | 0.4445 | 0.4222 | 1.049 |
| fno | s2 | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03494 | 0.2621 | -0.113 | 0.3791 | 1.113 | 1.521 |
| fno | s2 | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01166 | 0.05773 | -0.09051 | 0.3759 | 0.5315 | 1.027 |
| fno | s2 | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03972 | 0.2008 | -0.1163 | 0.3328 | 1.063 | 1.588 |
| fno | s2 | 256x97 | 896 | 2 | test_in | supercritical | 24 | 24 | 27 | 1 | 0.009002 | 0.05773 | -0.08214 | 0.5105 | 0.3932 | 0.8124 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.00775 | 0.01221 | -0.07594 | 0.2303 | 0.6892 | 1.002 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.0186 | -0.05571 | 0.368 | 0.1923 | 0.5409 |

## Physics checks

Mass flux is |net flux| through closed grid line j. n_nonpositive_state counts cases with rho <= 0 or p <= 0 somewhere in the near field.

| method | variant | grid | n_train | seed | split | regime | n_ok | n_nonpositive_state | entropy_max_median | entropy_max_max | entropy_mean_median | entropy_mean_max | enthalpy_max_dev_median | enthalpy_max_dev_max | wall_normal_rms_median | wall_normal_rms_max | mass_flux_line0_j | mass_flux_line0_median | mass_flux_line0_max | mass_flux_line1_j | mass_flux_line1_median | mass_flux_line1_max | mass_flux_line2_j | mass_flux_line2_median | mass_flux_line2_max | mass_flux_line3_j | mass_flux_line3_median | mass_flux_line3_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fno | abl-I | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1185 | 0.2082 | 0.001661 | 0.003237 | 0.03223 | 0.1151 | 0.01219 | 0.04366 | 4 | 0.001507 | 0.004486 | 12 | 0.004631 | 0.01437 | 24 | 0.007299 | 0.0188 | 36 | 0.04206 | 0.1351 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01875 | 0.04628 | 3.02e-06 | 2.755e-05 | 0.0009556 | 0.00267 | 0.0009192 | 0.002161 | 4 | 0.0001338 | 0.0004404 | 12 | 0.0001342 | 0.0006241 | 24 | 0.0001714 | 0.001321 | 36 | 0.001362 | 0.008941 |
| fno | abl-I | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04393 | 0.1024 | 0.0001129 | 0.003013 | 0.008111 | 0.03083 | 0.002472 | 0.01791 | 4 | 0.0002563 | 0.001689 | 12 | 0.0003726 | 0.001742 | 24 | 0.0009126 | 0.01203 | 36 | 0.002679 | 0.0293 |
| fno | abl-I | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.107 | 0.2728 | 0.00249 | 0.005825 | 0.02379 | 0.08831 | 0.01325 | 0.03045 | 4 | 0.003242 | 0.008294 | 12 | 0.00324 | 0.009158 | 24 | 0.004923 | 0.01362 | 36 | 0.08815 | 0.4973 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01802 | 0.04493 | 3.003e-06 | 3.071e-05 | 0.0009753 | 0.003654 | 0.0008641 | 0.00295 | 4 | 0.0001078 | 0.0008141 | 12 | 9.698e-05 | 0.0005887 | 24 | 0.0003798 | 0.002077 | 36 | 0.001861 | 0.01064 |
| fno | abl-I | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04109 | 0.09841 | 6.584e-05 | 0.002873 | 0.007786 | 0.02867 | 0.0027 | 0.01067 | 4 | 0.0003087 | 0.002621 | 12 | 0.0003232 | 0.002242 | 24 | 0.0009139 | 0.005616 | 36 | 0.005775 | 0.02075 |
| fno | abl-I | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1074 | 0.2269 | 0.001898 | 0.005822 | 0.02919 | 0.08109 | 0.01227 | 0.02626 | 4 | 0.002072 | 0.004521 | 12 | 0.00211 | 0.01213 | 24 | 0.003298 | 0.01561 | 36 | 0.01896 | 0.1282 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01811 | 0.04427 | 5.736e-07 | 3.522e-05 | 0.0007779 | 0.003983 | 0.0008165 | 0.002594 | 4 | 8.487e-05 | 0.0003529 | 12 | 0.0001173 | 0.0006276 | 24 | 0.0002297 | 0.001235 | 36 | 0.001472 | 0.008315 |
| fno | abl-I | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04092 | 0.1164 | 0.0001241 | 0.002392 | 0.007256 | 0.024 | 0.002791 | 0.0121 | 4 | 0.0003448 | 0.001579 | 12 | 0.0002853 | 0.003761 | 24 | 0.00133 | 0.003986 | 36 | 0.002918 | 0.03828 |
| fno | abl-I | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.115 | 0.2206 | 0.002476 | 0.007062 | 0.03269 | 0.1881 | 0.01211 | 0.03729 | 4 | 0.002272 | 0.005756 | 12 | 0.000745 | 0.02292 | 24 | 0.009486 | 0.07895 | 36 | 0.04802 | 0.4657 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01857 | 0.04612 | 1.723e-06 | 3.167e-05 | 0.0009721 | 0.003027 | 0.0009215 | 0.002755 | 4 | 9.809e-05 | 0.0008978 | 12 | 8.579e-05 | 0.0007569 | 24 | 0.0003974 | 0.002033 | 36 | 0.002355 | 0.008424 |
| fno | abl-I | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04245 | 0.1002 | 0.0001106 | 0.002278 | 0.006432 | 0.0252 | 0.00285 | 0.009877 | 4 | 0.000406 | 0.001249 | 12 | 0.0003836 | 0.003759 | 24 | 0.0008151 | 0.003517 | 36 | 0.006297 | 0.04474 |
| fno | abl-I | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1089 | 0.2059 | 0.001881 | 0.003169 | 0.03709 | 0.139 | 0.01423 | 0.03078 | 4 | 0.002925 | 0.01159 | 12 | 0.003726 | 0.01038 | 24 | 0.006478 | 0.03909 | 36 | 0.0186 | 0.06772 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01813 | 0.04746 | -2.079e-08 | 2.458e-05 | 0.0008833 | 0.002777 | 0.0009013 | 0.002648 | 4 | 0.000139 | 0.001009 | 12 | 0.0001501 | 0.0009323 | 24 | 0.00039 | 0.001062 | 36 | 0.00189 | 0.008767 |
| fno | abl-I | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04133 | 0.126 | 0.0001406 | 0.002529 | 0.008166 | 0.02683 | 0.002731 | 0.01319 | 4 | 0.0003611 | 0.001142 | 12 | 0.000665 | 0.002384 | 24 | 0.0009434 | 0.006866 | 36 | 0.007309 | 0.02757 |
| fno | abl-I | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1204 | 0.2214 | 0.001613 | 0.003075 | 0.02951 | 0.1029 | 0.01815 | 0.04875 | 8 | 0.001476 | 0.006959 | 24 | 0.002453 | 0.01219 | 48 | 0.006895 | 0.0167 | 72 | 0.0344 | 0.1452 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01785 | 0.04272 | 8.398e-07 | 3.072e-05 | 0.001545 | 0.003516 | 0.01452 | 0.01826 | 8 | 0.0002973 | 0.001075 | 24 | 0.0001527 | 0.0007076 | 48 | 0.0003685 | 0.002416 | 72 | 0.001515 | 0.008831 |
| fno | abl-I | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04273 | 0.1049 | 0.0001172 | 0.00293 | 0.008696 | 0.02474 | 0.01412 | 0.02162 | 8 | 0.0004241 | 0.001447 | 24 | 0.0006077 | 0.003624 | 48 | 0.001256 | 0.01161 | 72 | 0.003348 | 0.03257 |
| fno | abl-I | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1196 | 0.2762 | 0.0024 | 0.005918 | 0.02321 | 0.08437 | 0.01834 | 0.03103 | 8 | 0.001533 | 0.007251 | 24 | 0.0042 | 0.01145 | 48 | 0.007253 | 0.01547 | 72 | 0.09336 | 0.5106 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01802 | 0.04125 | -3.335e-07 | 2.743e-05 | 0.001779 | 0.003703 | 0.01408 | 0.01805 | 8 | 0.0001912 | 0.001135 | 24 | 0.0002167 | 0.0009755 | 48 | 0.0002975 | 0.001443 | 72 | 0.003085 | 0.01441 |
| fno | abl-I | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03983 | 0.09775 | 4.659e-05 | 0.002862 | 0.008416 | 0.0312 | 0.01357 | 0.0181 | 8 | 0.0006835 | 0.003898 | 24 | 0.0006021 | 0.002297 | 48 | 0.0007919 | 0.004503 | 72 | 0.00617 | 0.03022 |
| fno | abl-I | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1221 | 0.2252 | 0.001812 | 0.005836 | 0.02915 | 0.07056 | 0.01899 | 0.02526 | 8 | 0.002156 | 0.00537 | 24 | 0.001228 | 0.01161 | 48 | 0.00325 | 0.01947 | 72 | 0.02425 | 0.1448 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01704 | 0.0404 | -7.037e-07 | 2.898e-05 | 0.00142 | 0.002362 | 0.01299 | 0.01698 | 8 | 9.037e-05 | 0.0008075 | 24 | 0.00036 | 0.0008992 | 48 | 0.0003239 | 0.001434 | 72 | 0.002442 | 0.01148 |
| fno | abl-I | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04053 | 0.1122 | 0.0001321 | 0.002349 | 0.008143 | 0.02789 | 0.01296 | 0.01767 | 8 | 0.0002922 | 0.003174 | 24 | 0.0004323 | 0.002255 | 48 | 0.001179 | 0.007698 | 72 | 0.007732 | 0.03794 |
| fno | abl-I | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1203 | 0.2147 | 0.002595 | 0.00745 | 0.03782 | 0.1952 | 0.01734 | 0.03874 | 8 | 0.002279 | 0.006301 | 24 | 0.003172 | 0.02004 | 48 | 0.01102 | 0.06654 | 72 | 0.05389 | 0.4654 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01764 | 0.04306 | -7.028e-07 | 2.965e-05 | 0.001642 | 0.002911 | 0.01358 | 0.01866 | 8 | 0.0002403 | 0.0006834 | 24 | 0.0004698 | 0.001581 | 48 | 0.0009392 | 0.003057 | 72 | 0.002334 | 0.01065 |
| fno | abl-I | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04111 | 0.1066 | 0.0001153 | 0.002247 | 0.008566 | 0.02597 | 0.01351 | 0.01785 | 8 | 0.0004035 | 0.001984 | 24 | 0.0004865 | 0.005972 | 48 | 0.0009557 | 0.007179 | 72 | 0.007005 | 0.04407 |
| fno | abl-I | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1175 | 0.2153 | 0.001599 | 0.002489 | 0.03922 | 0.1294 | 0.01699 | 0.03574 | 8 | 0.003831 | 0.01388 | 24 | 0.003819 | 0.008099 | 48 | 0.006348 | 0.02656 | 72 | 0.01728 | 0.07444 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01757 | 0.04213 | -1.191e-06 | 4.115e-05 | 0.001376 | 0.002955 | 0.01323 | 0.01709 | 8 | 9.425e-05 | 0.0007317 | 24 | 0.0002002 | 0.001597 | 48 | 0.000396 | 0.001791 | 72 | 0.002997 | 0.007608 |
| fno | abl-I | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04181 | 0.1398 | 0.0001907 | 0.002305 | 0.009765 | 0.02782 | 0.0116 | 0.01587 | 8 | 0.0004109 | 0.002204 | 24 | 0.0008238 | 0.004377 | 48 | 0.00114 | 0.01574 | 72 | 0.008898 | 0.03534 |
| fno | abl-L | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1023 | 0.1882 | 0.00196 | 0.00367 | 0.03615 | 0.1801 | 0.01338 | 0.04308 | 4 | 0.001154 | 0.005838 | 12 | 0.00301 | 0.01549 | 24 | 0.009002 | 0.01728 | 36 | 0.01908 | 0.1034 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0183 | 0.04698 | 2.751e-06 | 4.297e-05 | 0.00139 | 0.004116 | 0.0009704 | 0.003452 | 4 | 0.0001387 | 0.0004726 | 12 | 0.0001176 | 0.0005911 | 24 | 0.0003016 | 0.002165 | 36 | 0.001855 | 0.009717 |
| fno | abl-L | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04243 | 0.1097 | 9.347e-05 | 0.003071 | 0.007446 | 0.02053 | 0.002742 | 0.01145 | 4 | 0.0004438 | 0.001537 | 12 | 0.0005078 | 0.001781 | 24 | 0.0009841 | 0.00408 | 36 | 0.005377 | 0.01988 |
| fno | abl-L | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 1 | 0.1172 | 0.1908 | 0.001904 | 0.005287 | 0.04022 | 0.7862 | 0.01286 | 0.05227 | 4 | 0.002156 | 0.01265 | 12 | 0.002913 | 0.01408 | 24 | 0.007712 | 0.01853 | 36 | 0.04021 | 0.1428 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01881 | 0.04503 | 3.122e-06 | 3.32e-05 | 0.001198 | 0.00435 | 0.001193 | 0.003211 | 4 | 8.961e-05 | 0.0006836 | 12 | 0.0001288 | 0.0005288 | 24 | 0.0002572 | 0.001182 | 36 | 0.001987 | 0.009487 |
| fno | abl-L | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03995 | 0.1116 | 0.0001362 | 0.002805 | 0.01213 | 0.04943 | 0.003269 | 0.009189 | 4 | 0.0003001 | 0.001661 | 12 | 0.0004845 | 0.001772 | 24 | 0.00101 | 0.003841 | 36 | 0.007788 | 0.02841 |
| fno | abl-L | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1144 | 0.1689 | 0.002165 | 0.005731 | 0.03056 | 0.1151 | 0.009975 | 0.03734 | 4 | 0.003057 | 0.01594 | 12 | 0.001605 | 0.01901 | 24 | 0.008592 | 0.03101 | 36 | 0.02637 | 0.07318 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01861 | 0.0445 | 1.603e-06 | 6.815e-05 | 0.001047 | 0.003503 | 0.0009221 | 0.002983 | 4 | 9.659e-05 | 0.0005267 | 12 | 0.0001112 | 0.0005819 | 24 | 0.0003223 | 0.001732 | 36 | 0.001444 | 0.00518 |
| fno | abl-L | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04203 | 0.1017 | 0.000111 | 0.003161 | 0.005913 | 0.02449 | 0.002628 | 0.01062 | 4 | 0.0002672 | 0.001287 | 12 | 0.0002498 | 0.001593 | 24 | 0.001036 | 0.007127 | 36 | 0.005263 | 0.03926 |
| fno | abl-L | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1011 | 0.1591 | 0.001822 | 0.002786 | 0.03898 | 0.09869 | 0.01734 | 0.03454 | 4 | 0.001796 | 0.00813 | 12 | 0.003457 | 0.009651 | 24 | 0.005973 | 0.01807 | 36 | 0.02433 | 0.07758 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01801 | 0.04494 | 3.421e-06 | 1.663e-05 | 0.001029 | 0.004514 | 0.001036 | 0.002647 | 4 | 8.08e-05 | 0.001088 | 12 | 0.0001085 | 0.0009733 | 24 | 0.0002485 | 0.0008587 | 36 | 0.001661 | 0.009864 |
| fno | abl-L | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04304 | 0.09894 | 0.0001444 | 0.002286 | 0.008558 | 0.02567 | 0.003325 | 0.009461 | 4 | 0.0002493 | 0.001458 | 12 | 0.0003271 | 0.003001 | 24 | 0.001114 | 0.005953 | 36 | 0.004895 | 0.02419 |
| fno | abl-L | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.116 | 0.1927 | 0.002268 | 0.004479 | 0.02519 | 0.1113 | 0.01227 | 0.03006 | 4 | 0.003238 | 0.005781 | 12 | 0.002951 | 0.009433 | 24 | 0.005734 | 0.01421 | 36 | 0.01127 | 0.04664 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01854 | 0.04482 | 2.016e-06 | 1.302e-05 | 0.0009006 | 0.00283 | 0.0009528 | 0.002778 | 4 | 0.0001285 | 0.0006932 | 12 | 0.0001507 | 0.001 | 24 | 0.0003021 | 0.001809 | 36 | 0.001635 | 0.008778 |
| fno | abl-L | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04128 | 0.1071 | 8.531e-05 | 0.002867 | 0.007146 | 0.02721 | 0.002683 | 0.01041 | 4 | 0.0002921 | 0.004789 | 12 | 0.000437 | 0.002206 | 24 | 0.0008316 | 0.005879 | 36 | 0.003864 | 0.0222 |
| fno | abl-L | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1066 | 0.2196 | 0.001986 | 0.003803 | 0.03373 | 0.159 | 0.01931 | 0.05222 | 8 | 0.0009151 | 0.006017 | 24 | 0.005105 | 0.01513 | 48 | 0.01292 | 0.02396 | 72 | 0.02522 | 0.1176 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01722 | 0.04199 | -1.869e-06 | 2.32e-05 | 0.001775 | 0.003749 | 0.01808 | 0.02311 | 8 | 0.0002981 | 0.0009198 | 24 | 0.000385 | 0.001479 | 48 | 0.000874 | 0.002159 | 72 | 0.00279 | 0.01316 |
| fno | abl-L | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04261 | 0.1152 | 6.944e-05 | 0.003205 | 0.008488 | 0.02398 | 0.01736 | 0.02029 | 8 | 0.000492 | 0.003955 | 24 | 0.0004151 | 0.003861 | 48 | 0.001112 | 0.004131 | 72 | 0.005191 | 0.02675 |
| fno | abl-L | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 1 | 0.1154 | 0.1852 | 0.002047 | 0.006106 | 0.04224 | 0.7987 | 0.02109 | 0.05377 | 8 | 0.002042 | 0.01148 | 24 | 0.00253 | 0.01171 | 48 | 0.01617 | 0.03029 | 72 | 0.03775 | 0.1461 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.017 | 0.03859 | 2.676e-06 | 3.404e-05 | 0.001765 | 0.00373 | 0.01643 | 0.02329 | 8 | 0.0006424 | 0.001426 | 24 | 0.0002969 | 0.001535 | 48 | 0.001328 | 0.002464 | 72 | 0.002871 | 0.01023 |
| fno | abl-L | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03915 | 0.1159 | 0.0001213 | 0.002892 | 0.01245 | 0.04033 | 0.01698 | 0.02063 | 8 | 0.0007271 | 0.002188 | 24 | 0.0007608 | 0.003713 | 48 | 0.001974 | 0.01519 | 72 | 0.009309 | 0.03316 |
| fno | abl-L | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1173 | 0.1675 | 0.002284 | 0.006062 | 0.03107 | 0.0911 | 0.02015 | 0.04753 | 8 | 0.003936 | 0.02051 | 24 | 0.001591 | 0.02206 | 48 | 0.009541 | 0.03979 | 72 | 0.04336 | 0.1013 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.0172 | 0.04141 | 2.691e-06 | 9.153e-05 | 0.002555 | 0.005141 | 0.01806 | 0.02294 | 8 | 0.000544 | 0.001207 | 24 | 0.0006443 | 0.001322 | 48 | 0.0007284 | 0.001941 | 72 | 0.004491 | 0.009573 |
| fno | abl-L | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04174 | 0.1114 | 0.0001387 | 0.003241 | 0.007725 | 0.02783 | 0.01773 | 0.02115 | 8 | 0.0006893 | 0.001288 | 24 | 0.0003922 | 0.0036 | 48 | 0.0007787 | 0.016 | 72 | 0.006914 | 0.06032 |
| fno | abl-L | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1 | 0.1582 | 0.001875 | 0.002856 | 0.03194 | 0.08889 | 0.02539 | 0.04242 | 8 | 0.002653 | 0.009511 | 24 | 0.004587 | 0.0104 | 48 | 0.005831 | 0.01355 | 72 | 0.02015 | 0.07355 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01746 | 0.04615 | 2.416e-06 | 9.159e-06 | 0.001487 | 0.003121 | 0.01662 | 0.02201 | 8 | 0.0001808 | 0.0009185 | 24 | 0.0002594 | 0.0004811 | 48 | 0.0004783 | 0.001847 | 72 | 0.00259 | 0.01367 |
| fno | abl-L | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04407 | 0.1026 | 0.0001286 | 0.0024 | 0.009203 | 0.02298 | 0.01769 | 0.02284 | 8 | 0.0006846 | 0.002449 | 24 | 0.0006492 | 0.003944 | 48 | 0.001417 | 0.005087 | 72 | 0.005663 | 0.02807 |
| fno | abl-L | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.117 | 0.1945 | 0.002464 | 0.005065 | 0.02697 | 0.09466 | 0.02311 | 0.03606 | 8 | 0.004067 | 0.006579 | 24 | 0.003311 | 0.01017 | 48 | 0.008001 | 0.02216 | 72 | 0.01741 | 0.04342 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01728 | 0.04448 | 2.831e-07 | 1.919e-05 | 0.001798 | 0.002923 | 0.01815 | 0.02237 | 8 | 0.0006991 | 0.001264 | 24 | 0.0007601 | 0.001865 | 48 | 0.001289 | 0.003263 | 72 | 0.004483 | 0.01171 |
| fno | abl-L | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04358 | 0.1091 | 0.0001057 | 0.003187 | 0.00761 | 0.0241 | 0.01709 | 0.02318 | 8 | 0.0007695 | 0.005867 | 24 | 0.0005519 | 0.002755 | 48 | 0.001944 | 0.01795 | 72 | 0.004586 | 0.03268 |
| fno | abl-M | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1133 | 0.1954 | 0.0009681 | 0.002913 | 0.03635 | 0.0989 | 0.01603 | 0.03622 | 4 | 0.00231 | 0.009444 | 12 | 0.003151 | 0.01459 | 24 | 0.006429 | 0.03576 | 36 | 0.03386 | 0.07953 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01852 | 0.04347 | 4.843e-06 | 2.229e-05 | 0.001015 | 0.003657 | 0.0009359 | 0.002723 | 4 | 9.944e-05 | 0.0005212 | 12 | 0.0001493 | 0.0007114 | 24 | 0.0002747 | 0.00114 | 36 | 0.002711 | 0.009117 |
| fno | abl-M | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04122 | 0.1194 | 0.0001123 | 0.002654 | 0.006502 | 0.02193 | 0.002612 | 0.01384 | 4 | 0.0002894 | 0.001502 | 12 | 0.0003175 | 0.005251 | 24 | 0.0008514 | 0.008856 | 36 | 0.006922 | 0.04471 |
| fno | abl-M | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1008 | 0.1521 | 0.002404 | 0.005105 | 0.0301 | 0.05158 | 0.01444 | 0.04147 | 4 | 0.003476 | 0.01276 | 12 | 0.003279 | 0.01238 | 24 | 0.006688 | 0.04668 | 36 | 0.01153 | 0.04458 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01858 | 0.04468 | 1.53e-06 | 4.418e-05 | 0.001215 | 0.003222 | 0.001063 | 0.003061 | 4 | 0.0002201 | 0.0008731 | 12 | 0.0002508 | 0.001106 | 24 | 0.0005147 | 0.001862 | 36 | 0.002298 | 0.009197 |
| fno | abl-M | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04295 | 0.1268 | 0.0001067 | 0.002844 | 0.008747 | 0.01902 | 0.003083 | 0.01237 | 4 | 0.0004975 | 0.002146 | 12 | 0.0005044 | 0.002757 | 24 | 0.001082 | 0.003002 | 36 | 0.005437 | 0.04681 |
| fno | abl-M | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1203 | 0.179 | 0.001528 | 0.003805 | 0.02927 | 0.09841 | 0.01321 | 0.06806 | 4 | 0.002043 | 0.01336 | 12 | 0.003611 | 0.01778 | 24 | 0.005642 | 0.009264 | 36 | 0.04741 | 0.102 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01847 | 0.046 | 7.999e-07 | 3.349e-05 | 0.00116 | 0.005036 | 0.001059 | 0.002869 | 4 | 0.0001341 | 0.001251 | 12 | 0.0001935 | 0.001067 | 24 | 0.0003885 | 0.002045 | 36 | 0.002528 | 0.01262 |
| fno | abl-M | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04143 | 0.1184 | 8.792e-05 | 0.002942 | 0.007948 | 0.02727 | 0.002808 | 0.01214 | 4 | 0.0003409 | 0.001554 | 12 | 0.0004117 | 0.002611 | 24 | 0.0006456 | 0.004266 | 36 | 0.003849 | 0.02608 |
| fno | abl-M | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1074 | 0.1552 | 0.002293 | 0.005101 | 0.03512 | 0.1306 | 0.01452 | 0.05469 | 4 | 0.001395 | 0.01061 | 12 | 0.001946 | 0.01833 | 24 | 0.005881 | 0.03535 | 36 | 0.02615 | 0.1056 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01878 | 0.04321 | 5.011e-06 | 8.843e-05 | 0.00111 | 0.004335 | 0.0009988 | 0.002746 | 4 | 0.0001501 | 0.001081 | 12 | 9.229e-05 | 0.000629 | 24 | 0.0003239 | 0.001737 | 36 | 0.001861 | 0.006816 |
| fno | abl-M | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04101 | 0.1151 | 0.0001714 | 0.002764 | 0.007408 | 0.02332 | 0.002792 | 0.01313 | 4 | 0.0003709 | 0.001504 | 12 | 0.0004966 | 0.002958 | 24 | 0.0007298 | 0.00541 | 36 | 0.004151 | 0.01617 |
| fno | abl-M | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1023 | 0.1777 | 0.002025 | 0.00388 | 0.03315 | 0.07017 | 0.01496 | 0.03936 | 4 | 0.002995 | 0.008642 | 12 | 0.001089 | 0.01163 | 24 | 0.004748 | 0.01178 | 36 | 0.02242 | 0.06516 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01861 | 0.04644 | -1.579e-06 | 1.003e-05 | 0.001018 | 0.00355 | 0.001014 | 0.003346 | 4 | 0.0001499 | 0.0007038 | 12 | 0.0001651 | 0.001087 | 24 | 0.0003072 | 0.002498 | 36 | 0.002121 | 0.01316 |
| fno | abl-M | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.0421 | 0.101 | 7.251e-05 | 0.003108 | 0.007588 | 0.02681 | 0.002879 | 0.009158 | 4 | 0.000248 | 0.00193 | 12 | 0.0002847 | 0.003887 | 24 | 0.001012 | 0.003845 | 36 | 0.004953 | 0.02168 |
| fno | abl-M | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.124 | 0.1955 | 0.0009395 | 0.002841 | 0.03728 | 0.0977 | 0.0209 | 0.03775 | 8 | 0.00186 | 0.009315 | 24 | 0.003648 | 0.01559 | 48 | 0.01047 | 0.03589 | 72 | 0.03428 | 0.08809 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0197 | 0.04897 | 4.283e-06 | 2.359e-05 | 0.001574 | 0.003475 | 0.01612 | 0.01947 | 8 | 0.0007905 | 0.00152 | 24 | 0.000688 | 0.001391 | 48 | 0.0004878 | 0.00238 | 72 | 0.003632 | 0.01236 |
| fno | abl-M | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04737 | 0.113 | 0.000111 | 0.002609 | 0.008573 | 0.02335 | 0.01471 | 0.01964 | 8 | 0.0008946 | 0.002229 | 24 | 0.0004456 | 0.006478 | 48 | 0.001157 | 0.01451 | 72 | 0.007729 | 0.04751 |
| fno | abl-M | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1032 | 0.1505 | 0.002436 | 0.005109 | 0.03054 | 0.05306 | 0.02153 | 0.0458 | 8 | 0.004379 | 0.01363 | 24 | 0.004213 | 0.01214 | 48 | 0.00742 | 0.04367 | 72 | 0.01489 | 0.03853 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01806 | 0.04436 | 2.288e-06 | 4.632e-05 | 0.001303 | 0.003479 | 0.01696 | 0.02018 | 8 | 0.0001829 | 0.0009653 | 24 | 0.0004714 | 0.001386 | 48 | 0.001308 | 0.004363 | 72 | 0.003206 | 0.008846 |
| fno | abl-M | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04452 | 0.1176 | 0.0001056 | 0.002787 | 0.008954 | 0.02059 | 0.01565 | 0.02017 | 8 | 0.0005007 | 0.002139 | 24 | 0.0009289 | 0.00274 | 48 | 0.002096 | 0.004534 | 72 | 0.007512 | 0.03635 |
| fno | abl-M | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1238 | 0.1799 | 0.001594 | 0.003987 | 0.02915 | 0.124 | 0.02102 | 0.06819 | 8 | 0.002149 | 0.01359 | 24 | 0.003562 | 0.02038 | 48 | 0.005696 | 0.01099 | 72 | 0.05033 | 0.09129 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01782 | 0.04311 | 3.527e-06 | 3.663e-05 | 0.001316 | 0.003959 | 0.01553 | 0.01899 | 8 | 0.0007576 | 0.001569 | 24 | 0.0005859 | 0.002383 | 48 | 0.0008186 | 0.00297 | 72 | 0.003221 | 0.01384 |
| fno | abl-M | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04294 | 0.1314 | 9.348e-05 | 0.003002 | 0.008898 | 0.02981 | 0.01483 | 0.01769 | 8 | 0.0009731 | 0.002894 | 24 | 0.000971 | 0.00242 | 48 | 0.001077 | 0.007077 | 72 | 0.004342 | 0.04475 |
| fno | abl-M | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1079 | 0.1552 | 0.002295 | 0.005011 | 0.03397 | 0.127 | 0.02088 | 0.05565 | 8 | 0.002046 | 0.01033 | 24 | 0.002292 | 0.01741 | 48 | 0.006869 | 0.04037 | 72 | 0.02595 | 0.1165 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01812 | 0.04362 | 5.732e-06 | 8.706e-05 | 0.001371 | 0.003956 | 0.0159 | 0.01869 | 8 | 0.0006148 | 0.002807 | 24 | 0.0006173 | 0.001047 | 48 | 0.0006118 | 0.001563 | 72 | 0.001879 | 0.009942 |
| fno | abl-M | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04493 | 0.1187 | 0.000172 | 0.002645 | 0.008356 | 0.02449 | 0.01473 | 0.01919 | 8 | 0.0005683 | 0.001984 | 24 | 0.0008701 | 0.003035 | 48 | 0.000889 | 0.006701 | 72 | 0.005677 | 0.02037 |
| fno | abl-M | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.09921 | 0.174 | 0.002005 | 0.003946 | 0.02845 | 0.06333 | 0.02222 | 0.04835 | 8 | 0.002577 | 0.01028 | 24 | 0.001634 | 0.01118 | 48 | 0.005524 | 0.01438 | 72 | 0.02508 | 0.05049 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01753 | 0.04291 | -1.195e-06 | 8.169e-06 | 0.00126 | 0.003959 | 0.01413 | 0.01742 | 8 | 0.0002313 | 0.0007815 | 24 | 0.0005903 | 0.001996 | 48 | 0.001383 | 0.002426 | 72 | 0.006023 | 0.01298 |
| fno | abl-M | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04223 | 0.1103 | 9.132e-05 | 0.003208 | 0.009585 | 0.03778 | 0.01443 | 0.01832 | 8 | 0.0005002 | 0.002827 | 24 | 0.0005713 | 0.004798 | 48 | 0.0008714 | 0.009571 | 72 | 0.005415 | 0.03477 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1005 | 0.1796 | 0.001806 | 0.003022 | 0.02631 | 0.1473 | 0.01088 | 0.02831 | 4 | 0.001628 | 0.006993 | 12 | 0.003098 | 0.01401 | 24 | 0.008851 | 0.05211 | 36 | 0.02925 | 0.1386 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01808 | 0.04341 | -6.359e-07 | 3.084e-05 | 0.001074 | 0.003434 | 0.0009203 | 0.002856 | 4 | 0.0002108 | 0.001166 | 12 | 0.0001639 | 0.0007214 | 24 | 0.0003847 | 0.002081 | 36 | 0.002015 | 0.007287 |
| fno | abl-MPIL | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04094 | 0.1056 | 0.0001334 | 0.00245 | 0.005574 | 0.01789 | 0.002719 | 0.006006 | 4 | 0.0004779 | 0.00253 | 12 | 0.0004583 | 0.00407 | 24 | 0.001408 | 0.007676 | 36 | 0.00401 | 0.02756 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1077 | 0.1722 | 0.002178 | 0.003198 | 0.02698 | 0.1187 | 0.01364 | 0.02616 | 4 | 0.001912 | 0.005937 | 12 | 0.003891 | 0.01947 | 24 | 0.005426 | 0.02346 | 36 | 0.02058 | 0.05575 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01878 | 0.04527 | -1.316e-06 | 6.568e-05 | 0.001205 | 0.004777 | 0.001076 | 0.002405 | 4 | 0.0001672 | 0.001983 | 12 | 0.0001546 | 0.0008018 | 24 | 0.0003908 | 0.001993 | 36 | 0.002441 | 0.0127 |
| fno | abl-MPIL | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03994 | 0.1177 | 0.0001395 | 0.0026 | 0.006867 | 0.02062 | 0.002675 | 0.009385 | 4 | 0.0004486 | 0.0015 | 12 | 0.0005179 | 0.004069 | 24 | 0.00077 | 0.004054 | 36 | 0.004523 | 0.02775 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1112 | 0.196 | 0.001856 | 0.004958 | 0.03457 | 0.06007 | 0.01243 | 0.03005 | 4 | 0.002115 | 0.01529 | 12 | 0.002942 | 0.01602 | 24 | 0.006836 | 0.02295 | 36 | 0.01445 | 0.0443 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01896 | 0.04329 | -6.907e-07 | 1.847e-05 | 0.001198 | 0.004204 | 0.0009907 | 0.003122 | 4 | 0.000168 | 0.0008603 | 12 | 0.0002799 | 0.001101 | 24 | 0.0004241 | 0.002437 | 36 | 0.001653 | 0.01052 |
| fno | abl-MPIL | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04171 | 0.1084 | 9.579e-05 | 0.002696 | 0.009285 | 0.03986 | 0.002468 | 0.006179 | 4 | 0.0003313 | 0.001383 | 12 | 0.0003696 | 0.003596 | 24 | 0.000606 | 0.00539 | 36 | 0.004067 | 0.03162 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1344 | 0.1762 | 0.002369 | 0.004995 | 0.04024 | 0.1885 | 0.01719 | 0.05145 | 4 | 0.004029 | 0.01125 | 12 | 0.006431 | 0.01756 | 24 | 0.009982 | 0.04557 | 36 | 0.03226 | 0.09948 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01867 | 0.04239 | 5.148e-06 | 2.856e-05 | 0.001244 | 0.003609 | 0.00108 | 0.002838 | 4 | 0.0001878 | 0.001398 | 12 | 0.0001951 | 0.001829 | 24 | 0.0004408 | 0.00144 | 36 | 0.002603 | 0.008769 |
| fno | abl-MPIL | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.0407 | 0.1486 | 5.77e-05 | 0.002999 | 0.00844 | 0.02987 | 0.002918 | 0.008385 | 4 | 0.0002736 | 0.002389 | 12 | 0.0005657 | 0.003868 | 24 | 0.001867 | 0.006326 | 36 | 0.008929 | 0.04591 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1049 | 0.1549 | 0.00187 | 0.002843 | 0.0271 | 0.101 | 0.01248 | 0.03969 | 4 | 0.00222 | 0.008545 | 12 | 0.002777 | 0.01076 | 24 | 0.01058 | 0.04569 | 36 | 0.01681 | 0.06419 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.0189 | 0.04377 | -2.086e-06 | 2.411e-05 | 0.001155 | 0.004483 | 0.0009406 | 0.002707 | 4 | 0.0001408 | 0.0008001 | 12 | 0.0001551 | 0.0006872 | 24 | 0.000343 | 0.001141 | 36 | 0.002184 | 0.009951 |
| fno | abl-MPIL | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.0403 | 0.1124 | 8.193e-05 | 0.002404 | 0.005716 | 0.03047 | 0.002622 | 0.007733 | 4 | 0.0004076 | 0.001525 | 12 | 0.0005588 | 0.004015 | 24 | 0.0008209 | 0.002847 | 36 | 0.003756 | 0.02135 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1096 | 0.1797 | 0.001867 | 0.003097 | 0.02763 | 0.1365 | 0.01326 | 0.02797 | 8 | 0.001344 | 0.007027 | 24 | 0.003127 | 0.01373 | 48 | 0.01036 | 0.05417 | 72 | 0.02673 | 0.1385 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01817 | 0.04199 | 2.154e-07 | 3.302e-05 | 0.001951 | 0.005088 | 0.006717 | 0.01004 | 8 | 0.0003333 | 0.001074 | 24 | 0.0002013 | 0.0008905 | 48 | 0.0005335 | 0.001593 | 72 | 0.002177 | 0.007705 |
| fno | abl-MPIL | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04176 | 0.1214 | 0.0001482 | 0.002484 | 0.007552 | 0.02603 | 0.008175 | 0.01215 | 8 | 0.000422 | 0.002553 | 24 | 0.0005776 | 0.005048 | 48 | 0.001419 | 0.01102 | 72 | 0.0056 | 0.03438 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1164 | 0.1744 | 0.002231 | 0.003245 | 0.02863 | 0.13 | 0.01421 | 0.02562 | 8 | 0.00261 | 0.00655 | 24 | 0.003904 | 0.02165 | 48 | 0.006261 | 0.0213 | 72 | 0.02315 | 0.05568 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01906 | 0.04606 | -4.128e-07 | 6.945e-05 | 0.001798 | 0.005912 | 0.008462 | 0.01014 | 8 | 0.0007183 | 0.002157 | 24 | 0.0004307 | 0.0008611 | 48 | 0.0009132 | 0.002997 | 72 | 0.002402 | 0.01163 |
| fno | abl-MPIL | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04139 | 0.1267 | 0.0001458 | 0.002673 | 0.006933 | 0.02578 | 0.008134 | 0.01035 | 8 | 0.0007466 | 0.001695 | 24 | 0.0005107 | 0.004211 | 48 | 0.001537 | 0.004633 | 72 | 0.004735 | 0.03462 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1179 | 0.1972 | 0.001905 | 0.004976 | 0.03737 | 0.05992 | 0.01578 | 0.03337 | 8 | 0.001727 | 0.01718 | 24 | 0.002735 | 0.01573 | 48 | 0.008442 | 0.01764 | 72 | 0.01208 | 0.03686 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.0191 | 0.04224 | -1.502e-06 | 1.8e-05 | 0.002074 | 0.005331 | 0.007684 | 0.01033 | 8 | 0.0001276 | 0.0008407 | 24 | 0.0003347 | 0.001624 | 48 | 0.0004872 | 0.002535 | 72 | 0.002688 | 0.008903 |
| fno | abl-MPIL | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04352 | 0.1207 | 9.691e-05 | 0.002723 | 0.01115 | 0.04243 | 0.008122 | 0.01249 | 8 | 0.0003746 | 0.001326 | 24 | 0.0006214 | 0.004045 | 48 | 0.0015 | 0.007898 | 72 | 0.002268 | 0.02625 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1402 | 0.1829 | 0.002383 | 0.005059 | 0.04096 | 0.1811 | 0.01996 | 0.04867 | 8 | 0.004445 | 0.01289 | 24 | 0.006879 | 0.01816 | 48 | 0.01127 | 0.0421 | 72 | 0.04047 | 0.1117 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01723 | 0.03841 | 4.712e-06 | 2.863e-05 | 0.002065 | 0.006734 | 0.008463 | 0.01188 | 8 | 0.0003828 | 0.001827 | 24 | 0.00057 | 0.001825 | 48 | 0.0005271 | 0.002138 | 72 | 0.002717 | 0.00772 |
| fno | abl-MPIL | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.03946 | 0.1704 | 5.226e-05 | 0.00299 | 0.009434 | 0.03687 | 0.008564 | 0.011 | 8 | 0.0007298 | 0.001738 | 24 | 0.0005991 | 0.004983 | 48 | 0.001838 | 0.009072 | 72 | 0.007938 | 0.03902 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1108 | 0.1583 | 0.001878 | 0.002859 | 0.02825 | 0.0999 | 0.01515 | 0.03958 | 8 | 0.002542 | 0.009951 | 24 | 0.003345 | 0.01134 | 48 | 0.01092 | 0.04727 | 72 | 0.01557 | 0.06414 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01899 | 0.04396 | -2.349e-06 | 2.146e-05 | 0.002018 | 0.007904 | 0.00776 | 0.00944 | 8 | 0.000254 | 0.0008161 | 24 | 0.0002158 | 0.00105 | 48 | 0.0004511 | 0.001553 | 72 | 0.002663 | 0.01093 |
| fno | abl-MPIL | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04303 | 0.1346 | 8.445e-05 | 0.002368 | 0.008238 | 0.03196 | 0.007495 | 0.01101 | 8 | 0.0004497 | 0.001751 | 24 | 0.0005536 | 0.003483 | 48 | 0.0009915 | 0.004059 | 72 | 0.004451 | 0.02044 |
| fno | abl-P | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1116 | 0.2281 | 0.001564 | 0.007395 | 0.03705 | 0.1663 | 0.01159 | 0.02229 | 4 | 0.001233 | 0.005051 | 12 | 0.002648 | 0.006724 | 24 | 0.008761 | 0.01645 | 36 | 0.07319 | 0.3132 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01849 | 0.04672 | 1.377e-06 | 1.652e-05 | 0.0007405 | 0.002509 | 0.0008076 | 0.002144 | 4 | 8.311e-05 | 0.0006564 | 12 | 6.902e-05 | 0.0007383 | 24 | 0.0001957 | 0.001458 | 36 | 0.001902 | 0.008239 |
| fno | abl-P | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04114 | 0.1065 | 8.77e-05 | 0.002533 | 0.00663 | 0.01812 | 0.002202 | 0.01417 | 4 | 0.0003594 | 0.002723 | 12 | 0.0002623 | 0.003506 | 24 | 0.0006749 | 0.004359 | 36 | 0.003651 | 0.01587 |
| fno | abl-P | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1027 | 0.1733 | 0.001913 | 0.00317 | 0.03523 | 0.07563 | 0.01309 | 0.02403 | 4 | 0.001477 | 0.005823 | 12 | 0.001769 | 0.009731 | 24 | 0.00415 | 0.01459 | 36 | 0.01803 | 0.04254 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01894 | 0.04565 | 7.685e-07 | 1.783e-05 | 0.0007548 | 0.002792 | 0.000827 | 0.002545 | 4 | 5.938e-05 | 0.0003807 | 12 | 0.0001114 | 0.0004445 | 24 | 0.0002312 | 0.001236 | 36 | 0.001424 | 0.005393 |
| fno | abl-P | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04255 | 0.09539 | 8.253e-05 | 0.00214 | 0.005324 | 0.02634 | 0.002649 | 0.01052 | 4 | 0.0003162 | 0.001212 | 12 | 0.0004939 | 0.001975 | 24 | 0.0009748 | 0.006313 | 36 | 0.00445 | 0.01462 |
| fno | abl-P | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1075 | 0.2007 | 0.002339 | 0.006214 | 0.04573 | 0.1488 | 0.01365 | 0.02163 | 4 | 0.001764 | 0.00908 | 12 | 0.001937 | 0.006484 | 24 | 0.004913 | 0.01213 | 36 | 0.02688 | 0.0728 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01843 | 0.04531 | 7.791e-07 | 4.665e-05 | 0.0008151 | 0.002179 | 0.0008022 | 0.002409 | 4 | 9.826e-05 | 0.001103 | 12 | 0.0001462 | 0.001116 | 24 | 0.0001731 | 0.001198 | 36 | 0.001531 | 0.01107 |
| fno | abl-P | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04472 | 0.1139 | 0.0001203 | 0.002646 | 0.00675 | 0.01785 | 0.002316 | 0.0138 | 4 | 0.0003521 | 0.001326 | 12 | 0.0004768 | 0.001209 | 24 | 0.0005616 | 0.002733 | 36 | 0.004728 | 0.0501 |
| fno | abl-P | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1049 | 0.241 | 0.001435 | 0.003605 | 0.02564 | 0.134 | 0.01281 | 0.03381 | 4 | 0.0008824 | 0.007043 | 12 | 0.002106 | 0.006855 | 24 | 0.00527 | 0.01978 | 36 | 0.009781 | 0.04451 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01797 | 0.04611 | 1.846e-06 | 1.831e-05 | 0.0007934 | 0.002736 | 0.0008823 | 0.002263 | 4 | 0.000129 | 0.0003898 | 12 | 7.127e-05 | 0.0006867 | 24 | 0.0002639 | 0.001465 | 36 | 0.001786 | 0.005297 |
| fno | abl-P | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04028 | 0.09876 | 9.674e-05 | 0.002338 | 0.005299 | 0.02055 | 0.00262 | 0.01609 | 4 | 0.0003638 | 0.002362 | 12 | 0.0003321 | 0.003435 | 24 | 0.0008474 | 0.005779 | 36 | 0.002923 | 0.01023 |
| fno | abl-P | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1193 | 0.199 | 0.002594 | 0.007856 | 0.02895 | 0.1237 | 0.01223 | 0.04146 | 4 | 0.002436 | 0.008628 | 12 | 0.002006 | 0.008336 | 24 | 0.004644 | 0.04101 | 36 | 0.0231 | 0.05115 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01882 | 0.04466 | 4.909e-07 | 3.701e-05 | 0.0007505 | 0.002295 | 0.0008575 | 0.002848 | 4 | 9.58e-05 | 0.0008874 | 12 | 9.898e-05 | 0.000627 | 24 | 0.0002525 | 0.00135 | 36 | 0.001733 | 0.004571 |
| fno | abl-P | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04271 | 0.1024 | 0.0001059 | 0.002145 | 0.007591 | 0.02029 | 0.002898 | 0.01376 | 4 | 0.0002809 | 0.003478 | 12 | 0.0004891 | 0.005094 | 24 | 0.0007563 | 0.005285 | 36 | 0.002142 | 0.02009 |
| fno | abl-P | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1256 | 0.2359 | 0.001584 | 0.007348 | 0.03743 | 0.1558 | 0.01113 | 0.02024 | 8 | 0.0011 | 0.004984 | 24 | 0.002669 | 0.006842 | 48 | 0.009902 | 0.01806 | 72 | 0.07478 | 0.3141 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01819 | 0.04655 | 6.788e-07 | 1.549e-05 | 0.003096 | 0.005703 | 0.006472 | 0.01054 | 8 | 0.0001994 | 0.0008008 | 24 | 0.0001239 | 0.0007152 | 48 | 0.000296 | 0.002222 | 72 | 0.002137 | 0.009082 |
| fno | abl-P | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04138 | 0.1177 | 9.233e-05 | 0.00254 | 0.007673 | 0.01832 | 0.00715 | 0.01324 | 8 | 0.0003462 | 0.00251 | 24 | 0.0003064 | 0.003682 | 48 | 0.001121 | 0.004658 | 72 | 0.004405 | 0.0137 |
| fno | abl-P | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.108 | 0.171 | 0.002015 | 0.003267 | 0.03714 | 0.07204 | 0.009856 | 0.02228 | 8 | 0.001545 | 0.006096 | 24 | 0.001429 | 0.01003 | 48 | 0.004411 | 0.01592 | 72 | 0.01912 | 0.04624 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01825 | 0.04505 | 4.279e-07 | 1.813e-05 | 0.002404 | 0.004604 | 0.006543 | 0.0102 | 8 | 0.0001055 | 0.0005064 | 24 | 0.0001272 | 0.0005258 | 48 | 0.0004127 | 0.001585 | 72 | 0.003414 | 0.007613 |
| fno | abl-P | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04221 | 0.09934 | 8.679e-05 | 0.002169 | 0.007183 | 0.02874 | 0.006912 | 0.009949 | 8 | 0.0003611 | 0.00136 | 24 | 0.0003841 | 0.001701 | 48 | 0.001166 | 0.007584 | 72 | 0.006162 | 0.01854 |
| fno | abl-P | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1243 | 0.2064 | 0.002375 | 0.006208 | 0.04595 | 0.1449 | 0.01237 | 0.02239 | 8 | 0.002188 | 0.00911 | 24 | 0.002176 | 0.006893 | 48 | 0.005573 | 0.01194 | 72 | 0.02634 | 0.06689 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01739 | 0.04258 | -1.476e-06 | 3.07e-05 | 0.002187 | 0.004679 | 0.007134 | 0.01086 | 8 | 0.0002696 | 0.001072 | 24 | 0.0001965 | 0.001047 | 48 | 0.00034 | 0.001406 | 72 | 0.001513 | 0.009237 |
| fno | abl-P | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04591 | 0.1239 | 0.0001142 | 0.002643 | 0.007887 | 0.02099 | 0.007695 | 0.01287 | 8 | 0.0003645 | 0.001329 | 24 | 0.0004789 | 0.001252 | 48 | 0.0005757 | 0.003566 | 72 | 0.004403 | 0.04484 |
| fno | abl-P | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1153 | 0.2506 | 0.001376 | 0.003622 | 0.02596 | 0.1293 | 0.01069 | 0.02942 | 8 | 0.001262 | 0.007506 | 24 | 0.002008 | 0.006968 | 48 | 0.006007 | 0.02093 | 72 | 0.01222 | 0.04493 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01787 | 0.04566 | 9.307e-08 | 1.464e-05 | 0.002208 | 0.004022 | 0.006587 | 0.0117 | 8 | 0.0002165 | 0.0006793 | 24 | 0.0002563 | 0.0006993 | 48 | 0.00027 | 0.001793 | 72 | 0.002019 | 0.008736 |
| fno | abl-P | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04224 | 0.1106 | 0.0001001 | 0.002269 | 0.006615 | 0.02028 | 0.007516 | 0.01109 | 8 | 0.0003867 | 0.002151 | 24 | 0.0003645 | 0.003348 | 48 | 0.001022 | 0.006842 | 72 | 0.00292 | 0.01146 |
| fno | abl-P | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1268 | 0.2074 | 0.00263 | 0.007959 | 0.03074 | 0.1159 | 0.01182 | 0.04084 | 8 | 0.002381 | 0.008732 | 24 | 0.002152 | 0.00988 | 48 | 0.004673 | 0.04314 | 72 | 0.02413 | 0.06093 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01879 | 0.04324 | -2.27e-07 | 4.63e-05 | 0.001667 | 0.004695 | 0.007603 | 0.01158 | 8 | 0.000159 | 0.0009464 | 24 | 0.0001543 | 0.000576 | 48 | 0.0002874 | 0.001143 | 72 | 0.002423 | 0.005764 |
| fno | abl-P | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04416 | 0.1146 | 0.000112 | 0.002159 | 0.00913 | 0.0266 | 0.008764 | 0.01226 | 8 | 0.0002391 | 0.002995 | 24 | 0.0003538 | 0.00512 | 48 | 0.0005767 | 0.00629 | 72 | 0.002606 | 0.01708 |
| fno | abl-none | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1121 | 0.2261 | 0.00159 | 0.007423 | 0.03961 | 0.1645 | 0.01125 | 0.02144 | 4 | 0.001215 | 0.005592 | 12 | 0.002568 | 0.007069 | 24 | 0.008206 | 0.01756 | 36 | 0.07189 | 0.3085 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01852 | 0.04671 | 1.307e-06 | 1.704e-05 | 0.0007292 | 0.002403 | 0.0008066 | 0.002117 | 4 | 8.631e-05 | 0.0006451 | 12 | 6.378e-05 | 0.0007823 | 24 | 0.0001814 | 0.001412 | 36 | 0.001631 | 0.008628 |
| fno | abl-none | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04113 | 0.1061 | 8.703e-05 | 0.002516 | 0.006661 | 0.01772 | 0.00214 | 0.01386 | 4 | 0.0003527 | 0.00288 | 12 | 0.0002449 | 0.003568 | 24 | 0.0005887 | 0.003947 | 36 | 0.003463 | 0.01538 |
| fno | abl-none | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1033 | 0.1742 | 0.001851 | 0.00293 | 0.03337 | 0.06874 | 0.01312 | 0.02363 | 4 | 0.001101 | 0.006382 | 12 | 0.00202 | 0.01046 | 24 | 0.004704 | 0.01085 | 36 | 0.01633 | 0.04624 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01896 | 0.04564 | 1.593e-06 | 1.745e-05 | 0.0007746 | 0.002883 | 0.0008173 | 0.002709 | 4 | 8.442e-05 | 0.000386 | 12 | 0.0001187 | 0.0003967 | 24 | 0.0002938 | 0.001315 | 36 | 0.00168 | 0.005287 |
| fno | abl-none | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04231 | 0.09599 | 8.408e-05 | 0.002108 | 0.005337 | 0.02651 | 0.002609 | 0.009969 | 4 | 0.0003141 | 0.001215 | 12 | 0.0004776 | 0.00188 | 24 | 0.0009036 | 0.00639 | 36 | 0.004576 | 0.01702 |
| fno | abl-none | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1096 | 0.199 | 0.002389 | 0.006138 | 0.04562 | 0.1505 | 0.0138 | 0.02188 | 4 | 0.002205 | 0.008318 | 12 | 0.00224 | 0.00567 | 24 | 0.005004 | 0.01117 | 36 | 0.02247 | 0.07528 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01842 | 0.04541 | 1.368e-06 | 4.817e-05 | 0.0008113 | 0.002121 | 0.0007783 | 0.002423 | 4 | 0.0001184 | 0.001024 | 12 | 0.0001418 | 0.001066 | 24 | 0.0001766 | 0.001199 | 36 | 0.001864 | 0.01 |
| fno | abl-none | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04491 | 0.115 | 0.0001255 | 0.002698 | 0.006759 | 0.02006 | 0.002296 | 0.01396 | 4 | 0.0003242 | 0.001243 | 12 | 0.0004687 | 0.001195 | 24 | 0.0005353 | 0.002869 | 36 | 0.005046 | 0.05375 |
| fno | abl-none | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1071 | 0.2413 | 0.001472 | 0.003791 | 0.02702 | 0.1381 | 0.01294 | 0.03268 | 4 | 0.001722 | 0.004592 | 12 | 0.002588 | 0.006532 | 24 | 0.004568 | 0.01382 | 36 | 0.01458 | 0.05107 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01795 | 0.04615 | 2.28e-06 | 1.429e-05 | 0.0007687 | 0.002753 | 0.0008962 | 0.002403 | 4 | 0.00011 | 0.0003804 | 12 | 5.793e-05 | 0.0007045 | 24 | 0.0002669 | 0.001473 | 36 | 0.001585 | 0.005049 |
| fno | abl-none | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04017 | 0.09741 | 9.107e-05 | 0.002195 | 0.005771 | 0.01938 | 0.002512 | 0.01669 | 4 | 0.0003817 | 0.002392 | 12 | 0.0003496 | 0.00316 | 24 | 0.0008882 | 0.006774 | 36 | 0.002749 | 0.01191 |
| fno | abl-none | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.119 | 0.1963 | 0.002574 | 0.007468 | 0.02922 | 0.1047 | 0.01206 | 0.03813 | 4 | 0.002635 | 0.008627 | 12 | 0.002017 | 0.009062 | 24 | 0.00435 | 0.04194 | 36 | 0.02259 | 0.05088 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01881 | 0.04458 | 7.286e-08 | 3.717e-05 | 0.0007821 | 0.002437 | 0.0008433 | 0.00295 | 4 | 8.821e-05 | 0.0009008 | 12 | 9.694e-05 | 0.0006701 | 24 | 0.0002564 | 0.00132 | 36 | 0.001812 | 0.004827 |
| fno | abl-none | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04279 | 0.1022 | 0.000107 | 0.002163 | 0.007468 | 0.02195 | 0.002846 | 0.01442 | 4 | 0.0003132 | 0.003402 | 12 | 0.0004815 | 0.005509 | 24 | 0.0007034 | 0.005939 | 36 | 0.00152 | 0.01316 |
| fno | abl-none | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1222 | 0.2206 | 0.001654 | 0.007429 | 0.03897 | 0.1572 | 0.01956 | 0.02833 | 8 | 0.001172 | 0.005183 | 24 | 0.0008755 | 0.007063 | 48 | 0.009725 | 0.01886 | 72 | 0.06683 | 0.2928 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01698 | 0.04459 | 6.823e-07 | 1.938e-05 | 0.001322 | 0.003059 | 0.01386 | 0.01796 | 8 | 0.0004964 | 0.001483 | 24 | 0.0002074 | 0.001378 | 48 | 0.0003879 | 0.003118 | 72 | 0.002253 | 0.008935 |
| fno | abl-none | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04063 | 0.111 | 9.767e-05 | 0.002677 | 0.007869 | 0.01973 | 0.01358 | 0.02014 | 8 | 0.0006725 | 0.002195 | 24 | 0.000846 | 0.004476 | 48 | 0.001469 | 0.005963 | 72 | 0.005611 | 0.01515 |
| fno | abl-none | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1037 | 0.1713 | 0.001936 | 0.003055 | 0.03131 | 0.06279 | 0.01844 | 0.02771 | 8 | 0.001187 | 0.005692 | 24 | 0.001369 | 0.01002 | 48 | 0.004859 | 0.01183 | 72 | 0.01702 | 0.04896 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01783 | 0.04435 | 1.039e-06 | 2.012e-05 | 0.001053 | 0.002741 | 0.01417 | 0.01877 | 8 | 0.0002033 | 0.0005864 | 24 | 0.000102 | 0.0006158 | 48 | 0.0007338 | 0.001999 | 72 | 0.006407 | 0.0121 |
| fno | abl-none | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04099 | 0.09612 | 7.758e-05 | 0.00219 | 0.00588 | 0.02503 | 0.01435 | 0.01951 | 8 | 0.0004566 | 0.001792 | 24 | 0.0004834 | 0.002016 | 48 | 0.001424 | 0.008339 | 72 | 0.007079 | 0.02042 |
| fno | abl-none | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1108 | 0.1858 | 0.002341 | 0.006082 | 0.04243 | 0.1253 | 0.02091 | 0.0286 | 8 | 0.002751 | 0.009327 | 24 | 0.003025 | 0.007152 | 48 | 0.00705 | 0.01299 | 72 | 0.02642 | 0.05851 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.0177 | 0.04044 | -6.065e-07 | 1.474e-05 | 0.001609 | 0.003143 | 0.01403 | 0.01939 | 8 | 0.0003768 | 0.001224 | 24 | 0.0002295 | 0.001571 | 48 | 0.0008277 | 0.002456 | 72 | 0.002222 | 0.007703 |
| fno | abl-none | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04495 | 0.1184 | 9.488e-05 | 0.002634 | 0.007687 | 0.02207 | 0.01513 | 0.02239 | 8 | 0.0005253 | 0.001667 | 24 | 0.0005447 | 0.002009 | 48 | 0.0008667 | 0.004082 | 72 | 0.003215 | 0.03964 |
| fno | abl-none | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.104 | 0.2437 | 0.001255 | 0.003369 | 0.0267 | 0.1365 | 0.02021 | 0.03675 | 8 | 0.00178 | 0.004797 | 24 | 0.00208 | 0.00705 | 48 | 0.006264 | 0.01865 | 72 | 0.01897 | 0.06588 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01772 | 0.04412 | -8.118e-07 | 1.163e-05 | 0.001824 | 0.002744 | 0.01388 | 0.01876 | 8 | 0.0005956 | 0.001353 | 24 | 0.0001759 | 0.0009749 | 48 | 0.0007869 | 0.002089 | 72 | 0.002131 | 0.008993 |
| fno | abl-none | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04212 | 0.09852 | 9.454e-05 | 0.001846 | 0.006681 | 0.01959 | 0.0143 | 0.02098 | 8 | 0.000763 | 0.001971 | 24 | 0.0006621 | 0.002865 | 48 | 0.00122 | 0.01127 | 72 | 0.002346 | 0.0121 |
| fno | abl-none | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1254 | 0.1963 | 0.002593 | 0.00759 | 0.03102 | 0.09683 | 0.01794 | 0.04054 | 8 | 0.002064 | 0.00898 | 24 | 0.003878 | 0.01575 | 48 | 0.005737 | 0.04805 | 72 | 0.01991 | 0.1007 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01814 | 0.04189 | -1.976e-06 | 4.543e-05 | 0.001328 | 0.002225 | 0.01403 | 0.01927 | 8 | 0.0003747 | 0.001718 | 24 | 0.0002311 | 0.0007961 | 48 | 0.0003628 | 0.001484 | 72 | 0.004298 | 0.00847 |
| fno | abl-none | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04422 | 0.1082 | 0.000134 | 0.002157 | 0.00858 | 0.02629 | 0.01355 | 0.01874 | 8 | 0.001267 | 0.003023 | 24 | 0.0004004 | 0.007069 | 48 | 0.0006716 | 0.01343 | 72 | 0.00349 | 0.01188 |
| fno | apno | 128x49 | 128 | 0 | test_ext | supercritical | 14 | 4 | 0.104 | 0.2142 | 0.002282 | 0.005122 | 0.1588 | 7.6 | 0.01531 | 0.04287 | 4 | 0.00412 | 0.007456 | 12 | 0.006257 | 0.02246 | 24 | 0.006993 | 0.02012 | 36 | 0.03774 | 0.07174 |
| fno | apno | 128x49 | 128 | 0 | test_in | subcritical | 37 | 0 | 0.0183 | 0.04043 | -2.07e-07 | 5.577e-05 | 0.002418 | 0.007178 | 0.003528 | 0.01348 | 4 | 0.0004357 | 0.004249 | 12 | 0.0005072 | 0.004253 | 24 | 0.001027 | 0.002968 | 36 | 0.004701 | 0.02213 |
| fno | apno | 128x49 | 128 | 0 | test_in | supercritical | 24 | 0 | 0.04041 | 0.1141 | 0.0001541 | 0.002996 | 0.01408 | 0.09891 | 0.006152 | 0.02796 | 4 | 0.001387 | 0.004574 | 12 | 0.001423 | 0.008014 | 24 | 0.00229 | 0.008954 | 36 | 0.009918 | 0.08415 |
| fno | apno | 128x49 | 256 | 0 | test_ext | supercritical | 14 | 0 | 0.1169 | 0.1883 | 0.001143 | 0.002843 | 0.07637 | 0.3682 | 0.01361 | 0.0258 | 4 | 0.002428 | 0.00617 | 12 | 0.00515 | 0.01003 | 24 | 0.003482 | 0.01215 | 36 | 0.0197 | 0.08884 |
| fno | apno | 128x49 | 256 | 0 | test_in | subcritical | 37 | 0 | 0.01939 | 0.04266 | 3.496e-06 | 5.018e-05 | 0.001708 | 0.004716 | 0.002294 | 0.01074 | 4 | 0.0002818 | 0.003002 | 12 | 0.0002974 | 0.001632 | 24 | 0.0004758 | 0.00353 | 36 | 0.004615 | 0.01908 |
| fno | apno | 128x49 | 256 | 0 | test_in | supercritical | 24 | 0 | 0.04246 | 0.1133 | 9.056e-05 | 0.002757 | 0.009386 | 0.05513 | 0.003943 | 0.02114 | 4 | 0.0003994 | 0.004038 | 12 | 0.0004809 | 0.007584 | 24 | 0.001414 | 0.004203 | 36 | 0.004687 | 0.0258 |
| fno | apno | 128x49 | 512 | 0 | test_ext | supercritical | 14 | 0 | 0.1112 | 0.191 | 0.002007 | 0.003974 | 0.03722 | 0.1453 | 0.01131 | 0.0259 | 4 | 0.003162 | 0.02076 | 12 | 0.003303 | 0.01726 | 24 | 0.004233 | 0.02622 | 36 | 0.01256 | 0.08066 |
| fno | apno | 128x49 | 512 | 0 | test_in | subcritical | 37 | 0 | 0.01751 | 0.04535 | 8.038e-07 | 2.505e-05 | 0.001135 | 0.003372 | 0.001227 | 0.004533 | 4 | 0.0001885 | 0.001606 | 12 | 0.0001794 | 0.001046 | 24 | 0.000348 | 0.002077 | 36 | 0.001689 | 0.008573 |
| fno | apno | 128x49 | 512 | 0 | test_in | supercritical | 24 | 0 | 0.04161 | 0.1099 | 7.708e-05 | 0.002712 | 0.008663 | 0.02395 | 0.003106 | 0.01741 | 4 | 0.0004839 | 0.004761 | 12 | 0.0006618 | 0.004273 | 24 | 0.001035 | 0.005837 | 36 | 0.003555 | 0.02293 |
| fno | apno | 128x49 | 64 | 0 | test_ext | supercritical | 14 | 5 | 0.08571 | 0.1262 | 0.003184 | 0.003704 | 0.2405 | 72.74 | 0.02409 | 0.03571 | 4 | 0.01037 | 0.01743 | 12 | 0.006083 | 0.04237 | 24 | 0.01031 | 0.03171 | 36 | 0.02071 | 0.1663 |
| fno | apno | 128x49 | 64 | 0 | test_in | subcritical | 37 | 0 | 0.01581 | 0.03919 | 4.277e-06 | 0.0005188 | 0.003792 | 0.009708 | 0.01091 | 0.03846 | 4 | 0.002111 | 0.006678 | 12 | 0.001213 | 0.005355 | 24 | 0.003274 | 0.01001 | 36 | 0.01162 | 0.04754 |
| fno | apno | 128x49 | 64 | 0 | test_in | supercritical | 24 | 0 | 0.03672 | 0.111 | 0.000251 | 0.003237 | 0.01358 | 0.1058 | 0.01398 | 0.02704 | 4 | 0.002108 | 0.006177 | 12 | 0.003101 | 0.01233 | 24 | 0.003736 | 0.01673 | 36 | 0.02325 | 0.04 |
| fno | apno | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1075 | 0.2127 | 0.001775 | 0.007449 | 0.04439 | 0.1521 | 0.0116 | 0.02435 | 4 | 0.0009704 | 0.005662 | 12 | 0.002685 | 0.006498 | 24 | 0.008164 | 0.02008 | 36 | 0.0616 | 0.3047 |
| fno | apno | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0186 | 0.04617 | 1.753e-06 | 1.597e-05 | 0.0007714 | 0.00242 | 0.0007852 | 0.002203 | 4 | 9.612e-05 | 0.0006371 | 12 | 6.922e-05 | 0.0008124 | 24 | 0.0001776 | 0.00101 | 36 | 0.001626 | 0.007111 |
| fno | apno | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04125 | 0.1052 | 9.04e-05 | 0.002627 | 0.006592 | 0.02057 | 0.00215 | 0.01422 | 4 | 0.0003621 | 0.003009 | 12 | 0.0003213 | 0.003955 | 24 | 0.0008128 | 0.002996 | 36 | 0.003454 | 0.01707 |
| fno | apno | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1028 | 0.1928 | 0.002229 | 0.003375 | 0.03094 | 0.07884 | 0.01315 | 0.02525 | 4 | 0.002834 | 0.00644 | 12 | 0.001692 | 0.008907 | 24 | 0.003867 | 0.02544 | 36 | 0.02218 | 0.08876 |
| fno | apno | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01891 | 0.04611 | 2.209e-05 | 8.91e-05 | 0.001362 | 0.005707 | 0.002527 | 0.005091 | 4 | 0.000544 | 0.00141 | 12 | 0.0003736 | 0.001164 | 24 | 0.000731 | 0.002533 | 36 | 0.004354 | 0.008355 |
| fno | apno | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04131 | 0.09596 | 0.0001916 | 0.002172 | 0.007682 | 0.02219 | 0.003119 | 0.012 | 4 | 0.0007244 | 0.003929 | 12 | 0.0005227 | 0.003436 | 24 | 0.0007589 | 0.004896 | 36 | 0.0058 | 0.02297 |
| fno | apno | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1142 | 0.2077 | 0.002253 | 0.005786 | 0.04322 | 0.1356 | 0.01359 | 0.02261 | 4 | 0.002581 | 0.007625 | 12 | 0.002469 | 0.005505 | 24 | 0.005145 | 0.01785 | 36 | 0.01801 | 0.08804 |
| fno | apno | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01847 | 0.04526 | 1.245e-06 | 3.945e-05 | 0.000816 | 0.002172 | 0.0007996 | 0.002181 | 4 | 0.0001074 | 0.001 | 12 | 0.0001251 | 0.0009872 | 24 | 0.0001325 | 0.0007451 | 36 | 0.001527 | 0.01226 |
| fno | apno | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04457 | 0.1111 | 0.0001136 | 0.002632 | 0.006166 | 0.01718 | 0.002386 | 0.01425 | 4 | 0.0003725 | 0.001183 | 12 | 0.0002942 | 0.001589 | 24 | 0.0005793 | 0.002643 | 36 | 0.003994 | 0.05973 |
| fno | apno | 256x97 | 128 | 0 | test_ext | supercritical | 14 | 4 | 0.1016 | 0.2111 | 0.002297 | 0.005386 | 0.169 | 16.02 | 0.02417 | 0.04932 | 8 | 0.003769 | 0.005273 | 24 | 0.003014 | 0.01855 | 48 | 0.006314 | 0.01384 | 72 | 0.04416 | 0.07251 |
| fno | apno | 256x97 | 128 | 0 | test_in | subcritical | 37 | 0 | 0.01824 | 0.0399 | -5.456e-06 | 4.255e-05 | 0.003031 | 0.008385 | 0.01873 | 0.02782 | 8 | 0.000365 | 0.004647 | 24 | 0.0005366 | 0.004424 | 48 | 0.0009829 | 0.00384 | 72 | 0.006669 | 0.02225 |
| fno | apno | 256x97 | 128 | 0 | test_in | supercritical | 24 | 0 | 0.04336 | 0.1135 | 0.0001271 | 0.003122 | 0.01306 | 0.09316 | 0.0192 | 0.02947 | 8 | 0.001564 | 0.005375 | 24 | 0.002062 | 0.008595 | 48 | 0.002533 | 0.01025 | 72 | 0.01089 | 0.08059 |
| fno | apno | 256x97 | 256 | 0 | test_ext | supercritical | 14 | 0 | 0.1171 | 0.1829 | 0.001239 | 0.003129 | 0.07016 | 0.3034 | 0.02074 | 0.02799 | 8 | 0.002189 | 0.006641 | 24 | 0.006813 | 0.01152 | 48 | 0.005363 | 0.01521 | 72 | 0.0152 | 0.04568 |
| fno | apno | 256x97 | 256 | 0 | test_in | subcritical | 37 | 0 | 0.01741 | 0.04233 | -2.591e-06 | 3.941e-05 | 0.002804 | 0.00633 | 0.01813 | 0.02456 | 8 | 0.0004625 | 0.003545 | 24 | 0.0004435 | 0.002237 | 48 | 0.0008836 | 0.003129 | 72 | 0.005303 | 0.01653 |
| fno | apno | 256x97 | 256 | 0 | test_in | supercritical | 24 | 0 | 0.0446 | 0.1121 | 9.466e-05 | 0.003064 | 0.01057 | 0.05119 | 0.01766 | 0.02553 | 8 | 0.0007326 | 0.003022 | 24 | 0.001047 | 0.008109 | 48 | 0.002079 | 0.007789 | 72 | 0.006283 | 0.02498 |
| fno | apno | 256x97 | 512 | 0 | test_ext | supercritical | 14 | 0 | 0.1191 | 0.1962 | 0.002182 | 0.004608 | 0.03635 | 0.1216 | 0.01831 | 0.03205 | 8 | 0.003809 | 0.02211 | 24 | 0.004645 | 0.01699 | 48 | 0.005085 | 0.02551 | 72 | 0.01852 | 0.08516 |
| fno | apno | 256x97 | 512 | 0 | test_in | subcritical | 37 | 0 | 0.01709 | 0.04355 | -2.768e-06 | 1.862e-05 | 0.002642 | 0.005683 | 0.01639 | 0.02082 | 8 | 0.0003946 | 0.002091 | 24 | 0.0004117 | 0.0009629 | 48 | 0.001086 | 0.003659 | 72 | 0.005373 | 0.009964 |
| fno | apno | 256x97 | 512 | 0 | test_in | supercritical | 24 | 0 | 0.04319 | 0.1064 | 9.055e-05 | 0.002915 | 0.009775 | 0.0241 | 0.0157 | 0.02494 | 8 | 0.0009355 | 0.003998 | 24 | 0.001065 | 0.004759 | 48 | 0.002076 | 0.008181 | 72 | 0.005217 | 0.02389 |
| fno | apno | 256x97 | 64 | 0 | test_ext | supercritical | 14 | 4 | 0.08825 | 0.1285 | 0.003092 | 0.003621 | 0.2221 | 540.8 | 0.02674 | 0.0351 | 8 | 0.01228 | 0.01924 | 24 | 0.007196 | 0.04312 | 48 | 0.006201 | 0.02381 | 72 | 0.0227 | 0.1742 |
| fno | apno | 256x97 | 64 | 0 | test_in | subcritical | 37 | 0 | 0.01583 | 0.03759 | -1.184e-05 | 0.0004443 | 0.003292 | 0.009339 | 0.02307 | 0.04176 | 8 | 0.002861 | 0.006567 | 24 | 0.001485 | 0.005377 | 48 | 0.002598 | 0.01261 | 72 | 0.01066 | 0.04202 |
| fno | apno | 256x97 | 64 | 0 | test_in | supercritical | 24 | 0 | 0.03782 | 0.1156 | 0.0001878 | 0.003166 | 0.01235 | 0.09931 | 0.02413 | 0.02931 | 8 | 0.002485 | 0.009236 | 24 | 0.002495 | 0.01421 | 48 | 0.005432 | 0.01895 | 72 | 0.02199 | 0.0487 |
| fno | apno | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1183 | 0.2097 | 0.001765 | 0.007349 | 0.04296 | 0.142 | 0.0187 | 0.03065 | 8 | 0.0006017 | 0.005547 | 24 | 0.001758 | 0.007181 | 48 | 0.008938 | 0.02274 | 72 | 0.05675 | 0.2903 |
| fno | apno | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01734 | 0.04259 | 1.688e-07 | 1.864e-05 | 0.001212 | 0.003353 | 0.01386 | 0.01827 | 8 | 0.000476 | 0.001533 | 24 | 0.0001645 | 0.001246 | 48 | 0.0003559 | 0.002796 | 72 | 0.00292 | 0.008189 |
| fno | apno | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04001 | 0.1113 | 9.91e-05 | 0.002757 | 0.00733 | 0.02096 | 0.01357 | 0.02053 | 8 | 0.0006967 | 0.00223 | 24 | 0.0008004 | 0.005172 | 48 | 0.001535 | 0.003635 | 72 | 0.006428 | 0.01791 |
| fno | apno | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1005 | 0.1887 | 0.002299 | 0.003472 | 0.02963 | 0.06853 | 0.01944 | 0.02878 | 8 | 0.002282 | 0.005357 | 24 | 0.002506 | 0.00931 | 48 | 0.00422 | 0.02767 | 72 | 0.02166 | 0.09463 |
| fno | apno | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.0178 | 0.04335 | 1.988e-05 | 9.874e-05 | 0.001675 | 0.005243 | 0.01592 | 0.02114 | 8 | 0.000713 | 0.001445 | 24 | 0.0003084 | 0.001044 | 48 | 0.001135 | 0.00246 | 72 | 0.004448 | 0.01645 |
| fno | apno | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03925 | 0.09285 | 0.0001767 | 0.002237 | 0.007633 | 0.02382 | 0.01469 | 0.01898 | 8 | 0.0008036 | 0.003757 | 24 | 0.0007573 | 0.004146 | 48 | 0.001151 | 0.00665 | 72 | 0.008246 | 0.02304 |
| fno | apno | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1109 | 0.1926 | 0.002236 | 0.005794 | 0.04076 | 0.1146 | 0.0208 | 0.03007 | 8 | 0.003275 | 0.008687 | 24 | 0.003475 | 0.006352 | 48 | 0.006641 | 0.02024 | 72 | 0.01993 | 0.07924 |
| fno | apno | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01771 | 0.04063 | -8.124e-07 | 9.397e-06 | 0.001542 | 0.003066 | 0.01404 | 0.01894 | 8 | 0.0002847 | 0.001145 | 24 | 0.0002536 | 0.001448 | 48 | 0.0006223 | 0.002045 | 72 | 0.002078 | 0.009644 |
| fno | apno | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04381 | 0.111 | 8.642e-05 | 0.002568 | 0.007701 | 0.01931 | 0.01497 | 0.02342 | 8 | 0.0004594 | 0.001449 | 24 | 0.0005996 | 0.002212 | 48 | 0.0008668 | 0.003892 | 72 | 0.002851 | 0.05124 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.102 | 0.1844 | 0.001934 | 0.003119 | 0.03581 | 0.09825 | 0.01276 | 0.02448 | 4 | 0.00153 | 0.005311 | 12 | 0.001347 | 0.008231 | 24 | 0.004708 | 0.02073 | 36 | 0.02081 | 0.05352 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01907 | 0.04609 | 1.481e-06 | 1.44e-05 | 0.0007868 | 0.002619 | 0.0008255 | 0.002693 | 4 | 8.992e-05 | 0.0004368 | 12 | 0.0001115 | 0.0005846 | 24 | 0.0002478 | 0.001531 | 36 | 0.001399 | 0.005379 |
| fno | apno-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04222 | 0.09458 | 7.132e-05 | 0.002103 | 0.005675 | 0.02563 | 0.002605 | 0.01138 | 4 | 0.0002456 | 0.001152 | 12 | 0.0003465 | 0.001621 | 24 | 0.0009427 | 0.007795 | 36 | 0.004745 | 0.02384 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1083 | 0.2333 | 0.001422 | 0.003955 | 0.02749 | 0.1246 | 0.01351 | 0.03029 | 4 | 0.001437 | 0.005981 | 12 | 0.001934 | 0.007641 | 24 | 0.003449 | 0.0107 | 36 | 0.009527 | 0.03538 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01803 | 0.04635 | 1.074e-06 | 2.113e-05 | 0.0007648 | 0.002572 | 0.0008901 | 0.002235 | 4 | 0.0001004 | 0.0003808 | 12 | 7.205e-05 | 0.0006255 | 24 | 0.0002476 | 0.001581 | 36 | 0.001822 | 0.0058 |
| fno | apno-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.0402 | 0.09922 | 0.0001026 | 0.002348 | 0.005976 | 0.01863 | 0.002496 | 0.01635 | 4 | 0.000353 | 0.00187 | 12 | 0.0003003 | 0.00322 | 24 | 0.0008949 | 0.005197 | 36 | 0.002793 | 0.01264 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1232 | 0.1939 | 0.002538 | 0.007546 | 0.02915 | 0.1021 | 0.01187 | 0.03866 | 4 | 0.002471 | 0.008774 | 12 | 0.001419 | 0.008838 | 24 | 0.00487 | 0.04365 | 36 | 0.01949 | 0.05613 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01876 | 0.04462 | 6.77e-08 | 4.365e-05 | 0.0007299 | 0.002382 | 0.0008346 | 0.002866 | 4 | 8.885e-05 | 0.0008983 | 12 | 8.9e-05 | 0.0006039 | 24 | 0.0002544 | 0.00125 | 36 | 0.001881 | 0.004278 |
| fno | apno-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04248 | 0.1008 | 9.809e-05 | 0.002191 | 0.007719 | 0.02193 | 0.002895 | 0.01452 | 4 | 0.0002955 | 0.003139 | 12 | 0.0005246 | 0.00532 | 24 | 0.0007086 | 0.006153 | 36 | 0.00179 | 0.01305 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1017 | 0.1795 | 0.002111 | 0.003205 | 0.03395 | 0.08876 | 0.01901 | 0.02876 | 8 | 0.001478 | 0.004522 | 24 | 0.001302 | 0.008007 | 48 | 0.004787 | 0.0225 | 72 | 0.01493 | 0.05181 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01794 | 0.04387 | 1.013e-06 | 1.88e-05 | 0.001131 | 0.002608 | 0.01406 | 0.01857 | 8 | 0.0002485 | 0.0007631 | 24 | 0.0001003 | 0.0008224 | 48 | 0.0007311 | 0.002207 | 72 | 0.00622 | 0.01238 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.0408 | 0.09428 | 7.18e-05 | 0.002155 | 0.006433 | 0.02246 | 0.0143 | 0.01918 | 8 | 0.0003882 | 0.001478 | 24 | 0.000566 | 0.001772 | 48 | 0.001352 | 0.00989 | 72 | 0.008694 | 0.01879 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1028 | 0.2388 | 0.001252 | 0.003616 | 0.02883 | 0.1238 | 0.02052 | 0.03445 | 8 | 0.001446 | 0.006995 | 24 | 0.001212 | 0.008681 | 48 | 0.006859 | 0.01535 | 72 | 0.01259 | 0.03691 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01772 | 0.04389 | -9.428e-07 | 2.042e-05 | 0.001858 | 0.002774 | 0.01376 | 0.01851 | 8 | 0.0005924 | 0.001269 | 24 | 0.0002344 | 0.00109 | 48 | 0.0007602 | 0.00193 | 72 | 0.002548 | 0.008585 |
| fno | apno-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04217 | 0.1022 | 0.0001106 | 0.002061 | 0.006454 | 0.01897 | 0.01384 | 0.02044 | 8 | 0.0008186 | 0.001949 | 24 | 0.0006848 | 0.002932 | 48 | 0.001009 | 0.008975 | 72 | 0.002864 | 0.01262 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1314 | 0.193 | 0.002535 | 0.007606 | 0.03138 | 0.09383 | 0.01803 | 0.04092 | 8 | 0.001871 | 0.00904 | 24 | 0.00391 | 0.01527 | 48 | 0.006749 | 0.04986 | 72 | 0.0196 | 0.0941 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01812 | 0.04234 | -1.266e-06 | 5.931e-05 | 0.001279 | 0.002335 | 0.01412 | 0.01924 | 8 | 0.000381 | 0.001697 | 24 | 0.0002318 | 0.0006866 | 48 | 0.0003681 | 0.0015 | 72 | 0.004275 | 0.009511 |
| fno | apno-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04401 | 0.1116 | 0.0001351 | 0.002185 | 0.007945 | 0.0262 | 0.01357 | 0.01906 | 8 | 0.001298 | 0.003136 | 24 | 0.0004946 | 0.006932 | 48 | 0.0007362 | 0.01425 | 72 | 0.004374 | 0.01267 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1042 | 0.1562 | 0.001932 | 0.003938 | 0.028 | 0.1175 | 0.01534 | 0.04152 | 4 | 0.002475 | 0.008578 | 12 | 0.004677 | 0.016 | 24 | 0.009488 | 0.0239 | 36 | 0.0502 | 0.09651 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01893 | 0.04482 | 9.082e-06 | 8.073e-05 | 0.001785 | 0.004627 | 0.002968 | 0.00613 | 4 | 0.001324 | 0.002396 | 12 | 0.0004618 | 0.002685 | 24 | 0.0009241 | 0.00305 | 36 | 0.004897 | 0.01411 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04412 | 0.1121 | 0.0001244 | 0.002639 | 0.007174 | 0.02628 | 0.005318 | 0.01135 | 4 | 0.002197 | 0.00419 | 12 | 0.001081 | 0.005315 | 24 | 0.001074 | 0.008293 | 36 | 0.003202 | 0.04449 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1112 | 0.1622 | 0.001962 | 0.004028 | 0.03077 | 0.1119 | 0.0172 | 0.03931 | 8 | 0.001854 | 0.01099 | 24 | 0.005617 | 0.01518 | 48 | 0.01041 | 0.0237 | 72 | 0.05073 | 0.097 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0199 | 0.04712 | 8.262e-06 | 7.77e-05 | 0.002351 | 0.005032 | 0.007703 | 0.00967 | 8 | 0.0009782 | 0.002047 | 24 | 0.0004677 | 0.002803 | 48 | 0.0008206 | 0.002667 | 72 | 0.005336 | 0.01327 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04637 | 0.1217 | 0.000127 | 0.002653 | 0.007342 | 0.0306 | 0.008065 | 0.01384 | 8 | 0.002072 | 0.004263 | 24 | 0.001547 | 0.00617 | 48 | 0.001926 | 0.01096 | 72 | 0.003358 | 0.04766 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.09693 | 0.1492 | 0.001888 | 0.003902 | 0.02579 | 0.1153 | 0.01468 | 0.04554 | 4 | 0.002749 | 0.007582 | 12 | 0.002949 | 0.01079 | 24 | 0.01047 | 0.03046 | 36 | 0.06546 | 0.1173 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01835 | 0.04306 | 2.183e-06 | 2.513e-05 | 0.001304 | 0.003328 | 0.001054 | 0.002228 | 4 | 9.551e-05 | 0.001295 | 12 | 0.0001804 | 0.0005396 | 24 | 0.0004336 | 0.001467 | 36 | 0.003372 | 0.009191 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04125 | 0.1109 | 8.006e-05 | 0.002613 | 0.006209 | 0.03261 | 0.002614 | 0.007626 | 4 | 0.0004784 | 0.002858 | 12 | 0.0006762 | 0.004052 | 24 | 0.0007703 | 0.004146 | 36 | 0.003181 | 0.02333 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.112 | 0.1549 | 0.001664 | 0.003677 | 0.03507 | 0.1626 | 0.01318 | 0.06036 | 4 | 0.001494 | 0.01129 | 12 | 0.004927 | 0.01904 | 24 | 0.01309 | 0.06185 | 36 | 0.03343 | 0.1121 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.019 | 0.04482 | 1.061e-06 | 3.731e-05 | 0.001072 | 0.00348 | 0.001121 | 0.002882 | 4 | 0.0001908 | 0.001769 | 12 | 0.0001741 | 0.001129 | 24 | 0.0005971 | 0.002524 | 36 | 0.002912 | 0.006287 |
| fno | neuralop-nopatience | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.03998 | 0.1153 | 6.04e-05 | 0.002649 | 0.006253 | 0.02764 | 0.002578 | 0.01148 | 4 | 0.0003752 | 0.003948 | 12 | 0.0005216 | 0.003921 | 24 | 0.001163 | 0.007837 | 36 | 0.005368 | 0.01721 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1013 | 0.1511 | 0.002399 | 0.003394 | 0.02701 | 0.04285 | 0.01361 | 0.02256 | 4 | 0.002007 | 0.009023 | 12 | 0.002833 | 0.02237 | 24 | 0.006285 | 0.02449 | 36 | 0.008219 | 0.05997 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01853 | 0.0459 | 4.086e-06 | 2.329e-05 | 0.001035 | 0.002841 | 0.0009895 | 0.002575 | 4 | 8.768e-05 | 0.0007358 | 12 | 0.0001894 | 0.000829 | 24 | 0.0002527 | 0.001347 | 36 | 0.001994 | 0.01211 |
| fno | neuralop-nopatience | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04228 | 0.1093 | 0.0001379 | 0.00254 | 0.005885 | 0.01846 | 0.002582 | 0.01155 | 4 | 0.0002929 | 0.002043 | 12 | 0.000471 | 0.003151 | 24 | 0.0008899 | 0.004987 | 36 | 0.003716 | 0.01684 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1053 | 0.1772 | 0.001944 | 0.003813 | 0.0292 | 0.05731 | 0.01211 | 0.03702 | 4 | 0.00245 | 0.007637 | 12 | 0.002219 | 0.007218 | 24 | 0.004478 | 0.01968 | 36 | 0.01101 | 0.04931 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.01911 | 0.04568 | 1.708e-06 | 3.055e-05 | 0.001155 | 0.004403 | 0.001059 | 0.003176 | 4 | 0.0001995 | 0.0007866 | 12 | 0.0002018 | 0.0006974 | 24 | 0.0005511 | 0.001442 | 36 | 0.003283 | 0.01124 |
| fno | neuralop-nopatience | 128x49 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.04139 | 0.1239 | 7.691e-05 | 0.002837 | 0.007258 | 0.03382 | 0.002559 | 0.007918 | 4 | 0.0003885 | 0.003219 | 12 | 0.0005943 | 0.004085 | 24 | 0.0009508 | 0.004845 | 36 | 0.004212 | 0.05195 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1075 | 0.1468 | 0.001801 | 0.002886 | 0.02489 | 0.03946 | 0.01556 | 0.05452 | 4 | 0.002804 | 0.009881 | 12 | 0.008184 | 0.02127 | 24 | 0.004617 | 0.03131 | 36 | 0.009278 | 0.0363 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01832 | 0.04469 | -7.781e-07 | 1.376e-05 | 0.001107 | 0.003965 | 0.0009506 | 0.003381 | 4 | 0.00017 | 0.0005494 | 12 | 0.0001493 | 0.0006591 | 24 | 0.000394 | 0.001138 | 36 | 0.002005 | 0.01229 |
| fno | neuralop-nopatience | 128x49 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.03941 | 0.1262 | 9.45e-05 | 0.002352 | 0.007639 | 0.02499 | 0.002366 | 0.009143 | 4 | 0.0004233 | 0.002252 | 12 | 0.0003959 | 0.002713 | 24 | 0.0007601 | 0.006933 | 36 | 0.003756 | 0.02886 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1064 | 0.1544 | 0.001922 | 0.003979 | 0.02724 | 0.1125 | 0.01526 | 0.04166 | 8 | 0.002468 | 0.01008 | 24 | 0.003567 | 0.01037 | 48 | 0.01021 | 0.03066 | 72 | 0.0658 | 0.1192 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01908 | 0.04455 | 1.704e-06 | 2.434e-05 | 0.001853 | 0.003998 | 0.006339 | 0.008263 | 8 | 0.0003097 | 0.001664 | 24 | 0.0002216 | 0.000616 | 48 | 0.0004558 | 0.001589 | 72 | 0.003076 | 0.009273 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04299 | 0.1191 | 8.49e-05 | 0.002606 | 0.006716 | 0.0327 | 0.00665 | 0.01086 | 8 | 0.0006342 | 0.002012 | 24 | 0.0003447 | 0.005002 | 48 | 0.001323 | 0.006729 | 72 | 0.002868 | 0.02233 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1153 | 0.1548 | 0.001687 | 0.003748 | 0.03771 | 0.1516 | 0.01347 | 0.06325 | 8 | 0.00175 | 0.01243 | 24 | 0.004714 | 0.01906 | 48 | 0.01393 | 0.06434 | 72 | 0.03456 | 0.1155 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.0192 | 0.04508 | 1.52e-06 | 3.53e-05 | 0.001982 | 0.00774 | 0.007401 | 0.01126 | 8 | 0.000549 | 0.002687 | 24 | 0.0001908 | 0.001268 | 48 | 0.000622 | 0.002706 | 72 | 0.002385 | 0.007615 |
| fno | neuralop-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04428 | 0.1284 | 6.157e-05 | 0.002677 | 0.007863 | 0.02679 | 0.00766 | 0.0104 | 8 | 0.0005606 | 0.003021 | 24 | 0.0005204 | 0.003984 | 48 | 0.001021 | 0.009378 | 72 | 0.005335 | 0.01831 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1134 | 0.1573 | 0.002393 | 0.003356 | 0.02779 | 0.04136 | 0.01574 | 0.02402 | 8 | 0.002184 | 0.009098 | 24 | 0.002575 | 0.02391 | 48 | 0.01024 | 0.02935 | 72 | 0.01451 | 0.06981 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01808 | 0.04622 | 3.354e-06 | 1.979e-05 | 0.001721 | 0.005974 | 0.006757 | 0.01146 | 8 | 0.0003871 | 0.001315 | 24 | 0.0004978 | 0.001049 | 48 | 0.0003513 | 0.001541 | 72 | 0.00217 | 0.01439 |
| fno | neuralop-nopatience | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04483 | 0.1199 | 0.0001345 | 0.00252 | 0.006948 | 0.03009 | 0.007015 | 0.0104 | 8 | 0.0002842 | 0.002294 | 24 | 0.0004983 | 0.004568 | 48 | 0.00179 | 0.00739 | 72 | 0.003144 | 0.01098 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_ext | supercritical | 14 | 0 | 0.1086 | 0.1763 | 0.00197 | 0.003904 | 0.03562 | 0.05666 | 0.01295 | 0.03486 | 8 | 0.002418 | 0.007929 | 24 | 0.001977 | 0.008511 | 48 | 0.005704 | 0.01966 | 72 | 0.01127 | 0.06055 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | subcritical | 37 | 0 | 0.02017 | 0.0479 | 2.265e-06 | 2.802e-05 | 0.002058 | 0.007619 | 0.007077 | 0.01047 | 8 | 0.0003313 | 0.0011 | 24 | 0.0009279 | 0.001564 | 48 | 0.0008567 | 0.002041 | 72 | 0.003742 | 0.01253 |
| fno | neuralop-nopatience | 256x97 | 896 | 3 | test_in | supercritical | 24 | 0 | 0.0456 | 0.125 | 8.713e-05 | 0.002837 | 0.008484 | 0.03583 | 0.007384 | 0.009472 | 8 | 0.0002409 | 0.003488 | 24 | 0.001128 | 0.004839 | 48 | 0.001418 | 0.00479 | 72 | 0.003972 | 0.06196 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_ext | supercritical | 14 | 0 | 0.1131 | 0.1513 | 0.001851 | 0.002893 | 0.02552 | 0.0392 | 0.01485 | 0.05371 | 8 | 0.002179 | 0.009058 | 24 | 0.008033 | 0.0221 | 48 | 0.00614 | 0.03022 | 72 | 0.01077 | 0.03765 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | subcritical | 37 | 0 | 0.01925 | 0.04639 | -1.014e-06 | 1.401e-05 | 0.002989 | 0.006524 | 0.006074 | 0.009461 | 8 | 0.0004129 | 0.0009117 | 24 | 0.0003861 | 0.001183 | 48 | 0.00051 | 0.001507 | 72 | 0.002803 | 0.01278 |
| fno | neuralop-nopatience | 256x97 | 896 | 4 | test_in | supercritical | 24 | 0 | 0.04288 | 0.1322 | 8.94e-05 | 0.002337 | 0.008397 | 0.02988 | 0.006531 | 0.01036 | 8 | 0.0004856 | 0.001788 | 24 | 0.0005157 | 0.00244 | 48 | 0.0009827 | 0.007959 | 72 | 0.003235 | 0.02173 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1108 | 0.3078 | 0.001448 | 0.003932 | 0.08479 | 0.2304 | 0.0188 | 0.03325 | 4 | 0.001909 | 0.008656 | 12 | 0.005462 | 0.02372 | 24 | 0.008693 | 0.02109 | 36 | 0.02137 | 0.07256 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0183 | 0.04527 | 1.117e-06 | 1.709e-05 | 0.001231 | 0.005184 | 0.001603 | 0.004569 | 4 | 0.0001434 | 0.001142 | 12 | 0.0001702 | 0.0009326 | 24 | 0.0003373 | 0.00149 | 36 | 0.001488 | 0.007589 |
| fno | physicsnemo | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04206 | 0.1149 | 7.565e-05 | 0.002788 | 0.01471 | 0.06824 | 0.003859 | 0.01296 | 4 | 0.0005319 | 0.004474 | 12 | 0.001186 | 0.004943 | 24 | 0.001101 | 0.01082 | 36 | 0.004635 | 0.02709 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1134 | 0.2867 | 0.001557 | 0.00409 | 0.08565 | 0.1813 | 0.02757 | 0.03644 | 8 | 0.002525 | 0.006791 | 24 | 0.007215 | 0.02674 | 48 | 0.01452 | 0.03445 | 72 | 0.02582 | 0.05451 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01825 | 0.04509 | -5.233e-06 | 5.045e-05 | 0.001764 | 0.005778 | 0.01855 | 0.02302 | 8 | 0.001587 | 0.003388 | 24 | 0.001323 | 0.002272 | 48 | 0.001026 | 0.00242 | 72 | 0.002797 | 0.00745 |
| fno | physicsnemo | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04487 | 0.1078 | 0.0001191 | 0.002899 | 0.01442 | 0.05737 | 0.01942 | 0.02519 | 8 | 0.001922 | 0.004155 | 24 | 0.001824 | 0.004921 | 48 | 0.00218 | 0.02263 | 72 | 0.006394 | 0.04446 |
| fno | s2 | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1074 | 0.206 | 0.001731 | 0.007165 | 0.04186 | 0.1664 | 0.01183 | 0.02381 | 4 | 0.0008141 | 0.005986 | 12 | 0.002884 | 0.008724 | 24 | 0.007913 | 0.02057 | 36 | 0.0667 | 0.3135 |
| fno | s2 | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01862 | 0.04629 | 4.714e-06 | 1.973e-05 | 0.0008137 | 0.00246 | 0.0008439 | 0.002334 | 4 | 8.049e-05 | 0.0005023 | 12 | 8.088e-05 | 0.0008616 | 24 | 0.000235 | 0.001065 | 36 | 0.001933 | 0.007479 |
| fno | s2 | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04117 | 0.1072 | 9.422e-05 | 0.002609 | 0.007312 | 0.02213 | 0.002192 | 0.01393 | 4 | 0.0003272 | 0.003293 | 12 | 0.0002572 | 0.00424 | 24 | 0.0009133 | 0.00329 | 36 | 0.0043 | 0.01984 |
| fno | s2 | 128x49 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1026 | 0.1874 | 0.001847 | 0.003129 | 0.03325 | 0.08028 | 0.01287 | 0.02412 | 4 | 0.001315 | 0.006066 | 12 | 0.001366 | 0.01071 | 24 | 0.005145 | 0.01689 | 36 | 0.0206 | 0.05244 |
| fno | s2 | 128x49 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01903 | 0.04599 | 1.418e-06 | 1.332e-05 | 0.0007676 | 0.002555 | 0.0008137 | 0.002634 | 4 | 8.751e-05 | 0.0004244 | 12 | 0.0001253 | 0.000544 | 24 | 0.0002525 | 0.001384 | 36 | 0.001371 | 0.00571 |
| fno | s2 | 128x49 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04221 | 0.09487 | 7.502e-05 | 0.002095 | 0.005274 | 0.02409 | 0.002593 | 0.01088 | 4 | 0.0002679 | 0.00124 | 12 | 0.0003918 | 0.00158 | 24 | 0.0009949 | 0.00795 | 36 | 0.005588 | 0.02476 |
| fno | s2 | 128x49 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1134 | 0.2026 | 0.002256 | 0.00597 | 0.04385 | 0.127 | 0.01367 | 0.02212 | 4 | 0.00248 | 0.008189 | 12 | 0.002649 | 0.004213 | 24 | 0.003972 | 0.01063 | 36 | 0.02023 | 0.09429 |
| fno | s2 | 128x49 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.01844 | 0.04531 | 9.011e-07 | 4.332e-05 | 0.0007931 | 0.002205 | 0.0007972 | 0.002484 | 4 | 9.61e-05 | 0.0009754 | 12 | 0.0001365 | 0.001064 | 24 | 0.0001519 | 0.0009223 | 36 | 0.001506 | 0.01201 |
| fno | s2 | 128x49 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04473 | 0.1086 | 0.0001164 | 0.002634 | 0.006406 | 0.01711 | 0.002419 | 0.01386 | 4 | 0.0003443 | 0.001147 | 12 | 0.0004051 | 0.001428 | 24 | 0.0005527 | 0.002411 | 36 | 0.003405 | 0.06741 |
| fno | s2 | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1186 | 0.2074 | 0.001712 | 0.007094 | 0.04056 | 0.1552 | 0.01918 | 0.02997 | 8 | 0.0006906 | 0.005746 | 24 | 0.001832 | 0.007669 | 48 | 0.008522 | 0.02323 | 72 | 0.06242 | 0.2994 |
| fno | s2 | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0174 | 0.04239 | 3.303e-06 | 2.644e-05 | 0.001205 | 0.003454 | 0.01393 | 0.01807 | 8 | 0.000394 | 0.001337 | 24 | 0.00019 | 0.001255 | 48 | 0.0004296 | 0.00282 | 72 | 0.00344 | 0.008353 |
| fno | s2 | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.03975 | 0.1114 | 0.0001028 | 0.002729 | 0.008234 | 0.02261 | 0.01363 | 0.02056 | 8 | 0.000698 | 0.002597 | 24 | 0.0007157 | 0.005425 | 48 | 0.001622 | 0.003581 | 72 | 0.006701 | 0.01892 |
| fno | s2 | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1028 | 0.1822 | 0.002068 | 0.003201 | 0.03181 | 0.07202 | 0.01867 | 0.02848 | 8 | 0.00134 | 0.005448 | 24 | 0.001354 | 0.01036 | 48 | 0.004058 | 0.01881 | 72 | 0.01593 | 0.05596 |
| fno | s2 | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01795 | 0.04386 | 6.978e-07 | 1.645e-05 | 0.001111 | 0.002672 | 0.01406 | 0.01873 | 8 | 0.0002486 | 0.0007356 | 24 | 0.0001186 | 0.0007968 | 48 | 0.0007674 | 0.002104 | 72 | 0.006133 | 0.01243 |
| fno | s2 | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.04061 | 0.09424 | 7.364e-05 | 0.002152 | 0.006289 | 0.0223 | 0.01435 | 0.0192 | 8 | 0.0003708 | 0.001546 | 24 | 0.000545 | 0.001755 | 48 | 0.001226 | 0.01017 | 72 | 0.007436 | 0.02193 |
| fno | s2 | 256x97 | 896 | 2 | test_ext | supercritical | 14 | 0 | 0.1103 | 0.187 | 0.002224 | 0.005964 | 0.04182 | 0.1064 | 0.02125 | 0.02984 | 8 | 0.003141 | 0.00942 | 24 | 0.003419 | 0.00521 | 48 | 0.005737 | 0.01398 | 72 | 0.01508 | 0.08736 |
| fno | s2 | 256x97 | 896 | 2 | test_in | subcritical | 37 | 0 | 0.0177 | 0.04055 | -4.057e-07 | 1.026e-05 | 0.001581 | 0.002984 | 0.01416 | 0.01894 | 8 | 0.0003152 | 0.001162 | 24 | 0.0002456 | 0.001502 | 48 | 0.0006689 | 0.00209 | 72 | 0.002173 | 0.01006 |
| fno | s2 | 256x97 | 896 | 2 | test_in | supercritical | 24 | 0 | 0.04382 | 0.1085 | 8.792e-05 | 0.002566 | 0.007999 | 0.01859 | 0.0149 | 0.02324 | 8 | 0.0004694 | 0.001417 | 24 | 0.0005249 | 0.001945 | 48 | 0.0009586 | 0.00346 | 72 | 0.003853 | 0.06016 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_ext | supercritical | 14 | 0 | 0.1376 | 0.2152 | 0.001979 | 0.003269 | 6.735e-06 | 1.168e-05 | 6.401e-09 | 7.386e-09 | 8 | 7.151e-05 | 0.000133 | 24 | 0.0003365 | 0.0005027 | 48 | 0.0001558 | 0.001262 | 72 | 7.321e-06 | 1.161e-05 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_in | subcritical | 37 | 0 | 0.03282 | 0.06447 | 1.709e-06 | 5.348e-06 | 5.96e-07 | 1.311e-06 | 6.96e-09 | 8.521e-09 | 8 | 3.108e-06 | 5.383e-06 | 24 | 6.151e-06 | 1.536e-05 | 48 | 1.71e-06 | 4.482e-06 | 72 | 1.308e-07 | 5.353e-07 |
| su2-256x97 | floor | 256x97 | 0 | 0 | test_in | supercritical | 24 | 0 | 0.06114 | 0.1507 | 9.09e-05 | 0.002755 | 1.669e-06 | 1.121e-05 | 6.861e-09 | 9.77e-09 | 8 | 5.929e-06 | 6.883e-05 | 24 | 1.747e-05 | 0.0002206 | 48 | 3.72e-06 | 0.0001289 | 72 | 3.854e-07 | 8.753e-06 |
