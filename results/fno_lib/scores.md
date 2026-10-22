# The library FNOs, scored as S4 scores its contestants

Command: `python scripts/probes/score_fno_libraries.py` over `runs/predictions_lib`, with S2 references fno-n896-s0, fno-n896-s1, fno-n896-s2 (variant `s2`). A library's variant is the library's name.

## Runs

best_val is loss A on the validation cases at 128x49; transfer is the relative L2 difference between the 256x97 prediction sampled at every second node and the 128x49 one. Wall times are on the device named: GPU wall time on different hardware is not one cost basis.

| tag | source | device | n_cases | n_params | epochs_run | best_epoch | best_val | val_rel_l2_256x97 | transfer_val_rel_l2 | wall_s | s_per_epoch | query_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apno-n128-s0 | apno 0.1.0 | cuda | 126 | 3555476 | 300 | 288 | 0.05065 | 0.07419 | 0.04758 | 73.82 | 0.2461 | 0.0009709 |
| apno-n256-s0 | apno 0.1.0 | cuda | 252 | 3555476 | 300 | 284 | 0.03322 | 0.06082 | 0.04744 | 127.8 | 0.4258 | 0.0009648 |
| apno-n512-s0 | apno 0.1.0 | cuda | 502 | 3555476 | 300 | 287 | 0.01781 | 0.04768 | 0.04312 | 238.3 | 0.7943 | 0.0009766 |
| apno-n64-s0 | apno 0.1.0 | cuda | 63 | 3555476 | 300 | 297 | 0.08493 | 0.1015 | 0.04425 | 50.15 | 0.1672 | 0.0009833 |
| apno-n896-s0 | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 284 | 0.01256 | 0.0405 | 0.03456 | 409.9 | 1.366 | 0.001097 |
| apno-n896-s1 | apno 0.1.0 | cuda | 881 | 3555476 | 191 | 160 | 0.02035 | 0.04464 | 0.03547 | 260.8 | 1.365 | 0.001097 |
| apno-n896-s1-nopatience | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 293 | 0.01264 | 0.04059 | 0.03551 | 408.4 | 1.361 | 0.001129 |
| apno-n896-s2 | apno 0.1.0 | cuda | 881 | 3555476 | 300 | 294 | 0.01169 | 0.04238 | 0.03808 | 408.6 | 1.362 | 0.0009714 |
| neuralop-n896-s0 | neuralop 2.0.0 | cuda | 881 | 3568468 | 183 | 152 | 0.0245 | 0.05083 | 0.02594 | 243.3 | 1.33 | 0.002119 |
| neuralop-n896-s0-nopatience | neuralop 2.0.0 | cuda | 881 | 3568468 | 300 | 294 | 0.01225 | 0.04222 | 0.0271 | 398.1 | 1.327 | 0.002333 |
| physicsnemo-n896-s0 | physicsnemo 2.2.2 | cuda | 881 | 7095452 | 300 | 282 | 0.02432 | 0.05583 | 0.04681 | 424.4 | 1.415 | 0.00119 |
| fno-n896-s0 | S2 FNO2d | mps | 881 |  | 300 | 278 | 0.0126 |  | 0.03477 | 2160 | 7.2 | 0.0536 |
| fno-n896-s1 | S2 FNO2d | mps | 881 |  | 300 | 293 | 0.01274 |  | 0.03548 | 2242 | 7.474 | 0.01032 |
| fno-n896-s2 | S2 FNO2d | mps | 881 |  | 300 | 294 | 0.01178 |  | 0.03788 | 2237 | 7.457 | 0.01077 |

## Near field against the floor

x_floor is the contestant's mean near-field relative L2 over the floor's (SU2 at 256x97) on the same split and regime. The floor exists at 256x97 only, so 128x49 rows are divided by it too.

| method | variant | grid | n_train | seed | split | regime | n_ok | near_field_rel_l2_mean | floor_mean | x_floor |
|---|---|---|---|---|---|---|---|---|---|---|
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
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0.2563 | 0.06511 | 3.936 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0.04217 | 0.007953 | 5.302 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0.06093 | 0.02413 | 2.525 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2733 | 0.06511 | 4.197 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01888 | 0.007953 | 2.374 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.05578 | 0.02413 | 2.311 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2667 | 0.06511 | 4.096 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.02853 | 0.007953 | 3.587 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.06584 | 0.02413 | 2.728 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0.2396 | 0.06511 | 3.681 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0.01171 | 0.007953 | 1.472 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0.03943 | 0.02413 | 1.634 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0.2338 | 0.06511 | 3.592 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0.01796 | 0.007953 | 2.258 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0.04445 | 0.02413 | 1.842 |
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
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0 | 0.2563 | 0.2436 | 171.7 | 1.281 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0 | 0.04217 | 0.02778 | 35.34 | 0.2343 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0 | 0.06093 | 0.04505 | 32.41 | 0.5294 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2733 | 0.2089 | 242.3 | 1.18 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01888 | 0.01194 | 23.19 | 0.2514 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.05578 | 0.05646 | 78.92 | 0.416 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2667 | 0.2163 | 268.3 | 1.19 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.02853 | 0.01827 | 13.56 | 0.4288 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.06584 | 0.05642 | 86.96 | 0.5443 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2396 | 0.2019 | 217.1 | 1.174 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01171 | 0.01052 | 9.813 | 0.2455 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.03943 | 0.02282 | 38.78 | 0.3416 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0 | 0.2338 | 0.2126 | 243.6 | 1.187 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0 | 0.01796 | 0.01519 | 20.49 | 0.4171 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0 | 0.04445 | 0.02348 | 47.5 | 0.4359 |
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
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03494 | 0.2621 | -0.09358 | 0.3921 | 1.114 | 1.521 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 24 | 27 | 2 | 0.01166 | 0.05773 | -0.08742 | 0.3704 | 0.5369 | 1.009 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04876 | 0.2614 | -0.1493 | 0.5134 | 1.05 | 1.537 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.05842 | -0.05398 | 0.2725 | 0.297 | 1.236 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.04657 | 0.2499 | -0.1742 | 0.5427 | 0.9947 | 1.537 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0.01034 | 0.06815 | -0.1028 | 0.3112 | 0.4707 | 1.179 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03544 | 0.2614 | -0.1039 | 0.4113 | 0.963 | 1.531 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06924 | -0.04971 | 0.2796 | 0.26 | 0.8703 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 14 | 21 | 0 | 0.03516 | 0.2499 | -0.1407 | 0.3684 | 1.005 | 1.531 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 24 | 27 | 0 | 0 | 0.06955 | -0.05443 | 0.2447 | 0.3536 | 0.7525 |
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
| fno | apno-nopatience | 256x97 | 896 | 1 | test_ext | supercritical | 14 | 0 | 0.1017 | 0.1795 | 0.002111 | 0.003205 | 0.03395 | 0.08876 | 0.01901 | 0.02876 | 8 | 0.001478 | 0.004522 | 24 | 0.001302 | 0.008007 | 48 | 0.004787 | 0.0225 | 72 | 0.01493 | 0.05181 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | subcritical | 37 | 0 | 0.01794 | 0.04387 | 1.013e-06 | 1.88e-05 | 0.001131 | 0.002608 | 0.01406 | 0.01857 | 8 | 0.0002485 | 0.0007631 | 24 | 0.0001003 | 0.0008224 | 48 | 0.0007311 | 0.002207 | 72 | 0.00622 | 0.01238 |
| fno | apno-nopatience | 256x97 | 896 | 1 | test_in | supercritical | 24 | 0 | 0.0408 | 0.09428 | 7.18e-05 | 0.002155 | 0.006433 | 0.02246 | 0.0143 | 0.01918 | 8 | 0.0003882 | 0.001478 | 24 | 0.000566 | 0.001772 | 48 | 0.001352 | 0.00989 | 72 | 0.008694 | 0.01879 |
| fno | neuralop | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1042 | 0.1562 | 0.001932 | 0.003938 | 0.028 | 0.1175 | 0.01534 | 0.04152 | 4 | 0.002475 | 0.008578 | 12 | 0.004677 | 0.016 | 24 | 0.009488 | 0.0239 | 36 | 0.0502 | 0.09651 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01893 | 0.04482 | 9.082e-06 | 8.073e-05 | 0.001785 | 0.004627 | 0.002968 | 0.00613 | 4 | 0.001324 | 0.002396 | 12 | 0.0004618 | 0.002685 | 24 | 0.0009241 | 0.00305 | 36 | 0.004897 | 0.01411 |
| fno | neuralop | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04412 | 0.1121 | 0.0001244 | 0.002639 | 0.007174 | 0.02628 | 0.005318 | 0.01135 | 4 | 0.002197 | 0.00419 | 12 | 0.001081 | 0.005315 | 24 | 0.001074 | 0.008293 | 36 | 0.003202 | 0.04449 |
| fno | neuralop | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1112 | 0.1622 | 0.001962 | 0.004028 | 0.03077 | 0.1119 | 0.0172 | 0.03931 | 8 | 0.001854 | 0.01099 | 24 | 0.005617 | 0.01518 | 48 | 0.01041 | 0.0237 | 72 | 0.05073 | 0.097 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.0199 | 0.04712 | 8.262e-06 | 7.77e-05 | 0.002351 | 0.005032 | 0.007703 | 0.00967 | 8 | 0.0009782 | 0.002047 | 24 | 0.0004677 | 0.002803 | 48 | 0.0008206 | 0.002667 | 72 | 0.005336 | 0.01327 |
| fno | neuralop | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04637 | 0.1217 | 0.000127 | 0.002653 | 0.007342 | 0.0306 | 0.008065 | 0.01384 | 8 | 0.002072 | 0.004263 | 24 | 0.001547 | 0.00617 | 48 | 0.001926 | 0.01096 | 72 | 0.003358 | 0.04766 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.09693 | 0.1492 | 0.001888 | 0.003902 | 0.02579 | 0.1153 | 0.01468 | 0.04554 | 4 | 0.002749 | 0.007582 | 12 | 0.002949 | 0.01079 | 24 | 0.01047 | 0.03046 | 36 | 0.06546 | 0.1173 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01835 | 0.04306 | 2.183e-06 | 2.513e-05 | 0.001304 | 0.003328 | 0.001054 | 0.002228 | 4 | 9.551e-05 | 0.001295 | 12 | 0.0001804 | 0.0005396 | 24 | 0.0004336 | 0.001467 | 36 | 0.003372 | 0.009191 |
| fno | neuralop-nopatience | 128x49 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04125 | 0.1109 | 8.006e-05 | 0.002613 | 0.006209 | 0.03261 | 0.002614 | 0.007626 | 4 | 0.0004784 | 0.002858 | 12 | 0.0006762 | 0.004052 | 24 | 0.0007703 | 0.004146 | 36 | 0.003181 | 0.02333 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_ext | supercritical | 14 | 0 | 0.1064 | 0.1544 | 0.001922 | 0.003979 | 0.02724 | 0.1125 | 0.01526 | 0.04166 | 8 | 0.002468 | 0.01008 | 24 | 0.003567 | 0.01037 | 48 | 0.01021 | 0.03066 | 72 | 0.0658 | 0.1192 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | subcritical | 37 | 0 | 0.01908 | 0.04455 | 1.704e-06 | 2.434e-05 | 0.001853 | 0.003998 | 0.006339 | 0.008263 | 8 | 0.0003097 | 0.001664 | 24 | 0.0002216 | 0.000616 | 48 | 0.0004558 | 0.001589 | 72 | 0.003076 | 0.009273 |
| fno | neuralop-nopatience | 256x97 | 896 | 0 | test_in | supercritical | 24 | 0 | 0.04299 | 0.1191 | 8.49e-05 | 0.002606 | 0.006716 | 0.0327 | 0.00665 | 0.01086 | 8 | 0.0006342 | 0.002012 | 24 | 0.0003447 | 0.005002 | 48 | 0.001323 | 0.006729 | 72 | 0.002868 | 0.02233 |
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
