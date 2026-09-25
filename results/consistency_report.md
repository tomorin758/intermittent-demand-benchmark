# Consistency report

Each cell below compares the LaTeX-table input (`tables/*_unified.csv`) with the
pooled metric file (`metrics/pooled/*.json`) produced by the metric implementation.

## parts (22 models)

| model | QL untr | MAE>0 untr | F1 trunc | tail untr | regimes untr | verdict |
|---|---:|---:|---:|---:|---:|---|
| Autoformer | 0.418 | 1.139 | 0.601 | 4.522 | ok | ✓ |
| Chronos | 0.305 | 1.388 | 0.743 | 5.255 | ok | ✓ |
| Croston | 0.435 | 1.048 | 0.230 | 5.125 | ok | ✓ |
| DLinear | 0.522 | 1.288 | 0.472 | 5.260 | ok | ✓ |
| DeepAR | 0.320 | 1.312 | 0.733 | 5.265 | ok | ✓ |
| DeepAR-negbin | 0.296 | 1.414 | 0.757 | 5.247 | ok | ✓ |
| FEDformer | 0.420 | 1.113 | 0.600 | 4.490 | ok | ✓ |
| HurdleDLinear | 0.364 | 1.544 | 0.715 | 5.524 | ok | ✓ |
| LSTM | 0.471 | 0.994 | 0.177 | 5.014 | ok | ✓ |
| MOMENT | 0.360 | 1.319 | 0.665 | 5.383 | ok | ✓ |
| Moirai | 0.333 | 1.156 | 0.676 | 4.887 | ok | ✓ |
| PatchMixer | 0.313 | 1.198 | 0.749 | 4.915 | ok | ✓ |
| PatchTCN | 0.329 | 1.136 | 0.714 | 4.778 | ok | ✓ |
| PatchTSMixer | 0.353 | 1.012 | 0.632 | 4.591 | ok | ✓ |
| PatchTST | 0.347 | 1.072 | 0.666 | 4.661 | ok | ✓ |
| SBA | 0.426 | 1.071 | 0.288 | 5.155 | ok | ✓ |
| TCN | 0.469 | 1.003 | 0.150 | 5.054 | ok | ✓ |
| TSB | 0.378 | 0.944 | 0.503 | 4.700 | ok | ✓ |
| TTM | 0.374 | 1.055 | 0.619 | 4.525 | ok | ✓ |
| TimesFM | 0.302 | 1.288 | 0.753 | 5.034 | ok | ✓ |
| TweedieGP | 0.316 | 1.338 | 0.711 | 5.377 | ok | ✓ |
| WaveNet-like | 0.455 | 1.003 | 0.199 | 5.033 | ok | ✓ |

## fresh_retail (22 models)

| model | QL untr | MAE>0 untr | F1 trunc | tail untr | regimes untr | verdict |
|---|---:|---:|---:|---:|---:|---|
| Autoformer | 0.328 | 0.718 | 0.534 | 2.876 | ok | ✓ |
| Chronos | 0.275 | 0.623 | 0.590 | 2.497 | ok | ✓ |
| Croston | 0.301 | 0.641 | 0.584 | 2.427 | ok | ✓ |
| DLinear | 0.368 | 0.828 | 0.507 | 4.525 | ok | ✓ |
| DeepAR | 0.296 | 0.683 | 0.590 | 2.619 | ok | ✓ |
| DeepAR-negbin | 0.301 | 0.773 | 0.608 | 2.752 | ok | ✓ |
| FEDformer | 0.326 | 0.713 | 0.537 | 2.820 | ok | ✓ |
| HurdleDLinear | 0.475 | 1.277 | 0.493 | 6.317 | ok | ✓ |
| LSTM | 0.361 | 0.804 | 0.539 | 4.346 | ok | ✓ |
| MOMENT | 0.459 | 0.895 | 0.086 | 7.578 | ok | ✓ |
| Moirai | 0.286 | 0.632 | 0.585 | 2.404 | ok | ✓ |
| PatchMixer | 0.284 | 0.619 | 0.608 | 2.374 | ok | ✓ |
| PatchTCN | 0.290 | 0.605 | 0.578 | 2.335 | ok | ✓ |
| PatchTSMixer | 0.297 | 0.639 | 0.590 | 2.520 | ok | ✓ |
| PatchTST | 0.290 | 0.618 | 0.592 | 2.421 | ok | ✓ |
| SBA | 0.307 | 0.672 | 0.594 | 2.650 | ok | ✓ |
| TCN | 0.355 | 0.787 | 0.545 | 4.337 | ok | ✓ |
| TSB | 0.289 | 0.617 | 0.596 | 2.427 | ok | ✓ |
| TTM | 0.298 | 0.644 | 0.583 | 2.566 | ok | ✓ |
| TimesFM | 0.270 | 0.611 | 0.605 | 2.289 | ok | ✓ |
| TweedieGP | 0.326 | 0.859 | 0.607 | 3.135 | ok | ✓ |
| WaveNet-like | 0.348 | 0.771 | 0.547 | 4.153 | ok | ✓ |

## m5 (23 models)

| model | QL untr | MAE>0 untr | F1 trunc | tail untr | regimes untr | verdict |
|---|---:|---:|---:|---:|---:|---|
| Autoformer | 0.598 | 1.848 | 0.671 | 9.521 | ok | ✓ |
| Chronos | 0.504 | 1.855 | 0.755 | 9.218 | ok | ✓ |
| Croston | 0.551 | 1.635 | 0.685 | 8.393 | ok | ✓ |
| DLinear | 0.512 | 1.596 | 0.716 | 8.110 | ok | ✓ |
| DeepAR | 0.496 | 2.016 | 0.774 | 9.044 | ok | ✓ |
| DeepAR-negbin | 0.490 | 1.823 | 0.766 | 8.940 | ok | ✓ |
| FEDformer | 0.608 | 1.884 | 0.668 | 9.843 | ok | ✓ |
| HurdleDLinear | 0.573 | 2.280 | 0.733 | 13.361 | ok | ✓ |
| LSTM | 0.532 | 1.706 | 0.715 | 8.859 | ok | ✓ |
| MOMENT | 0.809 | 2.254 | 0.390 | 15.057 | ok | ✓ |
| Moirai | 0.511 | 1.724 | 0.694 | 8.637 | ok | ✓ |
| PatchMixer | 0.494 | 1.637 | 0.742 | 8.157 | ok | ✓ |
| PatchTCN | 0.512 | 1.622 | 0.723 | 8.350 | ok | ✓ |
| PatchTSMixer | 0.505 | 1.589 | 0.724 | 8.013 | ok | ✓ |
| PatchTST | 0.507 | 1.597 | 0.722 | 8.100 | ok | ✓ |
| SBA | 0.543 | 1.642 | 0.694 | 8.486 | ok | ✓ |
| TCN | 0.522 | 1.652 | 0.714 | 8.477 | ok | ✓ |
| TSB | 0.520 | 1.626 | 0.713 | 8.281 | ok | ✓ |
| TTM | 0.508 | 1.603 | 0.723 | 8.148 | ok | ✓ |
| Timer | 0.785 | 2.238 | 0.341 | 15.755 | ok | ✓ |
| TimesFM | 0.478 | 1.768 | 0.768 | 8.462 | ok | ✓ |
| TweedieGP | 0.500 | 2.009 | 0.773 | 9.684 | ok | ✓ |
| WaveNet-like | 0.522 | 1.638 | 0.711 | 8.413 | ok | ✓ |

**335/335 checks passed.**
