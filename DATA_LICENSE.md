# Data provenance and licences

All three datasets are **third-party public data**; this bundle contains only derived tables.

| Path | Content | Source | Licence |
|---|---|---|---|
| `data/parts.csv` | 857 monthly car-parts series, 1998-01 … 2002-03 (51 points) | `carparts` data set of the **expsmooth** R package (Hyndman, Koehler, Ord & Snyder, *Forecasting with Exponential Smoothing*, Springer 2008); keeping the series with ≥16 non-zero months | GPL-2 or later (inherited from the source package) |
| `data/fresh_retail.csv` | 50,000 store-product daily sales, 2024-03-28 … 2024-06-25 (90 days) | **FreshRetailNet-50K** (`Dingdong-Inc/FreshRetailNet-50K` on Hugging Face; Wang et al., arXiv:2505.16319), daily-aggregated `sale_amount`, stockout annotation dropped | CC-BY-4.0 (inherited from the source dataset) |
| `metrics/**`, `tables/**` | derived metrics and tables for all three datasets | this work | CC-BY-4.0 (fresh retail, M5-derived) / GPL-2+ (car-parts-derived) |
| M5 raw data | **not redistributed** | M5 competition (Makridakis et al., 2022) | M5 competition terms; only derived per-series metrics are shipped |

Code (`code/`, `analysis/`) is MIT-licensed (see `LICENSE`).

Rebuilding the two dataset tables from source requires `datasets.load_dataset("Dingdong-Inc/FreshRetailNet-50K")`
for fresh retail and `library(expsmooth); carparts` in R for the car-parts series.
