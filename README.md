# Intermittent-demand benchmark — artefacts (anonymised)

Companion artefacts for *Scoring the Zero Decision: A Benchmark and Evaluation Protocol for
Intermittent Demand Forecasting*.
Author and institution names are withheld for double-blind review.

## What is here

| Path | Content |
|---|---|
| `code/pipeline/` | the evaluation pipeline: build the common-window prediction store, re-run the single-window baselines, recompute every metric, verify every array |
| `code/metrics_implementation.py` | the metric implementation used for every number (`metric_comprehensive`) |
| `code/gen_tables.py` | generates the LaTeX tables from `tables/*_unified.csv` |
| `analysis/` | paired bootstrap, Romano-Wolf stepdown, threshold sweep, rank/redundancy analysis, the three figure generators, number audit |
| `metrics/pooled/<dataset>/<model>.json` | every pooled metric per model: the 16 of the truncated variant and the 13 that are defined without truncation |
| `metrics/perseries/<dataset>/<model>.csv` | one row per series: 25 columns incl. TP/FP/FN counts, so both pooled and per-series-mean conventions can be recomputed |
| `tables/` | the three LaTeX result tables printed in the paper (Tables 2--4) and the sheet-format CSVs they are built from |
| `data/` | derived tables of the two non-M5 sources (see `DATA_LICENSE.md`): car-parts series from the `expsmooth` R package (GPL-2+), fresh-retail daily sales from FreshRetailNet-50K (CC-BY-4.0) |
| `data/INVENTORY.md` | file-by-file inventory of the derived artefacts, including the ARIMA(1,0,1) baseline that was run before the evaluation window was unified and is therefore reported nowhere in the paper |

Not included: prediction arrays (~30 GB) and the M5 history table, whose licence does not permit
redistribution; M5 enters only through derived per-series metrics.

## Licences

All code is Apache-2.0 (`LICENSE`); one file, `code/metrics_implementation.py`, is reproduced from the
PatchTST repository and was modified here (`THIRD_PARTY_NOTICES.md`). The redistributed dataset tables
keep the licences of their sources (`DATA_LICENSE.md`): the car-parts table is GPL-2+ (from the
`expsmooth` R package) and the fresh-retail table is CC-BY-4.0 (from FreshRetailNet-50K).

## Evaluation protocol in one paragraph

Every model forecasts the **final H periods** of its history table and nothing else:
Parts rows 48–50 (2002-01 … 2002-03, H=3), Fresh retail rows 83–89 (2024-06-19 … 2024-06-25, H=7),
M5 rows 1885–1912 (2016-03-28 … 2016-04-24, H=28). Ground truth is taken from the same table for
every model. Each model is scored twice: on its own point forecast, and on the same forecast after
both sides are truncated at c = 0.5 (`*_trunc` columns). The untruncated row, the original model, is the ranking reference. ADI–CV² classes and quantile segments are computed on the history window only.

## Data provenance

Both non-M5 tables are derived from public sources and are redistributed here under the source
licences (see `DATA_LICENSE.md`): the car-parts series come from the `carparts` data set of the
**expsmooth** R package (GPL-2 or later; Hyndman et al., *Forecasting with Exponential Smoothing*),
and the fresh-retail series are daily aggregates of **FreshRetailNet-50K**
(`Dingdong-Inc/FreshRetailNet-50K`, CC-BY-4.0; Wang et al., arXiv:2505.16319). The M5 history table
is not redistributed; M5 enters only through derived metrics.

## What is reproducible from this bundle

Everything below runs from the bundle alone (numpy + pandas; matplotlib only for the figure):

| Command | Produces | Paper claim it backs |
|---|---|---|
| `python code/gen_tables.py` | the LaTeX tables in `tables/` | every number in Tables 2--4 |
| `python analysis/check_consistency.py` | `results/consistency_report.md` | the tables agree cell-by-cell with `metrics/pooled/` (335/335 checks) |
| `python analysis/bootstrap_final.py bundle ql_raw` | `results/bootstrap_ql_raw.{txt,json}` | the leading groups and the "N of M pairs are separated" counts (Section 4) |
| `python analysis/bootstrap_final.py bundle f1_trunc` | `results/bootstrap_f1_trunc.{txt,json}` | the zero-detection leading groups |
| `python analysis/romano_wolf.py` | `results/romano_wolf_{ql_raw,f1_trunc}.{txt,json}` | the family-wise-error-controlled pair counts and leading groups of Section 4 |
| `python analysis/trivial_baselines.py` | `results/trivial_baselines.csv` | the trivial reference predictors (always-zero, naive, seasonal naive, two constant means) and the ranks they would take among the subjects; Section 5.1 and the limitations (Section 6.1) |
| `python analysis/rank_analysis.py tables/parts_unified.csv tables/fresh_retail_unified.csv tables/m5_unified.csv unified` | `results/rank_analysis_unified.txt` | metric redundancy and rank-reversal statistics (Section 4.2) |
| `python analysis/group_analysis.py` | `results/group_analysis.{csv,txt}` | within-family rankings and the separated-pair counts of Section 4.1 (runs from `metrics/perseries` alone) |
| `python analysis/make_fig_profiles.py` | `results/fig_profiles.pdf` | Figure 1 (needs the history tables; the M5 table is not redistributed, so the printed figure ships as `results/fig_profiles.pdf`) |
| `python analysis/make_fig_reorder_unified.py` | `results/fig_reorder.pdf` | Figure 2 |
| `python analysis/make_fig_csens.py` | `results/fig_csens.pdf` | Figure 3 |
| (pre-computed) `results/cscan.csv` | per-model zero-detection metrics at c in {0.1 ... 2.0} | the threshold-sensitivity section |
| (pre-computed) `results/zero_rule_analysis.csv` | exact-zero rates, zero detection under an exact-zero rule against the protocol rule, and the always-zero error decomposition | the abstention paragraph of Section 4.1 and the reference-predictor paragraph of Section 5.1 |
| (pre-computed) `results/window_stability.csv` | per-model QL(0.5) and zero-detection F1 on earlier non-overlapping windows of the held-out region | the window-stability paragraph of Section 4.1 |

## Scope of this bundle

This bundle contains the evaluation *outputs* and the code that turns them into the paper's numbers.
It does not contain model implementations, trained checkpoints or prediction arrays, so the forecasts
themselves cannot be regenerated from it; `analysis/cscan_final.py`,
`analysis/zero_rule_analysis.py` and `analysis/window_stability.py` are the three scripts that need
the arrays (they are kept so that the same computations can be re-run by anyone who has them, while
their output is shipped as `results/cscan.csv`, `results/zero_rule_analysis.csv` and
`results/window_stability.csv`). For a reader who does want to rebuild
the full pipeline, `code/pipeline/` documents exactly what was run, and the paper's Section 3 lists
the third-party repositories and datasets that the runs started from. Five shipped results are
pre-computed rather than regenerated: `results/cscan.csv`, `results/zero_rule_analysis.csv` and
`results/window_stability.csv` (all three need the prediction arrays), `results/fig_profiles.pdf`
(needs the M5 history table, which is not redistributed) and the M5 rows of
`results/trivial_baselines.csv` (same reason; the script recomputes the Parts and Fresh retail rows
from `data/` alone). The estimator-error sweeps quoted in
Section 3.4 ship as data where the draws were kept: `results/chronos_repeat_spread.csv` holds the ten
repeat-inference draws for Chronos on Parts, while the twenty-seed DeepAR-negbin sweep is quoted in
the paper and its run log is not part of this bundle.

## Reproducing a number

1. `metrics/pooled/<ds>/<model>.json` holds the pooled value of every metric.
2. `metrics/perseries/<ds>/<model>.csv` holds the per-series decomposition; summing the TP/FP/FN
   columns reproduces the pooled zero-detection metrics, and averaging `ql_raw` reproduces the
   pooled QL(0.5).
3. `MANIFEST.csv` maps every reported quantity to the file that contains it.
4. To rebuild the tables: copy `tables/<ds>_unified.csv` to `../data/<ds>.csv` and run
   `code/gen_tables.py`.

Sources to cite when reusing the data: `carparts`/expsmooth (GPL-2+) and FreshRetailNet-50K
(CC-BY-4.0, arXiv:2505.16319); M5 per Makridakis et al. (2022).

The paired bootstrap of Section 4 is `analysis/bootstrap_final.py` (B = 4000, Poisson weights,
resampling series); the Romano-Wolf stepdown that controls the family-wise error rate over the whole
family of model pairs is `analysis/romano_wolf.py` (same resampling, alpha = 5%); the threshold sweep
of Section 4.3 is `analysis/cscan_final.py`.
