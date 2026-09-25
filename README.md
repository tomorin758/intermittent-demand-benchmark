# Intermittent-demand benchmark — artefacts (anonymised)

Companion artefacts for *Intermittent-Demand Forecast Rankings Depend on the Metric*.
Author and institution names are withheld for double-blind review.

## What is here

| Path | Content |
|---|---|
| `code/pipeline/` | the evaluation pipeline: build the common-window prediction store, re-run the single-window baselines, recompute every metric, verify every array |
| `code/metrics_implementation.py` | the metric implementation used for every number (`metric_comprehensive`) |
| `code/gen_tables.py` | generates the LaTeX tables from `tables/*_unified.csv` |
| `analysis/` | paired bootstrap, threshold sweep, rank/redundancy analysis, figure generator, number audit |
| `metrics/pooled/<dataset>/<model>.json` | all 16 pooled metrics per model (untruncated and truncated) |
| `metrics/perseries/<dataset>/<model>.csv` | one row per series: 25 columns incl. TP/FP/FN counts, so both pooled and per-series-mean conventions can be recomputed |
| `tables/` | the six LaTeX tables of the paper and the sheet-format CSVs they are built from |
| `data/` | derived tables of the two non-M5 sources (see `DATA_LICENSE.md`): car-parts series from the `expsmooth` R package (GPL-2+), fresh-retail daily sales from FreshRetailNet-50K (CC-BY-4.0) |

Not included: prediction arrays (~30 GB) and the M5 history table, whose licence does not permit
redistribution; M5 enters only through derived per-series metrics.

## Evaluation protocol in one paragraph

Every model forecasts the **final H periods** of its history table and nothing else:
Parts rows 48–50 (2002-01 … 2002-03, H=3), Fresh retail rows 83–89 (2024-06-19 … 2024-06-25, H=7),
M5 rows 1885–1912 (2016-03-28 … 2016-04-24, H=28). Ground truth is taken from the same table for
every model. Each model is scored twice: on its own point forecast, and on the same forecast after
both sides are truncated at c = 0.5 (`*_trunc` columns). The untruncated row is the ranking
reference. ADI–CV² classes and quantile segments are computed on the history window only.

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
| `python code/gen_tables.py` | the six LaTeX tables in `tables/` | every number in Tables 2-7 |
| `python analysis/check_consistency.py` | `results/consistency_report.md` | the tables agree cell-by-cell with `metrics/pooled/` (335/335 checks) |
| `python analysis/bootstrap_final.py bundle ql_raw` | `results/bootstrap_ql_raw.{txt,json}` | the leading groups and the "N of M pairs are separated" counts (Section 4) |
| `python analysis/bootstrap_final.py bundle f1_trunc` | `results/bootstrap_f1_trunc.{txt,json}` | the zero-detection leading groups |
| `python analysis/rank_analysis.py tables/parts_unified.csv tables/fresh_retail_unified.csv tables/m5_unified.csv unified` | `results/rank_analysis_unified.txt` | metric redundancy and rank-reversal statistics (Section 4.4) |
| `python analysis/make_fig_reorder_unified.py` | `results/fig_reorder.pdf` | Figure 1 |
| (pre-computed) `results/cscan.csv` | per-model zero-detection metrics at c in {0.1 ... 2.0} | the threshold-sensitivity section |

**Not reproducible here, by design.** `code/pipeline/` documents the evaluation protocol (window
unification, baseline re-runs, metric recomputation) but needs the third-party repositories and the
trained checkpoints, which are not redistributed; `analysis/cscan_final.py` needs the prediction
arrays (~30 GB) and is kept only for provenance. No model can be re-trained or re-evaluated from
this bundle, and no reviewer is expected to: the released per-series tables are the level at which
the reported numbers are meant to be checked.

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
resampling series); the threshold sweep of Section 5.5 is `analysis/cscan_final.py`.
