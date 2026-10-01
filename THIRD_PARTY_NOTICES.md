# Third-party notices

This bundle contains one file that is **not** original work of the authors. Everything else — the
pipeline in `code/pipeline/`, `code/gen_tables.py`, and every script in `analysis/` — was written for
this paper and is licensed under the Apache License 2.0 (see `LICENSE`).

## `code/metrics_implementation.py`

* **Source**: reproduced from the PatchTST repository
  (`PatchTST_supervised/utils/metrics.py`), which is distributed under the **Apache License 2.0**
  (the upstream repository carries no separate `NOTICE` file).
* **Changes**: the file was **modified and extended** by the authors of this paper. The functions the
  reported numbers come from — `metric_comprehensive` and its helpers `_ql`, `_mae_nz`, `_rmse_nz`,
  `_zero_classification`, `_segment_rmse_by_percentile`, `_adi_cv2`, `_adi_cv2_group_rmse` — were
  added or rewritten here; the earlier, unused entry points (`metric`, `zero_metrics`,
  `segment_rmse`) are retained unchanged for reference.
* **Apache-2.0 obligations**: the licence text is included in full in `LICENSE`; this notice records
  the origin of the file and the fact that it was changed, as required by section 4 of that licence.

Metric definitions in the time-series forecasting literature share a common lineage (the
Informer/Autoformer evaluation code); where function bodies coincide with that lineage, the
attribution above (the immediate upstream repository) applies.
