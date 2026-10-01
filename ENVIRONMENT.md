# Environments used to produce the archived results

Package versions as recorded in the environments on the machine that produced the archived arrays and
metrics. They are listed so that a reader can judge whether a re-run uses the same stack; they are not
a claim that these are the versions the third-party repositories recommend.

| Purpose | Environment | Key packages |
|---|---|---|
| Supervised and probabilistic models (patch family, DLinear/LSTM/TCN/WaveNet-like, DeepAR, HurdleDLinear) and the metric recomputation for those runs | `patch_gpu` | torch 2.4.1, pytorch-lightning 2.6.5, gluonts 0.14.4, numpy 1.26.4, pandas 2.1.4, transformers 5.16.1 |
| Table and figure generation and every analysis shipped in `analysis/` | `analysis` | numpy 1.25.2, pandas 2.3.3, matplotlib 3.10.9, scikit-learn 1.7.2 |
| Foundation-model evaluations (TimesFM, Moirai, Chronos, TTM, MOMENT, Timer) | `chronos`, `momentft`, `timer`, `ga_pts`, `testenv` | torch 2.2.0–2.13.0 (env-dependent), transformers 4.33.3–5.17.0, numpy 1.25.2–2.4.6 |
| TweedieGP per-series fits | `tweediegp` | torch 2.2.0, numpy 1.26.4, scikit-learn 1.9.1 |

Caveats that matter for reproduction:

* The `chronos` and `tweediegp` environments were modified after the corresponding runs finished, so
  their listed versions are those installed at archiving time rather than provably those used.
* The foundation-model evaluations used the published implementations of each model; the model
  versions and adaptation regimes are recorded in the paper's Section 3.2 and in `code/pipeline/`.
* Two upstream data loaders silently dropped the final window of each series. Both were found while
  unifying the evaluation window across families and are fixed in the shipped pipeline
  (`code/pipeline/`); every number in the paper is computed after those fixes, on the final $H$
  periods of each table and nothing else.
* Hardware: an NVIDIA RTX 4090 (patch family and supervised baselines) and a Tesla P40
  (foundation-model evaluations). No per-run wall-clock was recorded, for the reason given in the
  paper's Section 3.2.
