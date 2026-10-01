import numpy as np
from collections import OrderedDict


# ============================================================================
#  Core metrics (kept for backward compatibility)
# ============================================================================

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MAE_non_zero(pred, true, threshold=0.0):
    """
    MAE computed only on points where the actual value is non-zero
    By default, true > 0 counts as non-zero
    """
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    mask = true > threshold
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs(pred[mask] - true[mask]))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    eps = 1e-12
    return np.mean(np.abs((pred - true) / (true + eps)))


def MSPE(pred, true):
    eps = 1e-12
    return np.mean(np.square((pred - true) / (true + eps)))


def QuantileLoss(pred, true, rho=0.5):
    """
    QL_rho = mean( max(rho*(y-ŷ), (1-rho)*(ŷ-y)) )
    With rho=0.5, QL(0.5) = 0.5 * MAE
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    return np.mean(np.maximum(rho * (true - pred), (1 - rho) * (pred - true)))


def zero_metrics(pred, true, threshold=0.5):
    """
    For the zero-value event:
    true <= threshold counts as a true zero event
    pred <= threshold counts as a predicted zero event
    """
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    true_zero = (true <= threshold).astype(int)
    pred_zero = (pred <= threshold).astype(int)

    tp = np.sum((pred_zero == 1) & (true_zero == 1))
    fp = np.sum((pred_zero == 1) & (true_zero == 0))
    fn = np.sum((pred_zero == 0) & (true_zero == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def segment_rmse(pred, true, zero_threshold=0.5):
    """
    Segmented by the actual value:
    1) true <= zero_threshold
    2) zero_threshold < true < 2
    3) true >= 2
    """
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    segments = {
        "seg_rmse_zero": true <= zero_threshold,
        "seg_rmse_one": (true > zero_threshold) & (true < 2),
        "seg_rmse_ge2": true >= 2,
    }

    results = {}
    for name, mask in segments.items():
        if np.sum(mask) == 0:
            results[name] = np.nan
        else:
            results[name] = np.sqrt(np.mean((pred[mask] - true[mask]) ** 2))
    return results


def metric(pred, true, zero_threshold=0.5, rho=0.5):
    """Legacy compatibility function."""
    mae_non_zero = MAE_non_zero(pred, true, threshold=zero_threshold)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    ql = QuantileLoss(pred, true, rho=rho)

    zero_precision, zero_recall, zero_f1 = zero_metrics(pred, true, threshold=zero_threshold)
    seg_dict = segment_rmse(pred, true, zero_threshold=zero_threshold)

    return {
        "mae_non_zero": mae_non_zero,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "rse": rse,
        "corr": corr,
        "ql": ql,
        "zero_precision": zero_precision,
        "zero_recall": zero_recall,
        "zero_f1": zero_f1,
        "seg_rmse_zero": seg_dict["seg_rmse_zero"],
        "seg_rmse_one": seg_dict["seg_rmse_one"],
        "seg_rmse_ge2": seg_dict["seg_rmse_ge2"],
    }


# ============================================================================
#  Metrics added for this benchmark: truncated and non-truncated groups
# ============================================================================

def _truncate(values, c):
    """Truncation: values below c are set to zero."""
    v = np.asarray(values, dtype=np.float64).copy()
    v[v < c] = 0.0
    return v


def _ql(pred, true, rho=0.5):
    """Internal: quantile loss over the flattened arrays."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    e = t - p
    return float(np.mean(np.where(e >= 0, rho * e, (rho - 1) * e)))


def _mae_nz(pred, true):
    """MAE computed only where the actual value is greater than zero."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    mask = t > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(p[mask] - t[mask])))


def _rmse_nz(pred, true):
    """RMSE computed only where the actual value is greater than zero."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    mask = t > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))


def _zero_classification(pred, true, c=0.5):
    """
    0Precision / 0Recall / 0F1
    Positive class = "actual value < c" (treated as no demand)
    TP0: actual < c and predicted < c
    FP0: actual >= c and predicted < c
    FN0: actual < c and predicted >= c
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    tp = np.sum((t < c) & (p < c))
    fp = np.sum((t >= c) & (p < c))
    fn = np.sum((t < c) & (p >= c))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


# ---- Segment RMSE (quantiles of the non-zero actual values) ----

_PERCENTILES = [10, 25, 50, 75, 90, 100]


def _segment_rmse_by_percentile(pred, true, c=None):
    """
    RMSE computed within segments defined by quantiles of the non-zero actual values.
    c=None -> non-truncated (non-zero: true > 0)
    c=float -> truncated version (non-zero: true >= c)
    Returns an OrderedDict whose keys carry the actual quantile values.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()

    if c is None:
        nz = t > 0
    else:
        nz = t >= c

    nz_t = t[nz]
    nz_p = p[nz]
    if len(nz_t) == 0:
        return OrderedDict()

    qs = np.percentile(nz_t, _PERCENTILES)
    if c is None:
        lbs = [0.0] + list(qs[:-1])
    else:
        lbs = [c] + list(qs[:-1])
    ubs = list(qs)

    seg_labels = [
        f"Seg1_(0,{qs[0]:.1f}]",
        f"Seg2_({qs[0]:.1f},{qs[1]:.1f}]",
        f"Seg3_({qs[1]:.1f},{qs[2]:.1f}]",
        f"Seg4_({qs[2]:.1f},{qs[3]:.1f}]",
        f"Seg5_({qs[3]:.1f},{qs[4]:.1f}]",
        f"Seg6_({qs[4]:.1f},{qs[5]:.1f}]",
    ]

    res = OrderedDict()
    for i, (lb, ub, name) in enumerate(zip(lbs, ubs, seg_labels)):
        if i == 0:
            m = (nz_t > lb) & (nz_t <= ub)
        else:
            m = (nz_t > lb) & (nz_t <= ub)
        if m.sum() == 0:
            res[name] = np.nan
        else:
            res[name] = float(np.sqrt(np.mean((nz_p[m] - nz_t[m]) ** 2)))
    return res


# ---- ADI-CV^2 classification and per-class RMSE ----

def _adi_cv2(hist_series, c=None):
    """One history series -> (ADI, CV^2)."""
    h = np.asarray(hist_series, dtype=np.float64)
    if c is not None:
        h = h.copy()
        h[h < c] = 0.0
    T = len(h)
    nz = np.sum(h > 0)
    if nz == 0:
        return np.inf, np.inf
    ADI = T / nz
    nz_vals = h[h > 0]
    eps = 1e-12
    CV2 = (np.std(nz_vals) / (np.mean(nz_vals) + eps)) ** 2
    return float(ADI), float(CV2)


def _classify(ADI, CV2):
    if ADI < 1.32 and CV2 < 0.49:
        return "Smooth"
    elif ADI >= 1.32 and CV2 < 0.49:
        return "Intermittent"
    elif ADI < 1.32 and CV2 >= 0.49:
        return "Erratic"
    else:
        return "Lumpy"


def _adi_cv2_group_rmse(preds, trues, hist_data, c=None):
    """
    preds/trues: (N, pred_len, C)  float32
    hist_data: (T_train, C)
    Returns an OrderedDict: {"Smooth RMSE": ..., ...}

    Memory-friendly: the SSE is accumulated window by window, avoiding an (N*pred_len, C) intermediate array.
    """
    N, pred_len, C = preds.shape
    labels = []
    for s in range(C):
        adi, cv2 = _adi_cv2(hist_data[:, s], c=c)
        labels.append(_classify(adi, cv2))

    if c is not None:
        tu = _truncate(trues, c)
        pu = _truncate(preds, c)
    else:
        tu, pu = trues, preds

    # accumulate the SSE of each series window by window (float64)
    sse = np.zeros(C, dtype=np.float64)
    for i in range(N):
        diff = pu[i].astype(np.float64) - tu[i].astype(np.float64)  # (pred_len, C)
        sse += (diff ** 2).sum(axis=0)  # sum over pred_len

    cnt_per_series = N * pred_len

    res = OrderedDict()
    for cat in ["Smooth", "Intermittent", "Erratic", "Lumpy"]:
        idx = [s for s in range(C) if labels[s] == cat]
        if not idx:
            res[f"{cat} RMSE"] = np.nan
        else:
            total_sse = sse[idx].sum()
            total_cnt = cnt_per_series * len(idx)
            res[f"{cat} RMSE"] = float(np.sqrt(total_sse / total_cnt))
    return res


# ============================================================================
#  Combined evaluation function
# ============================================================================

def metric_comprehensive(preds, trues, historical_data, c=0.5):
    """
    Computes every metric this benchmark defines and returns an OrderedDict.

    Arguments:
      preds:           (N, pred_len, C)  predictions on the original scale
      trues:           (N, pred_len, C)  actual values on the original scale
      historical_data: (T_train, C)      training-period history, used only for ADI-CV^2
      c:               truncation threshold, default 0.5

    Returns:
      OrderedDict mapping metric name to float
    """
    results = OrderedDict()
    pf = preds.ravel()
    tf = trues.ravel()

    # ==================== non-truncated metrics ====================
    results["QL(0.5)"] = _ql(pf, tf, rho=0.5)
    results["MAE_nonzero"] = _mae_nz(pf, tf)
    results["RMSE_nonzero"] = _rmse_nz(pf, tf)

    for k, v in _segment_rmse_by_percentile(pf, tf, c=None).items():
        results[f"NT_{k}"] = v

    for k, v in _adi_cv2_group_rmse(preds, trues, historical_data, c=None).items():
        results[f"NT_{k}"] = v

    # ====================== truncated metrics =======================
    tf_c = _truncate(tf, c)
    pf_c = _truncate(pf, c)

    results[f"QL(0.5,c={c})"] = _ql(pf_c, tf_c, rho=0.5)
    p0, r0, f0 = _zero_classification(pf, tf, c=c)
    results[f"0Precision(c={c})"] = p0
    results[f"0Recall(c={c})"] = r0
    results[f"0F1(c={c})"] = f0
    results[f"MAE_nonzero(c={c})"] = _mae_nz(pf_c, tf_c)
    results[f"RMSE_nonzero(c={c})"] = _rmse_nz(pf_c, tf_c)

    for k, v in _segment_rmse_by_percentile(pf, tf, c=c).items():
        results[f"T_{k}"] = v

    for k, v in _adi_cv2_group_rmse(preds, trues, historical_data, c=c).items():
        results[f"T_{k}"] = v

    return results


# ============================================================================
#  Formatted printing
# ============================================================================

def print_metrics_table(metrics_dict):
    """Print the metric table, separating the non-truncated and truncated groups."""
    print("\n" + "=" * 72)
    print("                    M E T R I C   S U M M A R Y")
    print("=" * 72)

    # grouping
    nt_keys = [k for k in metrics_dict if k.startswith("NT_") or
               (not k.startswith("T_") and not k.startswith("NT_") and "c=" not in k)]
    t_keys = [k for k in metrics_dict if k.startswith("T_") or "c=" in k]

    def _print_group(title, keys):
        print(f"\n  +- {title} --------------------------------------+")
        for k in keys:
            v = metrics_dict.get(k, "N/A")
            if isinstance(v, float):
                s = f"{v:.6f}" if not np.isnan(v) else "N/A"
            else:
                s = str(v)
            print(f"  | {k:<48s} = {s:>12s} |")
        print(f"  +{'-'*62}+")

    _print_group("Non-truncated metrics", nt_keys)
    _print_group("Truncated metrics", t_keys)
    print("=" * 72)