"""Regression metrics for drug-target affinity prediction.

Mirrors the metrics used by DeepDTAGen (see ../utils.py) so the simplified
models are evaluated on exactly the same footing as the baseline.
"""
import numpy as np
from math import sqrt
from scipy import stats


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def get_cindex(Y, P):
    """Concordance index: probability that predicted order matches true order."""
    P = P[:, np.newaxis] - P
    P = np.float32(P == 0) * 0.5 + np.float32(P > 0)
    Y = Y[:, np.newaxis] - Y
    Y = np.tril(np.float32(Y > 0), 0)
    P_sum = np.sum(P * Y)
    Y_sum = np.sum(Y)
    return 0.0 if Y_sum == 0 else P_sum / Y_sum


def _r_squared_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)


def _squared_error_zero(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    k = sum(y_obs * y_pred) / float(sum(y_pred * y_pred))
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - (k * y_pred)) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = _r_squared_error(ys_orig, ys_line)
    r02 = _squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


# Binary active/inactive cutoffs per dataset -- identical panel to ../test.py
# (lines 22-30). Balanced accuracy is evaluated at each of these thresholds.
THRESHOLDS = {
    "davis": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    "bindingdb": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    "kiba": [10.0, 10.5, 11.0, 11.5, 12.0, 12.1, 12.5],
}


def balanced_accuracy_full(G, P, thr):
    """Per-threshold balanced accuracy with its full confusion breakdown.

    Binarizes BOTH truth and prediction with '>=' thr (active if >= threshold),
    matching the reference implementation. sensitivity = TP/(TP+FN),
    specificity = TN/(TN+FP); bal_acc = mean of the two. Any component whose
    denominator is empty (no positives, or no negatives) makes bal_acc NaN.
    """
    yt = np.asarray(G) >= thr
    yp = np.asarray(P) >= thr
    tp = int(np.sum(yt & yp)); tn = int(np.sum(~yt & ~yp))
    fp = int(np.sum(~yt & yp)); fn = int(np.sum(yt & ~yp))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float("nan")
    return {"threshold": float(thr), "bal_acc": bal, "sensitivity": sens,
            "specificity": spec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def balanced_accuracy(G, P, thr):
    """Scalar balanced accuracy at one threshold (see balanced_accuracy_full)."""
    return balanced_accuracy_full(G, P, thr)["bal_acc"]


def balanced_accuracy_panel(G, P, dataset):
    """Balanced accuracy at every threshold for `dataset`, plus the mean over them.

    `balacc`      -> {threshold: bal_acc}                (compact, for ranking)
    `balacc_full` -> {threshold: {bal_acc,sens,spec,tp,tn,fp,fn}}  (full record)
    `balacc_mean` -> nan-mean of bal_acc over the threshold panel
    """
    thrs = THRESHOLDS[dataset]
    full = {f"{t:g}": balanced_accuracy_full(G, P, t) for t in thrs}
    per = {k: v["bal_acc"] for k, v in full.items()}
    mean = float(np.nanmean(list(per.values())))
    return {"balacc": per, "balacc_full": full, "balacc_mean": mean}


def all_metrics(G, P, dataset=None):
    """Standard DTA metric bundle. If `dataset` is given, also append the
    balanced-accuracy panel (per-threshold + mean) for that dataset."""
    out = {
        "MSE": float(mse(G, P)),
        "RMSE": float(rmse(G, P)),
        "CI": float(get_cindex(G, P)),
        "rm2": float(get_rm2(G, P)),
        "Pearson": float(pearson(G, P)),
        "Spearman": float(spearman(G, P)),
    }
    if dataset in THRESHOLDS:
        out.update(balanced_accuracy_panel(G, P, dataset))
    return out
