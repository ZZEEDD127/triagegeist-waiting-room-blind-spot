"""
Statistical helpers shared by the analysis: bootstrap confidence intervals,
cross-fitted calibration, calibration diagnostics, operating points and
decision-curve analysis.

Every interval in the write-up comes from these functions. Two bootstrap
flavours are used:
  * patient-level bootstrap (synthetic data, where visits are independent);
  * cluster bootstrap that resamples whole hospitals (NHAMCS, where visits are
    nested in hospitals and treating them as independent understates variance).
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.model_selection import KFold

SEED = 42


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def _resample_index(n, clusters, rng):
    """Row indices for one bootstrap replicate (cluster-aware if clusters given)."""
    if clusters is None:
        return rng.integers(0, n, n)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_by_cluster = np.split(np.argsort(inv, kind="stable"),
                               np.cumsum(np.bincount(inv))[:-1])
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows_by_cluster[k] for k in pick])


def bootstrap(stat, *arrays, clusters=None, n_boot=1000, seed=SEED, alpha=0.05):
    """Point estimate + percentile CI of stat(*arrays).

    stat receives the resampled arrays and returns a float (or NaN when the
    replicate is degenerate, e.g. a single class)."""
    arrays = [np.asarray(a) for a in arrays]
    point = float(stat(*arrays))
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    reps = []
    for _ in range(n_boot):
        idx = _resample_index(n, clusters, rng)
        try:
            v = stat(*[a[idx] for a in arrays])
        except ValueError:
            v = np.nan
        reps.append(v)
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"est": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def auc(y, p, w=None):
    return roc_auc_score(y, p, sample_weight=w)


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def delta_auc(y, p_new, p_base, w=None):
    return auc(y, p_new, w) - auc(y, p_base, w)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def crossfit_isotonic(p, y, n_splits=5, seed=SEED):
    """Calibrate out-of-fold scores without reusing the same rows.

    The isotonic map applied to each fold is fitted only on the other folds,
    so the reported Brier score is an honest out-of-sample estimate."""
    p = np.asarray(p); y = np.asarray(y)
    out = np.zeros_like(p, dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(p):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
        out[va] = iso.predict(p[va])
    return out


def calibration_stats(y, p, n_bins=10):
    """Brier, expected calibration error, calibration slope and
    calibration-in-the-large (mean predicted minus observed)."""
    y = np.asarray(y); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean()
              for k in range(n_bins) if (b == k).any())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = float(LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0])
    return {"brier": round(brier, 4), "ece": round(float(ece), 4),
            "slope": round(slope, 3), "citl": round(float(p.mean() - y.mean()), 4)}


# --------------------------------------------------------------------------
# operating points and decision curves
# --------------------------------------------------------------------------
def flag_top(p, frac):
    """Boolean flag for the top `frac` of scores."""
    return p >= np.quantile(p, 1 - frac)


def operating_point(y, p, frac, w=None):
    """Alert rate, precision, recall and lift when flagging the top `frac`."""
    y = np.asarray(y); p = np.asarray(p)
    w = np.ones_like(p, dtype=float) if w is None else np.asarray(w, dtype=float)
    f = flag_top(p, frac)
    prec = np.sum(w[f] * y[f]) / np.sum(w[f])
    rec = np.sum(w[f] * y[f]) / np.sum(w * y)
    base = np.sum(w * y) / np.sum(w)
    return {"alert_rate": round(float(np.sum(w[f]) / np.sum(w)), 3), "precision": round(float(prec), 3),
            "recall": round(float(rec), 3), "lift": round(float(prec / base), 2), "base_rate": round(float(base), 3)}


def recall_at(y, p, frac):
    f = flag_top(np.asarray(p), frac)
    y = np.asarray(y)
    return y[f].sum() / y.sum()


def lift_at(y, p, frac):
    f = flag_top(np.asarray(p), frac)
    y = np.asarray(y)
    return y[f].mean() / y.mean()


def net_benefit(y, flag, pt):
    """Vickers & Elkin net benefit: TP/N - FP/N * pt/(1-pt)."""
    y = np.asarray(y); flag = np.asarray(flag, dtype=bool); n = len(y)
    tp = np.sum(flag & (y == 1)); fp = np.sum(flag & (y == 0))
    return tp / n - fp / n * pt / (1 - pt)


def decision_curve(y, p, thresholds, clusters=None, n_boot=300, seed=SEED):
    """Net benefit of the model, of treating everyone and of treating no-one,
    with a bootstrap band for the model's advantage over the better default."""
    y = np.asarray(y); p = np.asarray(p)
    model = [net_benefit(y, p >= t, t) for t in thresholds]
    everyone = [net_benefit(y, np.ones_like(y, dtype=bool), t) for t in thresholds]
    rng = np.random.default_rng(seed)
    adv = []
    for _ in range(n_boot):
        idx = _resample_index(len(y), clusters, rng)
        yy, pp = y[idx], p[idx]
        adv.append([net_benefit(yy, pp >= t, t) - max(0.0, net_benefit(yy, np.ones_like(yy, dtype=bool), t))
                    for t in thresholds])
    adv = np.asarray(adv)
    return {"thresholds": [round(float(t), 3) for t in thresholds],
            "model": [round(float(v), 5) for v in model],
            "treat_all": [round(float(v), 5) for v in everyone],
            "adv_lo": [round(float(v), 5) for v in np.quantile(adv, 0.025, axis=0)],
            "adv_hi": [round(float(v), 5) for v in np.quantile(adv, 0.975, axis=0)]}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
