"""Task-agnostic, training-free trajectory description core (audit proof of concept, 2026-10-02).

Input contract (nothing task-specific in this file):
  X[n_steps, n_features]  per-step feature values (any units)
  ep[n_steps]             episode index of each step
  phase[n_steps]          task-declared phase variable used to align episodes (progress, time since event, ...)
  cond[n_episodes]        condition label (stratum); envelopes are built per condition
  success[n_episodes]     bool
Method:
  1. success envelope: per (condition, phase bin, feature) the [lo, hi] quantile band of successful episodes
  2. deviation: signed distance outside the band, in units of the band width (0 inside)
  3. excursion events: sustained runs outside the band (feature, sign, onset, duration, peak)
  4. per-failure signature: peak deviation per (feature, sign) inside a look-back window before the end
  5. grouping: k-medoids on the signature vectors
No thresholds are hand-set per feature: everything is relative to what successful episodes do at the same phase.
"""
import numpy as np


def phase_bins(phase, n_bins, lo=None, hi=None):
    lo = np.nanmin(phase) if lo is None else lo
    hi = np.nanmax(phase) if hi is None else hi
    return np.clip(((phase - lo) / max(hi - lo, 1e-9) * n_bins).astype(int), 0, n_bins - 1)


def build_envelope(X, ep, pbin, cond, success, fit_mask, q=(0.005, 0.995), n_bins=20, min_steps=200):
    """Quantile band of successful steps selected by fit_mask (bool per episode)."""
    conds = sorted(set(cond.tolist()))
    F = X.shape[1]
    lo = np.full((len(conds), n_bins, F), np.nan); hi = lo.copy(); med = lo.copy()
    step_ok = success[ep] & fit_mask[ep]
    cidx = np.array([conds.index(c) for c in cond])[ep]
    for ci in range(len(conds)):
        for b in range(n_bins):
            m = step_ok & (cidx == ci) & (pbin == b)
            if m.sum() < min_steps:                      # fall back to all conditions for sparse cells
                m = step_ok & (pbin == b)
            if m.sum() < 20:
                continue
            v = X[m]
            lo[ci, b] = np.nanquantile(v, q[0], axis=0); hi[ci, b] = np.nanquantile(v, q[1], axis=0)
            med[ci, b] = np.nanmedian(v, axis=0)
    return {"conds": conds, "lo": lo, "hi": hi, "med": med, "cidx": cidx}


def deviation(X, pbin, env, scale):
    lo = env["lo"][env["cidx"], pbin]; hi = env["hi"][env["cidx"], pbin]
    d = np.where(X > hi, (X - hi) / scale, np.where(X < lo, (X - lo) / scale, 0.0))
    return np.nan_to_num(d)


def excursions(dev_ep, min_len=5, min_peak=0.25):
    """Sustained runs outside the band for one episode: list of (feature, sign, onset, length, peak)."""
    out = []
    T, F = dev_ep.shape
    for f in range(F):
        for sign in (1, -1):
            a = (dev_ep[:, f] * sign) > 0
            if not a.any():
                continue
            edges = np.diff(np.concatenate([[0], a.astype(int), [0]]))
            for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                peak = float(np.max(dev_ep[s:e, f] * sign))
                if e - s >= min_len and peak >= min_peak:
                    out.append((f, sign, int(s), int(e - s), peak))
    return sorted(out, key=lambda r: r[2])


def kmedoids(V, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    D = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(-1))
    med = [int(rng.integers(len(V)))]
    while len(med) < k:                                   # farthest-point style seeding
        d = D[:, med].min(1); med.append(int(np.argmax(d)))
    med = np.array(med)
    for _ in range(iters):
        lab = D[:, med].argmin(1)
        new = np.array([np.nonzero(lab == j)[0][D[np.ix_(lab == j, lab == j)].sum(0).argmin()] if (lab == j).any() else med[j] for j in range(k)])
        if (new == med).all():
            break
        med = new
    lab = D[:, med].argmin(1)
    cost = float(D[np.arange(len(V)), med[lab]].sum())
    return lab, med, cost


def silhouette(V, lab):
    D = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(-1))
    s = []
    for i in range(len(V)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0); continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab.tolist()) if j != lab[i])
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))
