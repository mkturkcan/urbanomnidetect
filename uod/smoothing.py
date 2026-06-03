"""Non-causal (offline) trajectory smoothing for the BEV pipeline.

A file render can look ahead over the whole clip, so we replace the live causal
estimates with a zero-phase, outlier-robust filter. This removes both the
high-frequency jitter and the occasional single-frame jump ("sudden shift")
without the lag a causal EMA/Kalman would add.

Two stages per scalar channel:
  1. Hampel filter -- a rolling-median / MAD test that replaces single-frame
     spikes with the local median (kills sudden shifts).
  2. Savitzky-Golay -- a zero-phase local-polynomial low-pass (removes residual
     jitter while preserving genuine motion and acceleration).

The BEV footprint is smoothed by *pose*, not by corner: each frame is
decomposed into a centroid and a heading over one fixed (rigid) template, so
the rendered size stays exactly constant and only the pose is filtered -- the
same "keypoint noise goes into pose, never size" principle the live rigid
estimator uses. Track-absent frames are never written; short internal gaps are
bridged by linear interpolation so coasting does not create a step.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# --------------------------------------------------------------------------- #
def _hampel(a: np.ndarray, k: int = 3, nsig: float = 3.0) -> np.ndarray:
    """Replace spikes with the local median over a +/-k window (MAD test)."""
    n = a.shape[0]
    if n < 2 * k + 1:
        return a
    out = a.copy()
    for i in range(n):
        w = a[max(0, i - k):min(n, i + k + 1)]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and abs(a[i] - med) > nsig * 1.4826 * mad:
            out[i] = med
    return out


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal NaNs; hold the endpoints."""
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a)
    if good.sum() < 2:
        return a
    idx = np.arange(a.shape[0])
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def _smooth1d(a: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Hampel spike removal followed by zero-phase Savitzky-Golay."""
    a = _hampel(np.asarray(a, dtype=np.float64))
    n = a.shape[0]
    w = win if win <= n else n
    if w % 2 == 0:
        w -= 1
    if w < 3 or w < poly + 2:
        return a
    return savgol_filter(a, w, poly)


def _smooth_channels(arr: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Independently smooth each column of an ``(L, C)`` array (NaNs bridged)."""
    out = arr.astype(np.float64).copy()
    for c in range(arr.shape[1]):
        out[:, c] = _smooth1d(_interp_nan(arr[:, c]), win, poly)
    return out


def _rot_to(ref: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Proper rotation R minimising ``||c - ref @ R.T||`` (rows are points)."""
    U, _, Vt = np.linalg.svd(c.T @ ref)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    return U @ np.array([[1.0, 0.0], [0.0, d]]) @ Vt


def _unwrap_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a)
    if good.sum() >= 2:
        a[good] = np.unwrap(a[good])
    return a


def _smooth_footprint(q: np.ndarray, win: int, poly: int):
    """Smooth a footprint sequence ``(L, 4, 2)`` by rigid pose.

    Decompose every present frame into a centroid and a heading over a single
    fixed template (the de-rotated median of the corners), smooth the centroid
    and heading, then re-pose the template. Size is therefore exactly constant.
    Returns ``(L, 4, 2)`` (NaN where absent) or ``None`` if too few frames.
    """
    L = q.shape[0]
    present = np.isfinite(q.reshape(L, -1)).all(axis=1)
    if present.sum() < max(5, poly + 2):
        return None
    idxp = np.where(present)[0]
    t0, t1 = int(idxp[0]), int(idxp[-1])

    cent = np.full((L, 2), np.nan)
    cent[present] = q[present].mean(axis=1)
    centered = q[present] - cent[present][:, None, :]
    ref = centered[0]

    thetas = np.full(L, np.nan)
    derot = []
    for j, f in enumerate(idxp):
        R = _rot_to(ref, centered[j])
        thetas[f] = np.arctan2(R[1, 0], R[0, 0])
        derot.append(centered[j] @ R)          # observation rotated into ref frame
    template = np.median(np.stack(derot), axis=0)
    template -= template.mean(axis=0)

    seg = slice(t0, t1 + 1)
    cx = _smooth1d(_interp_nan(cent[seg, 0]), win, poly)
    cy = _smooth1d(_interp_nan(cent[seg, 1]), win, poly)
    th = _smooth1d(_interp_nan(_unwrap_nan(thetas[seg])), win, poly)

    out = np.full((L, 4, 2), np.nan)
    for k in range(t1 - t0 + 1):
        c, s = np.cos(th[k]), np.sin(th[k])
        R = np.array([[c, -s], [s, c]])
        out[t0 + k] = (template @ R.T) + np.array([cx[k], cy[k]])
    return out


# --------------------------------------------------------------------------- #
def _write_kpts(view, xy: np.ndarray) -> None:
    k = np.asarray(view.kpts, dtype=np.float64)
    if k.ndim == 2 and k.shape[0] == xy.shape[0] and k.shape[1] >= 2:
        k = k.copy()
        k[:, :2] = xy
        view.kpts = k
    else:
        view.kpts = xy


def smooth_states(states, win: int = 11, poly: int = 2,
                  smooth_H: bool = True) -> None:
    """Smooth track geometry (and the global homography) across ``states``.

    Mutates each ``RenderState`` in place: the ``_TrackView`` keypoints, ground
    quad, box and BEV footprint are replaced with their zero-phase filtered
    values, and ``state.H`` with an element-wise smoothed homography. Track
    identity, class and ``time_since_update`` are left untouched.
    """
    if not states or win < 3:
        return
    T = len(states)

    # 1. Global homography: the camera moves smoothly, so smooth its 8 free DoF.
    if smooth_H:
        Hs = np.stack([np.asarray(s.H, dtype=np.float64) for s in states])  # (T,3,3)
        denom = Hs[:, 2, 2].copy()
        denom[np.abs(denom) < 1e-12] = 1.0
        Hs /= denom[:, None, None]
        flat = Hs.reshape(T, 9)
        for j in range(8):                      # leave H[2,2] == 1
            if np.isfinite(flat[:, j]).all():
                flat[:, j] = _smooth1d(flat[:, j], win, poly)
        Hs = flat.reshape(T, 3, 3)
        for i, s in enumerate(states):
            s.H = Hs[i]

    # 2. Per-track geometry. Gather id -> {frame_index: view}.
    seen: dict = {}
    for ti, s in enumerate(states):
        for v in s.tracks:
            seen.setdefault(v.id, {})[ti] = v

    for tid, fr in seen.items():
        frames = sorted(fr)
        if len(frames) < max(5, poly + 2):
            continue
        t0, t1 = frames[0], frames[-1]
        L = t1 - t0 + 1

        # camera keypoints (K, 2)
        kshape = next((np.asarray(fr[f].kpts).shape for f in frames
                       if fr[f].kpts is not None and np.asarray(fr[f].kpts).size), None)
        if kshape is not None and len(kshape) == 2 and kshape[1] >= 2:
            K = kshape[0]
            arr = np.full((L, K, 2), np.nan)
            for f in frames:
                kv = np.asarray(fr[f].kpts, dtype=np.float64)
                if kv.shape == kshape:
                    arr[f - t0] = kv[:, :2]
            sm = _smooth_channels(arr.reshape(L, -1), win, poly).reshape(L, K, 2)
            for f in frames:
                _write_kpts(fr[f], sm[f - t0])

        # camera ground quad (4, 2)
        g = np.full((L, 4, 2), np.nan)
        for f in frames:
            gv = np.asarray(fr[f].ground, dtype=np.float64)
            if gv.shape == (4, 2):
                g[f - t0] = gv
        if np.isfinite(g).any():
            gs = _smooth_channels(g.reshape(L, -1), win, poly).reshape(L, 4, 2)
            for f in frames:
                if np.asarray(fr[f].ground).shape == (4, 2):
                    fr[f].ground = gs[f - t0]

        # camera box (4,)
        b = np.full((L, 4), np.nan)
        for f in frames:
            bv = np.asarray(fr[f].box_xyxy, dtype=np.float64)
            if bv.shape == (4,):
                b[f - t0] = bv
        if np.isfinite(b).any():
            bs = _smooth_channels(b, win, poly)
            for f in frames:
                fr[f].box_xyxy = bs[f - t0]

        # BEV footprint (4, 2) -- rigid pose smoothing
        q = np.full((L, 4, 2), np.nan)
        for f in frames:
            qv = fr[f].bev_quad
            if qv is not None and np.asarray(qv).shape == (4, 2):
                q[f - t0] = np.asarray(qv, dtype=np.float64)
        qs = _smooth_footprint(q, win, poly)
        if qs is not None:
            for f in frames:
                if fr[f].bev_quad is not None and np.all(np.isfinite(qs[f - t0])):
                    fr[f].bev_quad = qs[f - t0]
