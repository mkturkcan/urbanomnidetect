"""Causal, low-latency stabilisation -- the online counterpart of the offline
zero-phase passes in :mod:`uod.smoothing`.

The offline back-end looks both forward and back (zero-phase Savitzky-Golay +
Hampel) and pins the BEV with a global homography. The online back-end here adds
NO look-ahead: it splits the same job into

  * zero-lag outlier rejection -- a physical heading rate-limit on each rigid
    footprint pose, which kills the single-frame Kabsch flips without any delay,
  * a low-lag jitter filter -- a One-Euro filter (Casiez et al., CHI 2012) per
    smoothed channel (keypoints, ground quad, box, footprint pose, aux centre).
    It is speed-adaptive: near-still objects are smoothed hard, fast ones barely,
    so it removes jitter with ~0-1 frame of effective lag and no fixed delay.

Footprints are corrected rigidly (smoothed centroid + heading applied to the raw
quad), so the locked size is preserved exactly. The module is self-contained
(NumPy only, duck-typed on the RenderState/_TrackView the core emits) so it never
imports the pipeline.
"""
from __future__ import annotations

import numpy as np

__all__ = ["OneEuro", "OnlineStabilizer", "ViewportLock"]


class ViewportLock:
    """Causal world-lock for a static camera (the online global-H counterpart).

    Watches the BEV viewport similarity over a warm-up window; if it has held
    steady (origin + heading variance below tolerance), the camera is effectively
    fixed, so it returns a single frozen ``_Sim`` (the window mean) to pin the BEV
    frame thereafter. A moving camera never trips it, so the radar keeps following.
    """

    def __init__(self, fps: float, warm: int = 0, origin_tol: float = 0.004,
                 angle_tol: float = 0.009):
        from collections import deque
        self.warm = int(warm) if warm else max(20, int(round(fps or 30.0)))
        self.origin_tol = float(origin_tol)
        self.angle_tol = float(angle_tol)
        self._hist = deque(maxlen=self.warm)
        self.frozen = False

    def update(self, viewport, img_wh):
        """Return a frozen ``_Sim`` once the view is confirmed static, else None."""
        if self.frozen or viewport is None or getattr(viewport, "_sim", None) is None:
            return None
        s = viewport._sim
        self._hist.append((np.asarray(s.origin, float).copy(),
                           float(s.angle), float(s.scale)))
        if len(self._hist) < self.warm:
            return None
        O = np.array([h[0] for h in self._hist])
        A = np.array([h[1] for h in self._hist])
        diag = float(np.hypot(*img_wh)) or 1.0
        if O.std(0).max() / diag < self.origin_tol and A.std() < self.angle_tol:
            from uod.bev import _Sim
            self.frozen = True
            return _Sim(scale=float(np.median([h[2] for h in self._hist])),
                        angle=float(np.arctan2(np.mean(np.sin(A)), np.mean(np.cos(A)))),
                        origin=O.mean(0),
                        offset=np.array(viewport.camera_canvas, dtype=np.float64))
        return None


class OneEuro:
    """Scalar One-Euro filter (speed-adaptive causal low-pass)."""

    def __init__(self, fps: float, mincutoff: float = 1.5, beta: float = 0.02,
                 dcutoff: float = 1.0):
        self.fps = float(fps) or 30.0
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x = None
        self._dx = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        x = float(x)
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.fps
        a_d = self._alpha(self.dcutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class OnlineStabilizer:
    """Per-frame causal stabiliser; mutates each RenderState in place (lag 0).

    Parameters mirror the offline filter's intent: ``pos_*`` tune the One-Euro on
    pixel/footprint coordinates, ``head_*`` the heading, and ``head_max_deg`` is
    the zero-lag per-frame heading rate-limit that rejects Kabsch flips.
    """

    def __init__(self, fps: float, *, pos_mincutoff: float = 1.6,
                 pos_beta: float = 0.015, head_mincutoff: float = 1.2,
                 head_beta: float = 0.15, head_max_deg: float = 15.0,
                 max_age: int = 30):
        self.fps = float(fps) or 30.0
        self.pos_mc = float(pos_mincutoff)
        self.pos_beta = float(pos_beta)
        self.head_mc = float(head_mincutoff)
        self.head_beta = float(head_beta)
        self.head_max = np.radians(float(head_max_deg))
        self.max_age = int(max_age)
        self._tracks: dict = {}
        self._aux: dict = {}

    # ------------------------------------------------------------------ #
    def _euros(self, store: dict, key: str, n: int):
        f = store.get(key)
        if f is None or len(f) != n:
            f = [OneEuro(self.fps, self.pos_mc, self.pos_beta) for _ in range(n)]
            store[key] = f
        return f

    def _smooth_array(self, store: dict, key: str, arr) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        flat = a.reshape(-1)
        f = self._euros(store, key, flat.shape[0])
        out = np.array([fi(v) for fi, v in zip(f, flat)], dtype=np.float64)
        return out.reshape(a.shape)

    def _smooth_footprint(self, st: dict, quad) -> np.ndarray:
        quad = np.asarray(quad, dtype=np.float64)
        c0 = quad.mean(axis=0)
        e = quad[1] - quad[0]
        th0 = float(np.arctan2(e[1], e[0]))
        # Unwrap raw heading for continuity with the previous raw value.
        prev_raw = st.get("head_raw")
        thu = th0 if prev_raw is None else prev_raw + ((th0 - prev_raw + np.pi)
                                                       % (2 * np.pi) - np.pi)
        st["head_raw"] = thu
        # Zero-lag rate-limit against the last SMOOTHED heading (rejects flips).
        prev_s = st.get("head_s", thu)
        thu = prev_s + float(np.clip(thu - prev_s, -self.head_max, self.head_max))
        fc = st.setdefault("fc", [OneEuro(self.fps, self.pos_mc, self.pos_beta),
                                  OneEuro(self.fps, self.pos_mc, self.pos_beta)])
        fh = st.setdefault("fh", OneEuro(self.fps, self.head_mc, self.head_beta))
        cs = np.array([fc[0](c0[0]), fc[1](c0[1])])
        ths = fh(thu)
        st["head_s"] = ths
        d = ths - th0                       # small rigid rotation toward smoothed
        c, s = np.cos(d), np.sin(d)
        R = np.array([[c, -s], [s, c]])
        return (quad - c0) @ R.T + cs       # shape preserved exactly

    # ------------------------------------------------------------------ #
    def push(self, state, idx: int):
        """Stabilise one RenderState in place and return it (no added latency)."""
        for v in state.tracks:
            st = self._tracks.setdefault(v.id, {"filt": {}})
            st["last"] = idx
            if v.kpts is not None and np.asarray(v.kpts).size:
                v.kpts = self._smooth_array(st["filt"], "kpts", v.kpts)
            if v.ground is not None and np.asarray(v.ground).shape == (4, 2):
                v.ground = self._smooth_array(st["filt"], "ground", v.ground)
            if v.box_xyxy is not None and np.asarray(v.box_xyxy).shape == (4,):
                v.box_xyxy = self._smooth_array(st["filt"], "box", v.box_xyxy)
            if v.bev_quad is not None and np.asarray(v.bev_quad).shape == (4, 2):
                v.bev_quad = self._smooth_footprint(st, v.bev_quad)

        ac = np.asarray(getattr(state, "aux_centers", []), dtype=np.float64).reshape(-1, 2)
        if len(ac):
            for i, det in enumerate(getattr(state, "aux_dets", []) or []):
                tid = getattr(det, "track_id", None)
                if tid is None or i >= len(ac):
                    continue
                ast = self._aux.setdefault(tid, {"filt": {}})
                ast["last"] = idx
                ac[i] = self._smooth_array(ast["filt"], "c", ac[i])
            state.aux_centers = ac

        # Prune filters for tracks/aux not seen recently (bounded memory).
        for store in (self._tracks, self._aux):
            for k in [k for k, s in store.items() if idx - s.get("last", idx) > self.max_age]:
                del store[k]
        return state
