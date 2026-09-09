"""Real-time temporal tracker for 3D-keypoint detections.

Goals (in priority order)
-------------------------
1. **Persistence** -- an object that is briefly missed must stay where it was
   (or coast along its last trajectory), not blink out. This is what lets the
   BEV layout feel stable.
2. **Correct identity** -- a track ID must follow the *same* physical object.
   Associating a very differently sized/placed box to an existing ID makes the
   box teleport; preventing that is as important as persistence.
3. **Low jitter** -- the eye is very sensitive to wobble of BEV keypoints, so
   every track's box *and* eight keypoints are smoothed.
4. **Real time** -- pure NumPy + a single Hungarian assignment per stage.

Each track runs an **alpha-beta filter** on a state vector
``[cx, cy, w, h, kx_0, ky_0, ... kx_{K-1}, ky_{K-1}]`` (constant-velocity
prediction, measurement-nudged correction).

Association follows the modern SORT family rather than plain IoU matching, to
stop the identity swaps that make boxes jump:

* **Shape-aware gating** -- a track and a detection can only match if their
  sizes (log-area) and aspect ratios are compatible. A small far car can no
  longer steal a large near car's ID.
* **DIoU cost** (Zheng et al., AAAI 2020) -- distance-penalised IoU, which is
  better behaved than IoU when boxes barely overlap.
* **Observation-Centric Momentum** (OC-SORT, CVPR 2023) -- a velocity-direction
  consistency penalty that rejects matches inconsistent with a track's motion.
* **Camera-motion compensation** (BoT-SORT, 2022) -- sparse optical flow
  estimates the inter-frame global motion and warps every prediction into the
  current frame before matching, which matters under a moving/aerial camera.
* **Two-stage / BYTE association** (ByteTrack, ECCV 2022) -- high-confidence
  detections match first, low-confidence ones then recover coasting tracks;
  tracks are never spawned from low-confidence boxes.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .keypoints import Detection

__all__ = ["Track", "MultiObjectTracker"]


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1])


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b
    hw, hh = w * 0.5, h * 0.5
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> ``(len(a), len(b))``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _diou_matrix(a: np.ndarray, b: np.ndarray):
    """Distance-IoU and IoU between two sets of xyxy boxes.

    ``DIoU = IoU - rho^2(centres) / c^2`` where ``c`` is the diagonal of the
    smallest box enclosing both. Returns ``(diou, iou)``, each ``(len(a), len(b))``.
    """
    if len(a) == 0 or len(b) == 0:
        z = np.zeros((len(a), len(b)))
        return z, z
    iou = _iou_matrix(a, b)
    ca = np.stack([(a[:, 0] + a[:, 2]) * 0.5, (a[:, 1] + a[:, 3]) * 0.5], axis=1)
    cb = np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], axis=1)
    rho2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    c2 = ((rb - lt) ** 2).sum(axis=2) + 1e-9
    return iou - rho2 / c2, iou


def global_motion_series(frames, downscale: float = 640.0) -> np.ndarray:
    """Per-consecutive-frame global camera motion, for fixed-camera detection.

    Uses the same sparse-optical-flow + partial-affine stack as the tracker's
    camera-motion compensation. Returns an ``(M, 3)`` array of
    ``[translation_px, |rotation_deg|, |log scale|]`` (translation in full-res
    pixels), or an empty ``(0, 3)`` array if it could not be estimated.
    """
    out = []
    prev = None
    prev_pts = None
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = downscale / max(g.shape)
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
            cur, stt, _ = cv2.calcOpticalFlowPyrLK(prev, g, prev_pts, None)
            if cur is not None and stt is not None:
                ok = stt.ravel() == 1
                if int(ok.sum()) >= 8:
                    A, _ = cv2.estimateAffinePartial2D(
                        prev_pts[ok], cur[ok], method=cv2.RANSAC,
                        ransacReprojThreshold=3.0)
                    if A is not None and np.all(np.isfinite(A)):
                        R = A[:, :2]
                        sc = float(np.sqrt(abs(np.linalg.det(R)))) or 1.0
                        rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                        t = float(np.hypot(A[0, 2], A[1, 2]) / s)
                        out.append([t, abs(rot), abs(np.log(sc))])
        prev = g
        prev_pts = cv2.goodFeaturesToTrack(
            g, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=3)
    return np.array(out) if out else np.zeros((0, 3))


def camera_trajectory(frames, downscale: float = 640.0):
    """Cumulative inter-frame camera motion as 3x3 similarities (frame0 -> t).

    Returns ``(C, steps)`` where ``C[t]`` (one per frame) maps a point in frame 0
    to its location in frame ``t`` (so ``inv(C[t])`` brings frame ``t`` back into
    frame 0), and ``steps[t] = [translation_px, |rotation_deg|, |log scale|]`` is
    that frame's incremental motion. Detection-independent (pure image motion),
    so objects entering/leaving never perturb it -- the property that lets it
    both classify the camera and stabilise the global BEV frame.
    """
    C = [np.eye(3)]
    steps = [[0.0, 0.0, 0.0]]
    prev = None
    prev_pts = None
    cur = np.eye(3)
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = downscale / max(g.shape)
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        M = np.eye(3)
        step = [0.0, 0.0, 0.0]
        if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, prev_pts, None)
            if nxt is not None and st is not None:
                ok = st.ravel() == 1
                if int(ok.sum()) >= 8:
                    A, _ = cv2.estimateAffinePartial2D(
                        prev_pts[ok], nxt[ok], method=cv2.RANSAC,
                        ransacReprojThreshold=3.0)
                    if A is not None and np.all(np.isfinite(A)):
                        A = A.copy()
                        A[:, 2] /= s                       # un-downscale translation
                        M = np.vstack([A, [0.0, 0.0, 1.0]])
                        sc = float(np.sqrt(abs(np.linalg.det(A[:, :2])))) or 1.0
                        step = [float(np.hypot(A[0, 2], A[1, 2])),
                                abs(float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))),
                                abs(float(np.log(sc)))]
        if prev is not None:
            cur = M @ cur
            C.append(cur.copy())
            steps.append(step)
        prev = g
        prev_pts = cv2.goodFeaturesToTrack(
            g, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=3)
    return C, np.array(steps)


class Track:
    """A single tracked object with an alpha-beta smoothed state."""

    __slots__ = ("id", "cls", "n_kpts", "state", "vel", "conf",
                 "age", "hits", "time_since_update", "confirmed",
                 "ground_indices", "last_obs", "obs_centers")

    # Observations spanned when estimating the motion direction for OC-SORT's
    # observation-centric momentum (longer = smoother direction, more lag).
    OCM_DELTA = 3

    def __init__(self, tid: int, det: Detection, alpha_beta,
                 ground_indices: Optional[Sequence[int]]):
        self.id = tid
        self.cls = det.cls
        self.n_kpts = len(det.kpts)
        self.ground_indices = list(ground_indices) if ground_indices else None
        box = _xyxy_to_cxcywh(det.xyxy)
        kp = det.kpts.reshape(-1) if self.n_kpts else np.zeros(0)
        self.state = np.concatenate([box, kp])
        self.vel = np.zeros_like(self.state)
        self.conf = det.conf
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.last_obs = box[:2].copy()                 # centre at last real match
        self.obs_centers = deque([box[:2].copy()], maxlen=self.OCM_DELTA + 1)

    # ---- state views -------------------------------------------------- #
    def predicted_state(self) -> np.ndarray:
        return self.state + self.vel

    @property
    def box_xyxy(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.state[:4])

    @property
    def kpts(self) -> np.ndarray:
        if self.n_kpts == 0:
            return np.zeros((0, 2))
        return self.state[4:].reshape(self.n_kpts, 2)

    @property
    def ground(self) -> np.ndarray:
        if not self.ground_indices or self.n_kpts < max(self.ground_indices) + 1:
            return np.zeros((0, 2))
        return self.kpts[self.ground_indices]

    @property
    def center(self) -> np.ndarray:
        return self.state[:2].copy()

    def motion_dir(self) -> Optional[np.ndarray]:
        """Unit velocity direction over the last ``OCM_DELTA`` observations."""
        if len(self.obs_centers) < 2:
            return None
        d = self.obs_centers[-1] - self.obs_centers[0]
        n = float(np.hypot(d[0], d[1]))
        return d / n if n > 1e-6 else None

    # ---- filter steps ------------------------------------------------- #
    def predict(self) -> None:
        """Advance to the next frame (called once per frame, before matching)."""
        self.state = self.state + self.vel
        self.age += 1
        self.time_since_update += 1

    def apply_cmc(self, M: np.ndarray) -> None:
        """Warp the predicted state by a 2x3 similarity (camera-motion comp.)."""
        R, t = M[:, :2], M[:, 2]
        s = float(np.sqrt(abs(np.linalg.det(R)))) or 1.0
        self.state[:2] = R @ self.state[:2] + t
        self.state[2:4] *= s
        self.vel[:2] = R @ self.vel[:2]
        self.vel[2:4] *= s
        if self.n_kpts:
            k = self.state[4:].reshape(-1, 2)
            self.state[4:] = (k @ R.T + t).reshape(-1)
            v = self.vel[4:].reshape(-1, 2)
            self.vel[4:] = (v @ R.T).reshape(-1)
        self.last_obs = R @ self.last_obs + t
        for i in range(len(self.obs_centers)):
            self.obs_centers[i] = R @ self.obs_centers[i] + t

    def coast(self, vel_decay: float) -> None:
        """Damp velocity for a track that found no detection this frame."""
        self.vel *= vel_decay

    def correct(self, det: Detection, alpha: float, beta: float,
                min_hits: int) -> None:
        """Fuse a matched measurement into the (already predicted) state."""
        meas = np.concatenate([_xyxy_to_cxcywh(det.xyxy),
                               det.kpts.reshape(-1) if self.n_kpts else np.zeros(0)])
        residual = meas - self.state
        self.state = self.state + alpha * residual
        self.vel = self.vel + beta * residual
        self.conf = 0.7 * self.conf + 0.3 * det.conf
        self.cls = det.cls
        self.hits += 1
        self.time_since_update = 0
        self.last_obs = self.state[:2].copy()
        self.obs_centers.append(self.state[:2].copy())
        if self.hits >= min_hits:
            self.confirmed = True

    def to_detection(self) -> Detection:
        return Detection(cls=self.cls, conf=float(self.conf),
                        xyxy=self.box_xyxy, kpts=self.kpts,
                        ground=self.ground, track_id=self.id)


class MultiObjectTracker:
    """Shape-aware, motion-compensated, two-stage Hungarian tracker.

    Parameters
    ----------
    alpha, beta : float
        Alpha-beta filter gains (smaller alpha = smoother / laggier).
    max_age : int
        Frames a track may coast without a detection before being dropped.
    min_hits : int
        Matches required before a track is reported as confirmed.
    iou_weight, dist_weight, shape_weight, ocm_weight : float
        Blend of the cost terms: (1 - DIoU), centre distance, shape mismatch
        (log-size + log-aspect), and observation-centric momentum.
    diou_gate, dist_gate : float
        A pair is allowed (subject to the shape gates) only if its DIoU is at
        least ``diou_gate`` *or* its centre distance (normalised by the image
        diagonal) is within the coast-scaled ``dist_gate``.
    size_gate, aspect_gate : float
        Maximum ``|log|`` ratio of areas / aspect ratios for a legal match.
        These are the gates that stop "very different boxes" sharing an ID.
    track_high_conf : float
        Detections at or above this confidence form BYTE's first (primary)
        association stage; the rest only recover already-existing tracks.
    cmc : bool
        Enable camera-motion compensation (sparse optical flow).
    vel_decay : float
        Per-frame multiplicative decay applied to a coasting track's velocity.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.08,
                 max_age: int = 30, min_hits: int = 2,
                 iou_weight: float = 0.55, dist_weight: float = 0.10,
                 shape_weight: float = 0.20, ocm_weight: float = 0.15,
                 diou_gate: float = -0.2, dist_gate: float = 0.08,
                 size_gate: float = 0.92, aspect_gate: float = 0.79,
                 track_high_conf: float = 0.5, cmc: bool = True,
                 vel_decay: float = 0.85, conf_thresh: float = 0.25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_weight = float(iou_weight)
        self.dist_weight = float(dist_weight)
        self.shape_weight = float(shape_weight)
        self.ocm_weight = float(ocm_weight)
        self.diou_gate = float(diou_gate)
        self.dist_gate = float(dist_gate)
        self.size_gate = float(size_gate)
        self.aspect_gate = float(aspect_gate)
        self.track_high_conf = float(track_high_conf)
        self.cmc = bool(cmc)
        self.vel_decay = float(vel_decay)
        self.conf_thresh = float(conf_thresh)
        self.tracks: List[Track] = []
        self._next_id = 1
        self._diag = 1.0
        # CMC state (sparse optical flow on a downscaled grey frame).
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._cmc_scale = 1.0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1
        self._prev_gray = None
        self._prev_pts = None

    # ------------------------------------------------------------------ #
    def _estimate_cmc(self, frame) -> np.ndarray:
        """Estimate the inter-frame global motion as a 2x3 similarity."""
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if frame is None:
            return M
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 640.0 / max(g.shape)
        if scale < 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
        if (self._prev_gray is not None and self._prev_pts is not None
                and len(self._prev_pts) >= 8):
            cur, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, g,
                                                  self._prev_pts, None)
            if cur is not None and st is not None:
                ok = st.ravel() == 1
                if int(ok.sum()) >= 8:
                    A, _ = cv2.estimateAffinePartial2D(
                        self._prev_pts[ok], cur[ok], method=cv2.RANSAC,
                        ransacReprojThreshold=3.0)
                    if A is not None and np.all(np.isfinite(A)):
                        A = A.copy()
                        A[:, 2] /= scale            # un-downscale the translation
                        M = A
        self._prev_gray = g
        self._prev_pts = cv2.goodFeaturesToTrack(
            g, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=3)
        self._cmc_scale = scale
        return M

    # ------------------------------------------------------------------ #
    def update(self, detections: Sequence[Detection], img_wh,
               ground_indices: Optional[Sequence[int]] = None,
               frame=None) -> List[Track]:
        """Advance the tracker by one frame and return the confirmed tracks."""
        self._diag = float(np.hypot(img_wh[0], img_wh[1])) or 1.0
        dets = [d for d in detections if d.conf >= self.conf_thresh]

        if ground_indices is not None:
            gi = list(ground_indices)
            for t in self.tracks:
                t.ground_indices = gi

        # 1) Predict, then warp predictions into the current frame (CMC).
        for t in self.tracks:
            t.predict()
        if self.cmc and self.tracks is not None:
            M = self._estimate_cmc(frame) if self.cmc else None
            if M is not None and not np.allclose(M, [[1, 0, 0], [0, 1, 0]]):
                for t in self.tracks:
                    t.apply_cmc(M)
        elif self.cmc:
            self._estimate_cmc(frame)   # keep CMC history warm even with 0 tracks

        # 2) Two-stage (BYTE) association: high-confidence first, low after.
        hi = [i for i, d in enumerate(dets) if d.conf >= self.track_high_conf]
        lo = [i for i, d in enumerate(dets) if d.conf < self.track_high_conf]

        track_idx = list(range(len(self.tracks)))
        matches = []
        m1, un_t1, un_d1 = self._associate(track_idx, hi, dets, stage=1)
        matches += m1
        # Stage 2 recovers still-unmatched tracks from low-confidence boxes.
        m2, un_t2, un_d2 = self._associate(un_t1, lo, dets, stage=2)
        matches += m2

        matched_t = {ti for ti, _ in matches}

        # 3) Correct matched tracks; coast the rest.
        for ti, di in matches:
            self.tracks[ti].correct(dets[di], self.alpha, self.beta, self.min_hits)
        for ti in range(len(self.tracks)):
            if ti not in matched_t:
                self.tracks[ti].coast(self.vel_decay)

        # 4) Spawn tracks only from unmatched HIGH-confidence detections.
        for di in un_d1:
            self.tracks.append(
                Track(self._next_id, dets[di], (self.alpha, self.beta),
                      ground_indices))
            self._next_id += 1

        # 5) Cull stale tracks, and dispose of any that have left the scene
        #    through a camera edge: once an unseen track's centre is outside the
        #    image its object is gone from view, so we drop it immediately
        #    instead of coasting a phantom box at the border for ``max_age``.
        W_img, H_img = img_wh
        kept = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue
            cx, cy = t.state[0], t.state[1]
            left_frame = not (0.0 <= cx <= W_img and 0.0 <= cy <= H_img)
            if left_frame and t.time_since_update >= 1:
                continue
            kept.append(t)
        self.tracks = kept

        return [t for t in self.tracks if t.confirmed]

    # ------------------------------------------------------------------ #
    def _associate(self, track_idx, det_idx, dets, stage: int):
        """Gated Hungarian assignment over a subset of tracks/detections.

        ``track_idx`` / ``det_idx`` index into ``self.tracks`` / ``dets``.
        Returns ``(matches, unmatched_track_idx, unmatched_det_idx)`` with the
        original indices preserved.
        """
        if not track_idx or not det_idx:
            return [], list(track_idx), list(det_idx)

        tracks = [self.tracks[i] for i in track_idx]
        ds = [dets[i] for i in det_idx]
        tb = np.array([t.box_xyxy for t in tracks])
        db = np.array([d.xyxy for d in ds])
        diou, _ = _diou_matrix(tb, db)

        tc = np.array([t.center for t in tracks])
        dc = np.array([d.center for d in ds])
        dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / self._diag

        # Shape terms: log-area and log-aspect differences.
        tw = np.clip(tb[:, 2] - tb[:, 0], 1e-3, None)
        th = np.clip(tb[:, 3] - tb[:, 1], 1e-3, None)
        dw = np.clip(db[:, 2] - db[:, 0], 1e-3, None)
        dh = np.clip(db[:, 3] - db[:, 1], 1e-3, None)
        d_area = np.abs(np.log((dw * dh)[None, :]) - np.log((tw * th)[:, None]))
        d_aspect = np.abs(np.log((dw / dh)[None, :]) - np.log((tw / th)[:, None]))

        # Observation-centric momentum: direction(track) vs direction(track->det).
        ocm = np.zeros((len(tracks), len(ds)))
        for i, t in enumerate(tracks):
            md = t.motion_dir()
            if md is None:
                continue
            cand = dc - t.last_obs[None, :]
            n = np.linalg.norm(cand, axis=1)
            valid = n > 1e-6
            cos = np.zeros(len(ds))
            cos[valid] = (cand[valid] @ md) / n[valid]
            ocm[i] = (1.0 - cos) * 0.5

        # Cost (only meaningful for allowed pairs).
        cost = (self.iou_weight * (1.0 - diou) * 0.5
                + self.dist_weight * dist
                + self.shape_weight * 0.5 * (d_area / self.size_gate
                                             + d_aspect / self.aspect_gate)
                + self.ocm_weight * ocm)

        # Gating.
        tsu = np.array([t.time_since_update for t in tracks])[:, None]
        tcls = np.array([t.cls for t in tracks])[:, None]
        dcls = np.array([d.cls for d in ds])[None, :]
        # Anti-steal / anti-swap plausibility. A detection may only match a track
        # if it falls inside a velocity-aligned ellipse around the track's LAST
        # OBSERVATION (not its drifted prediction): generous ALONG the heading
        # (the object may have travelled), tight ACROSS it (it stays in its
        # lane), and tight isotropically when (near-)stationary. Tolerances scale
        # with the object's box size, so adjacent same-size cars in stopped
        # traffic -- one box-width apart -- can no longer swap IDs, while a
        # coasting track still cannot capture a different (cross-lane / far)
        # object -- that object keeps or gets its own ID instead.
        last_obs = np.array([t.last_obs for t in tracks])          # (T, 2)
        vel = np.array([t.vel[:2] for t in tracks])                # (T, 2)
        spd = np.linalg.norm(vel, axis=1)                          # (T,)
        scale = np.sqrt(np.maximum(tw * th, 1.0))                  # (T,) box size, px
        disp = dc[None, :, :] - last_obs[:, None, :]               # (T, D, 2)
        dn = np.linalg.norm(disp, axis=2)                          # (T, D)
        tsuf = tsu.astype(float)                                   # (T, 1)
        moving = (spd > 1.5)[:, None]                              # (T, 1)
        vh = vel / np.maximum(spd[:, None], 1e-6)                  # (T, 2)
        along = np.abs(np.einsum('td,tnd->tn', vh, disp))          # (T, D)
        cross = np.sqrt(np.maximum(dn ** 2 - along ** 2, 0.0))     # (T, D)
        sc = scale[:, None]
        along_tol = 0.6 * sc + 1.3 * spd[:, None] * tsuf + 0.15 * sc * tsuf
        cross_tol = 0.6 * sc + 0.15 * sc * tsuf
        iso_tol = 0.7 * sc + 0.2 * sc * tsuf
        plausible = np.where(moving,
                             (along <= along_tol) & (cross <= cross_tol),
                             dn <= iso_tol)
        loc_ok = (diou >= self.diou_gate) | (dist <= self.dist_gate)
        shape_ok = (d_area <= self.size_gate) & (d_aspect <= self.aspect_gate)
        if stage == 2:                       # low-conf recovery: require overlap
            loc_ok = diou >= max(self.diou_gate, 0.1)
        forbidden = (tcls != dcls) | (~loc_ok) | (~shape_ok) | (~plausible)

        BIG = 1e6
        cost = np.where(forbidden, BIG, cost)
        rows, cols = linear_sum_assignment(cost)
        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= BIG:
                continue
            matches.append((track_idx[r], det_idx[c]))
            matched_t.add(r)
            matched_d.add(c)
        un_t = [track_idx[i] for i in range(len(tracks)) if i not in matched_t]
        un_d = [det_idx[j] for j in range(len(ds)) if j not in matched_d]
        return matches, un_t, un_d
