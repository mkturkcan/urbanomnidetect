"""Real-time temporal tracker for 3D-keypoint detections.

Goals (in priority order)
-------------------------
1. **Persistence** -- an object that is briefly missed must stay where it was
   (or coast along its last trajectory), not blink out. This is what lets the
   BEV layout feel stable.
2. **Low jitter** -- the eye is very sensitive to wobble of BEV keypoints, so
   every track's box *and* eight keypoints are smoothed.
3. **Real time** -- pure NumPy + a single Hungarian assignment per frame;
   microsecond-scale for typical object counts.

Each track runs an **alpha-beta filter** on a state vector
``[cx, cy, w, h, kx_0, ky_0, ... kx_{K-1}, ky_{K-1}]``. The filter predicts the
next frame from a constant-velocity model, then (on a match) nudges the state
and velocity toward the measurement:

    predicted = state + velocity
    residual  = measurement - predicted
    state     = predicted + alpha * residual
    velocity  = velocity  + beta  * residual

On a miss the track *coasts* (``state = predicted``) and its velocity is gently
damped, so stationary objects stay put and moving ones continue plausibly until
``max_age`` frames elapse without a detection.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

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


class Track:
    """A single tracked object with an alpha-beta smoothed state."""

    __slots__ = ("id", "cls", "n_kpts", "state", "vel", "conf",
                 "age", "hits", "time_since_update", "confirmed",
                 "ground_indices")

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

    # ---- filter steps ------------------------------------------------- #
    def predict(self) -> None:
        """Advance to the next frame (called once per frame, before matching).

        Velocity is *not* decayed here -- decaying every frame (including
        matched tracks) would bias velocity low and lag moving objects. Decay
        is applied only while coasting (see :meth:`coast`).
        """
        self.state = self.state + self.vel
        self.age += 1
        self.time_since_update += 1

    def coast(self, vel_decay: float) -> None:
        """Damp velocity for a track that found no detection this frame."""
        self.vel *= vel_decay

    def correct(self, det: Detection, alpha: float, beta: float,
                min_hits: int) -> None:
        """Fuse a matched measurement into the (already predicted) state."""
        meas = np.concatenate([_xyxy_to_cxcywh(det.xyxy),
                               det.kpts.reshape(-1) if self.n_kpts else np.zeros(0)])
        # ``self.state`` already holds the prediction (predict() ran this frame).
        residual = meas - self.state
        self.state = self.state + alpha * residual
        self.vel = self.vel + beta * residual
        self.conf = 0.7 * self.conf + 0.3 * det.conf
        self.cls = det.cls
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= min_hits:
            self.confirmed = True

    def to_detection(self) -> Detection:
        return Detection(cls=self.cls, conf=float(self.conf),
                        xyxy=self.box_xyxy, kpts=self.kpts,
                        ground=self.ground, track_id=self.id)


class MultiObjectTracker:
    """Greedy-gated Hungarian tracker over alpha-beta smoothed tracks.

    Parameters
    ----------
    alpha, beta : float
        Alpha-beta filter gains. Smaller ``alpha`` = smoother but laggier;
        ``beta`` controls how fast velocity adapts. Defaults favour smoothness
        (the eye is jitter-sensitive) while the velocity term prevents lag on
        moving objects.
    max_age : int
        Frames a track may coast without a detection before being dropped.
    min_hits : int
        Matches required before a track is reported as confirmed.
    iou_weight, dist_weight : float
        Blend of (1 - IoU) and normalised centre distance in the match cost.
    iou_gate, dist_gate : float
        A track/detection pair is forbidden unless IoU >= ``iou_gate`` *or*
        the centre distance (normalised by image diagonal) <= ``dist_gate``.
    vel_decay : float
        Per-frame multiplicative decay applied to velocity (stabilises coasting).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.08,
                 max_age: int = 30, min_hits: int = 2,
                 iou_weight: float = 0.7, dist_weight: float = 0.3,
                 iou_gate: float = 0.1, dist_gate: float = 0.08,
                 vel_decay: float = 0.85, conf_thresh: float = 0.25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_weight = float(iou_weight)
        self.dist_weight = float(dist_weight)
        self.iou_gate = float(iou_gate)
        self.dist_gate = float(dist_gate)
        self.vel_decay = float(vel_decay)
        self.conf_thresh = float(conf_thresh)
        self.tracks: List[Track] = []
        self._next_id = 1
        self._diag = 1.0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def update(self, detections: Sequence[Detection], img_wh,
               ground_indices: Optional[Sequence[int]] = None) -> List[Track]:
        """Advance the tracker by one frame and return the confirmed tracks."""
        self._diag = float(np.hypot(img_wh[0], img_wh[1])) or 1.0
        dets = [d for d in detections if d.conf >= self.conf_thresh]

        # Propagate a (possibly newly-resolved) ground-index choice to all
        # existing tracks so their footprints populate once the resolver locks.
        if ground_indices is not None:
            gi = list(ground_indices)
            for t in self.tracks:
                t.ground_indices = gi

        # 1) Predict every existing track forward one frame (no velocity decay).
        for t in self.tracks:
            t.predict()

        # 2) Associate detections to predicted tracks (class-aware, gated).
        matches, un_tracks, un_dets = self._associate(dets)

        # 3) Correct matched tracks; damp velocity only on coasting tracks.
        for ti, di in matches:
            self.tracks[ti].correct(dets[di], self.alpha, self.beta, self.min_hits)
        for ti in un_tracks:
            self.tracks[ti].coast(self.vel_decay)

        # 4) Spawn tracks for unmatched detections.
        for di in un_dets:
            self.tracks.append(
                Track(self._next_id, dets[di], (self.alpha, self.beta),
                      ground_indices))
            self._next_id += 1

        # 5) Cull stale tracks (those that coasted past max_age unmatched).
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Report confirmed tracks; coasting ones (time_since_update > 0) are
        # still returned so missed objects persist at their last position.
        return [t for t in self.tracks if t.confirmed]

    # ------------------------------------------------------------------ #
    def _associate(self, dets: Sequence[Detection]):
        T, D = len(self.tracks), len(dets)
        if T == 0 or D == 0:
            return [], list(range(T)), list(range(D))

        track_boxes = np.array([t.box_xyxy for t in self.tracks])
        det_boxes = np.array([d.xyxy for d in dets])
        iou = _iou_matrix(track_boxes, det_boxes)

        tc = np.array([t.center for t in self.tracks])
        dc = np.array([d.center for d in dets])
        dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / self._diag

        cost = self.iou_weight * (1.0 - iou) + self.dist_weight * dist
        # Class mismatch or failing both gates -> forbidden.
        tcls = np.array([t.cls for t in self.tracks])[:, None]
        dcls = np.array([d.cls for d in dets])[None, :]
        gate = (iou >= self.iou_gate) | (dist <= self.dist_gate)
        forbidden = (tcls != dcls) | (~gate)
        BIG = 1e6
        cost = np.where(forbidden, BIG, cost)

        rows, cols = linear_sum_assignment(cost)
        matches, un_tracks, un_dets = [], [], []
        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= BIG:
                continue
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
        un_tracks = [i for i in range(T) if i not in matched_t]
        un_dets = [i for i in range(D) if i not in matched_d]
        return matches, un_tracks, un_dets
