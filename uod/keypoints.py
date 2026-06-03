"""Parsing of UrbanOmniDetect pose-model outputs into structured detections.

The model emits, per object, a 2D box and eight ordered keypoints that are the
projected corners of the object's 3D cuboid. Four of those eight are the
*ground-contact* corners that the BEV head consumes.

Ground-corner convention
-------------------------
The paper text states indices 4-7 are ground; however the *released*
``urbanomnidetect_*`` checkpoints emit keypoints with the opposite ordering
(indices 0-3 are the ground corners, verified empirically: they sit ~25 px
below their vertical partners on the released models). To be correct for any
checkpoint regardless of its baked-in ordering, :class:`GroundIndexResolver`
auto-detects which half is the ground by majority vote over the first frames
(the ground corners are lower in the image, i.e. larger ``y``, than their
matching top corners) and then *locks* the choice for temporal stability. The
choice can also be forced explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Detection", "GroundIndexResolver", "parse_pose_result",
           "parse_boxes_result"]


@dataclass
class Detection:
    """A single detected object in full-image pixel coordinates."""
    cls: int
    conf: float
    xyxy: np.ndarray                  # (4,) [x1, y1, x2, y2]
    kpts: np.ndarray = field(        # (K, 2) all keypoints, may be empty
        default_factory=lambda: np.zeros((0, 2)))
    ground: np.ndarray = field(      # (4, 2) ground footprint, cyclic order
        default_factory=lambda: np.zeros((0, 2)))
    track_id: int = -1
    is_aux: bool = False             # came from the auxiliary box-only detector

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])

    @property
    def has_ground(self) -> bool:
        return self.ground.shape == (4, 2)


class GroundIndexResolver:
    """Decide which 4 of the 8 keypoints are the ground-contact corners.

    Parameters
    ----------
    forced : sequence of int, optional
        If given, always use exactly these indices (e.g. ``[0, 1, 2, 3]`` or
        ``[4, 5, 6, 7]``) and skip auto-detection.
    vote_frames : int
        Number of frames worth of detections to accumulate before locking the
        auto-detected choice.
    """

    def __init__(self, forced: Optional[Sequence[int]] = None,
                 vote_frames: int = 5):
        self._forced = list(forced) if forced is not None else None
        self.vote_frames = int(vote_frames)
        self._votes_lower_half = 0      # votes that the *second* half is ground
        self._votes_total = 0
        self._frames_seen = 0
        self._locked: Optional[List[int]] = list(forced) if forced else None

    @property
    def locked(self) -> bool:
        return self._locked is not None

    def indices(self, n_kpts: int) -> Optional[List[int]]:
        """Return the resolved ground indices, or ``None`` if not yet known."""
        if self._locked is not None:
            return self._locked
        if n_kpts >= 8:
            return None  # undecided
        # Degenerate model with <8 keypoints: treat all as "ground".
        return list(range(n_kpts))

    def observe(self, kpts_batch: np.ndarray) -> None:
        """Accumulate evidence from one frame's keypoints ``(N, K, 2)``."""
        if self._locked is not None or kpts_batch is None or len(kpts_batch) == 0:
            return
        k = kpts_batch.shape[1]
        if k == 0:
            return  # degenerate frame; do not lock to an empty index set
        if k < 8:
            self._locked = list(range(k))
            return
        half = k // 2
        # Per-detection: is the second half (indices half:) lower (larger y)?
        y_first = kpts_batch[:, :half, 1].mean(axis=1)
        y_second = kpts_batch[:, half:, 1].mean(axis=1)
        self._votes_lower_half += int(np.sum(y_second > y_first))
        self._votes_total += len(kpts_batch)
        self._frames_seen += 1
        if self._frames_seen >= self.vote_frames and self._votes_total > 0:
            second_is_ground = self._votes_lower_half > self._votes_total / 2
            self._locked = (list(range(half, k)) if second_is_ground
                            else list(range(half)))


def _order_ground_quad(quad: np.ndarray) -> np.ndarray:
    """Order four ground corners CCW around their centroid.

    The model already emits a consistent cyclic order, but auxiliary sources or
    occlusion can scramble it; re-ordering guarantees a simple (non
    self-intersecting) polygon for the orthogonality loss.
    """
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    return quad[np.argsort(ang)]


def parse_pose_result(result, ground_indices: Optional[Sequence[int]],
                      reorder: bool = False) -> List[Detection]:
    """Convert one Ultralytics pose ``Results`` object into ``Detection``s.

    ``ground_indices`` selects the ground keypoints; if ``None`` the per-object
    ground footprint is left empty (used while the resolver is still voting).
    """
    dets: List[Detection] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return dets

    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    conf = result.boxes.conf.cpu().numpy().astype(np.float64)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    has_kp = result.keypoints is not None and result.keypoints.data is not None
    kp_all = (result.keypoints.data.cpu().numpy()[:, :, :2].astype(np.float64)
              if has_kp else None)

    gi = list(ground_indices) if ground_indices else None
    for i in range(len(xyxy)):
        kp = kp_all[i] if kp_all is not None else np.zeros((0, 2))
        ground = np.zeros((0, 2))
        if gi and kp.shape[0] >= max(gi) + 1:
            ground = kp[gi].copy()
            if reorder:
                ground = _order_ground_quad(ground)
        dets.append(Detection(cls=int(cls[i]), conf=float(conf[i]),
                              xyxy=xyxy[i], kpts=kp, ground=ground))
    return dets


def parse_boxes_result(result, conf_thresh: float = 0.0,
                       class_filter: Optional[Sequence[int]] = None
                       ) -> List[Detection]:
    """Convert a box-only (auxiliary) ``Results`` object into ``Detection``s."""
    dets: List[Detection] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    conf = result.boxes.conf.cpu().numpy().astype(np.float64)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    cf = set(class_filter) if class_filter is not None else None
    for i in range(len(xyxy)):
        if conf[i] < conf_thresh:
            continue
        if cf is not None and int(cls[i]) not in cf:
            continue
        dets.append(Detection(cls=int(cls[i]), conf=float(conf[i]),
                              xyxy=xyxy[i], is_aux=True))
    return dets
