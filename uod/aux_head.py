"""Auxiliary-detector BEV-center head (paper Sec. 3.5, Eqs. 4-5).

A frozen, COCO-trained 2D detector (default ``yolo26x``) broadens recall for
classes the pose model misses. Box-only detections have no keypoints, so we
cannot form a ground footprint for them directly. Instead we learn, *per
frame*, a closed-form ridge-regularised linear map from box features

    phi(b) = [cx, cy, w, h, w/h]

to the ground-footprint center of the pose detections (whose footprints we do
have), then apply it to the auxiliary boxes:

    W* = argmin_W  sum_n || c_n - W phi(b_n) ||^2 + lambda ||W||_F^2
    c_hat = W* phi(b)

The fit is per frame because the image->ground relationship is view-dependent;
solving in closed form keeps it sub-millisecond.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["AuxCenterRegressor"]


def _features(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Map ``(N, 4)`` xyxy boxes to ``(N, 6)`` features ``[cx,cy,w,h,w/h,1]``.

    A bias term (the trailing 1) is appended so the map can express an offset;
    it is excluded from the ridge penalty.
    """
    x1, y1, x2, y2 = (boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                      boxes_xyxy[:, 2], boxes_xyxy[:, 3])
    w = np.maximum(x2 - x1, 1e-6)
    h = np.maximum(y2 - y1, 1e-6)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return np.stack([cx, cy, w, h, w / h, np.ones_like(cx)], axis=1)


class AuxCenterRegressor:
    """Per-frame ridge map from auxiliary box features to ground centers."""

    def __init__(self, lam: float = 1.0):
        self.lam = float(lam)
        self._W: Optional[np.ndarray] = None  # (6, 2)

    @property
    def fitted(self) -> bool:
        return self._W is not None

    def fit(self, boxes_xyxy: np.ndarray, centers: np.ndarray) -> bool:
        """Fit ``W`` from anchor boxes and their ground-footprint centers.

        Returns ``True`` if a usable map was produced (needs >= 2 anchors).
        """
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        if len(boxes_xyxy) < 2 or len(boxes_xyxy) != len(centers):
            self._W = None
            return False
        Phi = _features(boxes_xyxy)                  # (N, 6)
        # Ridge on the non-bias columns only.
        reg = self.lam * np.eye(Phi.shape[1])
        reg[-1, -1] = 0.0
        try:
            A = Phi.T @ Phi + reg
            self._W = np.linalg.solve(A, Phi.T @ centers)   # (6, 2)
        except np.linalg.LinAlgError:
            self._W = None
            return False
        return True

    def predict(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Predict ground-footprint centers for ``(N, 4)`` boxes -> ``(N, 2)``."""
        if self._W is None or len(boxes_xyxy) == 0:
            return np.zeros((0, 2))
        return _features(np.asarray(boxes_xyxy, dtype=np.float64)) @ self._W
