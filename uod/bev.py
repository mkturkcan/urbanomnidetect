"""Bird's-eye-view projection, a temporally stable viewport, and rendering.

The orthogonality solver returns a homography ``H`` defined only up to a
similarity (it cannot observe global rotation / scale / translation). Rendering
naively with per-frame min/max bounds makes the whole BEV pan and zoom as
objects move -- the opposite of what we want for tracking, where a stationary
object must stay put on the canvas.

:class:`BEVViewport` solves this. Each frame it builds a similarity
``ViewSim`` from the homography's action on **fixed image anchor points** (the
image bottom-center = camera, and the image top-center = "forward"), so

    canvas = ViewSim( H( image_point ) )

maps any fixed image location to a fixed canvas location regardless of which
objects are currently detected. Because ``ViewSim`` is rebuilt from the same
anchors every frame, it automatically cancels ``H``'s similarity-gauge drift
(the gauge ambiguity and the solver's Hartley normaliser are both similarities,
which cancel exactly in ``ViewSim ∘ H`` for every point). The viewport is
therefore recomputed fresh each frame -- it is *not* EMA-blended, since its raw
parameters live in the drifting rectified gauge; temporal smoothness instead
comes from the solver's parameter smoothing and the tracker's keypoint filter,
which act in stable spaces. The net effect: the BEV ground is locked, and a
tracked object that is briefly lost simply stays where it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from homography_rt import apply_homography

__all__ = ["BEVViewport", "fit_min_area_rect", "convex_overlap_fraction",
           "DEFAULT_AUX_ASPECT"]

# Footprint length / width priors by COCO class id (the default auxiliary
# detector is COCO-trained). Used only to extrude a depth for the estimated
# auxiliary footprint; everything else comes from the homography.
DEFAULT_AUX_ASPECT = {
    0: 1.0,    # person
    1: 1.9,    # bicycle
    2: 2.2,    # car
    3: 1.9,    # motorcycle
    5: 3.2,    # bus
    7: 3.0,    # truck
}
_DEFAULT_ASPECT = 1.6


def fit_min_area_rect(points: np.ndarray) -> np.ndarray:
    """Return the 4 corners of the minimum-area rectangle around points."""
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    return cv2.boxPoints(rect)


def convex_overlap_fraction(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Intersection area of two convex polygons divided by ``poly_a``'s area.

    Uses ``cv2.intersectConvexConvex`` so it needs no shapely dependency.
    """
    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)
    area_a = abs(cv2.contourArea(a))
    if area_a < 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter) / area_a


@dataclass
class _Sim:
    """A 2D similarity ``x -> s * R(angle) @ (x - origin) + offset``."""
    scale: float
    angle: float
    origin: np.ndarray   # (2,) in rectified coords
    offset: np.ndarray   # (2,) in canvas coords

    def apply(self, pts: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        return (self.scale * ((pts - self.origin) @ R.T)) + self.offset


class BEVViewport:
    """Maintains a stable rectified-plane -> canvas mapping for video.

    Parameters
    ----------
    canvas_size : (W, H)
        Output canvas size in pixels.
    margin : int
        Border kept clear around content / camera marker.
    forward_frac : float
        Image fraction (from the top) used as the "forward" baseline point on
        the centre column. Must stay *below the horizon* so the homography maps
        it to a finite rectified point: the image top-centre lies on the ground
        plane's vanishing line and explodes. ``0.5`` (image mid-height) is a
        safe, road-level default.
    range_fraction : float
        Fraction of the canvas height that the near->forward baseline maps to;
        sets the zoom. Because the baseline is a *fixed image segment*, the
        scale is independent of which objects are present -- new objects
        entering the scene never rescale the view (radar-like behaviour).
    """

    def __init__(self, canvas_size: Tuple[int, int] = (480, 854),
                 margin: int = 60, forward_frac: float = 0.5,
                 range_fraction: float = 0.22, object_px: float = 40.0):
        self.W, self.H = int(canvas_size[0]), int(canvas_size[1])
        self.margin = int(margin)
        self.forward_frac = float(forward_frac)
        self.range_fraction = float(range_fraction)
        self.object_px = float(object_px)
        self._sim: Optional[_Sim] = None
        self._img_wh: Optional[Tuple[int, int]] = None
        self._radar_bg: Optional[np.ndarray] = None
        self._rings = []   # [(radius_px, range_in_object_units)] for labelling

    def reset(self) -> None:
        self._sim = None

    @property
    def camera_canvas(self) -> Tuple[int, int]:
        return (self.W // 2, self.H - self.margin)

    # ------------------------------------------------------------------ #
    def update(self, H: np.ndarray, img_wh: Tuple[int, int],
               unit: float = 0.0) -> None:
        """Rebuild the viewport from the current homography.

        Origin (ego) and orientation come from the homography's action on the
        camera-column anchors, so they follow the (possibly moving) camera. The
        SCALE is pinned to the objects: ``unit`` is the median footprint size in
        H's gauge, and the scale is chosen so that median footprint renders at a
        fixed ``object_px`` on screen. Because real vehicles are roughly constant
        size, this keeps object scale consistent across camera motion and across
        objects entering/leaving -- there is no metric scale to use otherwise,
        and tying scale to image/camera geometry would change with the camera.
        If ``unit`` is unavailable it falls back to the image-span heuristic.
        """
        self._img_wh = img_wh
        W_img, H_img = img_wh
        anchor = np.array([[W_img * 0.5, H_img - 1.0]])       # camera (near)
        forward = np.array([[W_img * 0.5, H_img * self.forward_frac]])  # road, ahead
        a_r = apply_homography(anchor, H)[0]
        f_r = apply_homography(forward, H)[0]
        d = f_r - a_r
        dn = float(np.hypot(d[0], d[1]))
        if not np.isfinite(dn) or dn < 1e-9 or dn > 1e6:
            return  # degenerate (e.g. baseline crosses the horizon); keep prev

        if unit and unit > 1e-9 and np.isfinite(unit):
            scale = self.object_px / unit           # object-grounded (preferred)
        else:
            scale = self.range_fraction * (self.H - 2 * self.margin) / dn
        # Rotate so the forward direction d points "up" on the canvas (0, -1).
        angle = (-np.pi / 2.0) - np.arctan2(d[1], d[0])
        offset = np.array(self.camera_canvas, dtype=np.float64)
        self._sim = _Sim(scale=scale, angle=angle, origin=a_r.copy(),
                         offset=offset)

    # ------------------------------------------------------------------ #
    def to_canvas(self, image_points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` image points to canvas pixels via ``ViewSim ∘ H``."""
        if self._sim is None:
            self.update(H, self._img_wh or (self.W, self.H))
            if self._sim is None:
                return np.zeros((0, 2))
        rect = apply_homography(image_points, H)
        return self._sim.apply(rect)

    def rect_to_canvas(self, rect_points: np.ndarray) -> np.ndarray:
        """Map already-rectified points (post-H) to canvas pixels via ViewSim."""
        if self._sim is None:
            return np.zeros((0, 2))
        return self._sim.apply(np.asarray(rect_points, dtype=np.float64))

    # ------------------------------------------------------------------ #
    def fov_polygon(self, H: np.ndarray, img_wh: Tuple[int, int],
                    per_edge: int = 48) -> Optional[np.ndarray]:
        """Canvas polygon of the ground actually seen by the camera.

        Projects the image border through ``H`` and keeps the part in front of
        the camera (below the horizon), clipped to the radar's max range. The
        convex hull of those points is the visible field of view -- the region
        of the BEV where detections can exist.
        """
        if self._sim is None:
            return None
        W, Hi = float(img_wh[0]), float(img_wh[1])
        n = int(per_edge)
        edges = [
            np.stack([np.linspace(0, W, n), np.full(n, Hi)], axis=1),    # bottom
            np.stack([np.full(n, W), np.linspace(Hi, 0, n)], axis=1),    # right
            np.stack([np.linspace(W, 0, n), np.full(n, 0.0)], axis=1),   # top
            np.stack([np.full(n, 0.0), np.linspace(0, Hi, n)], axis=1),  # left
        ]
        border = np.vstack(edges)
        ph = np.concatenate([border, np.ones((len(border), 1))], axis=1) @ np.asarray(H).T
        w = ph[:, 2]
        # "In front of camera" = same sign of w as the near anchor (image bottom).
        wa = float((np.asarray(H) @ np.array([W * 0.5, Hi - 1.0, 1.0]))[2])
        front = 1.0 if wa >= 0 else -1.0
        wsafe = np.where(np.abs(w) < 1e-9, 1e-9, w)
        rect = ph[:, :2] / wsafe[:, None]
        canv = self._sim.apply(rect)
        ok = (w * front > 1e-6) & np.all(np.isfinite(canv), axis=1)
        pts = canv[ok]
        if len(pts) < 3:
            return None
        cc = np.array(self.camera_canvas, dtype=np.float64)
        r_max = (self.H - 2 * self.margin) * 0.99
        d = pts - cc
        rad = np.hypot(d[:, 0], d[:, 1])
        scl = np.minimum(1.0, r_max / np.maximum(rad, 1e-6))
        pts = cc + d * scl[:, None]
        hull = cv2.convexHull(pts.astype(np.float32))
        return hull.reshape(-1, 2)

    def draw_fov(self, canvas: np.ndarray, H: np.ndarray,
                 img_wh: Tuple[int, int]) -> None:
        """Shade + outline the currently visible field of view on the radar."""
        from . import style
        poly = self.fov_polygon(H, img_wh)
        if poly is None or len(poly) < 3:
            return
        pts = poly.astype(np.int32)
        ov = canvas.copy()
        cv2.fillPoly(ov, [pts], (44, 46, 50), cv2.LINE_AA)   # faint lift over bg
        cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, canvas)
        ego = style.THEME["ego"]
        cv2.polylines(canvas, [pts], True, (ego[2], ego[1], ego[0]), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _build_radar_bg(self) -> np.ndarray:
        """Build the static dark radar background (rings, spokes, FOV, vignette).

        Everything here is fixed for a given canvas, so it is built once and
        copied per frame. The look is a dark automotive-HMI radar field.
        """
        from . import style
        H, W = self.H, self.W
        cx, cy = self.camera_canvas
        r_max = H - 2 * self.margin

        def _bgr(c):
            return (int(c[2]), int(c[1]), int(c[0]))

        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = _bgr(style.THEME["bev_bg"])

        # (The visible field of view is drawn per-frame from the live
        # homography by ``draw_fov``; the static background no longer bakes a
        # fixed wedge, so the radar reflects what the camera actually sees.)

        # Metric range rings. The BEV scale is pinned so a median footprint is
        # ``object_px`` on screen, hence canvas distance = range measured in
        # object-size units x object_px. Rings are therefore placed at fixed
        # multiples of object_px: they ARE the scale, so the viewer can read
        # range directly and see any scale change against them. ``self._rings``
        # records (radius, range-in-object-units) for labelling.
        self._rings = []
        max_units = r_max / max(self.object_px, 1.0)
        step = max(1, int(round(max_units / 5.0)))
        k = step
        while k * self.object_px <= r_max + 1:
            r = int(k * self.object_px)
            outer = (k + step) * self.object_px > r_max + 1
            col = style.THEME["ring_hi"] if outer else style.THEME["ring"]
            cv2.circle(canvas, (cx, cy), r, _bgr(col), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx - 3, cy - r), (cx + 3, cy - r),
                     _bgr(style.THEME["ring_hi"]), 1, cv2.LINE_AA)
            self._rings.append((r, k))
            k += step

        # Radial spokes across the forward fan.
        for deg in range(-60, 61, 30):
            a = np.deg2rad(deg)
            ex = int(cx + r_max * np.sin(a))
            ey = int(cy - r_max * np.cos(a))
            cv2.line(canvas, (cx, cy), (ex, ey), _bgr(style.THEME["spoke"]),
                     1, cv2.LINE_AA)

        # Soft radial vignette toward the edges.
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt((xx - W / 2.0) ** 2 + (yy - H / 2.0) ** 2)
        d /= d.max()
        vig = np.clip(1.0 - 0.45 * (d ** 2.2), 0.0, 1.0)[:, :, None]
        canvas = (canvas.astype(np.float32) * vig).astype(np.uint8)
        return canvas

    def blank_canvas(self, radar: bool = True) -> np.ndarray:
        """Return a fresh BEV canvas (a copy of the cached radar background)."""
        if not radar:
            from . import style
            c = style.THEME["bev_bg"]
            return np.full((self.H, self.W, 3), (c[2], c[1], c[0]), np.uint8)
        if self._radar_bg is None:
            self._radar_bg = self._build_radar_bg()
        return self._radar_bg.copy()

    def draw_camera(self, canvas: np.ndarray) -> None:
        """Draw the ego marker (a clean upward chevron with a soft glow)."""
        from . import style
        cx, cy = self.camera_canvas
        ego = (style.THEME["ego"][2], style.THEME["ego"][1], style.THEME["ego"][0])
        glow = canvas.copy()
        cv2.circle(glow, (cx, cy), 16, ego, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.18, canvas, 0.82, 0, canvas)
        chevron = np.array([(cx, cy - 11), (cx + 8, cy + 7), (cx, cy + 2),
                            (cx - 8, cy + 7)], np.int32)
        cv2.fillConvexPoly(canvas, chevron, ego, cv2.LINE_AA)

    @staticmethod
    def _fill_alpha(canvas, contour_int, bgr, alpha):
        ch, cw = canvas.shape[:2]
        bx, by, bw, bh = cv2.boundingRect(contour_int)
        x0, y0 = max(bx, 0), max(by, 0)
        x1, y1 = min(bx + bw, cw), min(by + bh, ch)
        if x1 <= x0 or y1 <= y0:
            return
        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.fillPoly(overlay, [contour_int - [x0, y0]], bgr, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def footprint_poly(self, image_quad: np.ndarray, H: np.ndarray,
                       snap_rect: bool = True) -> Optional[np.ndarray]:
        """Map an image-space ground quad to a canvas-space rectangle (or None)."""
        if image_quad is None or len(image_quad) != 4:
            return None
        canvas_quad = self.to_canvas(image_quad, H)
        if len(canvas_quad) != 4 or not np.all(np.isfinite(canvas_quad)):
            return None
        return fit_min_area_rect(canvas_quad) if snap_rect else canvas_quad

    def draw_footprint_poly(self, canvas: np.ndarray, poly: np.ndarray, color,
                            thickness: int = 2, fill_alpha: float = 0.25,
                            glow: bool = True) -> None:
        """Draw a canvas-space footprint rectangle: rounded corners + soft glow.

        ``color`` is RGB. The rounded corners and faint outer glow give an
        elevated, HMI-like look.
        """
        from . import style
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        e0 = float(np.linalg.norm(poly[1] - poly[0]))
        e1 = float(np.linalg.norm(poly[2] - poly[1]))
        radius = min(e0, e1) * 0.30
        from .viz import rounded_contour
        rc = np.round(rounded_contour(poly, radius)).astype(np.int32)
        if glow:
            # Soft halo: a dim, thick under-stroke (class colour lerped toward the
            # background). Cheaper than a blended fill and reads as elevation.
            bg = np.array(style.THEME["bev_bg"], dtype=np.float32)
            gc = bg * 0.6 + np.array(color, dtype=np.float32) * 0.4
            cv2.polylines(canvas, [rc], True,
                          (int(gc[2]), int(gc[1]), int(gc[0])),
                          thickness + 6, cv2.LINE_AA)
        if fill_alpha > 0.0:
            self._fill_alpha(canvas, rc, bgr, fill_alpha)
        cv2.polylines(canvas, [rc], True, bgr, thickness, cv2.LINE_AA)

    def draw_footprint(self, canvas: np.ndarray, image_quad: np.ndarray,
                       H: np.ndarray, color, snap_rect: bool = True,
                       thickness: int = 2, fill_alpha: float = 0.25,
                       label: Optional[str] = None, glow: bool = True
                       ) -> Optional[np.ndarray]:
        """Compute and draw a footprint; returns its canvas rectangle or None."""
        poly = self.footprint_poly(image_quad, H, snap_rect)
        if poly is None:
            return None
        self.draw_footprint_poly(canvas, poly, color, thickness, fill_alpha, glow)
        return poly

    def estimate_aux_footprint(self, box_xyxy: np.ndarray, H: np.ndarray,
                               aspect: float, width_scale: float = 0.9,
                               min_w: float = 4.0) -> Optional[np.ndarray]:
        """Estimate a ground footprint for an auxiliary (box-only) detection.

        We cannot recover the true cuboid corners from a 2D box, so we apply a
        height/ground-plane decomposition: the box's bottom edge is taken to lie
        on the ground (the standard inverse-perspective assumption), and the
        rest of the box height is attributed to the object's vertical extent and
        discarded. Mapping the two bottom corners through ``H`` gives the
        footprint's near edge, width, and orientation on the ground. The depth
        (length away from the camera) is not observable from one box, so it is
        extruded from a class ``aspect`` (length/width) prior.

        Returns a canvas-space ``(4, 2)`` rectangle, or ``None`` if degenerate.
        """
        x1, y1, x2, y2 = box_xyxy
        bottom = np.array([[x1, y2], [x2, y2]], dtype=np.float64)
        bc = self.to_canvas(bottom, H)
        if len(bc) != 2 or not np.all(np.isfinite(bc)):
            return None
        bl, br = bc[0], bc[1]
        u = br - bl
        w = float(np.hypot(u[0], u[1]))
        if w < min_w:
            return None
        u = u / w
        w *= width_scale
        # Re-centre the (possibly shrunk) near edge on its midpoint.
        mid = 0.5 * (bl + br)
        bl = mid - 0.5 * w * u
        br = mid + 0.5 * w * u
        v = np.array([-u[1], u[0]])                      # perpendicular
        cam = np.array(self.camera_canvas, dtype=np.float64)
        if np.dot(v, mid - cam) < 0:                     # point away from camera
            v = -v
        depth = max(aspect, 0.2) * w
        far_l = bl + v * depth
        far_r = br + v * depth
        return np.array([bl, br, far_r, far_l])
