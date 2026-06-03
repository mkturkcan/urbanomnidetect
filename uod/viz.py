"""Geometry drawing helpers for the camera and BEV panels.

These draw shapes only (boxes, cuboids, footprints). All text, chips, and UI
chrome are batched separately through :class:`uod.textdraw.UILayer` so they can
be rendered crisply in a single anti-aliased pass.

The signature trick for legibility on arbitrary backgrounds is the *halo*: every
coloured stroke is drawn over a slightly thicker dark stroke, so edges stay
readable whether they cross sky, asphalt, or a white truck.

Colours are passed as RGB (matching :mod:`uod.style`); conversion to OpenCV's
BGR happens here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from . import style

__all__ = ["stroke_polyline", "fill_poly_alpha", "draw_cuboid", "draw_box",
           "draw_ground_quad", "draw_dashed_polyline", "stack_side_by_side",
           "rounded_contour"]


def rounded_contour(pts, radius, steps=4):
    """Return a dense contour that rounds the corners of a convex polygon.

    Each sharp vertex is replaced by a short quadratic-bezier fillet (the corner
    is the control point), so it works for rotated rectangles, not just
    axis-aligned ones. ``radius`` is clamped per corner to half the shorter
    adjacent edge.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3 or radius <= 0:
        return pts
    out = []
    for i in range(n):
        prev, cur, nxt = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = prev - cur, nxt - cur
        l1, l2 = np.hypot(*v1), np.hypot(*v2)
        if l1 < 1e-6 or l2 < 1e-6:
            out.append(cur)
            continue
        r = min(radius, l1 * 0.5, l2 * 0.5)
        a = cur + v1 / l1 * r
        b = cur + v2 / l2 * r
        for t in np.linspace(0.0, 1.0, steps + 1):
            out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * cur + t ** 2 * b)
    return np.asarray(out)

_GROUND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
_TOP_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]
_VERT_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]


def _bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def stroke_polyline(img, pts, color_rgb, thickness=2, closed=True,
                    halo=True, halo_color=(8, 10, 14)):
    """Draw an anti-aliased polyline with a dark halo underneath."""
    p = np.round(np.asarray(pts)).astype(np.int32)
    if halo:
        cv2.polylines(img, [p], closed, _bgr(halo_color), thickness + 2, cv2.LINE_AA)
    cv2.polylines(img, [p], closed, _bgr(color_rgb), thickness, cv2.LINE_AA)


def _stroke_segments(img, segs, color_rgb, thickness, halo=True,
                     halo_color=(8, 10, 14)):
    cb, hb = _bgr(color_rgb), _bgr(halo_color)
    for a, b in segs:
        a = (int(a[0]), int(a[1])); b = (int(b[0]), int(b[1]))
        if halo:
            cv2.line(img, a, b, hb, thickness + 2, cv2.LINE_AA)
    for a, b in segs:
        a = (int(a[0]), int(a[1])); b = (int(b[0]), int(b[1]))
        cv2.line(img, a, b, cb, thickness, cv2.LINE_AA)


def fill_poly_alpha(img, pts, color_rgb, alpha):
    """Translucent polygon fill blended over just its bounding ROI."""
    if alpha <= 0:
        return
    p = np.round(np.asarray(pts)).astype(np.int32)
    ch, cw = img.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(p)
    x0, y0 = max(bx, 0), max(by, 0)
    x1, y1 = min(bx + bw, cw), min(by + bh, ch)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.fillPoly(ov, [p - [x0, y0]], _bgr(color_rgb), cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)


def draw_box(img, xyxy, color_rgb, thickness=2, halo=True, radius=0):
    """Draw a 2D bounding box (used for auxiliary detections), optionally rounded."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    pts = rounded_contour(rect, radius) if radius > 0 else rect
    stroke_polyline(img, pts, color_rgb, thickness, closed=True, halo=halo)


def draw_cuboid(img, kpts8, ground_indices: Optional[Sequence[int]], color_rgb,
                dim=False, fill=True):
    """Draw a 3D cuboid wireframe from 8 keypoints, ground corners first.

    ``dim`` (for coasting tracks) lightens the stroke and drops the fill.
    """
    if kpts8 is None or len(kpts8) < 8:
        if kpts8 is not None and len(kpts8) >= 4:
            draw_ground_quad(img, kpts8[:4], color_rgb, thickness=2)
        return
    if ground_indices is not None and len(ground_indices) == 4:
        top = [i for i in range(8) if i not in ground_indices]
        k = kpts8[list(ground_indices) + top]
    else:
        k = kpts8
    g_th = 1 if dim else 2
    o_th = 1
    if fill and not dim:
        fill_poly_alpha(img, k[:4], color_rgb, 0.18)
    _stroke_segments(img, [(k[a], k[b]) for a, b in _TOP_EDGES], color_rgb, o_th, halo=not dim)
    _stroke_segments(img, [(k[a], k[b]) for a, b in _VERT_EDGES], color_rgb, o_th, halo=not dim)
    _stroke_segments(img, [(k[a], k[b]) for a, b in _GROUND_EDGES], color_rgb, g_th, halo=not dim)


def draw_ground_quad(img, quad, color_rgb, thickness=2, fill_alpha=0.0):
    if quad is None or len(quad) != 4:
        return
    if fill_alpha > 0:
        fill_poly_alpha(img, quad, color_rgb, fill_alpha)
    stroke_polyline(img, quad, color_rgb, thickness, closed=True)


def draw_dashed_polyline(img, pts, color_rgb, thickness=1, dash=7, gap=5,
                         halo=False):
    """Closed dashed polygon outline (used for estimated aux footprints)."""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    cb, hb = _bgr(color_rgb), (10, 12, 16)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        seg = b - a
        length = float(np.hypot(seg[0], seg[1]))
        if length < 1e-6:
            continue
        d = seg / length
        t = 0.0
        while t < length:
            p0 = a + d * t
            p1 = a + d * min(t + dash, length)
            q0 = (int(p0[0]), int(p0[1])); q1 = (int(p1[0]), int(p1[1]))
            if halo:
                cv2.line(img, q0, q1, hb, thickness + 2, cv2.LINE_AA)
            cv2.line(img, q0, q1, cb, thickness, cv2.LINE_AA)
            t += dash + gap


def stack_side_by_side(left, right, gap=2, sep_color=style.THEME["separator"]):
    """Place two equal-height panels side by side with a thin separator."""
    h = max(left.shape[0], right.shape[0])

    def fit(im):
        if im.shape[0] == h:
            return im
        s = h / im.shape[0]
        return cv2.resize(im, (int(round(im.shape[1] * s)), h))

    l, r = fit(left), fit(right)
    sep = np.full((h, gap, 3), _bgr(sep_color), dtype=np.uint8)
    return np.hstack([l, sep, r])
