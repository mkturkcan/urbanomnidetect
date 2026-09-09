#!/usr/bin/env python3
"""bev_realtime for the **v2 hybrid** checkpoints (e.g. ``best_x_640.pt``).

The v1 pipeline (``bev_realtime.py``) assumes the released pose checkpoints:
3 classes ``{0: car, 1: person, 2: bike}`` and ``[8, 2]`` keypoints. The v2
hybrid detect+pose model differs, and this adapter absorbs the differences
WITHOUT editing any v1 file (it subclasses / re-points at import time):

1. **Classes are COCO-80 ids** (``0 person, 1 bicycle, 2 car, 3 motorcycle,
   5 bus, 7 truck`` ...). The style tables are re-pointed so labels, legend and
   colours are right; everything that is not a road user (traffic lights,
   benches, ...) is dropped before tracking.
2. **Keypoints are ``[8, 3]``**: the 3rd channel is the model's "this instance
   has a 3D box" confidence. Instances whose corners are not trusted (low
   visibility / a corner collapsed to the origin / far outside the box) are
   demoted to *box-only* detections and shown like auxiliary detections
   (dashed box, BEV dot) instead of drawing a garbage cuboid.
3. **It is its own auxiliary detector** (v2 is a full COCO detector), so the
   separate ``yolo26x`` pass is off by default; ``--aux-model`` still merges in.

Demo-reel extras (all optional, all offline / non-causal):

* ``--layout dashboard``: a 16:9 composite -- camera panel top-left, a tall
  BEV column on the right and an info strip (wordmark, scene caption, legend
  with live counts, timings) under the camera -- instead of the raw
  side-by-side that letterboxes to half the frame. ``--layout side`` keeps the
  v1 look. Portrait sources always use side-by-side (camera | BEV).
* ``--bev-fit``: auto-zoom the BEV so the clip's footprints fit the panel.
* ``--refine``: offline track/geometry repair (see :mod:`refine_v2`) -- merge
  re-spawned ids, fill detection gaps by interpolation, decide once per track
  whether its cuboid is usable, rescale cuboids to fill their 2D box, pose
  every footprint from a whole-track rigid shape with the heading of a moving
  vehicle taken from its own trajectory, freeze vehicles that are standing
  still, and redraw each cuboid as the projection of a rigid box (constant
  footprint + one height) through the ground homography and the vertical
  vanishing point.
* ``--class-conf 0:0.4,1:0.5,3:0.5``: per-class minimum confidence (COCO ids).
* ``--caption "Aerial view"``: one phrase describing the shot. The static
  or moving camera tag is appended automatically, measured from how far the
  view actually travels between the first frame and the last.

Everything else is passed straight through to ``bev_realtime``.

Run from anywhere (the module puts its own directory on ``sys.path``), but do
NOT run python from a directory that contains an ``ultralytics/`` source clone:
it shadows the installed package as a namespace package and the checkpoint
fails to load.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter, deque
from typing import Dict, List, Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import v2.hybrid_v2  # noqa: E402,F401  registers HybridPoseModel26 for unpickling
import bev_realtime as rt  # noqa: E402
import uod.bev as uod_bev  # noqa: E402
from uod import style  # noqa: E402
from uod.bev import BEVViewport  # noqa: E402
from uod.keypoints import Detection  # noqa: E402
from uod.textdraw import UILayer  # noqa: E402
from uod.tracking import _iou_matrix  # noqa: E402
import refine_v2  # noqa: E402
from uod.viz import (draw_cuboid, draw_dashed_polyline, draw_ground_quad)  # noqa: E402
from homography_rt import apply_homography  # noqa: E402

# COCO ids the v2 model may report as road users -> pipeline categories.
POSE_CATEGORY_V2: Dict[int, str] = {
    0: "person", 1: "bike", 2: "car", 3: "bike", 5: "bus", 7: "truck",
}
# Re-point the shared style tables at the COCO ids (process-local; no file edit).
style.POSE_CATEGORY.clear()
style.POSE_CATEGORY.update(POSE_CATEGORY_V2)

# --------------------------------------------------------------------------- #
# Visual identity. Black ground, one weight of white, and colour used only where
# it carries meaning: the object's class. Everything structural is a hairline in
# a neutral grey, so nothing competes with the footage.
style.PALETTE.update({
    "car":    (96, 150, 246),
    "person": (240, 178, 74),
    "bike":   (167, 139, 250),
    "bus":    (86, 196, 140),
    "truck":  (232, 110, 98),
    "other":  (150, 152, 160),
})
style.THEME.update({
    "bev_bg":      (0, 0, 0),
    "bev_bg_edge": (0, 0, 0),
    "ring":        (26, 26, 29),
    "ring_hi":     (40, 40, 45),
    "spoke":       (0, 0, 0),
    "grid_text":   (86, 86, 92),
    "ego":         (232, 232, 238),
    "panel":       (0, 0, 0),
    "panel_text":  (246, 246, 248),
    "muted_text":  (139, 139, 147),
    "faint_text":  (92, 92, 99),
    "accent":      (30, 30, 34),
    "separator":   (24, 24, 28),
    "aux":         (150, 152, 160),
})
style.CATEGORY_LABEL.update({
    "car": "car", "person": "person", "bike": "bike",
    "bus": "bus", "truck": "truck", "other": "object",
})
_PLURAL = {"car": "cars", "person": "people", "bike": "bikes",
           "bus": "buses", "truck": "trucks", "other": "objects"}

_base_instance_color = style.instance_color


def _instance_color(cls: int, instance_id: int, source: str = "pose"):
    """Per-track shade, but a narrow one: class identity should read first."""
    base = style.class_color(cls, source)
    if instance_id is None or instance_id < 0:
        return base
    import colorsys
    h, l, sat = colorsys.rgb_to_hls(*[c / 255.0 for c in base])
    j = (instance_id * 0.6180339887) % 1.0
    l = min(0.80, max(0.42, l + (j - 0.5) * 0.10))
    r, g, b = colorsys.hls_to_rgb(h, l, sat)
    return (int(r * 255), int(g * 255), int(b * 255))


style.instance_color = _instance_color


def _text_size(size: int, s: str, font: str = "regular"):
    """Measured width/height of a string in the UI font."""
    try:
        from PIL import Image, ImageDraw
        f = UILayer._font(font, size)
        d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        l, t, r, b = d.textbbox((0, 0), s, font=f)
        return r - l, b - t
    except Exception:
        return int(len(s) * size * 0.55), size


def _pill(ui, x, y, text, *, size=20, fg=(246, 246, 248), dot=None,
          bg=(0, 0, 0), alpha=0.66, pad=(12, 7), font="regular", anchor="lm",
          opacity=1.0):
    """A rounded label. A class colour appears as a dot, never as an edge."""
    tw, th = _text_size(size, text, font)
    dd = int(size * 0.40) if dot else 0
    gap = int(size * 0.44) if dot else 0
    w = pad[0] * 2 + dd + gap + tw
    h = pad[1] * 2 + th
    ax = x if anchor[0] == "l" else (x - w / 2 if anchor[0] == "m" else x - w)
    ay = y - h / 2 if anchor[1] == "m" else (y if anchor[1] == "a" else y - h)
    ui.panel((ax, ay), (ax + w, ay + h), bg, alpha=alpha * opacity,
             radius=int(h / 2))
    tx = ax + pad[0]
    if dot:
        cy = ay + h / 2
        ui.panel((tx, cy - dd / 2), (tx + dd, cy + dd / 2), dot, alpha=opacity,
                 radius=int(dd / 2))
        tx += dd + gap
    ui.text(tx, ay + h / 2 + 1, text, size=size, color=fg, font=font,
            anchor="lm", shadow=False, alpha=opacity)
    return w


def _count_label(cat: str, n: int) -> str:
    """"43 cars", never "Car 43" -- a count must not read like an identifier."""
    return f"{n} " + (_PLURAL[cat] if n != 1 else style.CATEGORY_LABEL[cat])


FINAL_W, FINAL_H = 1920, 1080          # dashboard layout output size


class _FFSink:
    """Write frames through x264 instead of OpenCV's MPEG-4 writer.

    ``cv2.VideoWriter`` with the ``mp4v`` fourcc is MPEG-4 Part 2 at a fixed,
    modest quality. The first things it loses are exactly what this render is
    made of: one-pixel cuboid edges, hairline rings and small type. The reel is
    then assembled from those files, so the loss compounds through the cut.
    Piping raw frames to x264 at a near-transparent quality costs a little disk
    and keeps every edge intact.
    """

    def __init__(self, args, fps_in):
        self.args = args
        self.fps = float(fps_in or 30.0)
        self.path = args.output
        self.proc = None
        self.frames = 0

    def emit(self, out) -> bool:
        if self.path:
            if self.proc is None:
                h, w = out.shape[:2]
                self.proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                     "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps:.6f}",
                     "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "12", "-pix_fmt", "yuv420p",
                     "-color_primaries", "bt709", "-color_trc", "bt709",
                     "-colorspace", "bt709", self.path],
                    stdin=subprocess.PIPE)
            self.proc.stdin.write(np.ascontiguousarray(out).tobytes())
            self.frames += 1
        if self.args.display:
            cv2.imshow("UrbanOmniDetect BEV", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


rt._Sink = _FFSink


# --------------------------------------------------------------------------- #
class V2Viewport(BEVViewport):
    """The bird's-eye panel, restyled.

    A radar is a diagram, not a picture: it should read instantly and add
    nothing the eye has to filter out. So the ground is black, the range rings
    are hairlines, the spokes are gone, and the only colour is the class of an
    object. Footprints are a flat fill with a crisp edge rather than a glow,
    and they carry no identifier -- the colour and the trail already say which
    object is which.

    ``object_px`` (the zoom) is a class attribute so ``--bev-fit`` can change it
    everywhere consistently, including the frozen world lock.
    """
    DEFAULT_OBJECT_PX = 40.0

    def __init__(self, canvas_size=(480, 854), margin=60, forward_frac=0.5,
                 range_fraction=0.22, object_px=None):
        if object_px is None:
            object_px = V2Viewport.DEFAULT_OBJECT_PX
        super().__init__(canvas_size=canvas_size, margin=margin,
                         forward_frac=forward_frac,
                         range_fraction=range_fraction, object_px=object_px)

    def _build_radar_bg(self) -> np.ndarray:
        from uod import style as st
        H, W = self.H, self.W
        cx, cy = self.camera_canvas
        r_max = H - 2 * self.margin
        canvas = np.zeros((H, W, 3), np.uint8)

        # Four hairline range rings. They are a scale reference, not data, so
        # they are unlabelled: a number in unnamed units explains nothing.
        self._rings = []
        step = max(1, int(round((r_max / max(self.object_px, 1.0)) / 4.0)))
        k = step
        while k * self.object_px <= r_max + 1:
            r = int(k * self.object_px)
            col = st.THEME["ring_hi"] if (k + step) * self.object_px > r_max + 1 \
                else st.THEME["ring"]
            cv2.circle(canvas, (cx, cy), r, (col[2], col[1], col[0]), 1, cv2.LINE_AA)
            self._rings.append((r, k))
            k += step

        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(max(r_max, 1))
        vig = np.clip(1.0 - 0.55 * np.clip(d - 0.55, 0, None) ** 1.6, 0.0, 1.0)
        return (canvas.astype(np.float32) * vig[:, :, None]).astype(np.uint8)

    def draw_fov(self, canvas: np.ndarray, H: np.ndarray,
                 img_wh: Tuple[int, int]) -> None:
        """The ground the camera can actually see: a barely-there lift."""
        poly = self.fov_polygon(H, img_wh)
        if poly is None or len(poly) < 3:
            return
        pts = poly.astype(np.int32)
        ov = canvas.copy()
        cv2.fillPoly(ov, [pts], (16, 16, 18), cv2.LINE_AA)
        cv2.addWeighted(ov, 0.9, canvas, 0.1, 0, canvas)
        cv2.polylines(canvas, [pts], True, (54, 54, 60), 1, cv2.LINE_AA)

    def draw_footprint_poly(self, canvas: np.ndarray, poly: np.ndarray, color,
                            thickness: int = 2, fill_alpha: float = 0.25,
                            glow: bool = True) -> None:
        from uod.viz import rounded_contour
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        e0 = float(np.linalg.norm(poly[1] - poly[0]))
        e1 = float(np.linalg.norm(poly[2] - poly[1]))
        rc = np.round(rounded_contour(poly, min(e0, e1) * 0.18)).astype(np.int32)
        self._fill_alpha(canvas, rc, bgr, max(fill_alpha, 0.18))
        cv2.polylines(canvas, [rc], True, bgr, max(1, thickness - 1), cv2.LINE_AA)

    def draw_camera(self, canvas: np.ndarray) -> None:
        """Where the camera is: a small open chevron, no halo."""
        from uod import style as st
        cx, cy = self.camera_canvas
        ego = st.THEME["ego"]
        c = (ego[2], ego[1], ego[0])
        pts = np.array([(cx, cy - 9), (cx + 7, cy + 6), (cx, cy + 2),
                        (cx - 7, cy + 6)], np.int32)
        cv2.fillConvexPoly(canvas, pts, c, cv2.LINE_AA)


uod_bev.BEVViewport = V2Viewport
rt.BEVViewport = V2Viewport


def _parse_class_conf(spec: Optional[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok:
            k, v = tok.split(":")
            out[int(k)] = float(v)
    return out


# --------------------------------------------------------------------------- #
VEHICLES_V2 = (2, 5, 7)   # car, bus, truck


def _ground_overlap(qa, qb):
    """Intersection of two ground quads as a fraction of the smaller one (convex hulls)."""
    ha = cv2.convexHull(np.asarray(qa, np.float32).reshape(-1, 1, 2))
    hb = cv2.convexHull(np.asarray(qb, np.float32).reshape(-1, 1, 2))
    small = min(cv2.contourArea(ha), cv2.contourArea(hb))
    if small <= 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    return float(inter) / float(small)


def _suppress_nested(dets, thresh, ground_idx):
    """Drop a vehicle whose 2D box lies >= ``thresh`` inside a LARGER vehicle's box and whose
    cuboid ground quad overlaps that vehicle's ground quad.

    The end2end head runs no NMS and reports a box truck twice: the whole vehicle as
    "truck" and its cab as a "car" nested inside it. The tracker then alternates between
    the two boxes and the offline refinement cuts and drops the fragments, so the truck
    disappears from the output although the detector saw it in every frame. The ground
    test keeps genuine neighbours: a car in front of a bus is nested in the image but
    stands on its own patch of road, so its footprint does not overlap the bus's.
    """
    veh = [d for d in dets if d.cls in VEHICLES_V2]
    gi = list(ground_idx)
    drop = set()
    for a in veh:
        ax1, ay1, ax2, ay2 = a.xyxy
        area_a = max(ax2 - ax1, 1.0) * max(ay2 - ay1, 1.0)
        for b in veh:
            if b is a or id(b) in drop:
                continue
            bx1, by1, bx2, by2 = b.xyxy
            if max(bx2 - bx1, 1.0) * max(by2 - by1, 1.0) <= area_a:
                continue
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            if iw * ih / area_a < thresh:
                continue
            if a.kpts.shape[0] < 8 or b.kpts.shape[0] < 8:
                continue                       # no cuboids to compare: keep both
            if _ground_overlap(a.kpts[gi], b.kpts[gi]) < 0.3:
                continue                       # nested in the image, elsewhere on the ground
            drop.add(id(a))
            break
    return [d for d in dets if id(d) not in drop]


class V2Pipeline(rt.RealtimePipeline):
    """``RealtimePipeline`` driven by one v2 hybrid forward pass per frame."""

    # Filled by ``main`` before ``rt.main`` constructs the pipeline.
    CONFIG: dict = dict(kp_vis=0.5, class_conf={}, bev_canvas=None, suppress_nested=0.0,
                        layout="side", caption="", show_coast_mark=False,
                        label_min_h=0.0, src_fps=30.0, title=(0.0, 0.0, 0.0, 0.0),
                        labels="", labels_meta=None, labels_first=0)

    def __init__(self, pose_model, aux_model=None, **kw):
        cfg = dict(V2Pipeline.CONFIG)
        if cfg.get("bev_canvas") is not None and kw.get("bev_canvas") is None:
            kw["bev_canvas"] = tuple(cfg["bev_canvas"])
        super().__init__(pose_model, aux_model, **kw)
        self.kp_vis = float(cfg.get("kp_vis", 0.5))
        self.suppress_nested = float(cfg.get("suppress_nested", 0.0))
        self.class_conf: Dict[int, float] = dict(cfg.get("class_conf") or {})
        self.layout = cfg.get("layout", "side")
        self.caption = cfg.get("caption", "") or ""
        self.show_coast_mark = bool(cfg.get("show_coast_mark", False))
        self.label_min_h = float(cfg.get("label_min_h", 0.0))
        self.camera_tag = ""                  # "static camera" / "moving camera"
        self.src_fps = float(cfg.get("src_fps", 30.0) or 30.0)
        self.title = tuple(cfg.get("title", (0.0, 0.0, 0.0, 0.0)))
        self.labels_path = cfg.get("labels", "") or ""
        self.labels_meta = dict(cfg.get("labels_meta") or {})
        self.labels_first = int(cfg.get("labels_first", 0) or 0)
        self._last_pose_dets: List[Detection] = []
        # v2 supplies its own auxiliary (box-only) channel, so the aux tracker
        # must exist even when no separate aux model is loaded.
        if self.aux_tracker is None:
            from uod.tracking import MultiObjectTracker
            self.aux_tracker = MultiObjectTracker(
                max_age=10, min_hits=2, conf_thresh=self.aux_conf,
                cmc=self.tracker.cmc)

    # ------------------------------------------------------------------ #
    def _detect_pose(self, frame) -> List[Detection]:
        """One forward pass -> every road-user detection, keypoints flagged.

        EVERY road user is tracked, whether or not its predicted cuboid is
        trustworthy: association uses boxes only, so a bad cuboid must not cost
        the object its identity. Each detection carries ``kp_ok`` (this frame's
        keypoints passed the trust test) and its raw corners; the *whole-track*
        decision is taken later, offline, in :mod:`refine_v2`. Only trusted
        corners produce a ground quad, so the homography solver never sees a
        degenerate footprint.
        """
        res = self.pose_model.predict(frame, imgsz=self.kp_imgsz,
                                      conf=self.kp_conf, device=self.device,
                                      verbose=False)[0]
        self._last_pose_dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = res.boxes.conf.cpu().numpy().astype(np.float64)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        kd = None
        if res.keypoints is not None and res.keypoints.data is not None:
            kd = res.keypoints.data.cpu().numpy().astype(np.float64)
            if kd.ndim != 3 or kd.shape[0] != len(xyxy):
                kd = None

        pose: List[Detection] = []
        trusted_kp = []
        for i in range(len(xyxy)):
            c = int(cls[i])
            if c not in POSE_CATEGORY_V2:
                continue                                   # not a road user
            if conf[i] < self.class_conf.get(c, 0.0):
                continue
            kp = np.zeros((0, 2))
            ok = False
            if kd is not None and kd.shape[1] >= 8:
                xy = kd[i, :, :2]
                vis = kd[i, :, 2] if kd.shape[2] >= 3 else np.ones(kd.shape[1])
                x1, y1, x2, y2 = xyxy[i]
                bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                inside = ((xy[:, 0] >= x1 - 0.75 * bw) & (xy[:, 0] <= x2 + 0.75 * bw)
                          & (xy[:, 1] >= y1 - 0.75 * bh) & (xy[:, 1] <= y2 + 0.75 * bh))
                nonzero = (np.abs(xy).sum(axis=1) > 1e-6)
                ok = bool(np.isfinite(xy).all() and (vis.min() >= self.kp_vis)
                          and inside.all() and nonzero.all())
                kp = xy.copy()
            det = Detection(cls=c, conf=float(conf[i]), xyxy=xyxy[i], kpts=kp)
            det.kp_ok = ok
            pose.append(det)
            if ok:
                trusted_kp.append(kp)

        if self.suppress_nested > 0:
            gi0 = self.resolver.indices(8) or [0, 1, 2, 3]
            pose = _suppress_nested(pose, self.suppress_nested, gi0)
            trusted_kp = [d.kpts for d in pose if d.kp_ok]
        if trusted_kp:
            self.resolver.observe(np.stack(trusted_kp))
        gi = self.resolver.indices(8)
        if gi:
            for d in pose:
                if d.kp_ok and d.kpts.shape[0] >= max(gi) + 1:
                    d.ground = d.kpts[list(gi)].copy()
        self._last_pose_dets = pose
        return pose

    def _detect_aux_raw(self, frame) -> List[Detection]:
        """Only an EXTERNAL auxiliary detector feeds this channel now.

        The v2 model's own low-confidence-cuboid instances are no longer
        demoted per frame (that is what made the 3D boxes flicker); they stay
        tracked and are demoted per track offline.
        """
        return super()._detect_aux_raw(frame) if self.aux_model is not None else []

    # ------------------------------------------------------------------ #
    def step(self, frame):
        """Core step + per-view annotation of this frame's raw observation.

        Records, per track view, whether it was backed by a real detection on
        this frame and whether that detection's keypoints were trusted, and
        replaces the tracker's causally-smoothed corners with the raw ones on
        trusted frames (the offline zero-phase filter smooths them afterwards,
        which is strictly better than a causal pre-filter here).
        """
        state = super().step(frame)
        raw = getattr(self, "_last_pose_dets", []) or []
        gi = self.resolver.indices(8)
        rb = np.array([d.xyxy for d in raw]) if raw else None
        for v in state.tracks:
            v.obs = False
            v.kp_ok = False
            v.interp = False
            if rb is None or v.time_since_update != 0:
                continue
            iou = _iou_matrix(np.asarray(v.box_xyxy, dtype=np.float64)[None], rb)[0]
            j = int(np.argmax(iou))
            if float(iou[j]) < 0.5:
                continue
            d = raw[j]
            v.obs = True
            v.kp_ok = bool(getattr(d, "kp_ok", False))
            if v.kp_ok and d.kpts.shape[0] >= 8:
                v.kpts = d.kpts.copy()
                if gi and v.kpts.shape[0] >= max(gi) + 1:
                    v.ground = v.kpts[list(gi)].copy()
        return state

    # ------------------------------------------------------------------ #
    # Rendering: dashboard layout (landscape sources) or the v1 side-by-side.
    def _render(self, state):
        if self.layout != "dashboard":
            return self._render_side(state)
        return self._render_dashboard(state)

    def _draw_aux_bev(self, bev, aux_dets, aux_centers, H, track_polys):
        """A detection with no usable 3D box: one small neutral dot.

        It is a lesser claim than a footprint, so it should look like one --
        no colour, no ring, no halo.
        """
        if not len(aux_centers):
            return
        cc = np.array(self.viewport.camera_canvas, dtype=np.float64)
        r_max = (self.viewport.H - 2 * self.viewport.margin) * 1.02
        canv = self.viewport.to_canvas(aux_centers, H)
        col = style.THEME["aux"]
        for i in range(len(canv)):
            c = canv[i]
            if not (np.all(np.isfinite(c)) and np.hypot(*(c - cc)) <= r_max):
                continue
            cv2.circle(bev, (int(c[0]), int(c[1])), 3,
                       (col[2], col[1], col[0]), -1, cv2.LINE_AA)

    def _draw_panels(self, state):
        """Draw the camera and BEV panels; returns ``(src, bev, present,
        src_labels, bev_labels)`` exactly like the v1 renderer does."""
        frame, tracks = state.frame, state.tracks
        aux_dets, aux_centers = state.aux_dets, state.aux_centers
        H, gi = state.H, state.gi
        src = frame.copy()
        src_labels = []
        if self.draw_aux_box:
            for d in aux_dets:
                acol = style.THEME["aux"]      # no 3D box is not a class claim
                x1, y1, x2, y2 = [float(v) for v in d.xyxy]
                draw_dashed_polyline(src, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                                     acol, thickness=2, dash=9, gap=6, halo=True)
        for t in tracks:
            col = style.instance_color(t.cls, t.id, source="pose")
            coasting = t.time_since_update > 0
            if self.show_3d and t.kpts.shape[0] >= 8:
                draw_cuboid(src, t.kpts, gi, col, dim=coasting)
            elif t.ground.shape == (4, 2):
                draw_ground_quad(src, t.ground, col, thickness=2)
            x1, y1 = t.box_xyxy[0], t.box_xyxy[1]
            bh = float(t.box_xyxy[3] - t.box_xyxy[1])
            src_labels.append((x1, y1, t.cls, t.id, coasting, bh))

        bev = self.viewport.blank_canvas(radar=True)
        if self.use_homography and self.show_fov:
            self.viewport.draw_fov(bev, H, state.img_wh)
        bev_labels = []
        present = Counter()
        if self.use_homography:
            fps = []
            for t in tracks:
                if t.bev_quad is None:
                    continue
                poly = self.viewport.rect_to_canvas(t.bev_quad)
                if len(poly) != 4 or not np.all(np.isfinite(poly)):
                    continue
                fps.append((t, poly))
                c = poly.mean(axis=0)
                bev_labels.append((c[0], c[1], t.id))
            if self.trails:
                cur = set()
                for (cx, cy, tid) in bev_labels:
                    cur.add(tid)
                    self._trails.setdefault(
                        tid, deque(maxlen=self.trail_len)).append((cx, cy))
                for tid in [k for k in self._trails if k not in cur]:
                    del self._trails[tid]
                self._draw_trails(bev, {t.id: t.cls for t, _ in fps})
            track_polys = []
            for (t, poly) in fps:
                col = style.instance_color(t.cls, t.id, source="pose")
                alpha = 0.30 if t.time_since_update == 0 else 0.12
                self.viewport.draw_footprint_poly(
                    bev, poly, col, fill_alpha=alpha,
                    thickness=2 if t.time_since_update == 0 else 1)
                track_polys.append(poly)
            self._draw_aux_bev(bev, aux_dets, aux_centers, H, track_polys)
        for t in tracks:
            present[style.class_category(t.cls, "pose")] += 1
        self.viewport.draw_camera(bev)
        return src, bev, present, src_labels, bev_labels

    def _render_side(self, state):
        from uod.viz import stack_side_by_side
        src, bev, present, src_labels, bev_labels = self._draw_panels(state)
        composite = stack_side_by_side(src, bev, gap=2)
        bev_x = src.shape[1] + 2
        ui = UILayer()
        self._ui_side(ui, composite, state, present, src_labels, bev_labels, bev_x)
        return ui.render(composite)

    def _render_dashboard(self, state):
        src, bev, present, src_labels, bev_labels = self._draw_panels(state)
        T = style.THEME
        Wc, Hc = FINAL_W, FINAL_H
        sh, sw = src.shape[:2]
        bh, bw = bev.shape[:2]
        canvas = np.empty((Hc, Wc, 3), np.uint8)
        canvas[:] = (T["panel"][2], T["panel"][1], T["panel"][0])
        # Camera panel top-left (already sized by the caller), BEV column right.
        canvas[:sh, :sw] = src
        bev_x = Wc - bw
        canvas[:bh, bev_x:bev_x + bw] = bev[:Hc]
        sep = T["separator"]
        cv2.line(canvas, (bev_x - 1, 0), (bev_x - 1, Hc), (sep[2], sep[1], sep[0]), 2)
        cv2.line(canvas, (0, sh), (sw, sh), (sep[2], sep[1], sep[0]), 2)
        ui = UILayer()
        self._ui_dashboard(ui, canvas, state, present, src_labels, bev_labels,
                           bev_x, (sw, sh))
        return ui.render(canvas)

    # ------------------------------------------------------------------ #
    def _caption_text(self) -> str:
        """Scene caption. Two clauses at most, joined by a comma."""
        parts = [p.split(",")[0].strip() for p in (self.caption, self.camera_tag) if p]
        return ", ".join(parts[:2])

    def _track_chips(self, ui, src_labels, y_min, max_labels: int = 6,
                     opacity: float = 1.0):
        """Label only what a viewer would not already assume.

        In an urban scene almost every box is a car, so labelling all of them
        writes the same word forty times across the footage and buries it. The
        classes worth pointing out are the others; the legend carries the full
        counts. Largest first, and never more than a handful.
        """
        rest = [r for r in src_labels
                if style.class_category(r[2], "pose") != "car" and r[5] >= self.label_min_h]
        rest.sort(key=lambda r: -r[5])
        for (x, y, cls, tid, coasting, bh) in rest[:max_labels]:
            col = style.instance_color(cls, tid, source="pose")
            name = style.CATEGORY_LABEL[style.class_category(cls, "pose")]
            _pill(ui, float(x) + 1, max(float(y), y_min) - 10, name, size=15,
                  dot=col, fg=(238, 238, 242), bg=(0, 0, 0),
                  alpha=0.55 if coasting else 0.74, pad=(9, 5), anchor="ld",
                  opacity=opacity)

    def _legend(self, ui, x, y, present, n_aux, size=20, align="l",
                opacity=1.0):
        """Live counts, at most three, worded so they cannot be read as ids."""
        cats = [c for c in style.PALETTE if present.get(c)]
        cats.sort(key=lambda c: -present[c])
        items = [(style.PALETTE[c], _count_label(c, present[c])) for c in cats[:3]]
        if n_aux and len(items) < 3:
            items.append((style.THEME["aux"], f"{n_aux} in 2D"))
        if align == "l":
            for dot, txt in items:
                x += _pill(ui, x, y, txt, size=size, dot=dot,
                           fg=(232, 232, 238), opacity=opacity) + 10
        else:
            for dot, txt in reversed(items):
                w, _ = _text_size(size, txt)
                x -= _pill(ui, x, y, txt, size=size, dot=dot,
                           fg=(232, 232, 238), anchor="rm", opacity=opacity) + 10

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _title_alpha(self, state) -> float:
        """How present the opening titles are on this frame.

        Timed in SOURCE seconds, because that is what the renderer sees; the cut
        trims and speeds up the clip afterwards, so the reel builder converts.
        """
        start, fin, hold, fout = self.title
        if fin <= 0 and hold <= 0:
            return 0.0
        t = (state.frame_idx - 1) / max(self.src_fps, 1e-6)
        if t <= start:
            return 0.0
        u = t - start
        if u < fin:
            return self._smoothstep(u / fin)
        if u < fin + hold:
            return 1.0
        if u < fin + hold + fout:
            return 1.0 - self._smoothstep((u - fin - hold) / fout)
        return 0.0

    def _hud_alpha(self, state) -> float:
        """The running readout: absent under the titles, arriving as they go."""
        start, fin, hold, fout = self.title
        if fin <= 0 and hold <= 0:
            return 1.0
        t = (state.frame_idx - 1) / max(self.src_fps, 1e-6)
        # The readout waits for the titles to LEAVE before arriving. Two blocks
        # of type dissolving through each other in the same place reads as a
        # double exposure; a beat of clean footage between them reads as intent.
        gone = start + fin + hold + max(fout, 1e-6)
        rise = max(0.6 * fout, 0.3)
        return self._smoothstep((t - gone) / rise)

    def _draw_titles(self, ui, x0, y0, h, sw, a: float) -> None:
        """The opening titles, in the strip, rising as they arrive."""
        T = style.THEME
        dy = (1.0 - a) * 0.055 * h
        ui.text(x0, y0 + 0.115 * h + dy, "CVPR 2026 DRIVEX WORKSHOP",
                size=int(0.052 * h), color=T["muted_text"], anchor="lm",
                shadow=False, alpha=a * 0.95)
        ui.text(x0, y0 + 0.350 * h + dy, "UrbanOmniDetect", font="bold",
                size=int(0.205 * h), color=T["panel_text"], anchor="lm",
                shadow=False, alpha=a)
        ui.text(x0, y0 + 0.585 * h + dy,
                "Calibration-free, view-agnostic monocular 3D detection",
                size=int(0.076 * h), color=(198, 198, 206), anchor="lm",
                shadow=False, alpha=a)
        ui.text(x0, y0 + 0.765 * h + dy,
                "Mehmet Kerem Turkcan, Devika Gumaste, Zoran Kostic",
                size=int(0.062 * h), color=T["muted_text"], anchor="lm",
                shadow=False, alpha=a)
        ui.text(sw - x0, y0 + 0.765 * h + dy, "Columbia University",
                size=int(0.062 * h), color=T["muted_text"], anchor="rm",
                shadow=False, alpha=a)

    def _fps_text(self, state) -> str:
        f = self.fps_core if self.fps_core > 0 else 0.0
        return f"{f:.0f} fps" if f > 0 else ""

    def _ui_side(self, ui, composite, state, present, src_labels, bev_labels, bev_x):
        """Chrome for the portrait layout: one top rule, nothing else."""
        T = style.THEME
        Wc, Hc = composite.shape[1], composite.shape[0]
        head = 46
        ui.panel((0, 0), (Wc, head), (0, 0, 0), alpha=0.88, radius=0)
        ui.line((0, head), (Wc, head), T["separator"], width=1, alpha=1.0)
        ui.text(26, head / 2, "UrbanOmniDetect", size=21, font="bold",
                color=T["panel_text"], anchor="lm", shadow=False)
        ui.text(Wc - 26, head / 2, self._fps_text(state), size=16, font="mono",
                color=T["muted_text"], anchor="rm", shadow=False)
        if self._caption_text():
            _pill(ui, 22, Hc - 26, self._caption_text(), size=16,
                  fg=(228, 228, 234), pad=(11, 6))
        self._legend(ui, Wc - 26, head + 34, present, len(state.aux_dets),
                     size=17, align="r")
        ui.text(bev_x + 22, Hc - 20, "Bird's-eye view", size=14,
                color=T["faint_text"], anchor="ld", shadow=False)
        self._track_chips(ui, src_labels, head + 18)

    def _ui_dashboard(self, ui, canvas, state, present, src_labels, bev_labels,
                      bev_x, src_wh):
        """Chrome for the landscape layout.

        One column of information under the footage, in three weights: what this
        is, what it is looking at, and what it found. Type is sized from the
        strip's own height, so the layout survives a different split between
        footage and chrome. During the opening the same strip carries the
        titles, and the two cross-fade rather than cutting.
        """
        T = style.THEME
        Wc, Hc = canvas.shape[1], canvas.shape[0]
        sw, sh = src_wh
        x0 = int(0.034 * sw)
        y0 = sh
        h = Hc - y0
        ui.line((0, y0), (sw, y0), T["separator"], width=1, alpha=1.0)

        a = self._title_alpha(state)
        if a > 0.002:
            self._draw_titles(ui, x0, y0, h, sw, a)
        # The strip opens AS the titles and becomes the readout only once they
        # have left. Fading the two through each other in the same place reads
        # as a double exposure, so the readout waits.
        o = self._hud_alpha(state)
        if o <= 0.002:
            return
        dy = a * 0.05 * h

        ui.text(x0, y0 + 0.215 * h + dy, "UrbanOmniDetect", size=int(0.148 * h),
                font="bold", color=T["panel_text"], anchor="lm", shadow=False,
                alpha=o)
        ui.text(sw - x0, y0 + 0.215 * h + dy, self._fps_text(state),
                size=int(0.066 * h), font="mono", color=T["muted_text"],
                anchor="rm", shadow=False, alpha=o)
        ui.text(x0, y0 + 0.44 * h + dy,
                "Calibration-free monocular 3D detection with a metric "
                "bird's-eye view", size=int(0.073 * h), color=T["muted_text"],
                anchor="lm", shadow=False, alpha=o)
        self._legend(ui, x0, y0 + 0.67 * h + dy, present, len(state.aux_dets),
                     size=int(0.069 * h), opacity=o)
        ui.text(x0, y0 + 0.885 * h + dy, "CVPR 2026 DriveX Workshop",
                size=int(0.059 * h), color=T["faint_text"], anchor="lm",
                shadow=False, alpha=o)
        ui.text(sw - x0, y0 + 0.885 * h + dy, "Columbia University",
                size=int(0.059 * h), color=T["faint_text"], anchor="rm",
                shadow=False, alpha=o)

        if self._caption_text():
            _pill(ui, 22, sh - 22, self._caption_text(), size=16,
                  fg=(228, 228, 234), pad=(11, 6), opacity=o)
        ui.text(bev_x + 22, Hc - 20, "Bird's-eye view", size=14,
                color=T["faint_text"], anchor="ld", shadow=False, alpha=o)
        self._track_chips(ui, src_labels, 18, opacity=o)


# --------------------------------------------------------------------------- #
# Offline post-processing: track clean-up + BEV auto-fit, run between the core
# pass and the v1 stabiliser (we wrap ``rt._stabilize_offline``).
POST: dict = dict(refine=False, min_track=12, coast_tail=8, merge_gap=30,
                  max_gap=45, min_fill=0.5, min_ok_frac=0.5, fill_box=True,
                  vel_heading=True, hold_still=True, rigid_cuboids=True,
                  world_lock=True, shape_smooth=True, shape_win=31, dedupe=True,
                  speed_ratio=3.0,
                  max_reproj=0.10, bev_fit=False, fit_max_px=40.0,
                  fit_min_px=22.0, fit_pct=90.0)


def _bev_fit(pipe, states, max_px, min_px, pct=90.0):
    """Zoom the BEV so the clip's footprints fit the panel (offline only).

    Measures every footprint centre in canvas units at the reference zoom, then
    scales ``object_px`` (and the frozen world-lock, if any) so the given
    percentile of the horizontal / forward extents fits inside the margins.
    """
    cw = pipe._bev_canvas or states[0].img_wh
    ref = V2Viewport.DEFAULT_OBJECT_PX
    vp = V2Viewport(canvas_size=cw, object_px=ref)
    cc = np.array(vp.camera_canvas, dtype=np.float64)
    dx, dy = [], []
    for s in states[::2]:
        if pipe._frozen_sim is not None:
            sim = pipe._frozen_sim
        else:
            vp.update(np.asarray(s.H, dtype=np.float64), s.img_wh, unit=s.bev_unit)
            sim = vp._sim
        if sim is None:
            continue
        for v in s.tracks:
            q = v.bev_quad
            if q is None:
                continue
            p = sim.apply(np.asarray(q, dtype=np.float64)).mean(axis=0)
            if np.all(np.isfinite(p)):
                dx.append(abs(p[0] - cc[0]))
                dy.append(cc[1] - p[1])
    if len(dx) < 10:
        print("[bev-fit] too few footprints; zoom unchanged")
        return
    dx, dy = np.array(dx), np.array(dy)
    ex = float(np.percentile(dx, pct)) + 0.8 * ref     # + half a car
    ey = float(np.percentile(dy[dy > 0], pct)) + 0.8 * ref if (dy > 0).any() else 1.0
    avail_x = vp.W / 2.0 - vp.margin - 6
    avail_y = vp.H - 2 * vp.margin - 6
    s = min(avail_x / max(ex, 1e-6), avail_y / max(ey, 1e-6))
    new_px = float(np.clip(ref * s, min_px, max_px))
    s = new_px / ref
    V2Viewport.DEFAULT_OBJECT_PX = new_px
    if pipe._frozen_sim is not None:
        pipe._frozen_sim.scale *= s
    pipe.viewport = None                       # rebuilt with the new zoom
    print(f"[bev-fit] extents x {ex:.0f}px  fwd {ey:.0f}px at {ref:.0f}px/object"
          f" -> object_px {new_px:.1f}")


def _dump_states(states, fixed, path):
    """Pickle a frame-free geometry snapshot so the offline maths can be tuned
    without re-running the detector (development helper only)."""
    import pickle
    snap = dict(fixed=bool(fixed), frames=[])
    for s in states:
        snap["frames"].append(dict(
            H=np.asarray(s.H, dtype=np.float64), bev_unit=float(s.bev_unit or 0.0),
            img_wh=tuple(s.img_wh),
            aux=[dict(cls=int(getattr(d, "cls", -1)),
                      box=np.asarray(d.xyxy, dtype=np.float64).copy()) for d in s.aux_dets],
            tracks=[dict(id=int(v.id), cls=int(v.cls),
                         quad=(np.asarray(v.bev_quad, dtype=np.float64).copy()
                               if getattr(v, "bev_quad", None) is not None else None),
                         kpts=np.asarray(v.kpts, dtype=np.float64).copy(),
                         ground=np.asarray(v.ground, dtype=np.float64).copy(),
                         box=np.asarray(v.box_xyxy, dtype=np.float64).copy(),
                         tsu=int(v.time_since_update),
                         obs=bool(getattr(v, "obs", v.time_since_update == 0)),
                         kp_ok=bool(getattr(v, "kp_ok", False)))
                    for v in s.tracks]))
    with open(path, "wb") as fh:
        pickle.dump(snap, fh)
    print(f"[dump] {len(snap['frames'])} frames -> {path}")


def _write_labels(pipe, states, fixed, path, first=0):
    """Write what the renderer drew as data: one file of pseudo-labels.

    These are model output, refined offline, not human annotation. What makes
    them worth keeping is that the refinement is whole-track: identity, class,
    footprint and height are each decided once over the clip and posed per
    frame, so the series is consistent in a way per-frame inference is not.

    Coordinates are pixels of the frames that were PROCESSED. When the source
    was scaled before rendering, ``source.scale`` maps them back: multiply by
    it to land in the original file's pixels.

    ``first`` drops the head of the clip from the file while keeping it in the
    computation. The refinement is whole-track and non-causal, so those frames
    still inform every estimate; they are simply not written out, which is what
    you want when the exported range has to match a range someone has actually
    watched. Frame numbering stays that of the source throughout.

    Keypoint order follows the released checkpoint: indices 0 to 3 are the
    ground-contact corners in cyclic order, 4 to 7 the roof corners above them
    in the same order. ``ground`` repeats the ground quad after refinement (the
    quad that is actually drawn), ``footprint`` is that quad mapped through the
    frame's homography onto the ground plane, where distances are comparable
    across the clip and ``bev_unit`` is one median vehicle length.
    """
    import gzip
    import json
    from datetime import datetime, timezone

    def r(a, nd=2):
        return [[round(float(x), nd) for x in row] for row in np.asarray(a)]

    frames = []
    for i, st in enumerate(states):
        if i < first:
            continue
        H = np.asarray(st.H, dtype=np.float64) if st.H is not None else None
        tracks = []
        for v in st.tracks:
            g = np.asarray(v.ground, dtype=np.float64)
            has_g = g.shape == (4, 2)
            fp = None
            if has_g and H is not None:
                try:
                    fp = r(apply_homography(g, H), 3)
                except Exception:
                    fp = None
            kp = np.asarray(v.kpts, dtype=np.float64)
            tracks.append(dict(
                id=int(v.id), cls=int(v.cls),
                name=style.CATEGORY_LABEL.get(POSE_CATEGORY_V2.get(int(v.cls), "other"),
                                              "object"),
                box=[round(float(x), 1) for x in np.asarray(v.box_xyxy).ravel()],
                kpts=r(kp, 1) if kp.size else None,
                ground=r(g, 1) if has_g else None,
                footprint=fp,
                observed=bool(getattr(v, "obs", v.time_since_update == 0)),
                kp_ok=bool(getattr(v, "kp_ok", False))))
        frames.append(dict(
            i=i, t=round(i / float(pipe.src_fps or 30.0), 4),
            H=[[float(x) for x in row] for row in H] if H is not None else None,
            bev_unit=round(float(st.bev_unit or 0.0), 4),
            tracks=tracks,
            boxes_2d=[dict(cls=int(getattr(d, "cls", -1)),
                           box=[round(float(x), 1) for x in np.asarray(d.xyxy).ravel()])
                      for d in st.aux_dets]))

    # Which half of the 8 keypoints touches the ground is a property of the
    # checkpoint, resolved by vote at run time, so it is recorded rather than
    # assumed.
    gi = list(pipe.resolver.indices(8) or [0, 1, 2, 3])
    roof = [k for k in range(8) if k not in gi]
    w, h = (int(v) for v in states[0].img_wh) if states else (0, 0)
    doc = dict(
        format="urbanomnidetect-pseudo-labels/1",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        note=("Model output refined offline by refine_v2, not human annotation. "
              "Treat as pseudo ground truth."),
        source=dict(processed_wh=[w, h], fps=round(float(pipe.src_fps or 30.0), 4),
                    frames=len(frames), first_frame=int(first),
                    last_frame=int(len(states) - 1), rendered_frames=len(states),
                    **pipe.labels_meta),
        scene=dict(camera=pipe.camera_tag or "unknown",
                   fixed_ground_plane=bool(fixed),
                   caption=pipe.caption or None),
        keypoints=dict(count=8, ground_indices=gi, roof_indices=roof,
                       order="the ground half is resolved per clip by majority vote; "
                             "each roof corner sits above the ground corner at the "
                             "same position within its half"),
        classes={str(k): style.CATEGORY_LABEL.get(vv, vv)
                 for k, vv in POSE_CATEGORY_V2.items()},
        fields=dict(
            box="2D box [x1,y1,x2,y2] in processed pixels",
            kpts="8 cuboid corners [[x,y]...] in processed pixels, null when 2D-only",
            ground="4 ground-contact corners in processed pixels",
            footprint="the same quad on the ground plane, through this frame's H",
            observed="false when the pose was filled across a detection gap",
            kp_ok="the keypoint head was trusted on this frame",
            boxes_2d="detections carried without a usable cuboid"),
        frames=frames)

    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    n_ids = len({t["id"] for f in frames for t in f["tracks"]})
    print(f"[labels] {len(frames)} frames, {n_ids} ids -> {path}")


_orig_stabilize = rt._stabilize_offline

def _shot_camera_tag(states, C, moving_source, shift_tol=0.03, zoom_tol=0.03):
    """Is this a static camera or a moving one, as a viewer would say it.

    Not the same question the solver answers. The solver asks whether one
    ground plane fits the whole clip, and a drone rising slowly enough still
    passes that test: over a few seconds its per-frame motion sits under every
    threshold, so the geometry is modelled as fixed. Saying "static camera"
    about that footage would simply be wrong.

    So the caption is measured from the NET transform between the first and the
    last frame, not from a per-frame median: a view that has travelled more than
    a few percent of the frame, or zoomed by more than a few percent, moved.
    Both tolerances are fractions of the frame, so nothing here is tied to a
    resolution, a frame rate or a clip length.
    """
    if moving_source:
        return "moving camera"
    if C is None or len(C) < 2 or not states:
        return "static camera"
    try:
        w, h = [float(v) for v in states[0].img_wh]
        M = np.asarray(C[-1], dtype=np.float64)
        if M.shape != (3, 3) or not np.all(np.isfinite(M)):
            return "static camera"
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, M).reshape(-1, 2)
    except Exception:
        return "static camera"
    src = src.reshape(-1, 2)
    diag = float(np.hypot(w, h)) or 1.0
    shift = float(np.linalg.norm(dst.mean(0) - src.mean(0))) / diag
    area = 0.5 * abs(float(np.dot(dst[:, 0], np.roll(dst[:, 1], -1))
                            - np.dot(dst[:, 1], np.roll(dst[:, 0], -1))))
    zoom = np.sqrt(area / max(w * h, 1.0)) - 1.0
    moved = shift > shift_tol or abs(zoom) > zoom_tol
    print(f"[caption] net camera motion over the clip: {shift * 100:.1f}% of the "
          f"frame, {zoom * 100:+.1f}% zoom -> "
          f"{'moving' if moved else 'static'} camera")
    return "moving camera" if moved else "static camera"




def _stabilize_offline_v2(pipe, states, fixed, traj_C, traj_steps, args):
    """Offline back-end: refine tracks/geometry, then the v1 stabilisation.

    The refinement that changes the geometry (gap fill, keypoint repair, box
    fill) runs BEFORE the v1 pass so its zero-phase smoothing and the global
    ground-homography solve both see the corrected footprints. The footprint
    shape/heading pass runs AFTER, because the global solve re-poses every
    footprint in its own reference gauge.
    """
    gi = pipe.resolver.indices(8)
    moving_source = not fixed          # what the SOURCE is, for the caption
    if (POST["world_lock"] and not fixed and states
            and getattr(states[0], "frame", None) is not None):
        # The camera was judged to be moving. That verdict comes from corners
        # tracked anywhere in the frame, so on a shot that is mostly traffic it
        # can be measuring the CARS. Re-measure away from them and re-decide
        # with the same thresholds: if the camera is really holding still (or
        # drifting slowly), the whole clip can share one ground plane instead of
        # a per-frame homography whose noise reaches every cuboid.
        C, steps = refine_v2.ego_motion(states)
        if len(steps) > 1:
            diag = float(np.hypot(*states[0].img_wh)) or 1.0
            tmed = float(np.median(steps[1:, 0])) / diag
            rmed = float(np.median(steps[1:, 1]))
            smed = float(np.median(steps[1:, 2]))
            if tmed < 0.003 and rmed < 0.15 and smed < 0.003:
                fixed = True
                traj_C, traj_steps = C, steps
                print("[refine] camera re-classified FIXED once the traffic is "
                      "excluded; solving one world-locked ground plane")
    cfg = refine_v2.RefineConfig.for_fps(getattr(pipe, "src_fps", 30.0))
    sets = refine_v2.TrackSets()
    if POST["refine"] and states:
        sets = refine_v2.refine_tracks(pipe, states, gi, cfg,
                                       dedupe=POST["dedupe"])

    _orig_stabilize(pipe, states, fixed, traj_C, traj_steps, args)
    # The caption describes the SHOT, not the solver.
    pipe.camera_tag = _shot_camera_tag(states, traj_C, moving_source)

    if os.environ.get("V2_DUMP_STATES"):      # dev hook: geometry-only snapshot
        _dump_states(states, fixed, os.environ["V2_DUMP_STATES"])
    if POST["refine"] and states:
        refine_v2.refine_geometry(
            pipe, states, gi, cfg, sets, fixed,
            smooth_win=max(3, int(getattr(args, "smooth", 11) or 11)),
            poly=int(getattr(args, "smooth_poly", 2) or 2),
            use_velocity=POST["vel_heading"], hold_still=POST["hold_still"],
            rigid_cuboids=POST["rigid_cuboids"], shape_smooth=POST["shape_smooth"],
            fill_box=POST["fill_box"])
        refine_v2.report_jitter(states)
        if os.environ.get("V2_DUMP_FINAL"):    # dev hook: what is actually drawn
            _dump_states(states, fixed, os.environ["V2_DUMP_FINAL"])
    if POST["bev_fit"] and states:
        _bev_fit(pipe, states, POST["fit_max_px"], POST["fit_min_px"], POST["fit_pct"])
    if states:
        n_ids = len({v.id for s in states for v in s.tracks})
        mean_trk = np.mean([len(s.tracks) for s in states])
        n_aux = np.mean([len(s.aux_dets) for s in states])
        print(f"[tracks] {n_ids} 3D ids over {len(states)} frames; mean "
              f"{mean_trk:.1f} cuboids + {n_aux:.1f} 2D-only per frame")
    if getattr(pipe, "labels_path", "") and states:
        _write_labels(pipe, states, fixed, pipe.labels_path,
                      first=int(getattr(pipe, "labels_first", 0)))


rt._stabilize_offline = _stabilize_offline_v2


# --------------------------------------------------------------------------- #
def build_v2_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kp-vis", type=float, default=0.5,
                   help="min per-corner keypoint visibility to draw a cuboid")
    p.add_argument("--suppress-nested", type=float, default=0.0,
                   help="drop a vehicle box nested >= this fraction inside a larger vehicle "
                        "box whose cuboid footprint it overlaps (0 = off; 0.85 recommended)")
    p.add_argument("--class-conf", default="",
                   help="per-class min confidence, COCO ids, e.g. '0:0.4,1:0.5,3:0.5'")
    p.add_argument("--bev-canvas", default=None,
                   help="BEV panel size WxH (default: source size; dashboard: 640x1080)")
    p.add_argument("--layout", choices=["side", "dashboard"], default="side")
    p.add_argument("--caption", default="", help="scene caption shown in the UI")
    p.add_argument("--title", default="",
                   help="opening titles over this clip, as start,fade-in,hold,fade-out "
                        "in SOURCE seconds (the cut trims and speeds up afterwards)")
    p.add_argument("--object-px", type=float, default=40.0,
                   help="BEV zoom: on-screen px of a median footprint (max with --bev-fit)")
    p.add_argument("--bev-fit", action="store_true",
                   help="offline: auto-zoom the BEV so the clip's footprints fit")
    p.add_argument("--bev-fit-min-px", type=float, default=22.0,
                   help="--bev-fit never zooms out below this many px per median object")
    p.add_argument("--bev-fit-pct", type=float, default=90.0,
                   help="--bev-fit fits this percentile of footprint extents")
    p.add_argument("--label-min-h", type=float, default=26.0,
                   help="hide the id chip of boxes shorter than this (px); cuboid still drawn")
    p.add_argument("--refine", "--track-cleanup", dest="refine", action="store_true",
                   help="offline: merge split ids, fill detection gaps, per-track cuboid "
                        "vote, box fill, whole-track rigid footprint + trajectory heading")
    p.add_argument("--no-fill-box", dest="fill_box", action="store_false",
                   help="do not rescale cuboids to fill their 2D box")
    p.add_argument("--no-vel-heading", dest="vel_heading", action="store_false",
                   help="do not take a moving vehicle's BEV heading from its trajectory")
    p.add_argument("--no-rigid-cuboids", dest="rigid_cuboids", action="store_false",
                   help="do not rebuild cuboids from (footprint, height, vertical VP)")
    p.add_argument("--refine-fps", type=float, default=0.0,
                   help="frame rate the refinement should assume (default: read from "
                        "the source). Its thresholds are in seconds, so this only "
                        "matters if the file's metadata is wrong")
    p.add_argument("--speed-ratio", type=float, default=3.0,
                   help="cut a track rather than bridge a gap that needs this many "
                        "times its own top speed")
    p.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                   help="do not merge tracks whose ground footprints coincide")
    p.add_argument("--no-shape-smooth", dest="shape_smooth", action="store_false",
                   help="do not smooth unmodelled cuboids in their own box frame")
    p.add_argument("--shape-win", type=int, default=31,
                   help="window for that box-frame shape smoothing")
    p.add_argument("--no-world-lock", dest="world_lock", action="store_false",
                   help="never re-measure ego-motion off the static scene; keep the "
                        "per-frame homography on a camera judged to be moving")
    p.add_argument("--no-hold-still", dest="hold_still", action="store_false",
                   help="do not freeze the position of a vehicle that is standing still")
    p.add_argument("--max-reproj", type=float, default=0.10,
                   help="a rebuilt cuboid is rejected above this reprojection error "
                        "(fraction of the 2D box diagonal)")
    p.add_argument("--max-gap", type=int, default=45,
                   help="longest detection gap (frames) that --refine interpolates")
    p.add_argument("--min-fill", type=float, default=0.5,
                   help="a track whose cuboid hull covers less than this fraction of its "
                        "2D box is drawn 2D-only for the whole clip")
    p.add_argument("--min-track", type=int, default=12)
    p.add_argument("--coast-tail", type=int, default=8)
    p.add_argument("--merge-gap", type=int, default=30)
    p.add_argument("--labels", default="",
                   help="also write the drawn geometry as pseudo-labels to this "
                        "path (.json, or .json.gz to compress)")
    p.add_argument("--labels-from", type=int, default=0,
                   help="write labels from this frame on. Earlier frames are still "
                        "rendered and still inform the offline refinement; they are "
                        "left out of the file")
    p.add_argument("--labels-source", default="",
                   help="original file the input was scaled from, recorded in the "
                        "label file so coordinates can be mapped back to it")
    p.add_argument("--coast-mark", action="store_true",
                   help="append '?' to the label of a coasting track (v1 look)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    v2args, rest = build_v2_argparser().parse_known_args(argv)
    # Probe the source once: the refinement is configured in seconds so it needs
    # the frame rate, and the dashboard gives the footage as many pixels as it
    # has, fitting the bird's-eye column into whatever is left.
    fps, src_w = 30.0, 0
    try:
        cap = cv2.VideoCapture(next(a for i, a in enumerate(rest)
                                    if rest[i - 1] == "--input"))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        cap.release()
    except Exception:
        pass
    canvas = None
    if v2args.bev_canvas:
        w, h = v2args.bev_canvas.lower().split("x")
        canvas = (int(w), int(h))
    elif v2args.layout == "dashboard":
        canvas = (max(360, FINAL_W - src_w) if src_w else 640, FINAL_H)
    # Provenance for the label file: which clip, which checkpoint, and the
    # factor that maps processed pixels back to the original footage.
    inp = next((a for i, a in enumerate(rest) if rest[i - 1] == "--input"), "")
    ckpt = next((a for i, a in enumerate(rest) if rest[i - 1] == "--kp-model"), "")
    meta = dict(path=os.path.abspath(v2args.labels_source or inp) if (inp or v2args.labels_source) else None,
                name=os.path.basename(v2args.labels_source or inp) or None,
                checkpoint=os.path.basename(ckpt) or None, scale=1.0)
    if v2args.labels_source:
        try:
            c = cv2.VideoCapture(v2args.labels_source)
            ow = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            oh = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            c.release()
            if ow and src_w:
                meta.update(width=ow, height=oh, scale=round(ow / float(src_w), 6))
        except Exception:
            pass
    V2Viewport.DEFAULT_OBJECT_PX = float(v2args.object_px)
    V2Pipeline.CONFIG = dict(kp_vis=v2args.kp_vis, suppress_nested=v2args.suppress_nested,
                             class_conf=_parse_class_conf(v2args.class_conf),
                             bev_canvas=canvas, layout=v2args.layout,
                             caption=v2args.caption,
                             labels=v2args.labels, labels_meta=meta,
                             labels_first=int(v2args.labels_from),
                             show_coast_mark=v2args.coast_mark,
                             label_min_h=v2args.label_min_h,
                             title=tuple(float(x) for x in v2args.title.split(","))
                             if v2args.title else (0.0, 0.0, 0.0, 0.0))
    POST.update(refine=v2args.refine, min_track=v2args.min_track,
                coast_tail=v2args.coast_tail, merge_gap=v2args.merge_gap,
                bev_fit=v2args.bev_fit, fit_max_px=float(v2args.object_px),
                fit_min_px=float(v2args.bev_fit_min_px), fit_pct=float(v2args.bev_fit_pct),
                max_gap=int(v2args.max_gap), min_fill=float(v2args.min_fill),
                fill_box=bool(v2args.fill_box), vel_heading=bool(v2args.vel_heading),
                hold_still=bool(v2args.hold_still),
                world_lock=bool(v2args.world_lock),
                shape_smooth=bool(v2args.shape_smooth), shape_win=int(v2args.shape_win),
                dedupe=bool(v2args.dedupe), speed_ratio=float(v2args.speed_ratio),
                rigid_cuboids=bool(v2args.rigid_cuboids),
                max_reproj=float(v2args.max_reproj))
    # v2 is its own auxiliary detector: default to none unless given explicitly.
    if not any(a.startswith("--aux-model") for a in rest):
        rest += ["--aux-model", "none"]
    if "--smooth" not in " ".join(rest) and (v2args.refine or v2args.bev_fit):
        print("[warn] --track-cleanup/--bev-fit are offline features; pass --smooth N")
    V2Pipeline.CONFIG["src_fps"] = float(v2args.refine_fps) or fps
    rt.RealtimePipeline = V2Pipeline          # rt.main() looks the name up at call time
    return rt.main(rest)


if __name__ == "__main__":
    main()
