#!/usr/bin/env python3
"""Real-time calibration-free BEV pipeline for video streams.

Ties the pieces together for live operation:

    pose model  -> detections (boxes + 8 keypoints)
    tracker     -> stable, jitter-damped, persistent tracks
    aux model   -> extra boxes -> ridge map to ground centres (paper Sec. 3.5)
    solver      -> orthogonality homography (warm-started, ~1-4 ms)
    viewport    -> temporally stable radar-style BEV
    renderer    -> source + BEV side by side with a timing HUD

Everything except the two neural nets runs on CPU in well under a millisecond,
so the achievable frame rate is set by detector inference. With TensorRT on a
modern GPU the pose model is the only meaningful cost (see paper Table 3).

Example
-------
    python bev_realtime.py \\
        --input assets/drone1hq.mp4 \\
        --kp-model UrbanOmniDetect/checkpoints/urbanomnidetect_yolo11x-p2_640.pt \\
        --kp-imgsz 640 --aux-model yolo26x.pt \\
        --device cuda:0 --export tensorrt --output out.mp4
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homography_rt import OrthoHomographySolver, apply_homography
from uod import style
from uod.aux_head import AuxCenterRegressor
from uod.bev import (BEVViewport, DEFAULT_AUX_ASPECT, convex_overlap_fraction)
from uod.keypoints import (Detection, GroundIndexResolver, parse_boxes_result,
                           parse_pose_result)
from uod.model import DEFAULT_AUX_MODEL, load_detector
from uod.textdraw import UILayer
from uod.tracking import MultiObjectTracker
from uod.viz import (draw_box, draw_cuboid, draw_dashed_polyline,
                     draw_ground_quad, rounded_contour, stack_side_by_side)
from uod.tracking import _iou_matrix

# COCO ids the auxiliary detector is allowed to report: pedestrian, bicycle,
# motorcycle, car, bus, truck. Everything else (traffic lights, etc.) is dropped.
AUX_CLASSES = (0, 1, 2, 3, 5, 7)


@dataclass
class _TrackView:
    """An immutable snapshot of a track, safe to hand to the render thread."""
    id: int
    cls: int
    kpts: np.ndarray
    ground: np.ndarray
    box_xyxy: np.ndarray
    time_since_update: int
    bev_quad: object = None   # rigid footprint in rectified coords (or None)


@dataclass
class RenderState:
    """Everything the renderer needs for one frame, fully decoupled from the
    live tracker/solver state (so rendering can run on its own thread)."""
    frame: np.ndarray
    tracks: List[_TrackView]
    aux_dets: list
    aux_centers: np.ndarray
    H: np.ndarray
    info: object
    gi: Optional[list]
    img_wh: tuple
    timing: dict
    frame_idx: int
    n_gated: int
    bev_unit: float = 0.0   # median footprint size in H's gauge (BEV scale anchor)
    class_dims: dict = field(default_factory=dict)  # cls -> (w, l) in object_px units
    fps_core: float = 0.0


class Stopwatch:
    """Accumulates per-stage timings and exposes EMA-smoothed values (ms)."""

    def __init__(self, ema: float = 0.9):
        self.ema = ema
        self.t: dict = {}
        self._mark = None

    def start(self):
        self._mark = time.perf_counter()

    def lap(self, name: str):
        now = time.perf_counter()
        dt = (now - self._mark) * 1000.0
        self._mark = now
        self.t[name] = self.ema * self.t.get(name, dt) + (1 - self.ema) * dt
        return dt


class RealtimePipeline:
    """Stateful per-frame pipeline; reusable for plain or sliced inference."""

    def __init__(self, pose_model, aux_model=None, *, kp_imgsz=640,
                 kp_conf=0.25, aux_conf=0.30, device="cpu",
                 ground_indices=None, use_homography=True, show_3d=True,
                 snap_rect=True, solver_ema=0.0, aux_lam=1.0,
                 aux_footprint=False, aux_overlap=0.12, aux_width_scale=0.9,
                 aux_aspect=None, aux_gate=False, aux_gate_iou=0.3, draw_aux_box=True,
                 trails=True, trail_len=16, stabilize=True, bev_window=512,
                 tracker_kwargs=None, bev_canvas=None, predict_fn=None):
        self.pose_model = pose_model
        self.aux_model = aux_model
        self.kp_imgsz = kp_imgsz
        self.kp_conf = kp_conf
        self.aux_conf = aux_conf
        self.device = device
        self.use_homography = use_homography
        self.show_3d = show_3d
        self.snap_rect = snap_rect
        self.aux_footprint = aux_footprint
        self.aux_overlap = float(aux_overlap)
        self.aux_width_scale = float(aux_width_scale)
        self.aux_aspect = dict(DEFAULT_AUX_ASPECT if aux_aspect is None else aux_aspect)
        self.aux_gate = aux_gate
        self.aux_gate_iou = float(aux_gate_iou)
        self.draw_aux_box = draw_aux_box
        self._n_gated = 0
        self.trails = trails
        self.trail_len = int(trail_len)
        self._trails: dict = {}     # track id -> deque of BEV-canvas centres
        # Footprint accumulation buffer: the ground plane is a scene constant,
        # so solving the homography over a rolling window of recent footprints
        # keeps it (and thus the BEV axis) stable when objects enter or leave.
        # A parallel class buffer lets us size every object of a class
        # identically (rigid same-class objects have equal footprints).
        self.stabilize = bool(stabilize)
        self._quad_buffer: deque = deque(maxlen=int(bev_window))
        self._cls_buffer: deque = deque(maxlen=int(bev_window))
        # Per-track rigid footprint estimates: id -> {S, W, locked, cls, rms}.
        self._rigid: dict = {}
        self._rigid_lock_after = 20   # observations before the shape is fixed
        # The BEV scale (object_px / bev_unit) must be a CONSTANT, not a per-frame
        # buffer median -- otherwise the median pulses as footprints enter/leave
        # the window and every object slides radially (physically impossible).
        # Since the solver's normaliser N fixes the gauge, we lock bev_unit once
        # the buffer is populated and re-lock only if N (the gauge) changes.
        self._bev_unit_locked = None
        self._last_N = None
        self._predict_fn = predict_fn  # override for sliced inference

        self.resolver = GroundIndexResolver(forced=ground_indices)
        self.tracker = MultiObjectTracker(**(tracker_kwargs or {}))
        self.solver = OrthoHomographySolver(ema=solver_ema)
        self.aux = AuxCenterRegressor(lam=aux_lam)
        self.viewport: Optional[BEVViewport] = None
        self._bev_canvas = bev_canvas
        self.sw = Stopwatch()
        self.frame_idx = 0
        self.fps_core = 0.0      # core (detect/track/solve) throughput, set by runner
        self.fps_render = 0.0    # render throughput, set by runner

    # ------------------------------------------------------------------ #
    def _detect_pose(self, frame) -> List[Detection]:
        if self._predict_fn is not None:
            return self._predict_fn(frame)
        res = self.pose_model.predict(frame, imgsz=self.kp_imgsz,
                                      conf=self.kp_conf, device=self.device,
                                      verbose=False)[0]
        # Observe raw keypoints to resolve ground indices, then parse.
        if res.keypoints is not None and res.keypoints.data is not None:
            self.resolver.observe(res.keypoints.data.cpu().numpy()[:, :, :2])
        return parse_pose_result(res, self.resolver.indices(8))

    def _detect_aux_raw(self, frame) -> List[Detection]:
        """Run the auxiliary detector and return all of its boxes (no dedup)."""
        if self.aux_model is None:
            return []
        res = self.aux_model.predict(frame, imgsz=self.kp_imgsz,
                                     conf=self.aux_conf, device=self.device,
                                     verbose=False)[0]
        return parse_boxes_result(res, conf_thresh=self.aux_conf)

    def _gate_by_aux(self, dets, aux_raw):
        """Drop pose detections the auxiliary detector cannot corroborate.

        A 3D detection is kept only if some auxiliary box overlaps it (IoU >=
        ``aux_gate_iou``), which removes hallucinated objects the high-recall
        auxiliary detector does not see. Frames where the auxiliary detector
        returned nothing at all are treated as an auxiliary miss (not evidence
        of absence) and are left ungated, so a single bad auxiliary frame does
        not wipe every track.
        """
        if not dets or not aux_raw:
            return dets
        ab = np.array([d.xyxy for d in aux_raw])
        kept = []
        for d in dets:
            iou = _iou_matrix(d.xyxy[None], ab)[0]
            if float(iou.max(initial=0.0)) >= self.aux_gate_iou:
                kept.append(d)
        return kept

    def _aux_display(self, aux_raw, track_boxes):
        """Auxiliary boxes that are NOT duplicates of a track, plus their
        ridge-predicted ground centres (for BEV display)."""
        aux = aux_raw
        if aux and len(track_boxes):
            ab = np.array([d.xyxy for d in aux])
            iou = _iou_matrix(ab, np.array(track_boxes))
            aux = [d for i, d in enumerate(aux) if iou[i].max(initial=0.0) < 0.45]
        centers = (self.aux.predict(np.array([d.xyxy for d in aux]))
                   if aux else np.zeros((0, 2)))
        return aux, centers

    # ------------------------------------------------------------------ #
    def step(self, frame) -> RenderState:
        """Run the real-time CORE for one frame (detect, track, solve).

        Returns an immutable :class:`RenderState` snapshot. This is the
        throughput-critical path; it does no rendering, so it can run on its
        own thread at the detector's native rate while the (heavier) renderer
        runs in parallel.
        """
        H_img, W_img = frame.shape[:2]
        img_wh = (W_img, H_img)
        self.sw.start()

        dets = self._detect_pose(frame)
        self.sw.lap("pose")

        # Auxiliary detector runs before tracking so it can gate hallucinations.
        aux_raw = self._detect_aux_raw(frame)
        self.sw.lap("aux")

        n_gated = 0
        if self.aux_gate and self.aux_model is not None:
            n_before = len(dets)
            dets = self._gate_by_aux(dets, aux_raw)
            n_gated = n_before - len(dets)

        gi = self.resolver.indices(8)
        tracks = self.tracker.update(dets, img_wh, ground_indices=gi)
        self.sw.lap("track")

        # Fit the auxiliary ground-centre map on confirmed tracks that carry a
        # footprint, then derive the auxiliary boxes to display (non-duplicates).
        anchor_boxes, anchor_centers = [], []
        for t in tracks:
            if t.ground.shape == (4, 2):
                anchor_boxes.append(t.box_xyxy)
                anchor_centers.append(t.ground.mean(axis=0))
        if len(anchor_boxes) >= 2:
            self.aux.fit(np.array(anchor_boxes), np.array(anchor_centers))
        track_boxes = [t.box_xyxy for t in tracks]
        aux_dets, aux_centers = self._aux_display(aux_raw, track_boxes)

        # Solve the orthogonality homography. With stabilisation enabled we
        # solve over a rolling buffer of recent footprints (the ground plane is
        # a scene constant), so the BEV axis does not lurch when the set of
        # visible objects changes.
        H = np.eye(3)
        info = None
        bev_unit = 0.0
        class_dims: dict = {}
        if self.use_homography:
            pairs = [(t.ground, t.cls) for t in tracks if t.ground.shape == (4, 2)]
            if self.stabilize:
                for g, cl in pairs:
                    self._quad_buffer.append(g)
                    self._cls_buffer.append(cl)
                solve_quads = list(self._quad_buffer)
                solve_cls = list(self._cls_buffer)
            else:
                solve_quads = [p[0] for p in pairs]
                solve_cls = [p[1] for p in pairs]
            H, info = self.solver.solve(solve_quads)
            class_dims, bev_unit_cur = self._class_footprint_dims(
                solve_quads, solve_cls, H)
            # If the gauge (normaliser N) changed, the locked scale and the
            # rigid shapes (expressed in that gauge) are no longer valid -> reset.
            if self.solver._N is not None:
                if (self._last_N is not None
                        and not np.allclose(self.solver._N, self._last_N)):
                    self._bev_unit_locked = None
                    self._rigid.clear()
                self._last_N = self.solver._N.copy()
            # Lock the scale once the buffer is populated; hold it thereafter.
            if self.stabilize:
                if (self._bev_unit_locked is None and bev_unit_cur > 1e-9
                        and len(self._quad_buffer) >= min(
                            256, self._quad_buffer.maxlen or 256)):
                    self._bev_unit_locked = bev_unit_cur
                bev_unit = (self._bev_unit_locked
                            if self._bev_unit_locked is not None else bev_unit_cur)
            else:
                bev_unit = bev_unit_cur
        self.sw.lap("solve")

        # Per-track rigid footprint: estimate each object's CONSTANT footprint
        # shape (generalised-Procrustes MLE) and render it at the current pose,
        # so a tracked object's BEV size never changes (only position/heading do).
        rigid = {}
        if self.use_homography and bev_unit > 1e-9:
            for t in tracks:
                if t.ground.shape == (4, 2):
                    rigid[t.id] = self._rigid_footprint(t, H, bev_unit)
            self._rigid = {k: v for k, v in self._rigid.items()
                           if k in {t.id for t in tracks}}

        # Snapshot the tracks (copy arrays) so the renderer is decoupled from
        # subsequent in-place tracker updates.
        views = []
        for t in tracks:
            v = _TrackView(t.id, t.cls, t.kpts.copy(), t.ground.copy(),
                           t.box_xyxy.copy(), t.time_since_update)
            v.bev_quad = rigid.get(t.id)
            views.append(v)
        self.frame_idx += 1
        return RenderState(frame, views, aux_dets, aux_centers, H, info, gi,
                           img_wh, dict(self.sw.t), self.frame_idx, n_gated,
                           bev_unit=bev_unit, class_dims=class_dims)

    @staticmethod
    def _class_footprint_dims(quads, classes, H):
        """Per-class median footprint (width, length) and the overall median size.

        A rigid object's footprint is a fixed real size, and objects of the same
        class are the same size, so we render every object of a class with one
        canonical box. We estimate that box as the robust per-class median of the
        footprint dimensions over the buffer (which spans many positions and
        frames), expressed relative to the overall median ``bev_unit``. The ratio
        is gauge-invariant and stable, so a class's rendered size is constant
        regardless of camera motion, an object's range, or scene turnover.
        Returns ``({cls: (w_ratio, l_ratio)}, bev_unit)``.
        """
        if not quads:
            return {}, 0.0
        pts = apply_homography(np.vstack(quads), H).reshape(len(quads), 4, 2)
        side_a = 0.5 * (np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1) +
                        np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1))
        side_b = 0.5 * (np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1) +
                        np.linalg.norm(pts[:, 0] - pts[:, 3], axis=1))
        w = np.minimum(side_a, side_b)
        l = np.maximum(side_a, side_b)
        sz = np.sqrt(np.maximum(w * l, 0.0))
        ok = np.isfinite(sz) & (sz > 1e-6)
        if not ok.any():
            return {}, 0.0
        overall = float(np.median(sz[ok]))
        cls = np.asarray(classes)
        dims = {}
        for c in np.unique(cls[ok]):
            m = ok & (cls == c)
            dims[int(c)] = (float(np.median(w[m])) / overall,
                            float(np.median(l[m])) / overall)
        return dims, overall

    # ------------------------------------------------------------------ #
    def render(self, state: RenderState) -> np.ndarray:
        """Render one :class:`RenderState` into the composite (source | BEV).

        This is the expensive, quality-heavy path; it owns the BEV viewport and
        the motion-trail history, so it can run on a separate thread without
        touching the core's tracker/solver state.
        """
        if self.viewport is None:
            cw = self._bev_canvas or state.img_wh
            self.viewport = BEVViewport(canvas_size=cw)
        if self.use_homography:
            self.viewport.update(state.H, state.img_wh, unit=state.bev_unit)
        return self._render(state)

    def process(self, frame) -> np.ndarray:
        """Synchronous convenience: core + render in one call (used by SAHI)."""
        return self.render(self.step(frame))

    # ------------------------------------------------------------------ #
    def _render(self, state: RenderState):
        frame, tracks = state.frame, state.tracks
        aux_dets, aux_centers = state.aux_dets, state.aux_centers
        H, info, gi = state.H, state.info, state.gi
        ui = UILayer()
        src = frame.copy()
        src_labels = []   # (x, y, text, color, coasting)

        # --- camera panel geometry ---
        if self.draw_aux_box:
            for d in aux_dets:
                draw_box(src, d.xyxy, style.THEME["aux"], thickness=1, halo=True, radius=4)
        for t in tracks:
            col = style.instance_color(t.cls, t.id, source="pose")
            coasting = t.time_since_update > 0
            if self.show_3d and t.kpts.shape[0] >= 8:
                draw_cuboid(src, t.kpts, gi, col, dim=coasting)
            elif t.ground.shape == (4, 2):
                draw_ground_quad(src, t.ground, col, thickness=2)
            x1, y1 = t.box_xyxy[0], t.box_xyxy[1]
            src_labels.append((x1, y1, t.cls, t.id, coasting))

        # --- BEV panel geometry ---
        bev = self.viewport.blank_canvas(radar=True)
        bev_labels = []   # (cx, cy, id)
        present = set()
        if self.use_homography:
            # Pass 1: map each track's RIGID (constant-size) footprint to canvas.
            fps = []   # (track, poly)
            for t in tracks:
                if t.bev_quad is None:
                    continue
                poly = self.viewport.rect_to_canvas(t.bev_quad)
                if len(poly) != 4 or not np.all(np.isfinite(poly)):
                    continue
                fps.append((t, poly))
                c = poly.mean(axis=0)
                bev_labels.append((c[0], c[1], t.id))

            # Update + draw fading motion trails UNDER the footprints.
            if self.trails:
                cur = set()
                for (cx, cy, tid) in bev_labels:
                    cur.add(tid)
                    self._trails.setdefault(
                        tid, deque(maxlen=self.trail_len)).append((cx, cy))
                for tid in [k for k in self._trails if k not in cur]:
                    del self._trails[tid]
                self._draw_trails(bev, {t.id: t.cls for t, _ in fps})

            # Pass 2: draw each track's constant-size footprint on top.
            track_polys = []
            for (t, poly) in fps:
                col = style.instance_color(t.cls, t.id, source="pose")
                present.add(style.class_category(t.cls, "pose"))
                alpha = 0.30 if t.time_since_update == 0 else 0.12
                self.viewport.draw_footprint_poly(
                    bev, poly, col, fill_alpha=alpha,
                    thickness=2 if t.time_since_update == 0 else 1)
                track_polys.append(poly)
            self._draw_aux_bev(bev, aux_dets, aux_centers, H, track_polys)
        self.viewport.draw_camera(bev)

        # --- compose, then a single anti-aliased UI pass ---
        composite = stack_side_by_side(src, bev, gap=2)
        bev_x = src.shape[1] + 2
        self._compose_ui(ui, composite, state, present,
                         src_labels, bev_labels, bev_x)
        return ui.render(composite)

    # ------------------------------------------------------------------ #
    def _compose_ui(self, ui, composite, state, present,
                    src_labels, bev_labels, bev_x):
        tracks, aux_dets, info = state.tracks, state.aux_dets, state.info
        Wc = composite.shape[1]
        Hc = composite.shape[0]
        T = style.THEME
        head_h = 30

        # Header bar across the full width.
        ui.panel((0, 0), (Wc, head_h), T["panel"], alpha=0.82, radius=0)
        ui.line((0, head_h), (Wc, head_h), T["accent"], width=1, alpha=0.5)
        ui.text(12, head_h / 2, "UrbanOmniDetect", size=15, font="bold",
                color=T["panel_text"], anchor="lm", shadow=False)
        ui.text(150, head_h / 2, "Calibration-Free 3D + BEV", size=12,
                font="regular", color=T["muted_text"], anchor="lm", shadow=False)
        timing = " ".join(f"{k} {v:.1f}" for k, v in state.timing.items())
        stats = (f"core {self.fps_core:4.1f} fps   render {self.fps_render:4.1f} fps"
                 f"   trk {len(tracks)}   aux {len(aux_dets)}")
        if self.aux_gate:
            stats += f"   gated {state.n_gated}"
        ui.text(Wc - 12, head_h / 2 - 6, stats, size=12, font="mono",
                color=T["panel_text"], anchor="rm", shadow=False)
        sub = f"{timing} ms"
        if info is not None:
            sub += f"   |  H {info.loss:.1e}  it{info.iterations}"
        ui.text(Wc - 12, head_h / 2 + 7, sub, size=10, font="mono",
                color=T["muted_text"], anchor="rm", shadow=False)

        # Metric range-ring labels (range in object-size units), so the scale
        # is explicit and readable directly off the radar.
        cx_b, cy_b = self.viewport.camera_canvas
        for (r, units) in getattr(self.viewport, "_rings", []):
            ui.text(bev_x + cx_b + 5, cy_b - r + 1, str(units), size=9,
                    font="mono", color=T["muted_text"], anchor="lm", shadow=False)

        # Panel captions.
        for x, label in ((10, "CAMERA"), (bev_x + 10, "BIRD'S-EYE VIEW")):
            ui.text(x, Hc - 10, label, size=11, font="bold",
                    color=T["muted_text"], anchor="ld")

        # Legend (classes present), top-right of the BEV panel.
        ly = head_h + 10
        for cat in [c for c in style.PALETTE if c in present]:
            ui.chip(Wc - 10, ly, style.CATEGORY_LABEL[cat], size=11,
                    fg=T["panel_text"], bg=T["panel"], alpha=0.8,
                    accent=style.PALETTE[cat], anchor="ra", radius=6)
            ly += 22
        if self.aux_footprint and aux_dets:
            ui.chip(Wc - 10, ly, "Aux (est.)", size=11, fg=T["panel_text"],
                    bg=T["panel"], alpha=0.8, accent=T["aux"], anchor="ra", radius=6)

        # Per-track chips on the camera panel.
        for (x, y, cls, tid, coasting) in src_labels:
            col = style.instance_color(cls, tid, source="pose")
            name = style.CATEGORY_LABEL[style.class_category(cls, "pose")]
            txt = f"{name} {tid}" + ("  ?" if coasting else "")
            yy = max(float(y), head_h + 16)
            ui.chip(float(x), yy, txt, size=11, fg=(240, 240, 240),
                    bg=T["panel"], alpha=0.55 if coasting else 0.85,
                    accent=col, anchor="ld", radius=6, font="bold")

        # Small id labels at BEV footprint centres.
        for (cx, cy, tid) in bev_labels:
            ui.text(bev_x + cx, cy, str(tid), size=10, font="bold",
                    color=(236, 236, 236), anchor="mm")

    # ------------------------------------------------------------------ #
    def _rigid_footprint(self, t, H, bev_unit):
        """Estimate a track's CONSTANT footprint and return it posed this frame.

        Physical model: a vehicle is a rigid body, so in the rectified ground
        plane its footprint is a constant shape S; only its planar pose (centre,
        heading) varies. We observe the 4 ground corners each frame (through the
        same H, normalised by ``bev_unit`` so frames under a moving camera are
        commensurable), fit the per-frame rotation by Kabsch (the corners are in
        a fixed cyclic order, so correspondence is known), and accumulate S as an
        incremental generalised-Procrustes mean -- the maximum-likelihood estimate
        of a constant under zero-mean noise. Its increments vanish as evidence
        grows (no EMA, no tuned gain); we fix it after ``_rigid_lock_after``
        observations so a tracked object's size then never changes. Keypoint noise
        flows entirely into the per-frame pose, never the size. Returns the posed
        footprint in rectified coords (or ``None``).
        """
        P = apply_homography(t.ground, H)
        if not np.all(np.isfinite(P)):
            return None
        Pn = P / bev_unit
        cP = Pn.mean(axis=0)
        Q = Pn - cP
        st = self._rigid.get(t.id)
        if st is None or st["cls"] != t.cls:        # new track (or class change)
            self._rigid[t.id] = {"S": Q.copy(), "W": 1.0, "locked": False,
                                 "cls": t.cls, "rms": deque(maxlen=30)}
            return P                                 # render raw first observation
        S = st["S"]
        # Kabsch rotation aligning the constant shape S to this frame's Q (Q ~ R S).
        C = Q.T @ S
        U, _, Vt = np.linalg.svd(C)
        dsign = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
        R = U @ np.array([[1.0, 0.0], [0.0, dsign]]) @ Vt
        posed = S @ R.T                              # constant shape at this heading
        # Update the shape estimate (MLE running mean) on matched, in-lier frames.
        if t.time_since_update == 0 and not st["locked"]:
            # Footprint area (normalised; median object ~= 1). Skip degenerate /
            # near-collinear (far, grazing) observations that would corrupt S.
            qa = 0.5 * abs(np.cross(Q[2] - Q[0], Q[3] - Q[1]))
            res = Q - posed
            rms = float(np.sqrt((res * res).mean()))
            med = float(np.median(st["rms"])) if len(st["rms"]) >= 5 else None
            if qa > 0.1 and (med is None or rms <= 4.0 * med):  # robust reject
                Qc = Q @ R                           # observation in S's frame
                w = st["W"]
                st["S"] = S + (Qc - S) / (w + 1.0)
                st["S"] -= st["S"].mean(axis=0)
                st["W"] = w + 1.0
                st["rms"].append(rms)
                if st["W"] >= self._rigid_lock_after:
                    st["locked"] = True
        return (posed + cP) * bev_unit               # constant size, current pose

    # ------------------------------------------------------------------ #
    def _draw_trails(self, bev, cls_of):
        """Draw fading per-track motion trails directly on the BEV canvas.

        Fade is faked by lerping the class colour toward the background, which
        avoids a per-segment alpha blend and keeps trails under the footprints.
        """
        bg = np.array(style.THEME["bev_bg"], dtype=np.float32)
        for tid, dq in self._trails.items():
            pj = list(dq)
            if len(pj) < 3:
                continue
            base = np.array(style.instance_color(cls_of.get(tid, -1), tid, "pose"),
                            dtype=np.float32)
            n = len(pj)
            for i in range(n - 1):
                f = i / (n - 1)
                col = bg * (1.0 - (0.12 + 0.5 * f)) + base * (0.12 + 0.5 * f)
                cv2.line(bev, (int(pj[i][0]), int(pj[i][1])),
                         (int(pj[i + 1][0]), int(pj[i + 1][1])),
                         (int(col[2]), int(col[1]), int(col[0])),
                         2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_aux_bev(self, bev, aux_dets, aux_centers, H, track_polys):
        """Render auxiliary detections in the BEV.

        With ``aux_footprint`` enabled, draw an estimated ground footprint per
        auxiliary box (dashed, to signal it is an estimate), suppressing any
        that would collide with a tracked-vehicle footprint or with an already
        drawn auxiliary footprint. Otherwise fall back to the ridge-predicted
        centre dots (paper Sec. 3.5).
        """
        aux_col = style.THEME["aux"]
        if self.aux_footprint:
            kept = list(track_polys)
            for d in aux_dets:
                aspect = self.aux_aspect.get(int(d.cls), 1.6)
                fp = self.viewport.estimate_aux_footprint(
                    d.xyxy, H, aspect, width_scale=self.aux_width_scale)
                if fp is None or not np.all(np.isfinite(fp)):
                    continue
                if any(convex_overlap_fraction(fp, p) > self.aux_overlap
                       for p in kept):
                    continue
                e0 = float(np.linalg.norm(fp[1] - fp[0]))
                e1 = float(np.linalg.norm(fp[2] - fp[1]))
                rc = rounded_contour(fp, min(e0, e1) * 0.30)
                draw_dashed_polyline(bev, rc, aux_col, thickness=1)
                kept.append(fp)
        elif len(aux_centers):
            # Auxiliary detections shown ONLY as their predicted ground-centre
            # (a clean marker), since the box->footprint estimate is unreliable.
            ac = (aux_col[2], aux_col[1], aux_col[0])
            for c in self.viewport.to_canvas(aux_centers, H):
                if not np.all(np.isfinite(c)):
                    continue
                p = (int(c[0]), int(c[1]))
                glow = bev.copy()
                cv2.circle(glow, p, 11, ac, -1, cv2.LINE_AA)
                cv2.addWeighted(glow, 0.18, bev, 0.82, 0, bev)
                cv2.circle(bev, p, 6, ac, 1, cv2.LINE_AA)       # outer ring
                cv2.circle(bev, p, 2, ac, -1, cv2.LINE_AA)      # centre dot


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="video path, image dir, or webcam index")
    p.add_argument("--kp-model", required=True, help="UrbanOmniDetect pose checkpoint")
    p.add_argument("--kp-imgsz", type=int, default=640)
    p.add_argument("--kp-conf", type=float, default=0.25)
    p.add_argument("--aux-model", default=DEFAULT_AUX_MODEL,
                   help=f"auxiliary COCO detector (default {DEFAULT_AUX_MODEL}); "
                        f"'none' to disable")
    p.add_argument("--aux-conf", type=float, default=0.30)
    p.add_argument("--homography", choices=["ortho", "adam", "none"], default="ortho",
                   help="BEV solver ('adam' is an alias for the fast LM "
                        "orthogonality solver kept for CLI compatibility)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--export", default="none",
                   choices=["none", "engine", "tensorrt", "onnx"])
    p.add_argument("--half", action="store_true", help="FP16 for the exported engine")
    p.add_argument("--ground-indices", default=None,
                   help="force ground keypoint indices, e.g. '0,1,2,3'")
    p.add_argument("--no-3d", action="store_true", help="draw only ground quads")
    p.add_argument("--no-snap-rect", action="store_true",
                   help="show raw (unsnapped) BEV footprints")
    p.add_argument("--aux-footprint", action="store_true",
                   help="estimate a BEV ground footprint per auxiliary box "
                        "(homography + class aspect prior) instead of a centre dot")
    p.add_argument("--no-aux-box", action="store_true",
                   help="do not draw the auxiliary 2D box on the camera panel "
                        "(auxiliary detections then appear only as a BEV centre marker)")
    p.add_argument("--aux-overlap", type=float, default=0.12,
                   help="drop an estimated aux footprint overlapping a vehicle "
                        "footprint by more than this fraction")
    p.add_argument("--aux-width-scale", type=float, default=0.9,
                   help="shrink factor for the estimated aux footprint width")
    p.add_argument("--aux-gate", action="store_true",
                   help="drop 3D detections the auxiliary detector cannot see "
                        "(removes hallucinations; requires an auxiliary model)")
    p.add_argument("--aux-gate-iou", type=float, default=0.3,
                   help="min IoU with an auxiliary box to keep a 3D detection")
    p.add_argument("--no-trails", action="store_true",
                   help="disable fading BEV motion trails")
    p.add_argument("--trail-len", type=int, default=16,
                   help="number of frames of BEV motion trail to keep")
    p.add_argument("--solver-ema", type=float, default=0.0,
                   help="optional temporal smoothing of the homography "
                        "parameters in [0,1); off by default (stability comes "
                        "from the footprint buffer and object-grounded scale)")
    p.add_argument("--no-stabilize", action="store_true",
                   help="solve the BEV homography per frame instead of over a "
                        "rolling footprint buffer (less stable axis)")
    p.add_argument("--bev-window", type=int, default=512,
                   help="rolling buffer size (footprints) for a stable BEV axis; "
                        "larger = steadier scale but more lag on camera motion")
    p.add_argument("--track-alpha", type=float, default=0.5)
    p.add_argument("--track-beta", type=float, default=0.08)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--output", default=None, help="write annotated mp4 here")
    p.add_argument("--display", action="store_true", help="show a live window")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--smooth", type=int, default=0,
                   help="non-causal offline smoothing window in frames (0 = off, "
                        "live streaming). Buffers the whole clip, then zero-phase "
                        "filters every track's pose and the homography for a "
                        "jitter-free render. Odd windows recommended (e.g. 11).")
    p.add_argument("--smooth-poly", type=int, default=2,
                   help="Savitzky-Golay polynomial order used by --smooth")
    p.add_argument("--sync", action="store_true",
                   help="render inline (no decoupling); simplest, lowest throughput")
    p.add_argument("--render-procs", type=int, default=1,
                   help="render on N separate PROCESSES for true (GIL-free) "
                        "decoupling so the core hits its native rate; 0 = use a "
                        "render thread instead")
    p.add_argument("--queue", type=int, default=4,
                   help="frames the core may run ahead of the renderer")
    return p


def open_source(spec: str):
    if spec.isdigit():
        return cv2.VideoCapture(int(spec))
    return cv2.VideoCapture(spec)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    gi = ([int(x) for x in args.ground_indices.split(",")]
          if args.ground_indices else None)

    pose = load_detector(args.kp_model, device=args.device, imgsz=args.kp_imgsz,
                         export=args.export, half=args.half)
    aux = None
    if args.aux_model and args.aux_model.lower() != "none":
        try:
            aux = load_detector(args.aux_model, device=args.device,
                                imgsz=args.kp_imgsz, export="none")
        except Exception as e:
            print(f"[warn] could not load auxiliary model {args.aux_model!r}: {e}")
    if args.aux_gate and aux is None:
        print("[warn] --aux-gate requires an auxiliary model; gating disabled")

    cap = open_source(args.input)
    if not cap.isOpened():
        raise SystemExit(f"could not open input: {args.input}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pipe = RealtimePipeline(
        pose, aux, kp_imgsz=args.kp_imgsz, kp_conf=args.kp_conf,
        aux_conf=args.aux_conf, device=args.device,
        ground_indices=gi, use_homography=(args.homography != "none"),
        show_3d=not args.no_3d, snap_rect=not args.no_snap_rect,
        solver_ema=args.solver_ema, aux_footprint=args.aux_footprint,
        aux_overlap=args.aux_overlap, aux_width_scale=args.aux_width_scale,
        aux_gate=args.aux_gate, aux_gate_iou=args.aux_gate_iou,
        draw_aux_box=not args.no_aux_box,
        trails=not args.no_trails, trail_len=args.trail_len,
        stabilize=not args.no_stabilize, bev_window=args.bev_window,
        tracker_kwargs=dict(alpha=args.track_alpha, beta=args.track_beta,
                            max_age=args.max_age, min_hits=args.min_hits,
                            conf_thresh=args.kp_conf))

    render_cfg = dict(
        kp_imgsz=args.kp_imgsz, use_homography=(args.homography != "none"),
        show_3d=not args.no_3d, snap_rect=not args.no_snap_rect,
        aux_footprint=args.aux_footprint, aux_overlap=args.aux_overlap,
        aux_width_scale=args.aux_width_scale, aux_gate=args.aux_gate,
        draw_aux_box=not args.no_aux_box,
        trails=not args.no_trails, trail_len=args.trail_len)

    if args.smooth and args.smooth >= 3:
        sink = _Sink(args, fps_in)
        n = _run_offline(pipe, cap, args, sink)
        sink.close()
    elif args.sync:
        sink = _Sink(args, fps_in)
        n = _run_sync(pipe, cap, args, sink)
        sink.close()
    elif args.render_procs and args.render_procs > 0:
        n = _run_process(pipe, cap, args, fps_in, render_cfg)
    else:
        sink = _Sink(args, fps_in)
        n = _run_threaded(pipe, cap, args, sink)
        sink.close()

    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"processed {n} frames")
    print(f"core {pipe.fps_core:.1f} fps  |  render {pipe.fps_render:.1f} fps")


class _Sink:
    """Lazily-opened video writer and/or display window."""

    def __init__(self, args, fps_in):
        self.args = args
        self.fps_in = fps_in
        self.path = args.output
        self.writer = None
        self.frames = 0

    def emit(self, out) -> bool:
        """Write/show one frame; returns False if the user asked to quit."""
        if self.path:
            if self.writer is None:
                h, w = out.shape[:2]
                self.writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps_in, (w, h))
            self.writer.write(out)
            self.frames += 1
        if self.args.display:
            cv2.imshow("UrbanOmniDetect BEV", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()


def _ema(prev, dt):
    f = 1.0 / dt if dt > 0 else 0.0
    return f if prev <= 0 else 0.9 * prev + 0.1 * f


def _run_sync(pipe, cap, args, sink) -> int:
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        out = pipe.process(frame)
        pipe.fps_core = pipe.fps_render = _ema(pipe.fps_core,
                                               time.perf_counter() - t0)
        n += 1
        if not sink.emit(out) or (args.max_frames and n >= args.max_frames):
            break
    return n


def _run_offline(pipe, cap, args, sink) -> int:
    """Two-pass NON-CAUSAL runner for the best-looking file render.

    Pass 1 runs the core (detect/track/solve) over the whole clip and buffers a
    :class:`RenderState` per frame. A zero-phase, outlier-robust filter then
    looks both forward and back to smooth every track's pose and the homography
    (see :mod:`uod.smoothing`). Pass 2 renders the cleaned states. This buffers
    all decoded frames in RAM, so it is for offline export, not live streaming.
    """
    from uod.smoothing import smooth_states

    states = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        states.append(pipe.step(frame))
        pipe.fps_core = _ema(pipe.fps_core, time.perf_counter() - t0)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    smooth_states(states, win=int(args.smooth), poly=int(args.smooth_poly),
                  smooth_H=(args.homography != "none"))

    for state in states:
        t0 = time.perf_counter()
        out = pipe.render(state)
        pipe.fps_render = _ema(pipe.fps_render, time.perf_counter() - t0)
        if not sink.emit(out):
            break
    return n


def _run_threaded(pipe, cap, args, sink) -> int:
    """Decoupled runner: the core (detect/track/solve) runs on a producer thread
    and the heavy renderer on the consumer (main) thread, overlapping GPU
    inference with CPU rendering. A bounded queue applies back-pressure so every
    frame is rendered (no drops) while each stage reports its own throughput."""
    q: "queue.Queue" = queue.Queue(maxsize=max(1, args.queue))
    stop = threading.Event()

    def producer():
        n = 0
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                state = pipe.step(frame)
                pipe.fps_core = _ema(pipe.fps_core, time.perf_counter() - t0)
                while not stop.is_set():
                    try:
                        q.put(state, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                n += 1
                if args.max_frames and n >= args.max_frames:
                    break
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    n = 0
    try:
        while True:
            try:
                state = q.get(timeout=1.0)
            except queue.Empty:
                if not th.is_alive():
                    break
                continue
            if state is None:
                break
            t0 = time.perf_counter()
            out = pipe.render(state)
            pipe.fps_render = _ema(pipe.fps_render, time.perf_counter() - t0)
            n += 1
            if not sink.emit(out):
                break
    finally:
        stop.set()
        # drain so a blocked producer can exit
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        th.join(timeout=2.0)
    return n


def _render_worker(cfg, in_q, fps_in, output_path, display, render_fps_val):
    """Render-only worker process: consumes RenderState, writes/shows frames.

    Runs in a separate process (spawn) so it does not contend with the core's
    Python work for the GIL. It builds a model-free render pipeline; rendering
    needs the viewport, trails, and style only -- never the detectors.
    """
    import time as _t
    from bev_realtime import RealtimePipeline, _ema  # re-import in child
    pipe = RealtimePipeline(None, None, **cfg)
    writer = None
    ema = 0.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    while True:
        state = in_q.get()
        if state is None:
            break
        pipe.fps_core = getattr(state, "fps_core", 0.0)
        pipe.fps_render = ema
        t0 = _t.perf_counter()
        out = pipe.render(state)
        ema = _ema(ema, _t.perf_counter() - t0)
        try:
            render_fps_val.value = ema
        except Exception:
            pass
        if output_path:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
            writer.write(out)
        if display:
            cv2.imshow("UrbanOmniDetect BEV", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()
    if display:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _run_process(pipe, cap, args, fps_in, render_cfg) -> int:
    """Decoupled runner with a separate render PROCESS (true parallelism).

    The core (this process) runs the detector/tracker/solver at its native rate
    and ships each RenderState to the render process over a bounded queue, which
    applies back-pressure so every frame is rendered (no drops)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue(maxsize=max(1, args.queue))
    render_fps = ctx.Value("d", 0.0)
    proc = ctx.Process(target=_render_worker,
                       args=(render_cfg, in_q, fps_in, args.output,
                             args.display, render_fps), daemon=True)
    proc.start()
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            state = pipe.step(frame)
            pipe.fps_core = _ema(pipe.fps_core, time.perf_counter() - t0)
            state.fps_core = pipe.fps_core
            in_q.put(state)            # blocks if renderer is behind (no drops)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        in_q.put(None)
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
    pipe.fps_render = render_fps.value
    return n


if __name__ == "__main__":
    main()
