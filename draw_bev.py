#!/usr/bin/env python3
"""Standalone bird's-eye-view reconstruction for a single image.

Runs the UrbanOmniDetect pose model (and an optional auxiliary COCO detector),
estimates the orthogonality homography from the ground-contact keypoints, and
produces two outputs:

* a publication-quality two-panel Matplotlib figure (input + BEV), and
* a lightweight OpenCV BEV image.

Example
-------
    python draw_bev.py --image frame.jpg \\
        --kp-model UrbanOmniDetect/checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \\
        --kp-imgsz 1920 --mode both --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homography_rt import OrthoHomographySolver
from uod.aux_head import AuxCenterRegressor
from uod.bev import BEVViewport, fit_min_area_rect
from uod.keypoints import (GroundIndexResolver, parse_boxes_result,
                           parse_pose_result)
from uod.model import DEFAULT_AUX_MODEL, load_detector
from uod.tracking import _iou_matrix
from uod import style


def run(image_path, kp_model, *, aux_model=None, device="cpu", kp_imgsz=1280,
        conf=0.1, aux_conf=0.3, ground_indices=None, snap_rect=True,
        out_prefix="bev"):
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"could not read image: {image_path}")
    H_img, W_img = image.shape[:2]

    pose = load_detector(kp_model, device=device, imgsz=kp_imgsz)
    res = pose.predict(image, imgsz=kp_imgsz, conf=conf, device=device,
                       verbose=False)[0]

    resolver = GroundIndexResolver(forced=ground_indices, vote_frames=1)
    if res.keypoints is not None and res.keypoints.data is not None:
        resolver.observe(res.keypoints.data.cpu().numpy()[:, :, :2])
    gi = resolver.indices(8)
    dets = parse_pose_result(res, gi)

    # Auxiliary detector + per-image ridge map to ground centres.
    aux_dets, aux_centers = [], np.zeros((0, 2))
    aux = AuxCenterRegressor(lam=1.0)
    anchors = [(d.xyxy, d.ground.mean(axis=0)) for d in dets
               if d.ground.shape == (4, 2)]
    if aux_model and len(anchors) >= 2:
        try:
            am = load_detector(aux_model, device=device, imgsz=kp_imgsz)
            ares = am.predict(image, imgsz=kp_imgsz, conf=aux_conf,
                              device=device, verbose=False)[0]
            aux_dets = parse_boxes_result(ares, conf_thresh=aux_conf)
            if aux_dets:
                ab = np.array([d.xyxy for d in aux_dets])
                tb = np.array([d.xyxy for d in dets])
                iou = _iou_matrix(ab, tb)
                aux_dets = [d for i, d in enumerate(aux_dets)
                            if iou[i].max(initial=0.0) < 0.45]
            aux.fit(np.array([a[0] for a in anchors]),
                    np.array([a[1] for a in anchors]))
            if aux_dets:
                aux_centers = aux.predict(np.array([d.xyxy for d in aux_dets]))
        except Exception as e:
            print(f"[warn] auxiliary model failed: {e}")

    quads = [d.ground for d in dets if d.ground.shape == (4, 2)]
    solver = OrthoHomographySolver(warm_start=False)
    H, info = solver.solve(quads)
    print(f"{len(dets)} detections, {len(quads)} ground footprints; "
          f"H loss {info.loss:.3e} in {info.iterations} iters")

    viewport = BEVViewport(canvas_size=(W_img, H_img))
    # Pin BEV scale to the objects (median footprint size) for consistency.
    unit = 0.0
    if quads:
        from bev_realtime import RealtimePipeline
        classes = [d.cls for d in dets if d.ground.shape == (4, 2)]
        _, unit = RealtimePipeline._class_footprint_dims(quads, classes, H)
    viewport.update(H, (W_img, H_img), unit=unit)

    # ---- OpenCV BEV ----
    bev = viewport.blank_canvas(radar=True)
    for i, d in enumerate(dets):
        if d.ground.shape != (4, 2):
            continue
        col = style.instance_color(d.cls, i, source="pose")
        viewport.draw_footprint(bev, d.ground, H, col, snap_rect=snap_rect,
                                fill_alpha=0.3)
    if len(aux_centers):
        ac = style.bgr(style.THEME["aux"])
        for c in viewport.to_canvas(aux_centers, H):
            if np.all(np.isfinite(c)):
                cv2.circle(bev, (int(c[0]), int(c[1])), 6, ac, 2, cv2.LINE_AA)
    viewport.draw_camera(bev)
    cv2_path = f"{out_prefix}_opencv.jpg"
    cv2.imwrite(cv2_path, bev)
    print(f"wrote {cv2_path}")

    return image, dets, aux_dets, aux_centers, H, viewport, gi


def save_figure(image, dets, aux_dets, aux_centers, viewport, H, gi,
                out_path="bev_figure.png", dpi=200, snap_rect=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: input + ground footprints + keypoints
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("(a) Input with ground keypoints", fontsize=13, fontweight="bold")
    ax1.axis("off")
    for i, d in enumerate(dets):
        c = [v / 255.0 for v in style.instance_color(d.cls, i, source="pose")]
        if d.ground.shape == (4, 2):
            ax1.add_patch(MplPoly(d.ground, closed=True, fill=False,
                                  edgecolor=c, linewidth=2))
            for (x, y) in d.ground:
                ax1.plot(x, y, "o", color=c, markersize=5)
    for d in aux_dets:
        x1, y1, x2, y2 = d.xyxy
        ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=(255/255,196/255,92/255), linewidth=2,
                                    linestyle="--"))

    # Panel B: BEV (vector polygons on a radar field)
    ax2.set_title("(b) Calibration-free bird's-eye view", fontsize=13, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.set_facecolor("#FAFAFA")
    cx, cy = viewport.camera_canvas
    r_max = viewport.H - 2 * viewport.margin
    for frac in (0.25, 0.5, 0.75, 1.0):
        ax2.add_patch(plt.Circle((cx, cy), r_max * frac, fill=False,
                                 edgecolor="#DDDDDD", linewidth=1))
    for i, d in enumerate(dets):
        if d.ground.shape != (4, 2):
            continue
        col = [v / 255.0 for v in style.instance_color(d.cls, i, source="pose")]
        poly = viewport.to_canvas(d.ground, H)
        if snap_rect:
            poly = fit_min_area_rect(poly)
        ax2.add_patch(MplPoly(poly, closed=True, facecolor=col, alpha=0.35,
                              edgecolor=col, linewidth=2))
    if len(aux_centers):
        for c in viewport.to_canvas(aux_centers, H):
            ax2.plot(c[0], c[1], "o", markerfacecolor="none",
                     markeredgecolor=(255/255,196/255,92/255), markersize=10, markeredgewidth=1.5)
    ax2.plot(cx, cy, "ks", markersize=8)
    ax2.annotate("CAM", (cx, cy), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=9)
    ax2.set_xlim(0, viewport.W)
    ax2.set_ylim(viewport.H, 0)   # image-like: camera at bottom
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--kp-model", required=True)
    p.add_argument("--kp-imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--aux-model", default=DEFAULT_AUX_MODEL,
                   help=f"auxiliary detector (default {DEFAULT_AUX_MODEL}); 'none' to disable")
    p.add_argument("--aux-conf", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=["figure", "opencv", "both"], default="both")
    p.add_argument("--ground-indices", default=None)
    p.add_argument("--no-snap-rect", action="store_true")
    p.add_argument("--out-prefix", default="bev")
    args = p.parse_args(argv)

    gi = ([int(x) for x in args.ground_indices.split(",")]
          if args.ground_indices else None)
    aux_model = None if (not args.aux_model or args.aux_model.lower() == "none") else args.aux_model

    image, dets, aux_dets, aux_centers, H, viewport, gi_used = run(
        args.image, args.kp_model, aux_model=aux_model, device=args.device,
        kp_imgsz=args.kp_imgsz, conf=args.conf, aux_conf=args.aux_conf,
        ground_indices=gi, snap_rect=not args.no_snap_rect,
        out_prefix=args.out_prefix)

    if args.mode in ("figure", "both"):
        save_figure(image, dets, aux_dets, aux_centers, viewport, H, gi_used,
                    out_path=f"{args.out_prefix}_figure.png",
                    snap_rect=not args.no_snap_rect)


if __name__ == "__main__":
    main()
