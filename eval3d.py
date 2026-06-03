#!/usr/bin/env python3
"""3D / BEV detection evaluation for UrbanOmniDetect pose models.

Computes AP at one or more IoU thresholds for three IoU flavours:

* ``2d``  -- axis-aligned image-box IoU
* ``bev`` -- IoU of the ground footprints after the orthogonality homography
* ``3d``  -- volumetric IoU of the vertical prisms whose bases are the BEV
            footprints and whose heights come from the keypoint vertical span

This is a cleaned-up, faster successor to the original ``yevalx`` script: the
homography is now the fast Levenberg-Marquardt orthogonality solver
(``homography_rt``) rather than a ~0.5 s Nelder-Mead run per image, ground
corners are resolved correctly per checkpoint, and the polygon maths is shared
with the runtime BEV code.

Labels are YOLO-pose text files: ``cls cx cy w h kp1x kp1y ... kp8x kp8y`` with
all values normalised to ``[0, 1]``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homography_rt import OrthoHomographySolver, apply_homography

try:
    from shapely.geometry import Polygon
    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False


@dataclass
class GTBox:
    cls: int
    kpts: np.ndarray   # (8, 2) pixel coords
    xyxy: np.ndarray   # (4,)
    conf: float = 1.0


def _ordered_polygon(corners: np.ndarray):
    c = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    poly = Polygon(corners[np.argsort(ang)])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.0


def iou_bev(g1: np.ndarray, g2: np.ndarray, H: Optional[np.ndarray]) -> float:
    if not _HAVE_SHAPELY:
        raise RuntimeError("BEV/3D IoU requires shapely")
    if H is not None:
        g1 = apply_homography(g1, H)
        g2 = apply_homography(g2, H)
    p1, p2 = _ordered_polygon(g1), _ordered_polygon(g2)
    if p1.area < 1e-9 or p2.area < 1e-9:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return float(np.clip(inter / union, 0.0, 1.0)) if union > 1e-9 else 0.0


def iou_3d(g1, t1, g2, t2, H) -> float:
    """Volumetric IoU using BEV footprints x image-space vertical extent."""
    if not _HAVE_SHAPELY:
        raise RuntimeError("BEV/3D IoU requires shapely")
    gg1, gg2 = (g1, g2)
    if H is not None:
        gg1 = apply_homography(g1, H); gg2 = apply_homography(g2, H)
    p1, p2 = _ordered_polygon(gg1), _ordered_polygon(gg2)
    if p1.area < 1e-9 or p2.area < 1e-9:
        return 0.0
    bev_inter = p1.intersection(p2).area
    if bev_inter < 1e-9:
        return 0.0
    # Heights from the vertical span between ground and top keypoints (image y).
    h1 = abs(g1[:, 1].mean() - t1[:, 1].mean())
    h2 = abs(g2[:, 1].mean() - t2[:, 1].mean())
    if h1 < 1e-9 or h2 < 1e-9:
        return 0.0
    # Both prisms rest on the ground plane (base z=0), so the vertical overlap
    # of [0,h1] and [0,h2] is simply min(h1, h2).
    h_inter = min(h1, h2)
    vol_inter = bev_inter * h_inter
    vol_union = p1.area * h1 + p2.area * h2 - vol_inter
    return float(np.clip(vol_inter / vol_union, 0.0, 1.0)) if vol_union > 1e-9 else 0.0


def parse_label(path: str, W: int, H: int) -> List[GTBox]:
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 21:
                continue
            cls = int(float(p[0]))
            # Use the labelled 2D box (cx,cy,w,h, normalised) for 2D IoU rather
            # than the keypoint extent, which would over-count the cuboid hull.
            cx, cy, bw, bh = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
            xyxy = np.array([(cx - bw / 2) * W, (cy - bh / 2) * H,
                             (cx + bw / 2) * W, (cy + bh / 2) * H])
            kp = np.array([float(v) for v in p[5:21]], dtype=np.float64).reshape(8, 2)
            kp[:, 0] *= W
            kp[:, 1] *= H
            boxes.append(GTBox(cls=cls, kpts=kp, xyxy=xyxy))
    return boxes


def resolve_ground(kpts_list: Sequence[np.ndarray],
                   forced: Optional[Sequence[int]]) -> List[int]:
    if forced:
        return list(forced)
    if not kpts_list:
        return [0, 1, 2, 3]
    arr = np.stack(kpts_list)
    half = arr.shape[1] // 2
    second_lower = np.mean(arr[:, half:, 1].mean(1) > arr[:, :half, 1].mean(1))
    return list(range(half, arr.shape[1])) if second_lower > 0.5 else list(range(half))


def solve_image_homographies(preds, gts, ground_idx, solver):
    """One orthogonality homography per image from all (pred+gt) footprints.

    Solved once and shared by the bev/3d flavours and every IoU threshold.
    """
    homs = []
    for pr, gt in zip(preds, gts):
        quads = [b.kpts[ground_idx] for b in (pr + gt)]
        solver.reset()
        H = solver.solve(quads)[0] if quads else None
        homs.append(H)
    return homs


def compute_ap(preds, gts, iou_thr, iou_type, ground_idx, homographies=None):
    """11-point AP for a single IoU threshold and flavour.

    ``homographies`` is the precomputed per-image list (ignored for ``2d``).
    """
    flat = []
    for img_idx, pr in enumerate(preds):
        for b in pr:
            flat.append((b.conf, img_idx, b))
    flat.sort(key=lambda x: x[0], reverse=True)
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return 0.0
    tp = np.zeros(len(flat)); fp = np.zeros(len(flat))
    used = [set() for _ in gts]
    for k, (_, img_idx, pred) in enumerate(flat):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts[img_idx]):
            if j in used[img_idx] or gt.cls != pred.cls:
                continue
            if iou_type == "2d":
                v = iou_2d(pred.xyxy, gt.xyxy)
            elif iou_type == "bev":
                v = iou_bev(pred.kpts[ground_idx], gt.kpts[ground_idx],
                            homographies[img_idx])
            else:
                top = [i for i in range(8) if i not in ground_idx]
                v = iou_3d(pred.kpts[ground_idx], pred.kpts[top],
                           gt.kpts[ground_idx], gt.kpts[top], homographies[img_idx])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[k] = 1; used[img_idx].add(best_j)
        else:
            fp[k] = 1
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        ap += prec[m].max() if m.any() else 0.0
    return ap / 11.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="directory of images")
    p.add_argument("--labels", required=True, help="directory of YOLO-pose labels")
    p.add_argument("--model", required=True)
    p.add_argument("--kp-imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--iou", default="0.5,0.7", help="comma-separated thresholds")
    p.add_argument("--ground-indices", default=None)
    p.add_argument("--max-images", type=int, default=0)
    args = p.parse_args(argv)

    import cv2
    from ultralytics import YOLO
    from uod.keypoints import parse_pose_result, GroundIndexResolver

    thresholds = [float(x) for x in args.iou.split(",")]
    forced = ([int(x) for x in args.ground_indices.split(",")]
              if args.ground_indices else None)
    model = YOLO(args.model)
    resolver = GroundIndexResolver(forced=forced, vote_frames=10)

    imgs = sorted(sum([glob.glob(os.path.join(args.images, e))
                       for e in ("*.jpg", "*.jpeg", "*.png")], []))
    if args.max_images:
        imgs = imgs[:args.max_images]

    preds_all, gts_all, raw_kpts = [], [], []
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        lp = os.path.join(args.labels, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        gt = parse_label(lp, W, H)
        res = model.predict(img, imgsz=args.kp_imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data):
            kp = res.keypoints.data.cpu().numpy()[:, :, :2]
            resolver.observe(kp)
            raw_kpts.extend(list(kp))
        dets = parse_pose_result(res, list(range(8)))  # keep all 8 kpts
        pr = [GTBox(cls=d.cls, kpts=d.kpts, xyxy=d.xyxy, conf=d.conf)
              for d in dets if len(d.kpts) >= 8]
        preds_all.append(pr)
        gts_all.append(gt)

    ground_idx = resolve_ground(raw_kpts, forced)
    print(f"evaluated {len(imgs)} images; ground indices = {ground_idx}")
    # Solve the per-image homographies once; reused for bev/3d at every threshold.
    homs = None
    if _HAVE_SHAPELY:
        solver = OrthoHomographySolver(warm_start=False)
        homs = solve_image_homographies(preds_all, gts_all, ground_idx, solver)
    print(f"{'IoU':>6} {'AP2D':>8} {'APBEV':>8} {'AP3D':>8}")
    for thr in thresholds:
        ap2 = compute_ap(preds_all, gts_all, thr, "2d", ground_idx)
        apb = (compute_ap(preds_all, gts_all, thr, "bev", ground_idx, homs)
               if _HAVE_SHAPELY else float("nan"))
        ap3 = (compute_ap(preds_all, gts_all, thr, "3d", ground_idx, homs)
               if _HAVE_SHAPELY else float("nan"))
        print(f"{thr:>6.2f} {ap2:>8.4f} {apb:>8.4f} {ap3:>8.4f}")


if __name__ == "__main__":
    main()
