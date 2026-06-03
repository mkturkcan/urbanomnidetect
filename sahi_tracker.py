#!/usr/bin/env python3
"""SAHI-style sliced inference + tracking for high-resolution streams.

Infrastructure and aerial cameras produce many small, distant objects that a
single full-frame forward pass at a fixed ``imgsz`` under-detects. SAHI (Slicing
Aided Hyper Inference) runs the detector on overlapping tiles and merges the
results, trading compute for small-object recall.

This module provides :class:`SlicedPoseDetector`, which slices a frame, runs the
pose model per tile, shifts boxes *and* keypoints back into full-image
coordinates, and merges duplicates from overlapping tiles with a keypoint-aware
greedy NMS. It plugs straight into the real-time pipeline via ``predict_fn``, so
tracking, the orthogonality solver, and the stable BEV all work unchanged.

Example
-------
    python sahi_tracker.py \\
        --input infra.mp4 \\
        --kp-model UrbanOmniDetect/checkpoints/urbanomnidetect_yolo11x-p2_640.pt \\
        --slice 640 --overlap 0.25 --kp-imgsz 640 --device cuda:0 --output out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uod.keypoints import Detection, GroundIndexResolver, parse_pose_result
from uod.model import DEFAULT_AUX_MODEL, load_detector
from uod.tracking import _iou_matrix


def compute_slices(w: int, h: int, slice_wh: Tuple[int, int],
                   overlap: float) -> List[Tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` tiles covering the image with overlap.

    The last row/column is snapped to the image edge so coverage is complete
    regardless of how the slice size divides the frame.
    """
    sw, sh = slice_wh
    sw, sh = min(sw, w), min(sh, h)
    step_x = max(1, int(sw * (1.0 - overlap)))
    step_y = max(1, int(sh * (1.0 - overlap)))
    xs = list(range(0, max(1, w - sw + 1), step_x))
    ys = list(range(0, max(1, h - sh + 1), step_y))
    if not xs or xs[-1] != w - sw:
        xs.append(max(0, w - sw))
    if not ys or ys[-1] != h - sh:
        ys.append(max(0, h - sh))
    tiles = []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            tiles.append((x0, y0, min(x0 + sw, w), min(y0 + sh, h)))
    return tiles


def _greedy_nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    """Class-aware greedy NMS keeping the highest-confidence detection."""
    if len(dets) <= 1:
        return dets
    order = sorted(range(len(dets)), key=lambda i: dets[i].conf, reverse=True)
    boxes = np.array([dets[i].xyxy for i in order])
    cls = np.array([dets[i].cls for i in order])
    keep, suppressed = [], np.zeros(len(order), dtype=bool)
    for a in range(len(order)):
        if suppressed[a]:
            continue
        keep.append(order[a])
        rest = np.arange(a + 1, len(order))
        rest = rest[~suppressed[rest]]
        if len(rest) == 0:
            continue
        iou = _iou_matrix(boxes[a:a + 1], boxes[rest])[0]
        same = cls[rest] == cls[a]
        suppressed[rest[(iou >= iou_thresh) & same]] = True
    return [dets[i] for i in keep]


class SlicedPoseDetector:
    """Runs the pose model over overlapping tiles and merges the results."""

    def __init__(self, model, *, slice_wh=(640, 640), overlap=0.25,
                 kp_imgsz=640, conf=0.25, device="cpu", iou_merge=0.55,
                 resolver: Optional[GroundIndexResolver] = None,
                 edge_margin: int = 4, full_frame: bool = True,
                 batch: bool = True):
        self.model = model
        self.slice_wh = slice_wh
        self.overlap = overlap
        self.kp_imgsz = kp_imgsz
        self.conf = conf
        self.device = device
        self.iou_merge = iou_merge
        self.resolver = resolver or GroundIndexResolver()
        self.edge_margin = edge_margin
        self.full_frame = full_frame
        self.batch = batch

    def __call__(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        tiles = compute_slices(w, h, self.slice_wh, self.overlap)
        crops, origins = [], []
        for (x0, y0, x1, y1) in tiles:
            crops.append(frame[y0:y1, x0:x1])
            origins.append((x0, y0, x1, y1))
        if self.full_frame:
            crops.append(frame)
            origins.append((0, 0, w, h))

        # Run inference (batched if the backend supports a list input).
        if self.batch:
            results = self.model.predict(crops, imgsz=self.kp_imgsz,
                                         conf=self.conf, device=self.device,
                                         verbose=False)
        else:
            results = [self.model.predict(c, imgsz=self.kp_imgsz, conf=self.conf,
                                          device=self.device, verbose=False)[0]
                       for c in crops]

        # First pass: observe keypoints for ground-index resolution.
        for r in results:
            if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data):
                self.resolver.observe(r.keypoints.data.cpu().numpy()[:, :, :2])
        gi = self.resolver.indices(8)

        merged: List[Detection] = []
        for r, (x0, y0, x1, y1) in zip(results, origins):
            tw, th = x1 - x0, y1 - y0
            local = parse_pose_result(r, gi)
            is_tile = not (x0 == 0 and y0 == 0 and tw == w and th == h)
            for d in local:
                # Drop detections hugging an interior tile edge -- but only when
                # a full-frame pass is also running to catch the whole object.
                # Without it, edge-filtering could erase an object larger than a
                # tile from every tile it appears in, so we keep all and let NMS
                # dedup instead.
                if is_tile and self.full_frame:
                    bx0, by0, bx1, by1 = d.xyxy
                    touches = ((bx0 < self.edge_margin and x0 > 0) or
                               (by0 < self.edge_margin and y0 > 0) or
                               (bx1 > tw - self.edge_margin and x1 < w) or
                               (by1 > th - self.edge_margin and y1 < h))
                    if touches:
                        continue
                shift = np.array([x0, y0], dtype=np.float64)
                d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], dtype=np.float64)
                if d.kpts.size:
                    d.kpts = d.kpts + shift
                if d.ground.size:
                    d.ground = d.ground + shift
                merged.append(d)
        return _greedy_nms(merged, self.iou_merge)


def main(argv=None):
    from bev_realtime import RealtimePipeline, open_source

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--kp-model", required=True)
    p.add_argument("--kp-imgsz", type=int, default=640)
    p.add_argument("--kp-conf", type=float, default=0.25)
    p.add_argument("--slice", type=int, default=640, help="square slice size")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--no-full-frame", action="store_true",
                   help="skip the extra whole-frame pass")
    p.add_argument("--aux-model", default="none")
    p.add_argument("--aux-conf", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ground-indices", default=None)
    p.add_argument("--no-3d", action="store_true")
    p.add_argument("--no-snap-rect", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args(argv)

    gi = ([int(x) for x in args.ground_indices.split(",")]
          if args.ground_indices else None)
    pose = load_detector(args.kp_model, device=args.device, imgsz=args.kp_imgsz)
    aux = None
    if args.aux_model and args.aux_model.lower() != "none":
        aux = load_detector(args.aux_model, device=args.device, imgsz=args.kp_imgsz)

    resolver = GroundIndexResolver(forced=gi)
    sliced = SlicedPoseDetector(pose, slice_wh=(args.slice, args.slice),
                                overlap=args.overlap, kp_imgsz=args.kp_imgsz,
                                conf=args.kp_conf, device=args.device,
                                resolver=resolver,
                                full_frame=not args.no_full_frame)

    pipe = RealtimePipeline(
        pose, aux, kp_imgsz=args.kp_imgsz, kp_conf=args.kp_conf,
        aux_conf=args.aux_conf, device=args.device, ground_indices=gi,
        show_3d=not args.no_3d, snap_rect=not args.no_snap_rect,
        predict_fn=sliced)
    pipe.resolver = resolver  # share the resolver used by the slicer

    cap = open_source(args.input)
    if not cap.isOpened():
        raise SystemExit(f"could not open input: {args.input}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipe.process(frame)
        if args.output:
            if writer is None:
                hh, ww = out.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_in, (ww, hh))
            writer.write(out)
        if args.display:
            cv2.imshow("UrbanOmniDetect SAHI BEV", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {n} frames to {args.output}")
    if args.display:
        cv2.destroyAllWindows()
    if pipe.sw.t:
        print("mean stage ms:", {k: round(v, 2) for k, v in pipe.sw.t.items()})


if __name__ == "__main__":
    main()
