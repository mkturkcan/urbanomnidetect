#!/usr/bin/env python3
"""Launcher for UrbanOmniDetect v2 hybrid detect+pose training.

Run as a module so the trainer class is importable in DDP subprocesses:
    cd code/urbanomnidetect
    python -m v2.train_v2 --device 0          # single GPU
    python -m v2.train_v2 --device 0,1        # multi-GPU DDP

The heavy lifting (masked hybrid loss, detection-weight transfer) lives in
``v2.hybrid_v2``; this only assembles the training overrides.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Make the `v2` package importable in this process AND in the DDP subprocesses
# ultralytics spawns (they run a temp file that does `from v2.hybrid_v2 import
# HybridPoseTrainer` and inherit this env), so multi-GPU works without hacks.
_PKG_PARENT = str(HERE.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_PKG_PARENT, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "v2_hybrid.yaml"))
    ap.add_argument("--model", default=str(HERE / "yolo26x-pose-v2.yaml"))
    ap.add_argument("--pretrained", default="yolo26x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="v2_hybrid")
    ap.add_argument("--project", default="runs/v2")
    # --- Augmentation profile (strong, for keypoint robustness) --------------
    # Geometric: a perspective warp is projective, so it maps a valid projected
    # cuboid to a valid projected cuboid and leaves the concurrency structure prior
    # consistent. Rotation is orientation-preserving, so the 8 corner labels stay
    # correct at ANY angle (no flip_idx remap needed -- that is only for
    # reflections), hence degrees can be sizable. scale=0.5/translate=0.1 stay at
    # ultralytics defaults: raising scale erases VisDrone's dominant tiny objects
    # (box_candidates drops sub-2px boxes).
    ap.add_argument("--perspective", type=float, default=0.0005)
    ap.add_argument("--degrees", type=float, default=15.0)
    # Shear is the affine skew DoF that rotation/scale don't cover, and is affine so
    # it preserves parallelism/concurrency -> sheared cuboids stay valid (structure
    # prior consistent). Probe: shear up to 5 adds ~0 tiny-object cost over the rest
    # of the strong profile and 0% extra demotion (shear 10 starts to cost ~2%).
    ap.add_argument("--shear", type=float, default=5.0)
    # Color: large hsv_h swings hue by +/-(hsv_h*180) to simulate any vehicle color;
    # grayscale occasionally drops a sample to monochrome (env var, read by our
    # RandomGrayscale in ultralytics v8_transforms; image-only so keypoint-safe).
    ap.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.5)
    ap.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.9)
    ap.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.5)
    ap.add_argument("--grayscale", type=float, default=0.05,
                    help="prob of converting a sample to grayscale (sets V2_GRAYSCALE_P)")
    # Multi-scale varies the network input size (peak imgsz = imgsz*(1+multi_scale))
    # for scale robustness; it RAISES peak memory, so OFF by default (the default
    # model is x-scale -- an OOM trap otherwise). For the n run enable it with a
    # smaller batch, e.g.  --multi-scale 0.5 --batch 32.
    ap.add_argument("--multi-scale", dest="multi_scale", type=float, default=0.0)
    ap.add_argument("--extra", default="", help="comma key=val overrides, e.g. lr0=0.001,close_mosaic=10")
    args = ap.parse_args()

    # RandomGrayscale reads this; set before the trainer import so DDP subprocesses
    # (which inherit os.environ at spawn) see it too.
    os.environ["V2_GRAYSCALE_P"] = str(args.grayscale)

    from v2.hybrid_v2 import HybridPoseTrainer

    overrides = dict(
        task="pose",
        model=args.model,
        data=args.data,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        perspective=args.perspective,
        degrees=args.degrees,
        shear=args.shear,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        multi_scale=args.multi_scale,
    )
    for kv in filter(None, args.extra.split(",")):
        k, v = kv.split("=", 1)
        if v in ("True", "False"):
            v = v == "True"
        elif v == "None":
            v = None
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        overrides[k] = v

    trainer = HybridPoseTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
