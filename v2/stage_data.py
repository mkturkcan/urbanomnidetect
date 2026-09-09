#!/usr/bin/env python3
"""Stage COCO + VisDrone + pose_dataset into one unified v2 pose dataset.

Writes a single dataset under OUT (default _work/v2data) whose every label line
is in the 29-column pose format ``cls cx cy w h (x y v)*8`` so the stock
ultralytics pose loader reads all three sources uniformly:

  * COCO     -> COCO class ids unchanged; segment polygons reduced to boxes;
               keypoints zero-padded with visibility 0 (bbox-only).
  * VisDrone -> already COCO class ids ({0,1,2,3,5,7}); boxes; v=0.
  * pose     -> remapped car/person/bike (0/1/2) -> COCO car/person/bicycle
               (2/0/1); 8 keypoints with visibility 1.

Images are symlinked (no copies); labels are small text. Non-destructive: the
source datasets are never modified. Build the train/val image lists too.

Usage:
  python stage_data.py [--limit N] [--out DIR]
``--limit`` stages only the first N images per (source, split) for a fast
end-to-end validation before the full run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("V2_DATA_ROOT", "/data/datasets"))
POSE_CLASS_MAP = {0: 2, 1: 0, 2: 1}    # pose car/person/bike -> COCO car/person/bicycle
NKPT = 8


def _box_from_tokens(t: np.ndarray) -> tuple:
    """Return normalized (cx, cy, w, h) from a label line's post-class tokens.

    5-col box lines pass through; segment polygons (>4 coords) reduce to their
    axis-aligned bounding box.
    """
    if len(t) == 4:
        cx, cy, w, h = t
    else:                                  # polygon: cls x1 y1 x2 y2 ...
        xy = t.reshape(-1, 2)
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return float(cx), float(cy), float(w), float(h)


def _convert_line(line: str, class_map: dict | None, has_kpts: bool):
    """Convert one source label line to a 29-column v2 pose line (or None)."""
    p = line.split()
    if len(p) < 5:
        return None
    cls = int(float(p[0]))
    if class_map is not None:
        if cls not in class_map:
            return None
        cls = class_map[cls]
    vals = np.array(p[1:], dtype=np.float64)
    if has_kpts:
        cx, cy, w, h = vals[:4]
        kp = vals[4:4 + NKPT * 2].reshape(-1, 2)
        kpts = [f"{x:.6f} {y:.6f} 1" for x, y in kp]
    else:
        cx, cy, w, h = _box_from_tokens(vals)
        kpts = ["0 0 0"] * NKPT
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kpts)


def _img_to_label(img: Path) -> Path:
    s = str(img)
    return Path(s.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
               ).with_suffix(".txt")


def stage_source(name, images, class_map, has_kpts, out: Path, limit=0):
    """Stage one (source, split): symlink images, write v2 labels, return paths."""
    img_dir = out / "images" / name
    lab_dir = out / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    n_inst = n_kpt_inst = 0
    if limit:
        images = images[:limit]
    for img in images:
        lab = _img_to_label(img)
        if not img.exists():
            continue
        lines = []
        if lab.exists():
            for ln in lab.read_text().splitlines():
                c = _convert_line(ln, class_map, has_kpts)
                if c is not None:
                    lines.append(c)
                    n_inst += 1
                    n_kpt_inst += int(has_kpts)
        # symlink image under a flat, unique name (source prefix avoids clashes)
        dst_img = img_dir / img.name
        if not dst_img.exists():
            os.symlink(img.resolve(), dst_img)
        (lab_dir / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        # List the SYMLINK path (under v2data/images) so img2label maps to the
        # staged v2 labels; do NOT resolve() it back to the source path.
        staged.append(str(dst_img))
    print(f"  {name}: {len(staged)} imgs, {n_inst} inst ({n_kpt_inst} w/ kpts)")
    return staged


def _list_dir(d: Path):
    return sorted([p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])


def _list_autosplit(txt: Path):
    out = []
    for ln in txt.read_text().splitlines():
        ln = ln.strip().lstrip("./")
        if ln:
            out.append((DATA / ln))
    return out


def _staged_imgs(out: Path, name: str):
    d = out / "images" / name
    return sorted(str(p) for p in d.glob("*")
                  if p.suffix.lower() in {".jpg", ".png", ".jpeg"}) if d.exists() else []


def _compose_train(per_source: dict, pose_frac: float):
    """Compose the train list, oversampling pose images to ``pose_frac``.

    Repeating pose image paths (rather than a custom sampler) keeps the stock
    DistributedSampler, so balancing is identical and safe single- or multi-GPU.
    """
    nonpose = list(per_source.get("coco_train", [])) + list(per_source.get("visdrone_train", []))
    pose = list(per_source.get("pose_train", []))
    rep = 1
    if pose and 0.0 < pose_frac < 1.0:
        rep = max(1, round(pose_frac * len(nonpose) / ((1.0 - pose_frac) * len(pose))))
    train = nonpose + pose * rep
    ach = (len(pose) * rep) / max(len(train), 1)
    print(f"train: {len(nonpose)} non-pose + {len(pose)}x{rep} pose = {len(train)} "
          f"(pose fraction {ach:.2f})")
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_work/v2data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pose-frac", type=float, default=0.33,
                    help="target fraction of pose images in the train list (oversample)")
    ap.add_argument("--lists-only", action="store_true",
                    help="rebuild train/val txt from already-staged dirs (fast; no re-staging)")
    args = ap.parse_args()
    out = Path(args.out)

    per = {}
    if args.lists_only:
        for n in ("coco_train", "coco_val", "visdrone_train", "visdrone_val",
                  "pose_train", "pose_val"):
            per[n] = _staged_imgs(out, n)
    else:
        print("Staging COCO ...")
        per["coco_train"] = stage_source("coco_train", _list_dir(DATA / "coco/images/train2017"), None, False, out, args.limit)
        per["coco_val"] = stage_source("coco_val", _list_dir(DATA / "coco/images/val2017"), None, False, out, args.limit)
        print("Staging VisDrone ...")
        per["visdrone_train"] = stage_source("visdrone_train", _list_dir(DATA / "VisDrone/images/train"), None, False, out, args.limit)
        per["visdrone_val"] = stage_source("visdrone_val", _list_dir(DATA / "VisDrone/images/val"), None, False, out, args.limit)
        print("Staging pose_dataset ...")
        per["pose_train"] = stage_source("pose_train", _list_autosplit(DATA / "pose_dataset/images/autosplit_train.txt"), POSE_CLASS_MAP, True, out, args.limit)
        per["pose_val"] = stage_source("pose_val", _list_autosplit(DATA / "pose_dataset/images/autosplit_val.txt"), POSE_CLASS_MAP, True, out, args.limit)

    train = _compose_train(per, args.pose_frac)
    val = list(per.get("coco_val", [])) + list(per.get("visdrone_val", [])) + list(per.get("pose_val", []))
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    print(f"train.txt: {len(train)} | val.txt: {len(val)}  -> {out}")


if __name__ == "__main__":
    main()
