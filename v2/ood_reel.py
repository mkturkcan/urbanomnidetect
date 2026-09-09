"""Deployment-proxy score for a checkpoint: the demo-reel clips, scored against the
reel's refined labels (urbanomnidetect-reel-labels).

Those labels are model output that was watched frame by frame, not ground truth, but
they are exactly the footage the model is judged on, and they separate the usable
epoch-1 checkpoints from the collapsed epoch-100 ones where val pose mAP does not.
48 frames (6 per clip), so it is cheap enough to run after every epoch.

Scored quantities, all on road-user detections at deployment settings (imgsz 640,
conf 0.1, frames pre-scaled like the reel):
  corner_err          mean corner distance to the matched label cuboid / label box diagonal
  corner_err_yawfree  same, allowing a 180-degree yaw (front/back is ambiguous from above)
  hull_box_med        median cuboid-hull area / 2D box area (collapse => small)
  trust_rate          fraction passing the renderer's keypoint trust gate
  recall3d            fraction of labelled cuboids matched by a detection (IoU >= 0.5)
"""
from __future__ import annotations

import csv
import gzip
import json
import os

import cv2
import numpy as np

ROAD = {0, 1, 2, 3, 5, 7}
# 180-degree yaw: front and back faces swap, so the corners are permuted, not moved.
YAW180 = [2, 3, 0, 1, 6, 7, 4, 5]


def load_frames(reel_dir, per_clip=6):
    """(clip, source frame, image at processed size, [(box, kpts|None, cls)]) per frame."""
    idx = json.load(open(os.path.join(reel_dir, "index.json")))
    out = []
    for c in idx["clips"]:
        d = json.load(gzip.open(os.path.join(reel_dir, c["labels"])))
        src = d["source"]
        pw, ph = src["processed_wh"]
        frames = d["frames"]
        picks = np.linspace(0, len(frames) - 1, per_clip).round().astype(int)
        cap = cv2.VideoCapture(os.path.join(reel_dir, c["video"]))
        for j in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
            f = frames[int(j)]
            gts = [(np.array(t["box"], float),
                    np.array(t["kpts"], float) if t["kpts"] else None, int(t["cls"]))
                   for t in f["tracks"] if t["observed"]]
            out.append((c["clip"], int(f["i"]), fr, gts))
        cap.release()
    return out


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(model, frames, imgsz=640, conf=0.1, device=None, kp_vis=0.5):
    errs, errs_flip, hull = [], [], []
    trusted = n_pred = matched = n_gt3d = 0
    for _clip, _fi, img, gts in frames:
        gt3 = [(b, k) for b, k, _ in gts if k is not None]
        n_gt3d += len(gt3)
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        kd = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
        keep = [i for i in range(len(xyxy)) if cls[i] in ROAD]
        n_pred += len(keep)
        if kd is None or not keep:
            continue
        for i in keep:
            k, v = kd[i, :, :2], kd[i, :, 2]
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            hull.append(cv2.contourArea(cv2.convexHull(k.astype(np.float32).reshape(-1, 1, 2))) / (bw * bh))
            inside = ((k[:, 0] >= x1 - .75 * bw) & (k[:, 0] <= x2 + .75 * bw)
                      & (k[:, 1] >= y1 - .75 * bh) & (k[:, 1] <= y2 + .75 * bh)).all()
            trusted += int(np.isfinite(k).all() and v.min() >= kp_vis and inside
                           and (np.abs(k).sum(1) > 1e-6).all())
        if not gt3:
            continue
        pb = xyxy[keep]
        ious = np.array([[_iou(b, p) for p in pb] for b, _ in gt3])   # (G, P)
        used = set()
        for g in np.argsort(-ious.max(1)):
            for p in np.argsort(-ious[g]):
                if ious[g, p] < 0.5:
                    break
                if p in used:
                    continue
                used.add(p)
                matched += 1
                gb, gk = gt3[g]
                pk = kd[keep[p], :, :2]
                diag = max(np.hypot(gb[2] - gb[0], gb[3] - gb[1]), 1.0)
                e = np.linalg.norm(pk - gk, axis=1).mean() / diag
                ef = np.linalg.norm(pk[YAW180] - gk, axis=1).mean() / diag
                errs.append(e)
                errs_flip.append(min(e, ef))
                break
    nan = float("nan")
    return dict(frames=len(frames), n_pred=n_pred, gt3d=n_gt3d, matched=matched,
                recall3d=matched / max(n_gt3d, 1),
                corner_err=float(np.mean(errs)) if errs else nan,
                corner_err_yawfree=float(np.mean(errs_flip)) if errs_flip else nan,
                hull_box_med=float(np.median(hull)) if hull else nan,
                trust_rate=trusted / max(len(hull), 1))


def make_callback(reel_dir, per_clip=6):
    """``on_model_save`` callback: score ``last.pt`` (EMA weights) and append a row to
    ``<save_dir>/ood_reel.csv``. Frames are loaded once, lazily."""
    state = {}

    def cb(trainer):
        import torch
        from ultralytics import YOLO
        if "frames" not in state:
            state["frames"] = load_frames(reel_dir, per_clip)
        ep = trainer.epoch + 1
        try:
            s = score(YOLO(str(trainer.last)), state["frames"], imgsz=trainer.args.imgsz,
                      device=trainer.device)
        except Exception as e:  # never let the proxy metric kill a run
            print(f"[ood_reel] epoch {ep}: failed: {e!r}", flush=True)
            return
        finally:
            torch.cuda.empty_cache()
        p = trainer.save_dir / "ood_reel.csv"
        new = not p.exists()
        with open(p, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["epoch"] + list(s))
            w.writerow([ep] + [f"{v:.5g}" if isinstance(v, float) else v for v in s.values()])
        print(f"[ood_reel] epoch {ep}: corner_err={s['corner_err']:.4f} "
              f"(yaw-free {s['corner_err_yawfree']:.4f}) hull/box={s['hull_box_med']:.2f} "
              f"trust={s['trust_rate']:.2f} recall3d={s['recall3d']:.2f}", flush=True)
    return cb


if __name__ == "__main__":  # python -m v2.ood_reel <reel_dir> <ckpt> [<ckpt> ...]
    import sys
    from ultralytics import YOLO
    import v2.hybrid_v2  # noqa: F401  registers HybridPoseModel26 for unpickling
    frames = load_frames(sys.argv[1])
    for ck in sys.argv[2:]:
        s = score(YOLO(ck), frames, device=0)
        print(os.path.basename(os.path.dirname(os.path.dirname(ck))) + "/" + os.path.basename(ck),
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
