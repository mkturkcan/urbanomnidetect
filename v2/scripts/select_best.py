#!/usr/bin/env python3
"""Materialise the best checkpoints of a run by EXPLICIT criteria from its per-epoch
epochN.pt files (save_period=1; epochN.pt is 0-based, i.e. results.csv epoch N+1):
  weights/best_pose.pt  argmax metrics/road_pose_mAP50-95  (val, road classes only)
  weights/best_reel.pt  argmin corner_err_yawfree in ood_reel.csv (deployment proxy; raw corner_err is dominated by front/back flips)
Writes weights/BEST.txt with the chosen epochs. Safe to re-run any time; only copies
when the choice changed.   select_best.py [run_dir ...]  (default: all runs/v2/v2*_640)"""
import csv, glob, os, shutil, sys
R = "runs/v2"
runs = sys.argv[1:] or sorted(glob.glob(f"{R}/v2*_640"))
for d in runs:
    d = d.rstrip("/"); w = f"{d}/weights"; csvp = f"{d}/results.csv"; oodp = f"{d}/ood_reel.csv"
    if not os.path.exists(csvp): continue
    picks = {}
    rows = [r for r in csv.DictReader(open(csvp)) if r.get("metrics/road_pose_mAP50-95") not in (None, "", "nan")]
    if rows:
        b = max(rows, key=lambda r: float(r["metrics/road_pose_mAP50-95"]))
        picks["best_pose"] = (int(float(b["epoch"])), f"road_pose_mAP50-95={float(b['metrics/road_pose_mAP50-95']):.4f}")
    if os.path.exists(oodp):
        o = [r for r in csv.DictReader(open(oodp)) if r.get("corner_err_yawfree") not in (None, "", "nan")]
        if o:
            b = min(o, key=lambda r: float(r["corner_err_yawfree"]))
            picks["best_reel"] = (int(float(b["epoch"])), f"corner_err_yawfree={float(b['corner_err_yawfree']):.4f}")
    lines = []
    for name, (ep, why) in picks.items():
        src = f"{w}/epoch{ep - 1}.pt"; dst = f"{w}/{name}.pt"; tag = f"{w}/.{name}.epoch"
        if not os.path.exists(src):
            lines.append(f"{name}: epoch {ep} ({why}) -- epoch{ep-1}.pt MISSING"); continue
        cur = open(tag).read().strip() if os.path.exists(tag) else ""
        if cur != str(ep):
            shutil.copyfile(src, dst); open(tag, "w").write(str(ep))
        lines.append(f"{name}: epoch {ep} ({why}) -> {name}.pt")
    open(f"{w}/BEST.txt", "w").write("\n".join(lines) + "\n")
    print(os.path.basename(d) + ": " + "; ".join(lines))
