# Command reference

All tools run from the repository root. The v2 checkpoints need `v2/hybrid_v2.py` importable
(`bev_realtime_v2.py` arranges that itself; in your own code `import v2.hybrid_v2` before
`YOLO(...)`). Never run Python from a directory that contains an `ultralytics/` source clone.

## `bev_realtime_v2.py` — v2 video pipeline

Detection + 3D cuboids from one v2 checkpoint, tracking, the orthogonality-homography
bird's-eye view, an optional offline refinement pass, and a rendered video. The v2 flags
below are handled by the adapter; everything else is passed to `bev_realtime.py` (next section).

```bash
python bev_realtime_v2.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_v2_x_640.pt \
    --kp-imgsz 960 --kp-conf 0.1 --kp-vis 0.5 --class-conf 0:0.4,1:0.5,3:0.5 --suppress-nested 0.85 \
    --smooth 11 --scale-lock --fixed-camera auto --bev-fit --refine --layout dashboard \
    --device cuda:0 --output out.mp4
```

| Flag | Default | Meaning |
|---|---|---|
| `--kp-vis F` | 0.5 | min per-corner cuboid confidence (3rd keypoint channel) to draw a cuboid; below it the object is tracked box-only |
| `--class-conf 'id:conf,...'` | none | per-class minimum confidence, COCO ids (`0:0.4,1:0.5,3:0.5` keeps traffic lights from becoming bikes) |
| `--suppress-nested F` | 0 (off) | drop a vehicle box nested >= F inside a larger vehicle box when their cuboid ground quads overlap; 0.85 recommended (one track per box truck) |
| `--layout side|dashboard` | side | `side`: camera and BEV side by side; `dashboard`: 16:9 composite with info strip (landscape sources) |
| `--bev-canvas WxH` | source size | BEV panel size (dashboard: 640x1080) |
| `--caption TEXT` | | scene caption in the UI; the static/moving camera tag is appended automatically |
| `--title s,in,hold,out` | | opening titles over this clip, timings in source seconds |
| `--object-px PX` / `--bev-fit` / `--bev-fit-min-px PX` / `--bev-fit-pct P` | 40 / off / 22 / 90 | BEV zoom: fixed px per median footprint, or auto-fit the clip's footprints (offline) |
| `--label-min-h PX` | 26 | no id chip on boxes shorter than this; the cuboid is still drawn |
| `--refine` | off | offline pass (`REFINE_README.md`): merge split ids, fill gaps, one cuboid decision per track, fill boxes, rigid footprint + trajectory heading, rigid cuboid rebuild, hold standing vehicles |
| `--no-fill-box` `--no-vel-heading` `--no-rigid-cuboids` `--no-dedupe` `--no-shape-smooth` `--no-world-lock` `--no-hold-still` | | switch off one refinement stage |
| `--refine-fps F` | from source | frame rate the refinement assumes (its thresholds are in seconds) |
| `--speed-ratio R` | 3.0 | cut a track instead of bridging a gap that needs R x its own top speed |
| `--max-reproj F` | 0.10 | reject a rebuilt cuboid above this reprojection error (fraction of the box diagonal) |
| `--max-gap N` `--min-fill F` `--min-track N` `--coast-tail N` `--merge-gap N` `--shape-win N` | 45 / 0.5 / 12 / 8 / 30 / 31 | refinement limits: longest interpolated gap, cuboid-hull fill below which a track is 2D-only, shortest kept track, coasting tail, re-spawn merge window, shape-smoothing window |
| `--labels PATH[.gz]` `--labels-from N` `--labels-source FILE` | | also write the drawn geometry as pseudo-labels (per frame and track: box, 8 corners, ground quad, ground-plane footprint, homography), from frame N on, recording the original file the input was scaled from |
| `--coast-mark` | off | mark coasting tracks with `?` |

Recommended on aerial footage: `--kp-imgsz 960` (or 1280). The models are trained at 640; a
larger input recovers far and small traffic and the cuboids stay tight.

### TensorRT

`--export tensorrt --half` exports an FP16 engine next to the checkpoint on first use (about
20 s for the x model) and loads it on every later run; the filename records imgsz and precision,
so an engine built for another resolution is never silently reused. `--export onnx` works the
same way. Requires **TensorRT 10.x** (`pip install "tensorrt>=10,<11"`) — 11.x removed
`BuilderFlag.FP16` and the Ultralytics exporter fails on it.

Loading an exported engine yourself needs the task passed explicitly:

```python
from ultralytics import YOLO
model = YOLO("urbanomnidetect_v2_x_640.engine", task="pose")   # without task=, keypoints are None
```

The exporter records the task as `detect` in the engine metadata, so without `task="pose"` the
pose predictor never runs: boxes still come back and `results.keypoints` is `None`, which is a
quiet failure. `load_detector(..., task="pose")` in `uod/model.py` does this for you.

Measured on the x model at 960 px, one RTX PRO 6000, 60 frames:

| | ms/frame | fps | cuboid corner agreement vs PyTorch |
|---|---|---|---|
| PyTorch FP32 | 7.7 | 130 | reference |
| TensorRT FP16 | 4.6 | 217 | median 0.9 px, p95 4.3 px |

End to end (detector + tracking + ground solve + refinement + render) the same clip runs at
48 fps with TensorRT against 45 with PyTorch: past this point the detector is not the bottleneck.

## `bev_realtime.py` — shared pipeline flags (v1 models directly, v2 through the adapter)

```
--input PATH|DIR|N          video path, image directory or webcam index
--kp-model PATH             checkpoint
--kp-imgsz N  --kp-conf F   inference size and confidence
--aux-model PATH|none       auxiliary COCO detector (v1 default yolo26x.pt; off by default in v2)
--aux-conf F  --aux-every N --aux-gate --aux-gate-iou F --aux-footprint --no-aux-box --aux-overlap F --aux-width-scale F
--homography ortho|adam|none   BEV solver
--device DEV  --export none|engine|tensorrt|onnx  --half
--ground-indices 0,1,2,3    force the ground keypoint indices (auto-detected by vote otherwise)
--no-3d  --no-snap-rect  --no-trails  --trail-len N  --no-fov
--solver-ema F  --no-stabilize  --bev-window N     homography smoothing / rolling footprint buffer
--track-alpha F  --track-beta F  --max-age N  --min-hits N  --no-cmc     tracker
--smooth N  --smooth-poly P   offline zero-phase smoothing of every track and the homography (odd N, e.g. 11; 0 = live)
--bev-freeze  --scale-lock  --fixed-camera auto|on|off   offline: world-locked viewport, constant radar scale, one global homography
--online                    causal streaming stabilisation (the live counterpart of --smooth)
--sync  --render-procs N  --queue N   rendering pipeline
--output PATH  --display  --max-frames N
```

## `ad/make_ad_v2.sh` — demo reel, `ad/make_labels.sh` — the same render as labels

```bash
bash ad/make_ad_v2.sh                        # every clip in test_videos/ -> _work/ad_v2/urbanomnidetect_v2_sequence.mp4
ORDER="03 05 14" SPEED=1.25 bash ad/make_ad_v2.sh
bash ad/make_labels.sh                       # -> _work/labels_v2/{labels/clip_NN.json.gz, clips/, vis/, index.json}
ORDER="$(seq -w 1 14)" bash ad/make_labels.sh
```

| Variable | Default | |
|---|---|---|
| `ROOT` / `PKG` / `PY` | repo / repo / `python` | data root, python package dir, interpreter |
| `KP` | `checkpoints/urbanomnidetect_v2_x_640.pt` | checkpoint |
| `VIDDIR` / `EXTRA` / `OUT` | `test_videos` / `assets/drone1hq.mp4` / `_work/ad_v2` | sources, one extra clip, output tree |
| `KPIMG` / `KPCONF` / `CLASSCONF` / `EXTRA_ARGS` | 960 / 0.1 / `0:0.4,1:0.5,3:0.5` / `--suppress-nested 0.85` | detector settings |
| `SMOOTH` / `FIXEDCAM` / `DEVICE` | 11 / auto / cuda:0 | |
| `ORDER` / `FRAMES` / `FRAMES_FIRST` / `TRIM` / `SPEED` / `XFADE` / `FPS` | reel order / 210 / 390 / 2 s / 1.5x / 0.4 s / 30 | the cut |
| `SRCW` / `LABELMINH` / `LABELMINH_P` / `BEVMINPX` | 1408 / 26 / 40 / 14 | layout |
| `MUSIC` / `TITLE_T` / `END_T` | `$OUT/*.mp3` / 3.5 s / 5 s | music, title and end card |

Clip indices are assigned alphabetically on the first run and frozen in `$OUT/manifest.tsv`;
captions are the `CAPTION` map at the top of each script. Requirements: `ffmpeg` with libx264
and an AAC encoder on `PATH`; `qrcode` for the end card; Liberation fonts optional.

## Label file format (`ad/make_labels.sh`, `--labels`)

`index.json` lists the clips (source, processed size, `scale` back to the original, `first_frame`).
Each `labels/clip_NN.json.gz` has `frames[]` with `i` (source frame), `H` (image -> ground
homography), `bev_unit` (median vehicle footprint, the object-length unit), `tracks[]`
(`id, cls, name, box [x1,y1,x2,y2], kpts [[x,y]]*8 or null, ground [[x,y]]*4, footprint` on the
ground plane, `observed`, `kp_ok`) and `boxes_2d[]` (detections without a usable cuboid).
Coordinates are in processed pixels; multiply by `scale` to reach the original file. Keypoints
0-3 are the ground corners, 4-7 the roof (roof i+4 above ground i).

## Training (v2)

```bash
python v2/stage_data.py --out _work/v2data                     # COCO + VisDrone + keypoint data -> 29-column labels
DEVICE=0,1 EPOCHS=50 bash v2/scripts/train_ladder_640.sh x l   # runs/v2/v2<scale>_640, every epoch saved
python v2/scripts/select_best.py runs/v2/v2x_640               # weights/best_pose.pt (road-class pose mAP), best_reel.pt
bash v2/scripts/heal_cls.sh x runs/v2/v2x_640/weights/best_pose.pt 0   # class branches only, on v2/v2_real_cls.yaml
python -m v2.train_v2 --help
```

Environment knobs read by `v2/hybrid_v2.py`: `V2_STRUCT_GAIN` (0.5), `V2_STRUCT_UNLABELED` (0:
prior on annotated instances only; 1 reproduces the collapsing recipe), `V2_STRUCT_UNLABELED_EPOCHS`,
`V2_KPT_REP` (`0:3,1:4,3:3` class-balanced keypoint loss), `V2_CONTAIN_GAIN`, `V2_KPT_L1_GAIN`,
`V2_TRAIN_ONLY` (`cv3,one2one_cv3` for the class heal), `V2_OOD_REEL` (labels dir from
`ad/make_labels.sh`; scores every saved epoch into `<run>/ood_reel.csv`), `V2_GRAYSCALE_P`
(needs `patches/ultralytics-random-grayscale.patch`).

## v1 tools

```bash
python draw_bev.py --image frame.jpg --kp-model checkpoints/urbanomnidetect_yolo11x-p2_1920.pt --kp-imgsz 1920 --mode both --device cuda:0
python bev_realtime.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_yolo11x-p2_640.pt --kp-imgsz 640 --aux-model yolo26x.pt --device cuda:0 --output bev.mp4
python sahi_tracker.py --input infra.mp4 --kp-model checkpoints/urbanomnidetect_yolo11x-p2_640.pt --slice 640 --overlap 0.25 --kp-imgsz 640 --device cuda:0 --output out.mp4
python eval3d.py --help
```

v1 checkpoints have 3 classes and `[8, 2]` keypoints with corners 0-3 on top and 4-7 on the
ground; the pipeline detects the ground half by vote, so both generations run unchanged.
