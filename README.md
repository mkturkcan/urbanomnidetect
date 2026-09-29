<p align="center">
  <h1 align="center">UrbanOmniDetect</h1>
  <h3 align="center">Calibration-Free View-Agnostic Monocular 3D Object Detection for Urban Scenes</h3>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=306TgWoAAAAJ">Mehmet Kerem Turkcan</a>&emsp;
    <a href="https://scholar.google.com/citations?user=GP7T1fgAAAAJ">Devika Gumaste</a>&emsp;
    <a href="https://scholar.google.com/citations?user=TlPI8yIAAAAJ">Zoran Kostic</a>
    <br/>
    <a href="https://www.aidl.ee.columbia.edu/">AIDL Lab</a>, <a href="https://www.ee.columbia.edu/">Department of Electrical Engineering</a>
    <br/>
    <b>Columbia University</b>
    <br/><br/>
    <a href="https://openaccess.thecvf.com/content/CVPR2026W/DriveX/papers/Turkcan_Calibration-Free_View-Agnostic_Monocular_3D_Object_Detection_for_Urban_Scenes_CVPRW_2026_paper.pdf"><img src="https://img.shields.io/badge/Paper-CVF%20Open%20Access-b31b1b.svg" alt="Paper"></a>
    <a href="https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniDetect--2-Models-blue" alt="UrbanOmniDetect-2 Models"></a>
    <a href="https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniDetect-Models-blue" alt="UrbanOmniDetect Models"></a>
    <a href="https://huggingface.co/datasets/mehmetkeremturkcan/UrbanOmniView"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniView-Dataset-green" alt="UrbanOmniView Dataset"></a>
    <a href="https://drivex-workshop.github.io/cvpr2026/"><img src="https://img.shields.io/badge/CVPR%202026-DriveX%20Workshop%20(Oral)-4b44ce.svg" alt="CVPR 2026 DriveX Workshop"></a>
    <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%203.0-orange.svg" alt="License"></a>
  </p>
  <p align="center">
    <b><a href="#urbanomnidetect-2">UrbanOmniDetect-2</a></b>, the current release, is developed and maintained by <a href="https://scholar.google.com/citations?user=306TgWoAAAAJ">Mehmet Kerem Turkcan</a>.
  </p>
</p>

<p align="center">
  <img src="https://github.com/mkturkcan/urbanomnidetect/raw/main/assets/urbanomniview.png" width="100%" alt="UrbanOmniView examples across ego-vehicle, infrastructure, and aerial viewpoints"/>
</p>

This repository contains two generations of calibration-free monocular 3D detection for urban scenes:

- **[UrbanOmniDetect-2](#urbanomnidetect-2)** (September 2026) is one network for COCO-80 detection and 3D cuboids, with tracking, a bird's-eye view, and offline trajectory refinement. Checkpoints are at [mehmetkeremturkcan/UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2).
- **[UrbanOmniDetect](#urbanomnidetect-cvpr-2026-drivex)** is the pose-only model from the CVPR 2026 DriveX paper, trained on the UrbanOmniView dataset. Checkpoints are at [mehmetkeremturkcan/UrbanOmniDetect](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect).

In file names and code, `v2` refers to UrbanOmniDetect-2 and `v1` to the paper model. The two sets of checkpoints are not interchangeable.

## UrbanOmniDetect-2

*Released September 2026. Developed and maintained by [Mehmet Kerem Turkcan](https://scholar.google.com/citations?user=306TgWoAAAAJ).*

UrbanOmniDetect-2 replaces the pose-only paper model with a single **hybrid detection and pose** network: a YOLO26 with a Pose26 head that is a full COCO-80 detector and, for every road user, also regresses the eight projected corners of its 3D box. It runs on any viewpoint without calibration, and the same forward pass feeds the tracker, the bird's-eye view, and the offline refinement that turns per-frame detections into rigid, physically consistent trajectories.

- **Detection and 3D cuboids from one forward pass.** All 80 COCO classes, with cuboids for person, bicycle, car, motorcycle, bus, and truck. No auxiliary detector is needed.
- **Trained on a mixture of 2D and 3D data.** COCO and VisDrone (boxes only) ground the detector. Real keypoint data (KITTI, DAIR-V2X), CDrone, and rendered vehicles (MeshFleet, Objaverse) teach the cuboids through a masked pose loss, so box-only images never push the keypoint head.
- **Offline refinement and a demo-reel builder.** `refine_v2.py` merges split tracks, cuts impossible joins, decides once per track whether its cuboid is trustworthy, and re-poses every object as a rigid box on the ground plane. `ad/make_ad_v2.sh` turns a folder of clips into a finished film, and `ad/make_labels.sh` writes the same geometry out as labels.
- **Five scales at 640 px.** Run them at 960 px on aerial footage to recover far, small traffic while the cuboids stay tight.

### Models

Every checkpoint is the epoch of its run with the highest road-class pose mAP, followed by a class-branch-only fine-tune on real box data (COCO + VisDrone) that corrects class boundaries learned from renders. The pose head is untouched by that step. Metrics are on the mixed validation set (COCO, VisDrone, KITTI and DAIR-V2X, CDrone, renders) at 640 px.

| Model | Params (M) | COCO AP | VisDrone AP @960 | KITTI 2D AP (Mod.) | KITTI AP<sub>3D</sub> E / M / H | KITTI AP<sub>BEV</sub> E / M / H | Download |
| --- | --: | --: | --: | --: | --: | --: | --- |
| v2-N | 2.6 | 14.3 | 14.6 | 91.5 | 31.5 / 22.2 / 18.8 | 37.0 / 26.5 / 22.4 | [urbanomnidetect_v2_n_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_n_640.pt) |
| v2-S | 9.9 | 21.3 | 19.1 | 94.6 | 42.9 / 32.5 / 27.6 | 49.7 / 37.5 / 31.0 | [urbanomnidetect_v2_s_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_s_640.pt) |
| v2-M | 21.3 | 28.1 | 25.1 | 95.4 | 46.3 / 34.8 / 30.7 | 50.7 / 39.6 / 35.3 | [urbanomnidetect_v2_m_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_m_640.pt) |
| v2-L | 25.7 | 28.7 | 25.1 | 96.5 | **53.6 / 39.4 / 33.1** | **56.7 / 42.5 / 36.9** | [urbanomnidetect_v2_l_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_l_640.pt) |
| v2-X | 57.6 | **30.4** | **26.7** | **96.6** | 47.5 / 37.1 / 32.9 | 52.1 / 41.1 / 36.4 | [urbanomnidetect_v2_x_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_x_640.pt) |

### Quick start

```bash
pip install -r requirements.txt          # ultralytics 8.4.61 (tested), torch, opencv, scipy, ...
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect-2 \
    checkpoints/urbanomnidetect_v2_x_640.pt --local-dir .
```

The checkpoints unpickle a class defined in `v2/hybrid_v2.py`, so run from the repository root (or put it on `PYTHONPATH`) and import it before loading. Do not run Python from a directory that contains an `ultralytics/` source clone, because it shadows the installed package.

```python
import v2.hybrid_v2                      # registers HybridPoseModel26 for unpickling
from ultralytics import YOLO

model = YOLO("checkpoints/urbanomnidetect_v2_x_640.pt")
r = model.predict("frame.jpg", imgsz=960, conf=0.1, device="cuda:0")[0]
boxes, classes = r.boxes.xyxy, r.boxes.cls          # COCO-80 ids
kpts = r.keypoints.data                              # (N, 8, 3): x, y, cuboid confidence
```

Indices 0 to 3 are the ground-contact corners and indices 4 to 7 are the top corners, the opposite of the paper model.

Video, with tracking, the bird's-eye view, and the offline refinement:

```bash
python bev_realtime_v2.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_v2_x_640.pt \
    --kp-imgsz 960 --kp-conf 0.1 --kp-vis 0.5 --class-conf 0:0.4,1:0.5,3:0.5 \
    --suppress-nested 0.85 --smooth 11 --scale-lock --fixed-camera auto --bev-fit --refine \
    --layout dashboard --device cuda:0 --output out.mp4
```

- `--kp-imgsz 960` runs the 640-trained model at a larger input. On drone footage this roughly doubles the far traffic that is tracked, and the cuboids stay tight (1280 works too).
- `--suppress-nested 0.85` keeps one track per box truck. The end-to-end head reports no NMS and can emit a whole truck and its cab as two detections. The flag drops a vehicle box nested inside a larger vehicle box when their cuboid footprints overlap, and leaves a car standing in front of a bus alone.
- `--refine` is the offline pass (`REFINE_README.md`). Drop it for a causal, streaming render. `--layout side` gives the plain camera and BEV view.
- `--export tensorrt --half` builds an FP16 engine on first use and reuses it afterwards. On the X model at 960 px on one RTX PRO 6000, this measures **4.6 ms per frame (217 fps) against 7.7 ms (130 fps) in PyTorch FP32**, a 1.7x speed-up, with cuboid corners agreeing to a median 0.9 px. End to end, the whole pipeline runs at 48 fps against 45, because the detector is no longer the bottleneck once tracking, the ground solve, and rendering are counted. TensorRT 10.x is required, since 11.x removed the builder-flag API that Ultralytics uses.
- Everything else is documented in `USAGE.md` (`python bev_realtime_v2.py --help`).

### Demo reel and labels

`ad/make_ad_v2.sh` renders every clip in a folder through the pipeline above, adds titles in the shot, crossfades, an end card, and music, and writes one film. `ad/make_labels.sh` runs the identical render but keeps the geometry instead: per frame and per track, the 2D box, the eight corners, the ground quad and its position on the ground plane, plus the homography, next to a cut of the original footage.

```bash
mkdir -p checkpoints test_videos assets _work
# checkpoints/urbanomnidetect_v2_x_640.pt, your clips in test_videos/, music at _work/ad_v2/<name>.mp3
bash ad/make_ad_v2.sh                       # -> _work/ad_v2/urbanomnidetect_v2_sequence.mp4
ORDER="01 02" bash ad/make_labels.sh        # -> _work/labels_v2/{labels,clips,vis,index.json}
```

Clips are addressed by a two-digit index assigned alphabetically on the first run (`manifest.tsv`), and captions live in the `CAPTION` map at the top of each script. Both scripts take their settings from environment variables (`KP`, `KPIMG`, `ORDER`, `TRIM`, `SPEED`, `DEVICE`, ...). `ffmpeg` with libx264 must be on `PATH`, and the end card needs `qrcode`.

### Training

The training code lives in `v2/`: `hybrid_v2.py` (the masked hybrid loss, the structure prior, the road-class fitness, and the head-only fine-tune), `train_v2.py` (launcher), `stage_data.py` (COCO, VisDrone, and keypoint data staged into one dataset in the 29-column format `cls cx cy w h (x y v)*8`), and the model and data YAMLs.

```bash
python v2/stage_data.py --out _work/v2data              # then point v2/v2_hybrid.yaml at it
DEVICE=0,1 bash v2/scripts/train_ladder_640.sh x        # 50 epochs, every epoch saved
python v2/scripts/select_best.py runs/v2/v2x_640        # best_pose.pt = highest road-class pose mAP
bash v2/scripts/heal_cls.sh x runs/v2/v2x_640/weights/best_pose.pt   # class-branch fine-tune on real boxes
```

Three things matter and are easy to get wrong:

- **Select by pose, not by the stock fitness, and never ship `last.pt`.** Road-class pose mAP peaks between epoch 25 and 47 depending on the scale and then fades while box mAP keeps rising. The last epochs (mosaic off) memorize the training data and detect visibly less on real drone footage, which is why every epoch is saved.
- **Keep the structure prior on annotated instances** (`V2_STRUCT_UNLABELED=0`, the default). Applied to box-only images it has a free minimum at a collapsed cuboid, and over a long run the head learns to draw slivers on real photographs.
- **Heal the class branch on real data at the end.** Renders over-represent buses (15% of their instances against under 2% in real data). Three epochs of the class branch alone on COCO + VisDrone fix the labels without touching the geometry.

`patches/ultralytics-random-grayscale.patch` adds the grayscale augmentation used in training (behind `V2_GRAYSCALE_P`). It is the only difference between the training fork and stock Ultralytics 8.4.61, and inference does not need it. Setting `V2_OOD_REEL=<labels dir>` scores every saved epoch on the output of `ad/make_labels.sh` (`ood_reel.csv`), which is how the released epochs were checked against footage.

### Repository structure

```
urbanomnidetect/
  bev_realtime_v2.py     # UrbanOmniDetect-2 pipeline: detection, tracking, BEV, dashboard, refinement hooks
  refine_v2.py           # offline track and 3D-box refinement (REFINE_README.md)
  bev_realtime.py        # paper-model pipeline (the v2 adapter subclasses it)
  draw_bev.py            # standalone BEV for a single image (paper models)
  sahi_tracker.py        # sliced inference and tracking for high-resolution streams (paper models)
  eval3d.py              # 2D, BEV, and 3D IoU evaluation
  homography_rt.py       # orthogonality-constrained BEV homography solver
  uod/                   # shared runtime: keypoints, tracking, BEV, smoothing, stabilization, visualization, HUD
  ad/                    # demo-reel builder, label export, title and end card
  v2/                    # UrbanOmniDetect-2 training: loss and trainer, launcher, data staging, YAMLs, scripts/
  patches/               # training-only Ultralytics augmentation patch
  cfg/, train.py         # paper experiments
  USAGE.md               # command reference for the inference tools
```

## UrbanOmniDetect (CVPR 2026 DriveX)

The paper model, presented as an oral at the [CVPR 2026 DriveX workshop](https://drivex-workshop.github.io/cvpr2026/) by Mehmet Kerem Turkcan, Devika Gumaste, and Zoran Kostic.

### Highlights

- **Calibration-free 3D detection.** A single model predicts 3D bounding box keypoints from a raw RGB image without camera intrinsics, depth estimation, or ground-plane priors.
- **View-agnostic.** One architecture works across ego-vehicle, infrastructure, and aerial drone viewpoints.
- **Strongest on the harder KITTI splits.** AP<sub>3D</sub> of 30.71 and AP<sub>BEV</sub> of 35.19 on the Moderate split at IoU ≥ 0.7, ahead of calibration-dependent baselines on Moderate and Hard.
- **Real-time.** Under 11 ms per image on an A100 with TensorRT at 640 × 640.
- **Robust to calibration error.** Calibration-dependent methods lose more than 80% of their accuracy with a 5% focal-length error. UrbanOmniDetect takes no intrinsics as input and is invariant to such errors by construction.

### Key results

#### Architecture comparison

Keypoint mAP<sub>50:95</sub> on the UrbanOmniView validation set at 640 × 640. Adding the P2 feature level improves every scale.

| Backbone | n | s | m | l | x |
|:---------|:---:|:---:|:---:|:---:|:---:|
| YOLO11 | 0.547 | 0.644 | 0.699 | 0.703 | 0.719 |
| YOLO11 + P6 | 0.548 | 0.639 | 0.693 | 0.698 | 0.718 |
| **YOLO11 + P2** | **0.559** | **0.656** | **0.717** | **0.729** | **0.751** |
| YOLO12 | 0.470 | 0.580 | 0.651 | 0.654 | 0.684 |
| YOLOv9 | 0.545 | 0.634 | 0.688 | 0.701 | 0.716 |
| YOLOv8 | 0.549 | 0.607 | 0.662 | 0.682 | 0.693 |

#### KITTI benchmark

Monocular 3D detection on KITTI at IoU ≥ 0.7. The baselines use camera calibration. UrbanOmniDetect does not.

| Method | AP<sub>3D</sub> Easy | AP<sub>3D</sub> Mod. | AP<sub>3D</sub> Hard | AP<sub>BEV</sub> Easy | AP<sub>BEV</sub> Mod. | AP<sub>BEV</sub> Hard |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MonoDGP | **30.76** | 22.34 | 19.02 | **39.40** | 28.20 | 24.42 |
| MonoCon | 26.33 | 19.01 | 15.98 | 34.65 | 25.39 | 21.93 |
| MonoLSS | 25.91 | 18.29 | 15.94 | 34.70 | 25.36 | 21.84 |
| DEVIANT | 24.63 | 16.54 | 14.52 | 32.60 | 23.04 | 19.99 |
| **UrbanOmniDetect** | 29.61 | **30.71** | **27.76** | 33.86 | **35.19** | **31.38** |

### Quick start

#### Prerequisites

A working CUDA and PyTorch environment is required. To set one up from scratch on Windows or Ubuntu, see [CUDA2025](https://github.com/mkturkcan/CUDA2025) for a step-by-step guide based on Miniconda.

Then install the remaining dependencies:

```bash
pip install ultralytics scipy scikit-learn opencv-python matplotlib
```

#### Download a pretrained model

The trained checkpoints are hosted on [Hugging Face](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect). The following command downloads the recommended YOLO11x model with the P2 feature-pyramid head.

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect \
    checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --local-dir .
```

To download every checkpoint at once:

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect --local-dir .
```

The full set covers YOLOv8, YOLOv9, YOLO11, and YOLO12 across all scales and head configurations.

#### Single-image inference

The model uses the standard Ultralytics prediction API. Point it at an image from any viewpoint and it predicts 3D bounding box keypoints without camera parameters.

```python
from ultralytics import YOLO

model = YOLO("checkpoints/urbanomnidetect_yolo11x-p2_1920.pt")
results = model.predict("your_image.jpg", imgsz=1920, conf=0.1, device="cuda:0")
```

Each detection has eight ordered keypoints, the 2D projections of its 3D bounding box corners. Indices 0 to 3 are the top corners and indices 4 to 7 are the ground-contact corners.

#### Bird's-eye view

Ground-contact keypoints are mapped to a top-down plane through an orthogonality-constrained homography. No camera calibration is needed.

```bash
python draw_bev.py \
    --image your_image.jpg \
    --kp-model checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --kp-imgsz 1920 \
    --mode both \
    --device cuda:0
```

This produces two outputs: a publication-quality Matplotlib figure and a lightweight OpenCV BEV image. Run `python draw_bev.py --help` for all options, including auxiliary model support, confidence thresholds, and output formats.

#### Real-time BEV on video

The video pipeline detects objects, tracks them across frames, solves the BEV homography, and renders the camera view next to a bird's-eye view. Tracked objects keep a stable identity and hold their last position when a detection is briefly missed, so the layout stays steady.

```bash
python bev_realtime.py \
    --input drone_manhattan.mp4 \
    --kp-model checkpoints/urbanomnidetect_yolo11x-p2_640.pt \
    --kp-imgsz 640 \
    --aux-model yolo26x.pt \
    --device cuda:0 \
    --export tensorrt \
    --output bev.mp4
```

Add `--display` for a live window. For high-resolution infrastructure or aerial streams, `sahi_tracker.py` runs the same pipeline over overlapping image tiles to recover small and distant objects. See `USAGE.md` for the full command reference.

### Training

#### Reproduce all experiments

The training script runs every experiment from the paper in sequence. All experiment definitions and augmentation hyperparameters live in `cfg/experiments.py`.

```bash
python train.py
```

To run a subset of experiments, filter by model name. For example, to train only YOLO11 variants:

```bash
python train.py --filter yolo11
```

Preview what would run without starting any training:

```bash
python train.py --filter p2 --dry-run
```

Override the default GPU configuration or epoch count:

```bash
python train.py --devices 0 1 2 3 --epochs 50
```

Resume an interrupted run from a specific experiment index:

```bash
python train.py --start-from 12
```

See `python train.py --help` for the complete CLI reference.

#### Dataset setup

Download the UrbanOmniView dataset from [Hugging Face](https://huggingface.co/datasets/mehmetkeremturkcan/UrbanOmniView) and place it according to the path in `cfg/dataset/urbanomniview.yaml`. The dataset combines three sources:

| Source | Frames | Viewpoint |
|:-------|-------:|:----------|
| KITTI | 15,022 | Ego-vehicle |
| DAIR-V2X | 12,424 | Infrastructure |
| UE5 Synthetic | 10,000 | Ground, infrastructure, drone |
| **Total** | **37,446** | |

The synthetic UE5 portion is released as part of this work. KITTI and DAIR-V2X should be downloaded from their original sources and formatted with the provided conversion scripts.

## Citation

If you use UrbanOmniDetect or the UrbanOmniView dataset in your research, please cite:

```bibtex
@inproceedings{turkcan2026urbanomnidetect,
  title     = {Calibration-Free View-Agnostic Monocular {3D} Object Detection for Urban Scenes},
  author    = {Turkcan, Mehmet Kerem and Gumaste, Devika and Kostic, Zoran},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages     = {786--795},
  year      = {2026}
}
```

If you use UrbanOmniDetect-2, please cite:

```bibtex
@software{turkcan2026urbanomnidetect2,
  author = {Turkcan, Mehmet Kerem},
  title  = {{UrbanOmniDetect-2}: One Network for Detection and Calibration-Free {3D} Cuboids from Any Viewpoint},
  year   = {2026},
  month  = {9},
  url    = {https://github.com/mkturkcan/urbanomnidetect}
}
```

## Acknowledgements

The UrbanOmniDetect paper and the UrbanOmniView dataset began while the first author was a postdoc in the [Department of Electrical Engineering](https://www.ee.columbia.edu/) ([AIDL Lab](https://www.aidl.ee.columbia.edu/)) at Columbia University. This work was supported by the NSF Engineering Research Center for Smart Streetscapes under Award EEC-2133516, NSF Grants CNS-2450567 and CNS-2038984, and by computing resources from the NVIDIA Academic Grant Program and the Empire AI Consortium.

## License

This project is released under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0).
