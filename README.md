<p align="center">
  <h1 align="center">UrbanOmniDetect</h1>
  <h3 align="center">Calibration-Free View-Agnostic Monocular 3D Object Detection for Urban Scenes</h3>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=306TgWoAAAAJ">Mehmet Kerem Turkcan</a> &bull;
    <a href="https://scholar.google.com/citations?user=GP7T1fgAAAAJ">Devika Gumaste</a> &bull;
    <a href="https://scholar.google.com/citations?user=TlPI8yIAAAAJ">Zoran Kostic</a>
    <br/>
    <b>Columbia University</b>
    <br/><br/>
    <a href="https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue" alt="HuggingFace Models"></a>
    <a href="https://huggingface.co/datasets/mehmetkeremturkcan/urbanomniview"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" alt="HuggingFace Dataset"></a>
    <a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR%202026-DriveX%20Workshop-4b44ce.svg" alt="CVPR 2026"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL%203.0-orange.svg" alt="License"></a>
  </p>
</p>

<p align="center">
  <img src="https://github.com/mkturkcan/urbanomnidetect/raw/main/assets/urbanomniview.png" width="100%" alt="UrbanOmniDetect pipeline overview"/>
</p>

## Highlights

- **Calibration-free 3D detection.** A single model predicts 3D bounding box keypoints from a raw RGB image without camera intrinsics, depth estimation, or ground-plane priors.
- **View-agnostic.** One unified architecture works across ego-vehicle, infrastructure, and aerial drone viewpoints.
- **State-of-the-art on monocular KITTI.** AP<sub>3D</sub> = 30.71 and AP<sub>BEV</sub> = 35.19 on the Moderate split at IoU >= 0.7, outperforming calibration-dependent baselines on Moderate and Hard.
- **Real-time.** Under 11 ms inference on an A100 GPU with TensorRT at 640x640.
- **Robust.** Calibration-dependent methods lose over 80% accuracy with a 5% focal-length error. Our method is invariant by construction.

## Key Results

### Architecture Comparison

| Backbone | n | s | m | l | x |
|:---------|:---:|:---:|:---:|:---:|:---:|
| YOLO11 | 0.547 | 0.644 | 0.699 | 0.703 | 0.719 |
| YOLO11 + P6 | 0.548 | 0.639 | 0.693 | 0.698 | 0.718 |
| **YOLO11 + P2** | **0.559** | **0.656** | **0.717** | **0.729** | **0.751** |
| YOLO12 | 0.470 | 0.580 | 0.651 | 0.654 | 0.684 |
| YOLOv9 | 0.545 | 0.634 | 0.688 | 0.701 | 0.716 |
| YOLOv8 | 0.549 | 0.607 | 0.662 | 0.682 | 0.693 |

### KITTI Benchmark

| Method | AP<sub>3D</sub> Easy | AP<sub>3D</sub> Mod. | AP<sub>3D</sub> Hard | AP<sub>BEV</sub> Easy | AP<sub>BEV</sub> Mod. | AP<sub>BEV</sub> Hard |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MonoDGP | **30.76** | 22.34 | 19.02 | **39.40** | 28.20 | 24.42 |
| MonoCon | 26.33 | 19.01 | 15.98 | 34.65 | 25.39 | 21.93 |
| MonoLSS | 25.91 | 18.29 | 15.94 | 34.70 | 25.36 | 21.84 |
| DEVIANT | 24.63 | 16.54 | 14.52 | 32.60 | 23.04 | 19.99 |
| **Ours** | 29.61 | **30.71** | **27.76** | 33.86 | **35.19** | **31.38** |

## Quick Start

### Prerequisites

A working CUDA + PyTorch environment is required. If you need to set one up from scratch on Windows or Ubuntu, see [CUDA2025](https://github.com/mkturkcan/CUDA2025) for a step-by-step miniconda-based guide.

Once your environment is ready, install the remaining dependencies:

```bash
pip install ultralytics scipy scikit-learn opencv-python matplotlib
```

### Download a Pretrained Model

Our 37 trained checkpoints are hosted on Hugging Face. The following command downloads the recommended XL model with the P2 feature pyramid head.

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect \
    checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --local-dir .
```

To download all checkpoints at once:

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect --local-dir .
```

The full set covers YOLOv8, YOLOv9, YOLO11, and YOLO12 across all scales and head configurations. See the [model repository on Hugging Face](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect) for the complete list.

### Run Inference on a Single Image

The model follows the standard Ultralytics prediction API. Point it at any image, from any viewpoint, and it will predict 3D bounding box keypoints without requiring camera parameters.

```python
from ultralytics import YOLO

model = YOLO("checkpoints/urbanomnidetect_yolo11x-p2_1920.pt")
results = model.predict("your_image.jpg", imgsz=1920, conf=0.1, device="cuda:0")
```

Each detected object will have 8 ordered keypoints representing the 2D projections of its 3D bounding box corners. Indices 0 to 3 are the top corners and indices 4 to 7 are the ground-contact corners.

### Generate a Bird's-Eye View

The BEV head maps ground-contact keypoints to a top-down plane through a learned homography. No camera calibration is needed.

```bash
python draw_bev.py \
    --image your_image.jpg \
    --kp-model checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --mode both \
    --device cuda:0
```

This produces two outputs: a publication-quality matplotlib figure and a lightweight OpenCV BEV image. Run `python draw_bev.py --help` for the full set of options, including auxiliary model support, confidence thresholds, and output format controls.

### Real-Time BEV on Video

For video streams with TensorRT acceleration:

```bash
python bev_realtime.py \
    --input drone_manhattan.mp4 \
    --kp-model checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --kp-imgsz 1920 \
    --homography adam \
    --export tensorrt
```

## Training

### Reproduce All Experiments

The training script launches every experiment from the paper in sequence. All experiment definitions and augmentation hyperparameters live in `cfg/experiments.py`.

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

If a long run is interrupted, resume from a specific experiment index:

```bash
python train.py --start-from 12
```

See `python train.py --help` for the complete CLI reference.

### Dataset Setup

Download the UrbanOmniView dataset from [Hugging Face Datasets](https://huggingface.co/datasets/mehmetkeremturkcan/urbanomniview) and place it according to the path in `cfg/dataset/urbanomniview.yaml`. The dataset combines three sources:

| Source | Frames | Viewpoint |
|:-------|-------:|:----------|
| KITTI | 15,022 | Ego-vehicle |
| DAIR-V2X | 12,424 | Infrastructure |
| UE5 Synthetic | 10,000 | Ground, infrastructure, drone |
| **Total** | **37,446** | |

The synthetic UE5 portion is released as part of this work. KITTI and DAIR-V2X should be downloaded from their respective sources and formatted using the provided conversion scripts.

## Repository Structure

```
urbanomniview/
  cfg/
    dataset/             # Dataset YAML configs
    models/              # YOLO model architecture YAMLs
    experiments.py       # All training experiment definitions
  train.py              # Training launcher
  draw_bev.py           # BEV visualization, standalone
  bev_realtime.py       # Real-time BEV pipeline with TensorRT
  homography_rt.py      # Homography solver
  sahi_tracker.py       # SAHI-based sliced inference and tracking
```

## Citation

If you use UrbanOmniDetect or the UrbanOmniView dataset in your research, please cite:

```bibtex
@inproceedings{turkcan2026urbanomnidetect,
  title     = {Calibration-Free View-Agnostic Monocular 3D Object Detection for Urban Scenes},
  author    = {Turkcan, Mehmet Kerem and Gumaste, Devika and Kostic, Zoran},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by the NSF Engineering Research Center for Smart Streetscapes under Award EEC-2133516, NSF Grants CNS-2450567 and CNS-2038984, and by computing resources from the NVIDIA Academic Grant Program and the Empire AI Consortium.

## License

This project is released under the [GNU Affero General Public License v3.0](LICENSE).
