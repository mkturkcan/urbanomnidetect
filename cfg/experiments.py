"""Experiment definitions for UrbanOmniView pose estimation training.

This module centralises every training configuration used to produce the results
in the paper.  Each experiment is represented as a plain dictionary so that
downstream code can filter, serialise, or extend entries without coupling to
positional tuple indices.

Typical usage from the training script::

    from cfg.experiments import EXPERIMENTS, AUGMENTATION

Dictionary keys per experiment
------------------------------
model_cfg : str
    Filename of the YOLO model YAML (resolved under ``cfg/models/``).
weights : str | None
    Pretrained checkpoint filename, or ``None`` for training from scratch.
imgsz : int
    Input image resolution (longest side).
batch_per_device : int
    Batch size **per GPU** — multiplied by the device count at runtime.
epochs : int
    Maximum training epochs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optimised augmentation hyperparameters
# ---------------------------------------------------------------------------

AUGMENTATION: dict[str, float | str] = {
    "hsv_h": 0.01315,
    "hsv_s": 0.35348,
    "hsv_v": 0.19383,
    "degrees": 0.00012,
    "translate": 0.27484,
    "scale": 0.95,
    "shear": 0.00136,
    "perspective": 0.00074,
    "flipud": 0.00653,
    "fliplr": 0.30393,
    "bgr": 0.0,
    "mosaic": 0.99182,
    "mixup": 0.42713,
    "cutmix": 0.00082,
    "copy_paste": 0.40413,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
}

# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

SCALES_ALL: list[str] = ["n", "s", "m", "l", "x"]
SCALES_V9: list[str] = ["t", "s", "m", "c", "e"]

# ---------------------------------------------------------------------------
# Helpers to build experiment batches
# ---------------------------------------------------------------------------


def _make_entries(
    arch: str,
    variant: str,
    scales: list[str],
    imgsz: int,
    batch_per_device: int,
    epochs: int,
    *,
    pretrained: bool,
) -> list[dict]:
    """Return a list of experiment dicts for *arch* across *scales*."""
    return [
        {
            "model_cfg": f"{arch}{s}-pose{variant}.yaml",
            "weights": f"{arch}{s}.pt" if pretrained else None,
            "imgsz": imgsz,
            "batch_per_device": batch_per_device,
            "epochs": epochs,
        }
        for s in scales
    ]


def pretrained(
    arch: str, variant: str, scales: list[str], imgsz: int, batch_per_device: int, epochs: int,
) -> list[dict]:
    """Experiment entries that fine-tune from COCO-pretrained weights."""
    return _make_entries(arch, variant, scales, imgsz, batch_per_device, epochs, pretrained=True)


def scratch(
    arch: str, variant: str, scales: list[str], imgsz: int, batch_per_device: int, epochs: int,
) -> list[dict]:
    """Experiment entries that train from scratch (random init)."""
    return _make_entries(arch, variant, scales, imgsz, batch_per_device, epochs, pretrained=False)


# ---------------------------------------------------------------------------
# Full experiment list — order matches the paper's ablation tables
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    # ── High-resolution pretrained ────────────────────────────────────────
    {"model_cfg": "yolo11n-pose-p2.yaml", "weights": "yolo11n.pt", "imgsz": 3840, "batch_per_device": 4,  "epochs": 100},
    {"model_cfg": "yolo11n-pose-p2.yaml", "weights": "yolo11n.pt", "imgsz": 1920, "batch_per_device": 8,  "epochs": 100},
    {"model_cfg": "yolo11x-pose-p1.yaml", "weights": "yolo11x.pt", "imgsz": 640,  "batch_per_device": 24, "epochs": 200},
    {"model_cfg": "yolo11x-pose-p2.yaml", "weights": "yolo11x.pt", "imgsz": 1280, "batch_per_device": 8,  "epochs": 100},

    # ── YOLO11 P2 pretrained (n/s/m/l) ───────────────────────────────────
    *pretrained("yolo11", "-p2", ["n", "s", "m", "l"], 640, 32, 100),

    # ── YOLO11 P6 / YOLOv9 P3 pretrained (single scale) ─────────────────
    {"model_cfg": "yolo11x-pose-p6.yaml", "weights": "yolo11x.pt", "imgsz": 640, "batch_per_device": 32, "epochs": 100},
    {"model_cfg": "yolov9e-pose-p3.yaml", "weights": "yolov9e.pt", "imgsz": 640, "batch_per_device": 32, "epochs": 100},

    # ── YOLOv8 pretrained (all scales, multiple heads) ───────────────────
    *pretrained("yolov8", "-p6", SCALES_ALL, 640, 32, 100),
    *pretrained("yolov8", "-p2", SCALES_ALL, 640, 32, 100),
    *pretrained("yolov8", "",    SCALES_ALL, 640, 32, 100),

    # ── YOLOv9 pretrained (all scales) ───────────────────────────────────
    *pretrained("yolov9", "", SCALES_V9, 640, 32, 100),

    # ── YOLO11 pretrained (all scales, base head) ────────────────────────
    *pretrained("yolo11", "", SCALES_ALL, 640, 32, 100),

    # ── From-scratch experiments ─────────────────────────────────────────
    *scratch("yolov9", "",    SCALES_V9,  640,  32, 100),
    *scratch("yolov8", "-p2", SCALES_ALL, 640,  32, 100),
    *scratch("yolov8", "",    SCALES_ALL, 640,  32, 100),
    *scratch("yolo12", "",    SCALES_ALL, 1280, 32, 100),
    *scratch("yolov8", "-p6", SCALES_ALL, 640,  32, 100),
    *scratch("yolo12", "",    SCALES_ALL, 640,  32, 100),
    *scratch("yolo11", "",    SCALES_ALL, 640,  32, 100),
]
