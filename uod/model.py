"""Model loading, optional TensorRT export, and a uniform predict wrapper.

Keeps Ultralytics specifics in one place so the pipeline code stays clean, and
so the (slow, one-off) TensorRT export is cached to disk and reused.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["load_detector", "maybe_export", "DEFAULT_AUX_MODEL"]

# Default frozen auxiliary 2D detector: a recent COCO-pretrained YOLO. The
# paper used a COCO model purely to broaden recall; ``yolo26x`` is the sane
# modern default (replacing the older, much slower ``drone_hyper_100.pt``).
DEFAULT_AUX_MODEL = "yolo26x.pt"


def maybe_export(weights: str, imgsz: int, device: str, fmt: str = "engine",
                 half: bool = True, dynamic: bool = False) -> str:
    """Export ``weights`` to an accelerated format if not already present.

    Returns the path to the exported artifact (e.g. a ``.engine`` file), or the
    original ``weights`` if ``fmt`` is falsy. The export is skipped when a
    matching artifact already exists next to the weights.
    """
    if not fmt or fmt in ("torch", "pytorch", "none"):
        return weights
    ext = {"engine": ".engine", "tensorrt": ".engine", "onnx": ".onnx"}.get(fmt, "." + fmt)
    # Encode the export-determining parameters in the cached artifact name so a
    # stale engine built at a different resolution/precision is never silently
    # reused (an engine is only valid for the imgsz/precision it was built for).
    base = os.path.splitext(weights)[0]
    tag = f"_{imgsz}{'_fp16' if half else '_fp32'}{'_dyn' if dynamic else ''}"
    out = f"{base}{tag}{ext}"
    if os.path.exists(out):
        return out
    from ultralytics import YOLO
    m = YOLO(weights)
    real_fmt = "engine" if fmt == "tensorrt" else fmt
    exported = str(m.export(format=real_fmt, imgsz=imgsz, half=half,
                            dynamic=dynamic, device=device, verbose=False))
    # Ultralytics writes a default name; rename to the parameterised cache name.
    if os.path.exists(exported) and os.path.abspath(exported) != os.path.abspath(out):
        try:
            os.replace(exported, out)
        except OSError:
            return exported
    return out if os.path.exists(out) else exported


def load_detector(weights: str, device: str = "cpu", imgsz: int = 640,
                  export: str = "none", half: bool = True,
                  fuse: bool = True):
    """Load a YOLO detector, optionally exporting to an accelerated backend.

    Parameters
    ----------
    weights : str
        Path to ``.pt`` (or a known model name for the auxiliary detector).
    device : str
        ``"cpu"``, ``"cuda:0"``, etc.
    export : {"none", "engine"/"tensorrt", "onnx"}
        Accelerated backend to export to and load. TensorRT requires a working
        CUDA + TensorRT install and a fixed ``imgsz``.
    half : bool
        Use FP16 for the exported engine (GPU only).
    """
    from ultralytics import YOLO
    path = weights
    if export and export not in ("none", "torch", "pytorch"):
        if "cuda" not in str(device):
            raise RuntimeError(
                f"export={export!r} requires a CUDA device, got device={device!r}")
        path = maybe_export(weights, imgsz=imgsz, device=device, fmt=export,
                            half=half)
    model = YOLO(path)
    # Engines are already fused/optimised; only fuse eager PyTorch graphs.
    if fuse and path.endswith(".pt"):
        try:
            model.fuse()
        except Exception:
            pass
    return model
