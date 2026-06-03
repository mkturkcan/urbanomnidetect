"""Visual style system: a class-based colour palette and theme constants.

Each object class has one base colour. An instance's colour is a deterministic
shade of its class colour (hue fixed, lightness varied by track id), so a class
reads as a coherent family while individual tracks stay distinguishable and a
given track keeps the same colour in both the camera and BEV panels.

Colours are stored as RGB tuples (0-255). Use :func:`bgr` at OpenCV call sites.
"""

from __future__ import annotations

import colorsys
from typing import Tuple

RGB = Tuple[int, int, int]

__all__ = ["bgr", "class_category", "class_color", "instance_color",
           "text_color_on", "PALETTE", "THEME", "COCO_CATEGORY",
           "POSE_CATEGORY", "CATEGORY_LABEL"]

# Curated, slightly desaturated palette (RGB). Distinct hues, similar value, so
# they coexist without one screaming louder than the rest.
PALETTE: dict = {
    "car":    (64, 156, 255),    # blue
    "person": (255, 168, 64),    # amber
    "bike":   (181, 99, 232),    # violet
    "bus":    (54, 200, 130),    # green
    "truck":  (255, 99, 99),     # coral
    "other":  (150, 162, 176),   # slate
}

CATEGORY_LABEL = {
    "car": "Car", "person": "Person", "bike": "Bike",
    "bus": "Bus", "truck": "Truck", "other": "Obj",
}

# Pose model classes: {0: car, 1: person, 2: bike}.
POSE_CATEGORY = {0: "car", 1: "person", 2: "bike"}

# COCO ids -> our categories (for the auxiliary detector).
COCO_CATEGORY = {
    0: "person", 1: "bike", 2: "car", 3: "bike", 5: "bus", 7: "truck",
}

# Dark "automotive HMI" theme for the BEV panel and UI chrome. Kept strictly
# neutral (R=G=B) -- a tinted background reads as a generated default. The only
# colour in the scene comes from the semantic class palette above.
THEME = {
    "bev_bg":      (23, 23, 23),
    "bev_bg_edge": (14, 14, 14),     # outer vignette
    "ring":        (56, 56, 56),
    "ring_hi":     (88, 88, 88),
    "spoke":       (40, 40, 40),
    "grid_text":   (128, 128, 128),
    "ego":         (222, 222, 222),  # neutral light marker
    "panel":       (20, 20, 20),     # HUD / chrome panel fill
    "panel_text":  (234, 234, 234),
    "muted_text":  (150, 150, 150),
    "accent":      (200, 200, 200),  # neutral light accent
    "separator":   (10, 10, 10),
    "aux":         (245, 211, 39),   # auxiliary detections (yellow)
}


def bgr(rgb: RGB) -> RGB:
    """RGB -> BGR for OpenCV."""
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def class_category(cls: int, source: str = "pose") -> str:
    table = POSE_CATEGORY if source == "pose" else COCO_CATEGORY
    return table.get(int(cls), "other")


def class_color(cls: int, source: str = "pose") -> RGB:
    """Base RGB colour for a class."""
    return PALETTE[class_category(cls, source)]


def instance_color(cls: int, instance_id: int, source: str = "pose") -> RGB:
    """A per-instance shade of the class colour (deterministic in the id)."""
    base = class_color(cls, source)
    h, l, s = colorsys.rgb_to_hls(*[c / 255.0 for c in base])
    if instance_id is None or instance_id < 0:
        instance_id = 0
    # Golden-ratio hop spreads ids; vary lightness most, hue/sat slightly.
    j = (instance_id * 0.6180339887) % 1.0
    l = min(0.74, max(0.40, l + (j - 0.5) * 0.26))
    h = (h + ((instance_id * 0.137) % 1.0 - 0.5) * 0.03) % 1.0
    s = min(1.0, max(0.45, s * (0.92 + 0.16 * j)))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def text_color_on(rgb: RGB) -> RGB:
    """Pick near-black or near-white text for legibility on ``rgb``."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (24, 26, 30) if lum > 140 else (244, 246, 250)
