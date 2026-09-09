#!/usr/bin/env bash
# Class-branch-only fine-tune on real bbox-only data (COCO + VisDrone x3, v2/v2_real_cls.yaml):
# trains model.23.cv3 + one2one_cv3 and freezes everything else, BatchNorm statistics included.
# Heals class boundaries learnt from synthetic renders (a van reported as "bus"). ~1 h for x.
#   bash v2/scripts/heal_cls.sh x runs/v2/v2x_640/weights/best_pose.pt [gpu] [epochs] [batch]
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
sc="$1"; ck="$(realpath "$2")"; gpu="${3:-0}"; ep="${4:-3}"; bs="${5:-64}"
export V2_TRAIN_ONLY="cv3,one2one_cv3" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PY" -m v2.train_v2 --data v2/v2_real_cls.yaml --model "v2/yolo26${sc}-pose-v2.yaml" --pretrained "$ck" \
    --epochs "$ep" --batch "$bs" --imgsz 640 --device "$gpu" --workers 16 --name "v2${sc}_640_clsheal" \
    --shear 0 --degrees 0 --perspective 0 \
    --extra "optimizer=AdamW,lr0=0.0002,lrf=0.1,warmup_epochs=0.5,close_mosaic=$ep,save_period=1"
