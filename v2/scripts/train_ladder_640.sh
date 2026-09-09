#!/usr/bin/env bash
# Train the v2 hybrid detect+pose models (n/s/m/l/x) at 640 with the recipe of the release:
# structure prior on annotated instances only, road-class pose fitness, every epoch saved,
# and (optionally) the demo-reel proxy metric. Run from the repository root.
#   bash v2/scripts/train_ladder_640.sh x            # one scale on GPU 0
#   DEVICE=0,1 bash v2/scripts/train_ladder_640.sh x # DDP
#   EPOCHS=50 DEVICE=2 bash v2/scripts/train_ladder_640.sh s m
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"; EPOCHS="${EPOCHS:-50}"; DEVICE="${DEVICE:-0}"; BATCH="${BATCH:-64}"; WORKERS="${WORKERS:-16}"
DATA="${DATA:-v2/v2_hybrid.yaml}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"       # DDP workers import v2.hybrid_v2
[ -n "${V2_OOD_REEL:-}" ] && export V2_OOD_REEL             # dir with index.json/labels/clips -> <run>/ood_reel.csv
cd "$ROOT"
for sc in "$@"; do
    "$PY" -m v2.train_v2 --data "$DATA" --model "v2/yolo26${sc}-pose-v2.yaml" --pretrained "yolo26${sc}.pt" \
        --epochs "$EPOCHS" --batch "$BATCH" --imgsz 640 --device "$DEVICE" --workers "$WORKERS" \
        --name "v2${sc}_640" --shear 0 --degrees 0 --perspective 0 \
        --extra "close_mosaic=10,save_period=1"
done
