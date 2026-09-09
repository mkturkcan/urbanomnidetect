"""UrbanOmniDetect v2: a single hybrid detect+pose model.

The model is the stock YOLO26 pose architecture, but trained on a mix of
bbox-only data (COCO, VisDrone) and our keypoint data (pose_dataset). The data
pipeline marks each instance's keypoint availability through the keypoint
*visibility* channel (1 = annotated, 0 = bbox-only; see ``stage_data.py``), and
this module supplies the only framework change: a pose loss that supervises
keypoints ONLY on instances that have them.

Two pieces:
  * ``HybridPoseLoss26`` -- the end-to-end YOLO26 pose loss with keypoint
    location / objectness / RLE losses restricted to instances with at least one
    visible keypoint, so COCO/VisDrone vehicles never get pushed toward "no
    keypoints" (which would wreck out-of-distribution keypoint transfer). It also
    keeps the keypoint head and the flow model in the autograd graph every step
    (a zero-weighted touch), so DistributedDataParallel needs no
    ``find_unused_parameters`` even on batches/ranks with no annotated keypoints.
  * ``HybridPoseTrainer`` -- a PoseTrainer that builds the pose model, attaches
    the hybrid loss, and (by ``pretrained=...``) transfers the YOLO26 detection
    weights so box/cls start strong and only the keypoint head must be learned.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import distributed as dist

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.utils.ops import xyxy2xywh

# --- lens-invariant cuboid structure prior --------------------------------------
# Strength of the structure prior, folded into the keypoint loss (so it is further
# scaled by the pose gain). Tunable per run via the env var so it propagates to DDP
# subprocesses without touching the validated cfg plumbing. 0 disables it.
# Effective weight is _STRUCT_GAIN * pose_gain * residual; the residual is small
# (~0 for valid cuboids, ~0.03 for random), so a gain near 1 makes it a gentle but
# real regularizer. Raise V2_STRUCT_GAIN to enforce cuboid structure more strongly.
_STRUCT_GAIN = float(os.environ.get("V2_STRUCT_GAIN", "0.5"))
# Where the prior is applied. Until 2026-09-07 it ran on EVERY positive anchor,
# including the ~125k bbox-only COCO/VisDrone images that carry no keypoint loss at
# all. On those the residual has a free global minimum -- a collapsed cuboid (coincident
# corners give zero lines, hence zero determinants) -- and nothing pulls the other way,
# so over a long run the head learns "real photograph => degenerate box". Measured on
# the 640 n/s/m/x runs: epoch-100 last.pt has residual 0.0000 in every domain, and its
# cuboid-hull/box area fell 0.43->0.22 on VisDrone and 0.34->0.10 on COCO while
# keypoint-supervised domains were unaffected (pose_val 0.87->0.94). That is why the
# epoch-1 best.pt checkpoints were the only usable ones in deployment. Default is now
# annotated instances only, where OKS pins the corners; set V2_STRUCT_UNLABELED=1 to
# reproduce the old recipe.
_STRUCT_UNLABELED = bool(int(os.environ.get("V2_STRUCT_UNLABELED", "0")))
# Hedge: the unlabeled prior HELPS early (epoch-1 reel corner error 0.094 with it vs
# 0.157 without, x scale) and only collapses the head over many epochs. With
# V2_STRUCT_UNLABELED_EPOCHS=N it is applied to bbox-only anchors for the first N
# epochs only (the trainer flips _struct_unlabeled_active at each epoch start), then
# the annotated-only default takes over. 0 = never (default).
_STRUCT_UNLABELED_EPOCHS = int(os.environ.get("V2_STRUCT_UNLABELED_EPOCHS", "0"))
_struct_unlabeled_active = _STRUCT_UNLABELED or _STRUCT_UNLABELED_EPOCHS > 0

# Class-balanced keypoint supervision: the synth pose data is ALL vehicles, so
# person/bicycle/motorcycle keypoints are heavily outnumbered (car ~80% of kpt
# instances) and the head collapses to a vehicle-cuboid prior -> garbage cuboids
# on those classes. Repeat their instances inside the keypoint loss to upweight
# them (integer reps reuse the OKS/RLE/obj losses as-is). Override via env, e.g.
# V2_KPT_REP="0:3,1:4,3:3" (coco_class:reps); empty string disables.
def _parse_reps(s):
    out = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split(":"); out[int(k)] = int(v)
    return out
_KPT_REP = _parse_reps(os.environ.get("V2_KPT_REP", "0:3,1:4,3:3"))  # person, bicycle, motorcycle
_KPT_DBG = bool(os.environ.get("V2_KPT_DEBUG"))

# Non-saturating keypoint regularizers that HEAL an exploded keypoint head -- the
# imgsz-1280 failure where pedestrian/bike corners spill ~2x outside the 2D box
# (vehicles unaffected). Root cause: the OKS keypoint loss is 1-exp(-d^2/area), which
# SATURATES once a corner is grossly displaced -> gradient ~0 -> a finetune can never
# pull it back; and it is area-normalized, so at 1280 (objects ~4x the area) the
# gradient is weaker still. Both terms below are non-saturating (constant per-error
# gradient) and fold into kpts_loss (so they are further scaled by the pose gain):
#   V2_CONTAIN_GAIN -- relu penalty on corners OUTSIDE the target 2D box (every
#                      positive anchor; box-diag-normalized). A hard guard that pulls
#                      exploded corners back inside. The 2D box == bbox(GT corners)
#                      (measured GT spread/box == 1.00), so corners are never
#                      legitimately outside it.
#   V2_KPT_L1_GAIN  -- box-diag-normalized L1 on annotated corners: scale-invariant,
#                      non-saturating tightening that OKS loses on large/1280 objects.
# ON by default (like _STRUCT_GAIN / _KPT_REP); set to 0 to reproduce the pre-heal
# recipe exactly. Tune up (e.g. V2_CONTAIN_GAIN=2) for a more aggressive heal.
_CONTAIN_GAIN = float(os.environ.get("V2_CONTAIN_GAIN", "0.0"))
_KPT_L1_GAIN = float(os.environ.get("V2_KPT_L1_GAIN", "0.0"))

# The 8 corners form a cuboid: 3 families of 4 mutually parallel 3D edges. Each
# family's 4 image lines meet at one vanishing point under ANY pinhole camera
# (finite for wide-angle, at infinity for telephoto), so "the 4 lines are
# concurrent" is a focal-length / FoV invariant -- unlike "verticals are parallel",
# which only holds for weak perspective. Corner order: ground 0-3, top 4-7,
# verticals (i, i+4); ground/top cycles per the renderer's edges.
_FAMILIES = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),   # length edges
    ((1, 2), (3, 0), (5, 6), (7, 4)),   # width edges
    ((0, 4), (1, 5), (2, 6), (3, 7)),   # vertical edges
)
_TRIPLES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _structure_residual(kpts_xy: torch.Tensor, box_diag: torch.Tensor) -> torch.Tensor:
    """Per-instance cuboid-concurrency residual (lens-invariant), shape (M,).

    For each of the 3 parallel-edge families, three lines are concurrent iff the
    determinant of their stacked homogeneous coordinates is 0. We sum the squared
    determinants over every triple of the family's 4 lines. Keypoints are centered
    and scaled by the box size first, so the measure is invariant to image scale
    and to the camera's focal length (it targets 0 for any valid projected cuboid).
    """
    c = kpts_xy.mean(dim=1, keepdim=True)                         # (M,1,2) center proxy
    k = (kpts_xy - c) / box_diag[:, None, None].clamp_min(1e-6)   # scale-normalized
    h = torch.cat([k, torch.ones_like(k[..., :1])], dim=-1)       # (M,8,3) homogeneous
    res = kpts_xy.new_zeros(kpts_xy.shape[0])
    for fam in _FAMILIES:
        lines = torch.stack(
            [torch.cross(h[:, i], h[:, j], dim=-1) for i, j in fam], dim=1)  # (M,4,3)
        lines = lines / lines[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for a, b, d in _TRIPLES:
            res = res + torch.linalg.det(
                torch.stack([lines[:, a], lines[:, b], lines[:, d]], dim=1)).pow(2)
    return res / (len(_FAMILIES) * len(_TRIPLES))


class HybridPoseLoss26(PoseLoss26):
    """Masks keypoints for bbox-only instances + class-balances the (vehicle-
    dominated) keypoint supervision."""

    def loss(self, preds, batch):
        """Stash gt classes (aligned with keypoints by batch_idx) so the keypoint
        loss can class-balance, then run the normal pose loss."""
        self._cls_flat = batch["cls"]
        return super().loss(preds, batch)

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        """Keypoint loss restricted to instances with >=1 annotated keypoint."""
        selected = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        # DDP-safe touch: keep the keypoint head (+ flow model) in the graph with
        # zero gradient even when this batch/rank has no annotated keypoints.
        kpts_loss = 0.0 * pred_kpts.sum()
        if self.flow_model is not None:
            kpts_loss = kpts_loss + 0.0 * sum(p.sum() for p in self.flow_model.parameters())
        kpts_obj_loss = 0.0 * pred_kpts.sum()
        rle_loss = (0.0 * pred_kpts.sum()) if self.rle_loss is not None else 0

        if masks.any():
            tboxes = target_bboxes / stride_tensor
            tb_m = tboxes[masks]                                   # (P,4) xyxy, grid units
            gt_kpt = selected[masks]
            wh = xyxy2xywh(tb_m)[:, 2:]
            area = wh.prod(1, keepdim=True)
            box_diag = wh.norm(dim=-1).clamp_min(1e-6)            # (P,) grid units
            pred_kpt = pred_kpts[masks]
            # Lens-invariant cuboid structure prior. Old recipe (V2_STRUCT_UNLABELED=1):
            # on EVERY positive anchor, bbox-only included -- see the note at
            # _STRUCT_UNLABELED for why that collapses the head on real photographs.
            # Default: applied below, on the annotated instances only.
            if _STRUCT_GAIN > 0 and _struct_unlabeled_active:
                kpts_loss = kpts_loss + _STRUCT_GAIN * _structure_residual(
                    pred_kpt[..., :2], box_diag).mean()
            # Box-containment prior on EVERY positive anchor. The 2D box is the exact
            # bbox of the GT corners (measured GT spread/box == 1.00), so a predicted
            # corner OUTSIDE the box is always wrong. OKS saturates (1-exp) once a
            # corner is grossly displaced -> ~0 gradient, so a finetune cannot pull an
            # exploded corner back; this relu penalty keeps a constant restoring
            # gradient (the imgsz-1280 pedestrian heal). Box-diag-normalized.
            if _CONTAIN_GAIN > 0:
                x1, y1, x2, y2 = tb_m.unbind(-1)                   # each (P,)
                kx, ky = pred_kpt[..., 0], pred_kpt[..., 1]        # (P,8)
                over = ((x1[:, None] - kx).clamp_min(0) + (kx - x2[:, None]).clamp_min(0)
                        + (y1[:, None] - ky).clamp_min(0) + (ky - y2[:, None]).clamp_min(0))
                kpts_loss = kpts_loss + _CONTAIN_GAIN * (over / box_diag[:, None]).mean()
            full_mask = (gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] >= 3
                         else torch.full_like(gt_kpt[..., 0], True))
            inst = full_mask.any(dim=1)              # instances WITH annotated keypoints
            if inst.any():
                gt_kpt, pred_kpt, area = gt_kpt[inst], pred_kpt[inst], area[inst]
                bd_i = box_diag[inst]                             # box_diag for the scale-invariant L1
                kpt_mask = full_mask[inst]
                # Class-balanced upweighting: repeat rare-class (person/bike/moto)
                # instances so the vehicle-dominated synth pose data doesn't drown
                # them out. Same gather as keypoints, with gt class as a 1-keypoint.
                if _KPT_REP and getattr(self, "_cls_flat", None) is not None:
                    cls11 = self._cls_flat.to(pred_kpts.device).float().view(-1, 1, 1)
                    sel_cls = self._select_target_keypoints(cls11, batch_idx, target_gt_idx, masks)[..., 0, 0]
                    inst_cls = sel_cls[masks][inst].long()
                    reps = torch.ones(inst_cls.shape[0], dtype=torch.long, device=inst_cls.device)
                    for c, w in _KPT_REP.items():
                        reps[inst_cls == c] = w
                    if (reps > 1).any():
                        ridx = torch.repeat_interleave(torch.arange(inst_cls.shape[0], device=inst_cls.device), reps)
                        gt_kpt, pred_kpt, area, kpt_mask, bd_i = (
                            gt_kpt[ridx], pred_kpt[ridx], area[ridx], kpt_mask[ridx], bd_i[ridx])
                        if _KPT_DBG:
                            print(f"[kptbal] cls0-7={inst_cls.bincount(minlength=8).tolist()[:8]} "
                                  f"inst {inst_cls.shape[0]}->{ridx.shape[0]}", flush=True)
                kpts_loss = kpts_loss + self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
                # Structure prior on the annotated (class-balanced) instances, where OKS
                # holds the corners in place so the prior can only tidy, not collapse.
                if _STRUCT_GAIN > 0 and not _struct_unlabeled_active:
                    kpts_loss = kpts_loss + _STRUCT_GAIN * _structure_residual(
                        pred_kpt[..., :2], bd_i).mean()
                # Scale-invariant L1 on annotated (class-balanced) corners. OKS
                # normalizes by AREA and saturates, so on large / imgsz-1280 objects
                # it yields little gradient; this box-diag-normalized L1 is non-
                # saturating with constant per-error gradient and invariant to object
                # scale -- the tightening OKS loses at 1280, and the restoring pull a
                # heal-finetune needs. Same annotated + class-balanced subset as OKS.
                if _KPT_L1_GAIN > 0:
                    d = (pred_kpt[..., :2] - gt_kpt[..., :2]).abs().sum(-1)         # (M,8) L1/corner
                    d = (d * kpt_mask).sum(-1) / kpt_mask.sum(-1).clamp_min(1)      # (M,) mean/visible
                    kpts_loss = kpts_loss + _KPT_L1_GAIN * (d / bd_i).mean()
                if self.rle_loss is not None and pred_kpt.shape[-1] in (4, 5):
                    rle_loss = rle_loss + self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask).clamp(min=0)
                if pred_kpt.shape[-1] in (3, 5):
                    kpts_obj_loss = kpts_obj_loss + self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss, rle_loss


class HybridPoseModel26(PoseModel):
    """YOLO26 pose model whose criterion masks keypoints for bbox-only instances.

    A real subclass (not a bound-method patch) so EMA ``deepcopy`` and DDP
    pickling stay correct.
    """

    def init_criterion(self):
        return E2ELoss(self, HybridPoseLoss26)


# COCO ids of the classes the keypoint head exists for. Fitness (=> best.pt) is taken
# over these only: the stock pose fitness averages pose mAP over all 80 classes, ~74 of
# which never carry keypoints and score 0, so pose contributed ~0.01 to a box-dominated
# fitness and best.pt froze at epoch 1 in every 640 run (the COCO-transfer box blip).
ROAD_CLASSES = (0, 1, 2, 3, 5, 7)   # person bicycle car motorcycle bus truck

# Head-only fine-tuning. V2_TRAIN_ONLY="cv3,one2one_cv3" trains just the named children
# of the detection head (here: the one2many and one2one CLASS branches) and freezes
# everything else, BatchNorm statistics included. Purpose (2026-09-08): heal class
# boundaries damaged by the synthetic renders -- bus is 14.8% of synth instances vs
# 0.6% of COCO, and the deployed x model labels a white van "bus" -- by a short pass
# over real bbox-only data (v2_real_cls.yaml: COCO + VisDrone x3) with the pose head
# untouched.
_TRAIN_ONLY = [c for c in os.environ.get("V2_TRAIN_ONLY", "").split(",") if c]


def _schedule_struct_prior(trainer):
    """Unlabeled prior for the first V2_STRUCT_UNLABELED_EPOCHS epochs, then annotated-only."""
    global _struct_unlabeled_active
    active = trainer.epoch < _STRUCT_UNLABELED_EPOCHS
    if active != _struct_unlabeled_active and RANK in {-1, 0}:
        print(f"[struct prior] epoch {trainer.epoch + 1}: unlabeled anchors "
              f"{'ON' if active else 'OFF (annotated only)'}", flush=True)
    _struct_unlabeled_active = active


class HybridPoseTrainer(yolo.pose.PoseTrainer):
    """PoseTrainer that uses the hybrid masked model and transfers detect weights."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Deployment proxy: score every saved epoch on the demo-reel clips against the
        # reel labels (v2/ood_reel.py). Env var so DDP subprocesses inherit it.
        reel = os.environ.get("V2_OOD_REEL", "")
        if reel and RANK in {-1, 0}:
            from v2.ood_reel import make_callback
            self.add_callback("on_model_save", make_callback(reel))
        if _STRUCT_UNLABELED_EPOCHS > 0 and not _STRUCT_UNLABELED:
            self.add_callback("on_train_epoch_start", _schedule_struct_prior)   # every rank

    def _setup_train(self):
        """Stock setup, then (V2_TRAIN_ONLY) restrict training to the named head branches."""
        super()._setup_train()
        if not _TRAIN_ONLY:
            return
        m = getattr(self.model, "module", self.model)   # unwrap DDP
        hi = len(m.model) - 1
        head = m.model[hi]
        missing = [c for c in _TRAIN_ONLY if not hasattr(head, c)]
        if missing:
            raise ValueError(f"V2_TRAIN_ONLY: head has no children {missing}; "
                             f"it has {[n for n, _ in head.named_children()]}")
        keep = tuple(f"model.{hi}.{c}." for c in _TRAIN_ONLY)
        n_train = n_frozen = 0
        for k, v in m.named_parameters():
            v.requires_grad = k.startswith(keep)
            n_train += v.numel() if v.requires_grad else 0
            n_frozen += 0 if v.requires_grad else v.numel()
        # _model_train() puts the BatchNorms of every module named here into eval mode,
        # so frozen layers keep their running statistics as well as their weights.
        self.freeze_layer_names = list(self.freeze_layer_names) + [f"model.{i}." for i in range(hi)] + [
            f"model.{hi}.{n}." for n, _ in head.named_children() if f"model.{hi}.{n}." not in keep]
        LOGGER.info(f"[train_only] trainable: {list(keep)} = {n_train / 1e6:.2f}M params; "
                    f"frozen {n_frozen / 1e6:.2f}M (weights and BN stats)")

    def validate(self):
        """Stock validation, but fitness = road-class pose mAP50-95 (+0.1 x road box
        mAP50-95), and both are logged to results.csv as metrics/road_*."""
        if self.ema and self.world_size > 1:
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        pm, bm = self.validator.metrics.pose, self.validator.metrics.box
        pose_ap = {int(c): float(a) for c, a in zip(pm.ap_class_index, pm.ap)}   # per-class AP50-95
        box_ap = {int(c): float(a) for c, a in zip(bm.ap_class_index, bm.ap)}
        present = [c for c in ROAD_CLASSES if c in pose_ap]
        road_pose = float(np.mean([pose_ap[c] for c in present])) if present else 0.0
        road_box = float(np.mean([box_ap.get(c, 0.0) for c in present])) if present else 0.0
        metrics["metrics/road_pose_mAP50-95"] = road_pose
        metrics["metrics/road_box_mAP50-95"] = road_box
        fitness = road_pose + 0.1 * road_box
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True) -> PoseModel:
        model = HybridPoseModel26(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        # Transfer YOLO26 detection weights (backbone+neck+box+cls); the keypoint
        # head and flow model stay freshly initialized. ``setup_model`` resolves
        # ``pretrained=...pt`` into a loaded model and passes it here as ``weights``.
        if weights:
            model.load(weights)
        return model
