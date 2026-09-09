#!/usr/bin/env python3
"""Offline (non-causal) refinement of tracks and 3D-box geometry.

A detector and a frame-to-frame tracker produce a per-frame *opinion*. This
module turns a whole clip of those opinions into the trajectory of a rigid
object, using only facts that hold for any road scene rather than anything
tuned to a particular video. It is offline by construction: every stage looks
both forwards and backwards. The live/streaming path is untouched.

The principles, in the order they do work
-----------------------------------------

1. **An object has one identity.** A track lost and re-spawned is the same
   object, but only if the join is consistent with how fast that object
   actually moves, judged on the ground rather than in pixels.
2. **The frame has edges, and they are not part of the world.** An object last
   seen hard against the border, moving outwards, has LEFT: nothing later in
   the clip is that object, because where it went cannot be observed. Without
   this a car exiting one side adopts an unrelated car entering a second later,
   since any tolerance that grows with the gap eventually reaches it.
3. **An identity cannot teleport.** A step that would need far more speed than
   the track ever shows is not the same object on both sides; the track is CUT
   there rather than smoothly interpolated, which would slide one identity onto
   another in plain view. A gap is not required: trackers swap targets between
   adjacent frames too.
4. **Physical speed limits are set by the scene, not by a constant.** Speed is
   measured on the ground in object lengths per second, so the traffic in the
   same shot is the reference and no calibration is needed. Nothing on foot
   outruns the cars around it.
5. **A judgement about an object should be made once, not per frame.** Whether
   its predicted cuboid is usable, what class it is, how big it is: taken by
   vote over the whole track, so nothing can flicker.
6. **Two objects cannot occupy the same ground.** Duplicate detections are
   resolved in the bird's-eye plane, where a real pair is separated and a
   double detection is stacked. Where a track has no usable footprint, image
   overlap is the fallback.
7. **A rigid object has ONE shape and ONE height.** They are estimated once
   from the whole track and posed per frame, so size is right on the first
   frame instead of converging, and per-corner noise cannot survive.
8. **Occlusion is asymmetric.** Truncation, occlusion and an object still
   emerging all make a measurement SMALLER; almost nothing makes it larger. So
   the reference size comes from the upper part of a track's distribution, not
   its median, and a rigid model is never judged on frames where the object was
   only half seen.
9. **A vehicle travels along its own length axis.** Where the trajectory proves
   motion, it gives the heading far better than four noisy corners do -- but
   only over a window short enough not to span a corner, since the chord of a
   turn is not its tangent. Where there is no motion evidence, the heading is
   HELD rather than estimated: a near-square footprint has no identifiable
   heading at all.
10. **What never moved never rotated.** A parked object gets one position and
   one orientation for its whole life.
11. **Trust a model only as far as the data can judge it.** The rigid
    reconstruction replaces the measurement when it explains the frames where
    the object was fully seen; a measurement that shakes is allowed to disagree
    more, because that disagreement is the shake being removed.

Everything is expressed in physical units -- seconds, object lengths, fractions
of the object's own box -- so the same configuration transfers to another frame
rate, resolution, or camera. :class:`RefineConfig` holds them all; call
:meth:`RefineConfig.for_fps` and the frame counts follow.

Entry points: :func:`refine_tracks` (identity and 2D geometry, before the
ground plane is solved) and :func:`refine_geometry` (3D reconstruction, after).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["RefineConfig", "refine_tracks", "refine_geometry",
           "merge_split_tracks", "majority_class", "interpolate_gaps",
           "classify_tracks", "repair_keypoints", "fill_boxes", "mute_tracks",
           "demote_tracks", "rigid_and_heading", "estimate_vertical_vp",
           "rebuild_cuboids", "report_jitter", "ego_motion", "smooth_shape",
           "dedupe_tracks", "stabilize_boxes", "drop_impossible"]


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RefineConfig:
    """Every threshold the refinement uses, in units that transfer.

    Nothing here is in frames or pixels except where the quantity really is one.
    Durations are SECONDS, speeds are per SECOND, distances on the ground are
    OBJECT LENGTHS (the clip's own median vehicle footprint, which the pipeline
    already estimates), distances in the image are fractions of the object's own
    2D box, and the rest are dimensionless ratios. Build one with
    :meth:`for_fps` and the frame counts are derived.
    """

    fps: float = 30.0

    # -- identity ---------------------------------------------------------- #
    min_track_s: float = 0.40        # shorter than this is detector flicker
    coast_tail_s: float = 0.27       # how long a lost object keeps being drawn
    merge_gap_s: float = 1.00        # longest re-spawn gap that can be one object
    max_gap_s: float = 1.50          # longest detection gap worth interpolating
    speed_span_s: float = 0.67       # window for "how fast does this track move"
    speed_ratio: float = 3.0         # a step this many times its own speed = a cut
    merge_speed_tol: float = 1.6     # ...and this many to rejoin a re-spawned id
    edge_frac: float = 0.004         # a box this close to the border is at the edge
    entry_gap_s: float = 0.23        # a miss this soon after entering is ordinary
    speed_floor_sps: float = 4.5     # ...or this many of its own box sizes / s
    ped_speed_frac: float = 0.7      # a pedestrian cannot exceed this x traffic
    veh_speed_mult: float = 2.5      # nor a vehicle this many x the traffic

    # -- one judgement per track ------------------------------------------- #
    min_fill: float = 0.5            # cuboid must cover this much of its 2D box
    min_ok_frac: float = 0.5         # ...on this fraction of its frames
    min_box_frac: float = 0.008      # ignore boxes smaller than this x image diagonal
    fill_lo: float = 0.85            # size correction is clamped to this range
    fill_hi: float = 1.8
    fill_dead: float = 0.04          # ...and skipped this close to 1

    # -- duplicates -------------------------------------------------------- #
    dup_overlap: float = 0.4         # footprints overlapping more than this...
    dup_frac: float = 0.5            # ...on this fraction of shared frames
    dup_pair_s: float = 0.33         # minimum shared time to judge a pair
    dup_box_iou: float = 0.6         # image fallback when there is no footprint
    dup_max_sep: float = 1.5         # object lengths; further apart is never one object

    # -- rigid shape and height -------------------------------------------- #
    scale_quantile: float = 75.0     # occlusion only shrinks: take size from the top
    scale_band: Tuple[float, float] = (0.75, 1.3)    # frames consistent with it
    whole_band: Tuple[float, float] = (0.75, 1.35)   # ...seen whole enough to judge on
    max_reproj: float = 0.10         # model error budget, fraction of box diagonal
    shake_cap: float = 1.5           # a shaky measurement may disagree this much more
    height_band: Tuple[float, float] = (0.5, 2.0)    # per-frame height sanity

    # -- pose -------------------------------------------------------------- #
    move_min_len: float = 1.2        # object lengths of travel before a heading counts
    straight_min: float = 0.95       # chord/path; 0.90 is a 90 deg turn, so stay high
    evidence_s: float = 0.27         # least motion evidence before trusting it
    heading_weight: float = 1.0      # a vehicle cannot rotate without moving
    max_disagree_deg: float = 45.0   # trajectory vs geometry sanity check
    still_speed_lps: float = 0.60    # object lengths per second that counts as still
    still_tol: float = 0.20          # object lengths a hold may move a footprint
    still_run_s: float = 0.27        # shortest stretch worth holding

    # -- windows ----------------------------------------------------------- #
    heading_windows_s: Tuple[float, ...] = (0.10, 0.13, 0.20, 0.30, 0.43, 0.60, 0.83)
    still_win_s: float = 0.30
    vp_win_s: float = 0.50
    shape_win_s: float = 1.03
    box_win_s: float = 0.50
    box_max_dev: float = 1.35        # a box this much over its own size is wrong
    shape_freeze_pct: float = 60.0   # freeze on the most completely seen frames

    @classmethod
    def for_fps(cls, fps: float, **kw) -> "RefineConfig":
        return cls(fps=float(fps) if fps and fps > 1e-6 else 30.0, **kw)

    # -- derived ----------------------------------------------------------- #
    def n(self, seconds: float, lo: int = 1) -> int:
        """Seconds -> frames."""
        return max(lo, int(round(seconds * self.fps)))

    def odd(self, seconds: float, lo: int = 3) -> int:
        """Seconds -> an odd frame count (filters want a centred window)."""
        w = self.n(seconds, lo)
        return w if w % 2 else w + 1

    def per_frame(self, per_second: float) -> float:
        return per_second / self.fps

    def ladder(self) -> Tuple[int, ...]:
        """Half-window sizes for the travel-direction search, shortest first."""
        out, seen = [], set()
        for t in self.heading_windows_s:
            k = self.n(t, 2)
            if k not in seen:
                seen.add(k)
                out.append(k)
        return tuple(out)


# --------------------------------------------------------------------------- #
# helpers
def _boxes_of(v) -> np.ndarray:
    return np.asarray(v.box_xyxy, dtype=np.float64)


def _bc_wh(box: np.ndarray):
    """Box centre and size (w, h), size floored at 1 px."""
    c = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    s = np.array([max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)])
    return c, s


def _norm_kpts(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    c, s = _bc_wh(box)
    return (np.asarray(kp, dtype=np.float64) - c) / s


def _denorm_kpts(kn: np.ndarray, box: np.ndarray) -> np.ndarray:
    c, s = _bc_wh(box)
    return np.asarray(kn, dtype=np.float64) * s + c


def _lerp_rows(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return a * (1.0 - w) + b * w


def _track_index(states) -> Dict[int, Dict[int, object]]:
    """id -> {frame index: view}."""
    per: Dict[int, Dict[int, object]] = {}
    for fi, s in enumerate(states):
        for v in s.tracks:
            per.setdefault(v.id, {})[fi] = v
    return per


def _observed(v) -> bool:
    """Was this view backed by a real detection on its frame?"""
    return bool(getattr(v, "obs", v.time_since_update == 0))


def _kp_ok(v) -> bool:
    return bool(getattr(v, "kp_ok", False))


def _hull_wh(kp: np.ndarray):
    kp = np.asarray(kp, dtype=np.float64)
    return (float(kp[:, 0].max() - kp[:, 0].min()),
            float(kp[:, 1].max() - kp[:, 1].min()))


# --------------------------------------------------------------------------- #
def _exit_state(states, edge_frac: float = 0.004):
    """Which tracks left the field of view, and which entered it.

    An object whose last sighting is hard against the frame border, moving
    outwards, has GONE. Nothing later in the clip is that object: it is off
    camera, and where it went cannot be observed. Re-attaching a new track to it
    is the one identity error a viewer always notices, because the box crosses
    open ground to reach a car that was never there.
    """
    if not states:
        return {}, {}
    W, H = states[0].img_wh
    m = max(2.0, edge_frac * float(np.hypot(W, H)))
    per: Dict[int, list] = {}
    for fi, s in enumerate(states):
        for v in s.tracks:
            if _observed(v):
                per.setdefault(v.id, []).append((fi, _boxes_of(v)))
    exited, entered = {}, {}
    for tid, obs in per.items():
        if len(obs) < 2:
            continue
        (fa, ba), (fp, bp) = obs[-1], obs[max(0, len(obs) - 4)]
        ca, _ = _bc_wh(ba)
        cp, _ = _bc_wh(bp)
        vel = (ca - cp) / max(fa - fp, 1)
        out = ((ba[0] <= m and vel[0] < 0) or (ba[2] >= W - m and vel[0] > 0)
               or (ba[1] <= m and vel[1] < 0) or (ba[3] >= H - m and vel[1] > 0))
        exited[tid] = bool(out)
        (fb, bb), (fn, bn) = obs[0], obs[min(len(obs) - 1, 3)]
        cb, _ = _bc_wh(bb)
        cn, _ = _bc_wh(bn)
        vin = (cn - cb) / max(fn - fb, 1)
        entered[tid] = bool((bb[0] <= m and vin[0] > 0) or (bb[2] >= W - m and vin[0] < 0)
                            or (bb[1] <= m and vin[1] > 0) or (bb[3] >= H - m and vin[1] < 0))
    return exited, entered


def merge_split_tracks(pipe, states, min_track: int = 12, coast_tail: int = 8,
                       merge_gap: int = 30, speed_tol: float = 1.6,
                       edge_frac: float = 0.004, entry_gap: int = 7,
                       verbose: bool = True) -> None:
    """Merge re-spawned ids, drop flicker tracks, trim coasting tails.

    A new id continues an old one only when every one of these holds: the class
    matches, the sizes are comparable, the old track did not leave the frame,
    the new one did not arrive through the border, the join sits near where the
    old track was heading, and -- the test that actually decides it -- the ground
    distance between them implies a speed this object could really have.

    The image-space check alone is not enough. A tolerance that grows with the
    gap lets a car that exited the right edge adopt an unrelated car entering a
    second later: measured on one clip, 171 px of separation for an object 21 px
    across. On the GROUND that join is plainly impossible, and the object's own
    speed says so without any threshold picked by hand.
    """
    world = _world_tracks(states)
    classes = {v.id: v.cls for s in states for v in s.tracks}
    scene = _scene_speed(world, classes)
    exited, entered = _exit_state(states, edge_frac)

    def _own_world_speed(tid):
        pts = world.get(tid, {})
        if len(pts) < 3:
            return 0.0
        f = sorted(pts)
        P = np.stack([pts[i] for i in f])
        d = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(f), 1)
        return float(np.percentile(d, 90)) if len(d) else 0.0

    per: Dict[int, dict] = {}
    for fi, s in enumerate(states):
        for v in s.tracks:
            d = per.setdefault(v.id, dict(cls=v.cls, obs=[]))
            if _observed(v):
                d["obs"].append((fi, _boxes_of(v)))
    n_before = len(per)

    parent = {tid: tid for tid in per}

    def root(t):
        while parent[t] != t:
            t = parent[t]
        return t

    ends = sorted((d["obs"][-1][0], tid) for tid, d in per.items() if d["obs"])
    starts = sorted((d["obs"][0][0], tid) for tid, d in per.items() if d["obs"])
    used_start, n_merged, n_exit = set(), 0, 0
    for (_, ta) in ends:
        da = per[ta]
        obs = da["obs"]
        fa, ba = obs[-1]
        if exited.get(ta):
            n_exit += 1
            if os.environ.get("V2_DEBUG_MERGE"):
                print(f"    [exit] track {ta} left the frame at {fa}, box "
                      f"{np.round(ba, 0).astype(int).tolist()}; closed")
            continue                          # off camera; it is not coming back
        ca, sa = _bc_wh(ba)
        size = float(np.sqrt(sa[0] * sa[1]))
        vel = np.zeros(2)
        if len(obs) >= 3:
            f0, b0 = obs[max(0, len(obs) - 6)]
            c0, _ = _bc_wh(b0)
            if fa > f0:
                vel = (ca - c0) / (fa - f0)
        own_w = _own_world_speed(ta)
        limit = (0.7 if da["cls"] not in (1, 2, 3, 5, 7) else 2.5) * scene
        best = None
        for (fs, tb) in starts:
            if tb == ta or tb in used_start or per[tb]["cls"] != da["cls"]:
                continue
            gap = fs - fa
            if gap <= 0:
                continue
            if gap > merge_gap:
                break
            if root(tb) == root(ta):
                continue
            if entered.get(tb) and gap > entry_gap:
                # It arrived through the border. A miss of a frame or two just
                # after an object appears is ordinary; a new track at the border
                # a second later is a new object that happens to be where the
                # old one was heading.
                continue
            bb = per[tb]["obs"][0][1]
            cb, sb = _bc_wh(bb)
            if not (0.5 < float(np.sqrt(sb[0] * sb[1])) / size < 2.0):
                continue
            wa, wb = world.get(ta, {}).get(fa), world.get(tb, {}).get(fs)
            if wa is not None and wb is not None:
                implied = float(np.hypot(*(wb - wa))) / gap
                if implied > max(speed_tol * own_w, 0.25 * limit if limit else 0.0):
                    continue                  # no object of this kind moves that fast
            err = float(np.hypot(*(cb - (ca + vel * gap))))
            tol = 0.6 * size + 2.0 * float(np.hypot(*vel)) * gap
            if err <= tol and (best is None or err < best[0]):
                best = (err, tb)
        if best is not None:
            parent[root(best[1])] = root(ta)
            used_start.add(best[1])
            n_merged += 1
            if os.environ.get("V2_DEBUG_MERGE"):
                fb = per[best[1]]["obs"][0][0]
                print(f"    [merge] {best[1]} -> {ta}: gap {fb - fa} frames, "
                      f"err {best[0]:.1f}px, size {size:.0f}px, last obs {fa}")

    if n_merged:
        for s in states:
            seen = {}
            for v in s.tracks:
                v.id = root(v.id)
                prev = seen.get(v.id)
                if prev is None or (not _observed(prev) and _observed(v)):
                    seen[v.id] = v
            s.tracks = list(seen.values())

    stat: Dict[int, dict] = {}
    for fi, s in enumerate(states):
        for v in s.tracks:
            d = stat.setdefault(v.id, dict(n=0, last=-1))
            if _observed(v):
                d["n"] += 1
                d["last"] = fi
    drop = {tid for tid, d in stat.items() if d["n"] < min_track}
    for fi, s in enumerate(states):
        keep = []
        for v in s.tracks:
            if v.id in drop:
                continue
            last = stat[v.id]["last"]
            if not _observed(v) and fi > last and fi - last > coast_tail:
                continue
            keep.append(v)
        s.tracks = keep
    if verbose:
        n_after = len({v.id for s in states for v in s.tracks})
        print(f"[refine] merge/prune: ids {n_before} -> {n_after} "
              f"(merged {n_merged}, dropped {len(drop)} short <{min_track} obs, "
              f"{n_exit} left the frame and were closed)")


def majority_class(states, verbose: bool = True) -> None:
    """Give every track one class (its per-frame argmax flickers)."""
    votes: Dict[int, Dict[int, int]] = {}
    for s in states:
        for v in s.tracks:
            votes.setdefault(v.id, {})
            votes[v.id][v.cls] = votes[v.id].get(v.cls, 0) + 1
    fixed = {tid: max(c.items(), key=lambda kv: kv[1])[0] for tid, c in votes.items()}
    n = 0
    for s in states:
        for v in s.tracks:
            if v.cls != fixed[v.id]:
                v.cls = fixed[v.id]
                n += 1
    if verbose and n:
        print(f"[refine] class vote: {n} view(s) relabelled to the track majority")


def _step_speeds(fr, obs):
    """Per-frame speeds between ADJACENT observed frames, in own box sizes.

    Only adjacent frames, because a pair separated by a gap may be the very
    teleport under test -- letting it into the baseline would raise the bar
    enough to excuse itself.
    """
    out = {}
    for a, b in zip(obs[:-1], obs[1:]):
        if b != a + 1:
            continue
        ca, sa = _bc_wh(_boxes_of(fr[a]))
        cb, _ = _bc_wh(_boxes_of(fr[b]))
        out[a] = float(np.hypot(*(cb - ca))) / float(np.sqrt(max(sa[0] * sa[1], 1.0)))
    return out


def _local_speed(speeds, a, b, span: int = 20) -> float:
    """The object's own top speed just before and just after a gap."""
    v = [s for f, s in speeds.items() if a - span <= f <= b + span]
    if len(v) < 3:
        v = list(speeds.values())
    return float(np.percentile(v, 90)) if v else 0.0


def _world_tracks(states):
    """Ground-contact position of every track per frame, in object lengths.

    The bottom edge of a 2D box lies on the ground whatever the object is, so
    this works for tracks with no usable cuboid too.
    """
    from homography_rt import apply_homography
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fi, s in enumerate(states):
        unit = float(s.bev_unit or 0.0)
        if unit <= 1e-9:
            continue
        H = np.asarray(s.H, dtype=np.float64)
        for v in s.tracks:
            b = _boxes_of(v)
            p = apply_homography(np.array([[(b[0] + b[2]) * 0.5, b[3]]]), H)[0] / unit
            if np.isfinite(p).all():
                out.setdefault(v.id, {})[fi] = p
    return out


def _scene_speed(world, classes, vel_classes=(1, 2, 3, 5, 7)) -> float:
    """How fast the traffic in this scene actually moves, in lengths per frame."""
    v = []
    for tid, pts in world.items():
        if classes.get(tid) not in vel_classes or len(pts) < 10:
            continue
        f = sorted(pts)
        P = np.stack([pts[i] for i in f])
        d = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(f), 1)
        v.extend(d.tolist())
    return float(np.percentile(v, 90)) if len(v) >= 20 else 0.0


def interpolate_gaps(states, gi: Optional[List[int]], max_gap: int = 45,
                     speed_ratio: float = 3.0, speed_floor: float = 0.15,
                     span: int = 20, min_track: int = 12,
                     verbose: bool = True) -> None:
    """Fill missing/coasted frames inside a track's life by interpolation.

    Box corners interpolate linearly; keypoints interpolate in the *box frame*
    and are re-attached to the interpolated box, so a filled frame follows the
    object's real motion rather than a decaying constant-velocity guess.
    """
    next_id = max((v.id for s in states for v in s.tracks), default=0) + 1
    n_filled = n_split = 0

    # Pass 1: cut identities that cannot be the same object. Would getting from
    # one observation to the next require this object to move far faster than it
    # ever does? Then it is not the same object on both sides -- and it does not
    # need a gap for that: the tracker can swap targets between adjacent frames.
    # Interpolating across such a step slides one identity onto another in plain
    # view; two honest identities beat one smooth lie. Repeated until stable,
    # since one track can swap more than once.
    for _ in range(8):
        per = _track_index(states)
        cuts = {}
        for tid, fr in per.items():
            obs = sorted(f for f, v in fr.items() if _observed(v))
            if len(obs) < 3:
                continue
            speeds = _step_speeds(fr, obs)
            for a_, b_ in zip(obs[:-1], obs[1:]):
                if b_ - a_ > max_gap:
                    continue
                ca, sa = _bc_wh(_boxes_of(fr[a_]))
                cb, _ = _bc_wh(_boxes_of(fr[b_]))
                sz = float(np.sqrt(max(sa[0] * sa[1], 1.0)))
                implied = float(np.hypot(*(cb - ca))) / sz / (b_ - a_)
                if implied > max(speed_ratio * _local_speed(speeds, a_, b_, span),
                                 speed_floor):
                    cuts[tid] = b_
                    break
        if not cuts:
            break
        for tid, cut in cuts.items():
            for f, v in per[tid].items():
                if f >= cut:
                    v.id = next_id
            next_id += 1
            n_split += 1

    # Fragments left by the cuts are not objects; drop them.
    if n_split:
        seen: Dict[int, int] = {}
        for s in states:
            for v in s.tracks:
                seen[v.id] = seen.get(v.id, 0) + int(_observed(v))
        gone = {t for t, n in seen.items() if n < min_track}
        if gone:
            for s in states:
                s.tracks = [v for v in s.tracks if v.id not in gone]

    # Pass 2: fill the gaps that survived.
    per = _track_index(states)
    for tid, fr in per.items():
        obs = sorted(f for f, v in fr.items() if _observed(v))
        if len(obs) < 2:
            continue
        ref = fr[obs[0]]
        for a, b in zip(obs[:-1], obs[1:]):
            if b - a <= 1 or b - a > max_gap:
                continue
            va, vb = fr[a], fr[b]
            box_a, box_b = _boxes_of(va), _boxes_of(vb)
            ka = np.asarray(va.kpts, dtype=np.float64)
            kb = np.asarray(vb.kpts, dtype=np.float64)
            has_kp = ka.shape == kb.shape and ka.ndim == 2 and ka.shape[0] >= 8
            if has_kp:
                na, nb = _norm_kpts(ka, box_a), _norm_kpts(kb, box_b)
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                box = _lerp_rows(box_a, box_b, w)
                kp = (_denorm_kpts(_lerp_rows(na, nb, w), box) if has_kp
                      else np.zeros((0, 2)))
                ground = (kp[list(gi)].copy() if (has_kp and gi and
                                                  kp.shape[0] >= max(gi) + 1)
                          else np.zeros((0, 2)))
                v = fr.get(f)
                if v is None:
                    v = type(ref)(tid, ref.cls, kp, ground, box, 0)
                    states[f].tracks.append(v)
                    fr[f] = v
                else:
                    v.kpts, v.ground, v.box_xyxy = kp, ground, box
                v.cls = ref.cls
                v.time_since_update = 0
                v.obs = False
                v.interp = True
                v.kp_ok = _kp_ok(va) and _kp_ok(vb)
                n_filled += 1
    if verbose:
        extra = f", {n_split} track(s) cut at an implausible jump" if n_split else ""
        print(f"[refine] gap fill: {n_filled} frame(s) interpolated inside tracks{extra}")


def classify_tracks(states, min_fill: float = 0.5, min_ok_frac: float = 0.5,
                    min_box_px: float = 12.0, verbose: bool = True):
    """Decide ONCE per track whether its predicted cuboid is usable.

    Returns ``(solid_ids, boxonly_ids, fill_ratio)`` where ``fill_ratio`` maps a
    solid track id to the per-axis (w, h) correction that makes its cuboid fill
    its 2D box.
    """
    per = _track_index(states)
    solid, boxonly, ratios = set(), set(), {}
    for tid, fr in per.items():
        ok_flags, rw, rh = [], [], []
        for f, v in fr.items():
            if not _observed(v):
                continue
            ok_flags.append(_kp_ok(v))
            kp = np.asarray(v.kpts, dtype=np.float64)
            if not _kp_ok(v) or kp.ndim != 2 or kp.shape[0] < 8:
                continue
            box = _boxes_of(v)
            _, s = _bc_wh(box)
            if min(s) < min_box_px:
                continue
            hw, hh = _hull_wh(kp)
            if hw > 1e-6 and hh > 1e-6:
                rw.append(s[0] / hw)
                rh.append(s[1] / hh)
        frac_ok = float(np.mean(ok_flags)) if ok_flags else 0.0
        if not rw:                       # never measurable: fall back to the vote
            (solid if frac_ok >= min_ok_frac else boxonly).add(tid)
            continue
        mw, mh = float(np.median(rw)), float(np.median(rh))
        # Judge coverage on the geometric mean of the two axes (i.e. on AREA),
        # not the worse axis: some classes are legitimately narrow in one
        # dimension -- a pedestrian's 3D box really is much thinner than its 2D
        # silhouette -- while a broken cuboid, like the tangle the keypoint head
        # emits for a bus at an oblique drone angle, is small in BOTH.
        usable = (frac_ok >= min_ok_frac
                  and float(np.sqrt(mw * mh)) <= (1.0 / max(min_fill, 1e-3)))
        if usable:
            solid.add(tid)
            ratios[tid] = (mw, mh)
        else:
            boxonly.add(tid)
    if verbose:
        from uod import style
        cls_of = {v.id: v.cls for s in states for v in s.tracks}
        tally: Dict[str, int] = {}
        for tid in boxonly:
            cat = style.CATEGORY_LABEL[style.class_category(cls_of.get(tid, -1), "pose")]
            tally[cat] = tally.get(cat, 0) + 1
        why = ", ".join(f"{v} {k}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
        print(f"[refine] cuboid vote: {len(solid)} solid, {len(boxonly)} demoted "
              f"to 2D-only (whole-track decision)" + (f" [{why}]" if why else ""))
    return solid, boxonly, ratios


def repair_keypoints(states, solid_ids, gi: Optional[List[int]],
                     verbose: bool = True) -> None:
    """Replace untrusted keypoints on solid tracks by box-frame interpolation."""
    per = _track_index(states)
    n = 0
    for tid, fr in per.items():
        if tid not in solid_ids:
            continue
        frames = sorted(fr)
        good = [f for f in frames
                if _kp_ok(fr[f]) and np.asarray(fr[f].kpts).ndim == 2
                and np.asarray(fr[f].kpts).shape[0] >= 8]
        if not good or len(good) == len(frames):
            continue
        norm = {f: _norm_kpts(np.asarray(fr[f].kpts, dtype=np.float64),
                              _boxes_of(fr[f])) for f in good}
        for f in frames:
            if f in norm:
                continue
            before = [g for g in good if g < f]
            after = [g for g in good if g > f]
            if before and after:
                a, b = before[-1], after[0]
                w = (f - a) / (b - a)
                kn = _lerp_rows(norm[a], norm[b], w)
            else:
                kn = norm[before[-1]] if before else norm[after[0]]
            box = _boxes_of(fr[f])
            kp = _denorm_kpts(kn, box)
            fr[f].kpts = kp
            if gi and kp.shape[0] >= max(gi) + 1:
                fr[f].ground = kp[list(gi)].copy()
            fr[f].kp_ok = True
            n += 1
    if verbose and n:
        print(f"[refine] keypoint repair: {n} frame(s) rebuilt from trusted neighbours")


def fill_scale(ratio, lo: float = 0.85, hi: float = 1.8, dead: float = 0.04):
    """One isotropic size correction for a track, or None if it already fits.

    ``ratio`` is the per-axis (width, height) ratio of the detector's 2D box to
    the cuboid's 2D hull. Their geometric mean is the linear correction implied
    by AREA, which is the honest reading of "the estimated box is too small":
    the object's 3D box is under-sized by this factor.
    """
    if ratio is None:
        return None
    s = float(np.clip(np.sqrt(max(ratio[0], 1e-6) * max(ratio[1], 1e-6)), lo, hi))
    return None if abs(s - 1.0) < dead else s


def fill_boxes(states, ratios, gi: Optional[List[int]], lo: float = 0.85,
               hi: float = 1.8, dead: float = 0.04, verbose: bool = True) -> None:
    """2D fallback: scale a cuboid in the image so its hull fills its 2D box.

    Used only for tracks the rigid reconstruction could not model, where there
    is no 3D box to enlarge. It scales about the hull centre, which is NOT a
    valid ground-plane operation -- the scaled ground corners stop being the
    projection of any rectangle on the ground -- so it must never be applied
    before the reconstruction. Tracks that ARE reconstructed get the size
    correction applied to the 3D box instead (footprint and height), which is
    what the same evidence actually means.
    """
    per = _track_index(states)
    n_tracks = 0
    for tid, (mw, mh) in ratios.items():
        sw = float(np.clip(mw, lo, hi))
        sh = float(np.clip(mh, lo, hi))
        if abs(sw - 1.0) < dead and abs(sh - 1.0) < dead:
            continue
        n_tracks += 1
        for f, v in per.get(tid, {}).items():
            kp = np.asarray(v.kpts, dtype=np.float64)
            if kp.ndim != 2 or kp.shape[0] < 8:
                continue
            c = np.array([(kp[:, 0].min() + kp[:, 0].max()) * 0.5,
                          (kp[:, 1].min() + kp[:, 1].max()) * 0.5])
            kp = c + (kp - c) * np.array([sw, sh])
            v.kpts = kp
            if gi and kp.shape[0] >= max(gi) + 1:
                v.ground = kp[list(gi)].copy()
    if verbose:
        print(f"[refine] box fill: {n_tracks} track(s) rescaled to fill their 2D box")


def mute_tracks(states, boxonly_ids, verbose: bool = True) -> None:
    """Strip the unusable cuboids, but keep the tracks in the 3D channel for now.

    Their corners are wrong, so they must not reach the ground-plane solver; but
    leaving the track in place lets the v1 zero-phase pass smooth its 2D box
    before it is handed to the 2D-only channel, which is what stops the dashed
    marker jittering.
    """
    if not boxonly_ids:
        return
    n = 0
    for s in states:
        for v in s.tracks:
            if v.id in boxonly_ids:
                v.kpts = np.zeros((0, 2))
                v.ground = np.zeros((0, 2))
                v.bev_quad = None
                n += 1
    if verbose and n:
        print(f"[refine] muted {len(boxonly_ids)} unusable cuboid(s) before the "
              f"ground-plane solve")


def demote_tracks(pipe, states, boxonly_ids, verbose: bool = True) -> None:
    """Move whole tracks out of the 3D channel into the 2D-only (aux) channel.

    They keep the tracker's smoothed box, so the dashed marker is stable for
    the track's whole life instead of flickering with the per-frame gate.
    """
    if not boxonly_ids:
        return
    from uod.keypoints import Detection
    n = 0
    for s in states:
        keep, moved = [], []
        for v in s.tracks:
            if v.id in boxonly_ids:
                moved.append(v)
            else:
                keep.append(v)
        if not moved:
            continue
        s.tracks = keep
        dets = list(s.aux_dets)
        cen = [np.asarray(s.aux_centers, dtype=np.float64).reshape(-1, 2)]
        extra = []
        for v in moved:
            box = _boxes_of(v)
            d = Detection(cls=int(v.cls), conf=1.0, xyxy=box)
            d.is_aux = True
            d.track_id = int(v.id)
            dets.append(d)
            extra.append([(box[0] + box[2]) * 0.5, box[3]])
            n += 1
        if extra:
            cen.append(np.asarray(extra, dtype=np.float64))
        s.aux_dets = dets
        s.aux_centers = np.vstack([c for c in cen if len(c)])
    if verbose:
        print(f"[refine] demoted {len(boxonly_ids)} track(s) to 2D-only "
              f"({n} frame markers)")


# --------------------------------------------------------------------------- #
# rigid footprint (whole-track MLE) + trajectory heading
def _rot(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _kabsch_angle(Q: np.ndarray, S: np.ndarray) -> float:
    """Angle t minimising ||Q - S R(t)^T|| for matched, centred 4x2 shapes."""
    num = float((S[:, 0] * Q[:, 1] - S[:, 1] * Q[:, 0]).sum())
    den = float((S[:, 0] * Q[:, 0] + S[:, 1] * Q[:, 1]).sum())
    return float(np.arctan2(num, den))


def _best_alignment(Q: np.ndarray, S: np.ndarray):
    """Best cyclic corner shift + rotation of ``S`` onto ``Q``.

    The model can emit the four ground corners starting from a different corner
    on a later frame; without the shift search that shows up as a 90 degree
    flip of the rendered footprint.
    """
    best = (0, 0.0, np.inf)
    for k in range(4):
        Qk = np.roll(Q, -k, axis=0)
        t = _kabsch_angle(Qk, S)
        r = float(((Qk - S @ _rot(t).T) ** 2).sum())
        if r < best[2]:
            best = (k, t, r)
    return best


def _quad_scale(q: np.ndarray) -> float:
    """Linear size of a centred quad (root of its area)."""
    return float(np.sqrt(abs(np.cross(q[2] - q[0], q[3] - q[1])) * 0.5))


def _shape_mle(Qs: List[np.ndarray], iters: int = 4, hi_q: float = 75.0,
               band: Tuple[float, float] = (0.75, 1.3),
               min_keep: int = 6) -> np.ndarray:
    """The constant footprint of a rigid object, from its best observations.

    A rigid object has ONE footprint, so any frame-to-frame change in the
    measured one is error -- and that error is NOT symmetric. Occlusion,
    truncation at the frame edge, and an object still emerging from behind
    something all make the measured footprint SMALLER; almost nothing makes it
    larger. Averaging over the whole track therefore biases the shape down and
    lands on a size the object never had: a bus entering from behind a tree was
    measured at 0.88 object lengths early and 4.38 late, and the plain median
    settled at 3.92 -- fitting neither half.

    So the reference scale is taken from the UPPER part of the distribution
    (``hi_q``), the frames whose footprint is consistent with it are selected,
    and the generalised-Procrustes mean is computed from those alone. That is
    the object seen whole; the rest were partial views of the same object.
    """
    if not Qs:
        return np.zeros((4, 2))
    sc = np.array([_quad_scale(q) for q in Qs])
    good = sc > 1e-9
    keep = np.zeros(len(Qs), bool)
    if good.sum() >= min_keep:
        target = float(np.percentile(sc[good], hi_q))
        keep = good & (sc >= band[0] * target) & (sc <= band[1] * target)
    if keep.sum() < min_keep:
        keep = np.ones(len(Qs), bool)
    sel = [Qs[i] for i in np.flatnonzero(keep)]

    areas = [abs(np.cross(q[2] - q[0], q[3] - q[1])) * 0.5 for q in sel]
    S = sel[int(np.argsort(areas)[len(areas) // 2])].copy()
    for _ in range(iters):
        acc = []
        for Q in sel:
            k, t, _ = _best_alignment(Q, S)
            acc.append(np.roll(Q, -k, axis=0) @ _rot(t))    # bring Q into S's frame
        S = np.median(np.stack(acc), axis=0)
        S -= S.mean(axis=0)
    return S


def _axis_angle(S: np.ndarray) -> float:
    """Angle of the shape's long (length) axis in its own frame."""
    e = np.roll(S, -1, axis=0) - S
    u = 0.5 * (e[0] - e[2])
    v = 0.5 * (e[1] - e[3])
    q = v if (v @ v) >= (u @ u) else u
    return float(np.arctan2(q[1], q[0]))


def _smooth(a: np.ndarray, win: int, poly: int) -> np.ndarray:
    from uod.smoothing import _smooth1d
    if len(a) < max(5, poly + 2) or win < 3:
        return a
    return _smooth1d(np.asarray(a, dtype=np.float64), win, poly)


def _wrap_pi(a):
    """Wrap an angle (or array) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _wrap_half(a):
    """Wrap to (-pi/2, pi/2]: a rectangle's heading is 180-degree symmetric,
    so this is the SMALLEST rotation that achieves a given orientation."""
    return np.arctan2(np.sin(2.0 * a), np.cos(2.0 * a)) * 0.5


def _unwrap_half(a: np.ndarray) -> np.ndarray:
    """Continuous representative of an orientation series, modulo PI.

    A rectangle's heading is 180-degree symmetric, so a series of orientations
    may legitimately fold at +/-90 degrees. Smoothing across such a fold averages
    two ends of the same orientation and returns one 90 degrees out -- the single
    most persistent bug in this file. Anything that is about to be filtered in
    time must go through here first.
    """
    out = np.empty(len(a), dtype=np.float64)
    prev = None
    for i, x in enumerate(a):
        out[i] = prev = float(x) if prev is None else prev + _wrap_half(x - prev)
    return out


def _traj_heading(cs: np.ndarray, d_min: float, straight: float,
                  ladder=(3, 4, 6, 9, 13, 18, 25)) -> np.ndarray:
    """Per-frame direction of travel, or NaN where the motion proves nothing.

    Evidence for frame ``i`` is the shortest centred window over which the
    object's NET displacement reaches ``d_min`` object lengths while staying
    straight (net >= ``straight`` x path). The straightness bar has to be high:
    the chord of a 90 degree turn is still 0.90 of its arc, so a loose gate
    happily returns the CHORD of a cornering vehicle instead of its tangent, and
    the box then sits at an angle to the vehicle for the whole corner. A parked vehicle, or one
    creeping inside detector noise, never clears the bar and keeps NaN -- which
    is the whole point: its apparent "velocity" is noise, and an earlier version
    of this that used a plain per-frame gradient rotated every stopped car in a
    queue to a random heading.
    """
    T = len(cs)
    phi = np.full(T, np.nan)
    if T < 3:
        return phi
    step = np.hypot(*np.diff(cs, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    for i in range(T):
        for h in ladder:
            a, b = max(0, i - h), min(T - 1, i + h)
            if b - a < 2:
                continue
            d = cs[b] - cs[a]
            n = float(np.hypot(d[0], d[1]))
            path = float(cum[b] - cum[a])
            if n >= d_min and n >= straight * path:
                phi[i] = np.arctan2(d[1], d[0])
                break
    return phi


def _fill_mod_pi(values: np.ndarray, known: np.ndarray, T: int) -> np.ndarray:
    """Interpolate/extend an orientation series known on ``known`` frames.

    Interpolation happens on the DOUBLED angle, so it is continuous modulo pi
    and cannot be corrupted by the arbitrary +/-180 degree branch of each
    sample. Ends are held.
    """
    idx = np.flatnonzero(known)
    x = np.arange(T, dtype=np.float64)
    c2 = np.interp(x, idx, np.cos(2.0 * values[idx]))
    s2 = np.interp(x, idx, np.sin(2.0 * values[idx]))
    return np.arctan2(s2, c2) * 0.5


def rigid_and_heading(pipe, states, fixed: bool, use_velocity: bool = True,
                      vel_classes=(1, 2, 3, 5, 7), win: int = 11, poly: int = 2,
                      d_min: float = 1.2, straight: float = 0.95,
                      min_evidence: int = 8, weight: float = 1.0,
                      hold_still: bool = True, v_still: float = 0.02,
                      still_tol: float = 0.2, max_disagree: float = np.pi / 4,
                      ladder: Tuple[int, ...] = (3, 4, 6, 9, 13, 18, 25),
                      still_run: int = 8, still_win: int = 9,
                      verbose: bool = True) -> None:
    """Constant per-track footprint + a heading that cannot flip (offline).

    Two physical facts do the work:

    * A vehicle is RIGID, so its ground footprint is one constant shape for the
      whole clip. Estimating it from every observation at once (generalised
      Procrustes with a cyclic-corner search) gives the same size on the first
      frame as on the last -- the causal estimator only converged after ~20
      observations, which is why large objects like buses visibly settled part
      way through a shot.
    * A vehicle travels along its length axis. Where a track's own trajectory
      proves it moved (see :func:`_traj_heading`), that direction is a far
      better orientation estimate than four noisy corners on one frame:
      measured over these clips the corner geometry agrees with the trajectory
      to ~4 degrees in the median but flips by up to 89 degrees on isolated
      frames (the length and width axes swapping). Where a track never moves,
      its orientation is instead held CONSTANT at the circular median of its
      geometric heading -- it did not move, so it did not rotate.

    ``fixed`` says the rectified ground frame is world-consistent across the
    clip (static or ego-motion-compensated camera). Trajectory headings are
    only meaningful then, so on a freely moving camera this falls back to the
    smoothed per-frame geometry.
    """
    from homography_rt import apply_homography

    per: Dict[int, dict] = {}
    for fi, s in enumerate(states):
        unit = float(s.bev_unit or 0.0)
        if unit <= 1e-9:
            continue
        for v in s.tracks:
            g = np.asarray(v.ground, dtype=np.float64)
            if g.shape != (4, 2) or not np.isfinite(g).all():
                continue
            P = apply_homography(g, np.asarray(s.H, dtype=np.float64))
            if not np.isfinite(P).all():
                continue
            d = per.setdefault(v.id, dict(P=[], v=[], unit=[], f=[], cls=v.cls))
            d["P"].append(P / unit)
            d["v"].append(v)
            d["unit"].append(unit)
            d["f"].append(fi)

    # Pass 1: the constant shape and the raw orientation series of every track.
    for tid, d in per.items():
        P = np.stack(d["P"])                       # (T,4,2), object-size units
        d["c"] = P.mean(axis=1)
        Q = P - d["c"][:, None, :]
        S = _shape_mle([q for q in Q])
        d["S"] = pipe._rect_shape(S) if getattr(pipe, "snap_rect", True) else S
        # Unwrap modulo PI, not 2*pi: the footprint is a rectangle, so t and
        # t+180deg draw the SAME polygon, and letting a 180deg step survive
        # would make the smoother average across it and return an orientation
        # 90deg out (the length and width axes swapped).
        d["theta"] = _unwrap_half(
            np.array([_best_alignment(q, d["S"])[1] for q in Q]))

    # Common-mode rotation of the rectified frame. When the camera moves, every
    # object's apparent orientation turns with it; the median inter-frame turn
    # over all tracks IS that camera rotation. Removing it lets each object's
    # own orientation -- which really does change slowly -- be smoothed without
    # the camera's motion being smoothed away with it. It is ~0 for a static
    # camera, so this is a no-op there.
    n_frames = len(states)
    steps = [[] for _ in range(n_frames)]
    for d in per.values():
        f, th = d["f"], d["theta"]
        for i in range(1, len(f)):
            if f[i] == f[i - 1] + 1:
                steps[f[i]].append(th[i] - th[i - 1])
    delta = np.array([np.median(x) if len(x) >= 3 else 0.0 for x in steps])
    G = np.cumsum(delta)

    n_shape = n_vel = n_const = n_still = 0
    for tid, d in per.items():
        if len(d["P"]) < 3:
            continue
        c, S, theta, f = d["c"], d["S"], d["theta"], np.asarray(d["f"])
        cs = np.stack([_smooth(c[:, 0], win, poly),
                       _smooth(c[:, 1], win, poly)], axis=1)
        cs_cam = cs
        # Motion evidence, measured on the UNFROZEN trajectory: it decides both
        # the heading below and whether this object ever went anywhere at all.
        is_vehicle = int(d["cls"]) in vel_classes
        phi = have = None
        if fixed and len(cs) >= 6:
            phi = _traj_heading(cs_cam, d_min, straight, ladder)
            have = ~np.isnan(phi)
        if hold_still and fixed:
            # A vehicle that is standing still is not moving at all, so freeze
            # it rather than let the estimate wander (measured drift on parked
            # vehicles reached most of a car length over a single clip). A
            # vehicle with NO motion evidence anywhere in its life is parked, so
            # it is pinned outright; anything else is only flattened over the
            # stretches where it is stopped, and by a bounded amount, so a
            # stop-and-go vehicle never jumps when it pulls away.
            parked = bool(is_vehicle and have is not None and have.sum() == 0)
            held = _still_flatten(cs, v_still, still_tol, min_run=still_run,
                                  win=still_win, full_freeze=parked)
            if not np.allclose(held, cs):
                n_still += 1
            cs = np.stack([_smooth(held[:, 0], win, poly),
                           _smooth(held[:, 1], win, poly)], axis=1)
        if fixed:
            # Smooth the object's OWN rotation (camera rotation added back
            # after), so a turning camera is not smoothed away with it.
            g = G[f]
            ts = _smooth(theta - g, win, poly) + g
        else:
            # Freely moving camera: the rectified frame is not a world frame,
            # so re-deriving the orientation here is noisier than what the v1
            # zero-phase pose smoother already produced. Keep ITS heading and
            # only swap in the constant whole-track shape (which is what fixes
            # a large object's size still converging part way through a shot).
            a_s = _axis_angle(S)
            ts = _unwrap_half(np.array([
                (_axis_angle(np.asarray(v.bev_quad, dtype=np.float64))- a_s)
                if getattr(v, "bev_quad", None) is not None
                and np.asarray(v.bev_quad).shape == (4, 2) else theta[i]
                for i, v in enumerate(d["v"])]))

        if fixed and have is not None:
            # A near-square footprint has no identifiable heading -- the fit can
            # sit at either axis -- so estimating one per frame just tracks
            # noise. Anything without solid trajectory evidence is held at a
            # single orientation instead; for a pedestrian, whose footprint is
            # essentially square, that is all the information there is.
            tgt = (_fill_mod_pi(phi - _axis_angle(S), have, len(cs))
                   if have.any() else None)
            # Two independent estimates of the same angle: where the object
            # went, and how its corners lie. They should broadly agree -- a
            # vehicle travels along its own length. When they disagree grossly
            # the trajectory is not measuring this object's motion (a long
            # vehicle whose footprint centre slides along its body, an
            # imperfectly compensated camera), and forcing it produces a box
            # visibly across the vehicle. Fall back to one held orientation.
            agree = True
            if tgt is not None:
                dev = np.abs(_wrap_half(tgt[have] - ts[have]))
                agree = float(np.median(dev)) <= max_disagree
                if os.environ.get("V2_DEBUG_HEAD") and int(os.environ["V2_DEBUG_HEAD"]) == tid:
                    print(f"    [head] id {tid} trajectory-vs-geometry median "
                          f"{np.degrees(np.median(dev)):.1f}deg -> "
                          f"{'trajectory' if agree else 'HELD (disagree)'}")
            if (use_velocity and is_vehicle and tgt is not None
                    and have.sum() >= min_evidence and agree):
                # Travel direction -> the shape's length axis must be parallel.
                ts = _smooth(_unwrap_half(ts + weight * _wrap_half(tgt - ts)),
                             win, poly)
                n_vel += 1
            else:
                # No usable travel direction: one constant orientation.
                m = np.arctan2(np.sin(2 * theta).mean(), np.cos(2 * theta).mean()) * 0.5
                ts = np.full(len(ts), ts[0] + _wrap_half(m - ts[0]))
                n_const += 1

        if os.environ.get("V2_DEBUG_HEAD") and int(os.environ["V2_DEBUG_HEAD"]) == tid:
            e = _axis_angle(S)
            asp = None
            ee = np.roll(S, -1, axis=0) - S
            u2, v2 = 0.5 * (ee[0] - ee[2]), 0.5 * (ee[1] - ee[3])
            a1, b1 = np.hypot(*u2), np.hypot(*v2)
            asp = max(a1, b1) / max(min(a1, b1), 1e-9)
            print(f"    [head] id {tid} branch={'traj' if n_vel and have is not None and use_velocity and is_vehicle and have.sum() >= min_evidence else 'const'}"
                  f" evidence={0 if have is None else int(have.sum())}/{len(cs)}"
                  f" aspect={asp:.2f} axis={np.degrees(e):+.1f}"
                  f" heading span={np.degrees(ts.max()-ts.min()):.1f}deg"
                  f" median={np.degrees(np.median(ts)):+.1f}")
        for i, v in enumerate(d["v"]):
            v.bev_quad = (S @ _rot(ts[i]).T + cs[i]) * d["unit"][i]
            v.cam_quad = (S @ _rot(ts[i]).T + cs_cam[i]) * d["unit"][i]
        n_shape += 1
    if verbose:
        tail = ("" if fixed else "; trajectory headings skipped "
                "(camera frame is not world-locked)")
        print(f"[refine] rigid shape: {n_shape} track(s) re-posed from a whole-track "
              f"MLE; heading from trajectory on {n_vel}, held constant on "
              f"{n_const}, standing still held on {n_still}{tail}")

# --------------------------------------------------------------------------- #
# Physical cuboid reconstruction: a rigid box on a ground plane, seen through
# the recovered homography and a vertical vanishing point.
def _still_flatten(c: np.ndarray, v_still: float = 0.02, tol: float = 0.2,
                   min_run: int = 8, win: int = 9,
                   full_freeze: bool = False) -> np.ndarray:
    """Hold a track's ground position CONSTANT while it is standing still.

    A parked vehicle does not move, so every wiggle in its estimated position is
    error. Runs of frames whose windowed speed is below ``v_still`` are replaced
    by their median position -- but only in chunks whose spread stays within
    ``tol``, so a slowly drifting estimate (an imperfectly compensated moving
    camera) becomes a gentle staircase that the following smoother turns into a
    slow drift, instead of being snapped somewhere it visibly does not belong.
    Distances are in object lengths.
    """
    T = len(c)
    if T < min_run:
        return c
    h = max(1, win // 2)
    disp = np.empty(T)
    for i in range(T):
        a, b = max(0, i - h), min(T - 1, i + h)
        disp[i] = float(np.hypot(*(c[b] - c[a]))) / max(b - a, 1)
    still = disp < v_still
    out = c.copy()
    if full_freeze:
        # Never moved at any point in its life: there is no later motion for a
        # jump to be visible against, so pin it to one position outright.
        out[:] = np.median(c, axis=0)
        return out
    i = 0
    while i < T:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and still[j + 1]:
            j += 1
        if j - i + 1 >= min_run:
            s0 = i
            while s0 <= j:
                e = s0
                while e + 1 <= j:
                    seg = c[s0:e + 2]
                    if float(np.hypot(*(seg.max(axis=0) - seg.min(axis=0)))) > tol:
                        break
                    e += 1
                if e - s0 + 1 >= min_run:
                    out[s0:e + 1] = np.median(c[s0:e + 1], axis=0)
                s0 = e + 1
        i = j + 1
    return out


def _pose_quad(v):
    """The footprint pose the CAMERA panel should use (see rigid_and_heading)."""
    q = getattr(v, "cam_quad", None)
    return v.bev_quad if q is None else q


def _vp_from_lines(L: np.ndarray, iters: int = 8) -> np.ndarray:
    """Robust homogeneous least-squares intersection of image lines (IRLS)."""
    w = np.ones(len(L))
    v = np.array([0.0, 1.0, 0.0])
    for _ in range(iters):
        _, _, Vt = np.linalg.svd(L * w[:, None], full_matrices=False)
        v = Vt[-1]
        r = np.abs(L @ v)
        sc = float(np.median(r)) + 1e-9
        w = 1.0 / (1.0 + (r / (3.0 * sc)) ** 2)
    return v


def estimate_vertical_vp(states, gi, win: int = 15, min_lines: int = 16,
                         verbose: bool = True):
    """Per-frame vertical vanishing point from the cuboids' own vertical edges.

    The four vertical edges of a cuboid are parallel in 3D, so under any pinhole
    camera their images are CONCURRENT -- they meet at the vertical vanishing
    point. Pooling every object's edges over a sliding temporal window estimates
    that point far better than any single object can, and it is exactly the
    invariant the v2 model was trained to respect. Returned in homogeneous
    coordinates, so a near-orthographic aerial view (vanishing point at
    infinity) needs no special case.
    """
    gset = set(gi or [])
    pairs = list(zip(sorted(gset), sorted(i for i in range(8) if i not in gset)))
    if len(pairs) != 4:
        return None
    per = []
    for s in states:
        L = []
        for v in s.tracks:
            kp = np.asarray(v.kpts, dtype=np.float64)
            if kp.shape != (8, 2) or not np.isfinite(kp).all():
                continue
            for (a, b) in pairs:
                g = np.array([kp[a, 0], kp[a, 1], 1.0])
                t = np.array([kp[b, 0], kp[b, 1], 1.0])
                if np.hypot(*(t[:2] - g[:2])) < 3.0:
                    continue
                line = np.cross(g, t)
                n = float(np.linalg.norm(line[:2]))
                if n > 1e-9:
                    L.append(line / n)
        per.append(np.array(L) if L else np.zeros((0, 3)))
    total = sum(len(x) for x in per)
    if total < min_lines:
        if verbose:
            print("[refine] vertical VP: too few vertical edges; cuboids left as measured")
        return None
    pool = np.vstack([x for x in per if len(x)])
    out = np.empty((len(states), 3))
    prev = None
    for i in range(len(states)):
        window = [x for x in per[max(0, i - win):i + win + 1] if len(x)]
        chunk = np.vstack(window) if window else pool
        if len(chunk) < min_lines:
            chunk = pool
        v = _vp_from_lines(chunk)
        v = v / (np.linalg.norm(v) + 1e-12)
        if prev is not None and float(prev @ v) < 0:
            v = -v                                  # keep the sign continuous
        out[i] = prev = v
    if verbose:
        g = _vp_from_lines(pool)
        g = g / (np.linalg.norm(g) + 1e-12)
        res = np.abs(pool @ g)
        loc = ("at infinity" if abs(g[2]) < 1e-6
               else f"({g[0] / g[2]:+.0f}, {g[1] / g[2]:+.0f}) px")
        inl = float((res < 3 * np.median(res)).mean()) * 100
        print(f"[refine] vertical VP {loc} from {total} vertical edges, {inl:.0f}% inliers")
    return out


def _height_ls(G: np.ndarray, V: np.ndarray, top: np.ndarray,
               iters: int = 3) -> float:
    """Height parameter of one cuboid on one frame, by weighted least squares.

    Each observed roof corner gives two linear equations in the height ``c``:
    ``(V[ax] - top[ax]*V[2]) * c = top[ax]*G[2] - G[ax]``. Solving them
    individually and taking a median does NOT work, and this was the cause of
    the wild cuboid heights: the coefficient vanishes wherever a corner lies on
    the vanishing point's own row or column, and the vertical vanishing point is
    often INSIDE the frame (on the drone clip it sits at x = 235 px of a 608 px
    wide image). Any corner near that column then produces an unbounded -- and,
    on the far side of the singularity, NEGATIVE -- estimate, and several of the
    eight can be corrupt at once, which no median survives.

    Solving all eight jointly by least squares fixes it, because a near-singular
    equation has a near-zero coefficient and therefore almost no influence. The
    weights then turn the algebraic residual into an image-space one (dividing
    by the reconstructed point's homogeneous denominator), with a Huber
    reweighting so one bad corner cannot drag the fit.
    """
    a = np.concatenate([V[ax] - top[:, ax] * V[2] for ax in (0, 1)])
    b = np.concatenate([top[:, ax] * G[:, 2] - G[:, ax] for ax in (0, 1)])
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or float(a[ok] @ a[ok]) < 1e-18:
        return float("nan")
    a, b = a[ok], b[ok]
    c = float(a @ b / (a @ a))
    zz = np.concatenate([G[:, 2], G[:, 2]])[ok]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(zz + c * V[2]), 1e-9)   # algebraic -> image
        r = (a * c - b) * w
        sc = float(np.median(np.abs(r))) + 1e-12
        ww = (w / (1.0 + (r / (3.0 * sc)) ** 2)) ** 2       # Huber-ish
        den = float((ww * a * a).sum())
        if den < 1e-18:
            break
        c = float((ww * a * b).sum() / den)
    return c


def _fit_height(items, Gs, vps, torder, win, poly, fixed: bool = True,
                band: Tuple[float, float] = (0.5, 2.0)):
    """Height parameter of a track over time.

    A rigid object has ONE height, so on a clip that shares a single ground
    plane the whole track gets a single robust value; only when the ground frame
    itself changes per frame is a (spike-filtered, smoothed) series used.
    """
    from uod.smoothing import _hampel
    cs = np.full(len(items), np.nan)
    for i, ((fi, v), G) in enumerate(zip(items, Gs)):
        kp = np.asarray(v.kpts, dtype=np.float64)
        if kp.shape != (8, 2) or not np.isfinite(kp).all():
            continue
        cs[i] = _height_ls(G, vps[fi], kp[torder])
    good = np.isfinite(cs)
    if good.sum() < 3:
        return None
    med = float(np.median(cs[good]))
    if not np.isfinite(med) or abs(med) < 1e-12:
        return None
    # Discard frames that disagree with the track's own height by more than a
    # factor of two, or that come out with the opposite sign (a cuboid growing
    # DOWNWARDS through the road).
    ratio = cs / med
    keep = good & np.isfinite(ratio) & (ratio > band[0]) & (ratio < band[1])
    if keep.sum() < 3:
        return None
    if fixed:
        return np.full(len(items), float(np.median(cs[keep])))
    cs[~keep] = np.nan
    idx = np.flatnonzero(keep)
    cs = np.interp(np.arange(len(cs), dtype=np.float64), idx, cs[idx])
    return _smooth(_hampel(cs, k=5, nsig=3.0), win, poly)


def _project_box(Gs, cs, vps, items, gorder, torder):
    """Image corners of the rigid box: ground face from G, top face from G + cV."""
    K = np.empty((len(items), 8, 2))
    for i, ((fi, _), G) in enumerate(zip(items, Gs)):
        g = G[:, :2] / np.where(np.abs(G[:, 2:3]) < 1e-12, 1e-12, G[:, 2:3])
        T = G + cs[i] * vps[fi][None, :]
        t = T[:, :2] / np.where(np.abs(T[:, 2:3]) < 1e-12, 1e-12, T[:, 2:3])
        K[i, gorder] = g
        K[i, torder] = t
    return K


def rebuild_cuboids(pipe, states, gi, vps, ratios=None, max_err: float = 0.10,
                    win: int = 11, poly: int = 2, fixed: bool = True,
                    shake_ref: Optional[float] = None, shake_cap: float = 1.5,
                    whole_band: Tuple[float, float] = (0.75, 1.35),
                    height_band: Tuple[float, float] = (0.5, 2.0),
                    verbose: bool = True):
    """Redraw every cuboid as the projection of a rigid box, not 8 loose points.

    A vehicle is a box of CONSTANT size standing on the ground. Given the ground
    homography, the vertical vanishing point and the track's constant footprint,
    the whole cuboid follows from one further number: its height. So 16 noisy
    coordinates per frame are replaced by a one-parameter fit on top of a pose
    that has already been smoothed -- the corners become exactly rigid, and per
    corner jitter cannot survive.

    Where the 2D evidence says the predicted box is too small, the FOOTPRINT is
    grown (about its own centre, so the ground contact does not slide) and the
    height is then re-fitted to the measured roof corners. Growing the height by
    the same factor was tried and left boxes standing visibly above their
    vehicles: the under-coverage is mostly in the ground plane, and the roof is
    the one part the model already places well.

    A track whose reconstruction does not reproduce its own measured corners to
    within ``max_err`` (as a fraction of the 2D box diagonal) keeps the measured
    ones: the model is a strong prior, not a licence to invent geometry.
    """
    if vps is None:
        return set()
    from homography_rt import apply_homography
    ratios = ratios or {}
    gset = set(gi or [])
    gorder = sorted(gset)
    torder = sorted(i for i in range(8) if i not in gset)
    if len(gorder) != 4 or len(torder) != 4:
        return set()
    per: Dict[int, list] = {}
    for fi, s in enumerate(states):
        for v in s.tracks:
            if getattr(v, "bev_quad", None) is None:
                continue
            per.setdefault(v.id, []).append((fi, v))

    # Reference shake for this clip: how much a typical track's corners move
    # frame to frame here. Using the clip's own median keeps the test relative
    # -- "this measurement is noisier than usual for this footage" -- instead of
    # hard-coding a number that only holds at one resolution and frame rate.
    if shake_ref is None:
        allsh = []
        for items in per.values():
            M = [np.asarray(v.kpts, dtype=np.float64) for _, v in items]
            M = [m for m in M if m.shape == (8, 2) and np.isfinite(m).all()]
            if len(M) < 3:
                continue
            M = np.stack(M)
            d = np.array([max(np.hypot(*(_boxes_of(v)[2:] - _boxes_of(v)[:2])), 1.0)
                          for _, v in items][:len(M)])
            acc = np.linalg.norm(M[2:] - 2 * M[1:-1] + M[:-2], axis=2).mean(1)
            allsh.append(float(np.median(acc / d[1:-1])))
        shake_ref = float(np.median(allsh)) if allsh else 1.0e-3
        shake_ref = max(shake_ref, 1e-6)

    n_ok = n_skip = n_scaled = 0
    errs = []
    rebuilt = set()
    for tid, items in per.items():
        if len(items) < 5:
            continue
        try:
            Ainv = [np.linalg.inv(np.asarray(states[fi].H, dtype=np.float64))
                    for fi, _ in items]
        except np.linalg.LinAlgError:
            continue
        # The footprint's corner origin is arbitrary (it comes from a Procrustes
        # template): find the cyclic roll that matches the measured corner order,
        # or ground corner k gets paired with the wrong top corner.
        best_roll, best_cost = 0, np.inf
        for r in range(4):
            cost = n = 0.0
            for fi, v in items:
                g = np.asarray(v.ground, dtype=np.float64)
                if g.shape != (4, 2):
                    continue
                P = apply_homography(g, np.asarray(states[fi].H, dtype=np.float64))
                if not np.isfinite(P).all():
                    continue
                cost += float(np.linalg.norm(
                    np.roll(np.asarray(_pose_quad(v), dtype=np.float64), -r, axis=0) - P,
                    axis=1).mean())
                n += 1
            if n and cost / n < best_cost:
                best_roll, best_cost = r, cost / n

        def _quads(scale):
            out = []
            for (fi, v), A in zip(items, Ainv):
                q = np.roll(np.asarray(_pose_quad(v), dtype=np.float64),
                            -best_roll, axis=0)
                if scale != 1.0:
                    q = q.mean(axis=0) + (q - q.mean(axis=0)) * scale
                out.append((A @ np.c_[q, np.ones(4)].T).T)
            return out

        Gs = _quads(1.0)
        cs = _fit_height(items, Gs, vps, torder, win, poly, fixed, height_band)
        if cs is None:
            continue
        K0 = _project_box(Gs, cs, vps, items, gorder, torder)

        boxes = np.array([_boxes_of(v) for _, v in items])
        diag = np.maximum(np.hypot(boxes[:, 2] - boxes[:, 0],
                                   boxes[:, 3] - boxes[:, 1]), 1.0)
        meas = [np.asarray(v.kpts, dtype=np.float64) for _, v in items]
        ok = np.array([m.shape == (8, 2) and np.isfinite(m).all() for m in meas])
        if ok.sum() < 3 or not np.isfinite(K0).all():
            n_skip += 1
            continue
        e = np.array([np.linalg.norm(K0[i] - meas[i], axis=1).mean() / diag[i]
                      if ok[i] else np.nan for i in range(len(items))])
        # Judge the model only where the object was seen WHOLE. A partial view --
        # something still emerging from behind a tree, or clipped by the frame --
        # is a smaller measurement of the same object, and a rigid model cannot
        # be faulted for disagreeing with it. Without this the bus that is only
        # fully visible late in a shot is rejected on the strength of the frames
        # where it was half hidden, and loses the correct box it had earned.
        ref = 0.0
        u0 = float(states[items[0][0]].bev_unit or 0.0)
        if u0 > 1e-9:
            q0 = np.asarray(_pose_quad(items[0][1]), dtype=np.float64) / u0
            ref = _quad_scale(q0 - q0.mean(axis=0))
        whole = np.zeros(len(items), bool)
        if ref > 1e-9:
            for i, (fi, v) in enumerate(items):
                g = np.asarray(v.ground, dtype=np.float64)
                unit = float(states[fi].bev_unit or 0.0)
                if g.shape != (4, 2) or not np.isfinite(g).all() or unit <= 1e-9:
                    continue
                P = apply_homography(g, np.asarray(states[fi].H, dtype=np.float64))
                if not np.isfinite(P).all():
                    continue
                r = _quad_scale(P / unit - (P / unit).mean(axis=0)) / ref
                whole[i] = whole_band[0] <= r <= whole_band[1]
        use = ok & whole
        if use.sum() < max(8, 0.15 * len(items)):
            use = ok
        err = float(np.nanmedian(e[use])) if use.any() else float("nan")
        # How jittery is the MEASUREMENT itself? A fixed error budget asks the
        # model to agree with corners that do not agree with themselves, and
        # rejecting on that basis leaves the jitter on screen -- which is where
        # the remaining wobble came from. A measurement that shakes is allowed
        # to disagree more, because that disagreement IS the shake. The budget
        # is capped, so a model that is genuinely wrong (a footprint at the
        # wrong heading, say) still cannot slip through.
        shake = float("nan")
        if ok.sum() >= 5:
            M = np.stack([meas[i] for i in np.flatnonzero(ok)])
            d = np.maximum(diag[np.flatnonzero(ok)], 1.0)
            if len(M) >= 3:
                acc = np.linalg.norm(M[2:] - 2 * M[1:-1] + M[:-2], axis=2).mean(1)
                shake = float(np.median(acc / d[1:-1]))
        gain = 1.0
        if np.isfinite(shake):
            gain += float(np.clip(shake / shake_ref - 1.0, 0.0, shake_cap))
        budget = max_err * gain
        if os.environ.get("V2_DEBUG_TRACK") and int(os.environ["V2_DEBUG_TRACK"]) == tid:
            np.savez(os.environ.get("V2_DEBUG_OUT", "/tmp/track.npz"),
                     K0=K0, meas=np.stack([m if m.shape == (8, 2) else np.full((8, 2), np.nan)
                                           for m in meas]),
                     boxes=boxes, frames=np.array([fi for fi, _ in items]),
                     err=np.array([err]))
        if not np.isfinite(err) or err > budget:
            n_skip += 1
            if os.environ.get("V2_DEBUG_FIT"):
                print(f"    [fit] id {tid:3d} REJECT err {err:.3f} shake {shake:.4f} "
                      f"budget {budget:.3f} n={len(items)} whole={int(whole.sum())} "
                      f"used={int(use.sum())}")
            continue
        if os.environ.get("V2_DEBUG_FIT"):
            print(f"    [fit] id {tid:3d} accept err {err:.3f} shake {shake:.4f} "
                  f"budget {budget:.3f} n={len(items)}")

        sc_fill = fill_scale(ratios.get(tid))
        if sc_fill is not None:
            Gs2 = _quads(sc_fill)
            cs2 = _fit_height(items, Gs2, vps, torder, win, poly, fixed, height_band)
            if cs2 is not None:
                K = _project_box(Gs2, cs2, vps, items, gorder, torder)
                n_scaled += 1
            else:
                K, sc_fill = K0, None
        else:
            K = K0
        if not np.isfinite(K).all():
            n_skip += 1
            continue
        # Light zero-phase smoothing: the reconstruction removed per-corner
        # noise, this removes what a per-frame homography injects when the
        # camera moves.
        if len(K) >= max(5, poly + 2):
            flat = K.reshape(len(K), -1)
            K = np.stack([_smooth(flat[:, j], win, poly)
                          for j in range(flat.shape[1])], axis=1).reshape(K.shape)

        errs.append(err)
        rebuilt.add(tid)
        for i, (fi, v) in enumerate(items):
            if sc_fill is not None and getattr(v, "bev_quad", None) is not None:
                q = np.asarray(v.bev_quad, dtype=np.float64)
                v.bev_quad = q.mean(axis=0) + (q - q.mean(axis=0)) * sc_fill
            v.kpts = K[i]
            v.ground = K[i][gorder].copy()
            v.box_xyxy = np.array([K[i][:, 0].min(), K[i][:, 1].min(),
                                   K[i][:, 0].max(), K[i][:, 1].max()])
        n_ok += 1
    if verbose:
        if errs:
            e = np.array(errs) * 100
            med = (f"{np.median(e):.1f}% of the box diagonal "
                   f"(p75 {np.percentile(e, 75):.1f}%, max {e.max():.1f}%)")
        else:
            med = "n/a"
        print(f"[refine] rigid cuboids: {n_ok} track(s) rebuilt from "
              f"(footprint, height, vertical VP), {n_skip} kept as measured; "
              f"reprojection {med}; footprint grown on {n_scaled}")
    return rebuilt


def report_jitter(states, verbose: bool = True):
    """Residual jitter of what is actually drawn.

    Uses the second difference of each cuboid corner (and of each BEV footprint
    centre): smooth motion has none, so what is left is estimation noise. Corner
    figures are a fraction of the object's own 2D box diagonal, footprint
    figures a fraction of an object length, both averaged per track so a few
    long tracks cannot dominate.
    """
    per: Dict[int, dict] = {}
    for s in states:
        unit = float(s.bev_unit or 0.0)
        for v in s.tracks:
            kp = np.asarray(v.kpts, dtype=np.float64)
            if kp.shape != (8, 2) or not np.isfinite(kp).all():
                continue
            d = per.setdefault(v.id, dict(k=[], b=[], c=[]))
            d["k"].append(kp)
            d["b"].append(_boxes_of(v))
            q = getattr(v, "bev_quad", None)
            d["c"].append(np.asarray(q, dtype=np.float64).mean(axis=0) / unit
                          if q is not None and unit > 1e-9 else np.full(2, np.nan))
    kj, cj = [], []
    for tid, d in per.items():
        if len(d["k"]) < 5:
            continue
        K = np.stack(d["k"])
        B = np.stack(d["b"])
        diag = np.maximum(np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]), 1.0)
        acc = np.linalg.norm(K[2:] - 2 * K[1:-1] + K[:-2], axis=2)
        kj.append(float(np.median(acc / diag[1:-1, None])))
        C = np.stack(d["c"])
        if np.isfinite(C).all():
            a = np.linalg.norm(C[2:] - 2 * C[1:-1] + C[:-2], axis=1)
            cj.append(float(np.median(a)))
    # How far does a vehicle that never really moves still wander? This is the
    # slow defect the eye actually reads as "the box is crawling around".
    wander = []
    n_moving = 0
    for tid, d in per.items():
        C = np.stack(d["c"])
        C = C[np.isfinite(C).all(axis=1)]     # frames whose footprint was drawn
        if len(C) < 20:
            continue
        cs = np.stack([_smooth(C[:, 0], 11, 2), _smooth(C[:, 1], 11, 2)], axis=1)
        if np.isnan(_traj_heading(cs, 1.2, 0.7)).all():          # never moves
            wander.append(float(np.hypot(*(cs.max(axis=0) - cs.min(axis=0)))))
        else:
            n_moving += 1
    if verbose and kj:
        w = (f"standing vehicles wander {np.median(wander):.3f} object lengths "
             f"(max {max(wander):.2f}, n={len(wander)})" if wander
             else "no standing vehicles")
        print(f"[jitter] cuboid corners {np.median(kj) * 1000:.2f} milli-diagonals/frame"
              f"  |  BEV footprint centres {np.median(cj) * 1000:.2f} milli-lengths/frame"
              f"  ({len(kj)} tracks)")
        print(f"[jitter] {w}, {n_moving} moving")
    return (float(np.median(kj)) if kj else float("nan"),
            float(np.median(cj)) if cj else float("nan"))

def ego_motion(states, downscale: float = 960.0, pad: float = 0.15,
               ransac: float = 2.0, verbose: bool = True):
    """Inter-frame camera motion measured on the STATIC scene only.

    The stock estimator tracks corners anywhere in the frame. On a shot that is
    mostly traffic -- a drone flying low along a busy avenue, against the flow --
    the strongest, most numerous corners belong to the VEHICLES, so what comes
    back is partly the traffic's motion and not the camera's, and the clip gets
    misjudged as a freely moving view. Masking out every tracked object first
    leaves the road, kerbs and buildings, which really are static.

    A homography (not a similarity) is fitted, because the dominant static
    surface here is the ground plane and its inter-frame image motion is exactly
    a homography; RANSAC pushes the off-plane parallax of building facades into
    the outliers. Returns ``(C, steps)`` in the same form the v1 offline path
    expects: ``C[t]`` maps frame 0 to frame ``t``.
    """
    import cv2
    C = [np.eye(3)]
    steps = [[0.0, 0.0, 0.0]]
    prev = prev_pts = None
    cur = np.eye(3)
    n_ok = 0
    for s in states:
        fr = getattr(s, "frame", None)
        if fr is None:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sc = downscale / max(g.shape)
        if sc < 1.0:
            g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        else:
            sc = 1.0
        M = np.eye(3)
        step = [0.0, 0.0, 0.0]
        if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, prev_pts, None)
            if nxt is not None and st is not None:
                ok = st.ravel() == 1
                if int(ok.sum()) >= 8:
                    src, dst = prev_pts[ok], nxt[ok]
                    A = None
                    if len(src) >= 12:
                        A, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
                    if A is None or not np.all(np.isfinite(A)):
                        B, _ = cv2.estimateAffinePartial2D(
                            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        A = (np.vstack([B, [0.0, 0.0, 1.0]])
                             if B is not None and np.all(np.isfinite(B)) else None)
                    if A is not None:
                        # undo the downscale: M_full = S^-1 A S
                        S = np.diag([sc, sc, 1.0])
                        M = np.linalg.inv(S) @ A @ S
                        M = M / M[2, 2] if abs(M[2, 2]) > 1e-12 else M
                        lin = M[:2, :2]
                        det = float(np.sqrt(abs(np.linalg.det(lin)))) or 1.0
                        step = [float(np.hypot(M[0, 2], M[1, 2])),
                                abs(float(np.degrees(np.arctan2(lin[1, 0], lin[0, 0])))),
                                abs(float(np.log(det)))]
                        n_ok += 1
        if prev is not None:
            cur = M @ cur
            C.append(cur.copy())
            steps.append(step)
        # Corners for the NEXT step, taken outside every tracked object.
        mask = np.full(g.shape, 255, np.uint8)
        for v in list(s.tracks) + list(s.aux_dets):
            b = np.asarray(getattr(v, "box_xyxy", getattr(v, "xyxy", None)),
                           dtype=np.float64)
            if b is None or b.shape != (4,):
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            x0 = int(max(0, (b[0] - pad * w) * sc))
            y0 = int(max(0, (b[1] - pad * h) * sc))
            x1 = int(min(g.shape[1], (b[2] + pad * w) * sc))
            y1 = int(min(g.shape[0], (b[3] + pad * h) * sc))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 0
        prev = g
        prev_pts = cv2.goodFeaturesToTrack(g, maxCorners=600, qualityLevel=0.01,
                                           minDistance=7, blockSize=3, mask=mask)
        if prev_pts is None or len(prev_pts) < 8:      # scene fully covered
            prev_pts = cv2.goodFeaturesToTrack(g, maxCorners=600, qualityLevel=0.01,
                                               minDistance=7, blockSize=3)
    steps = np.array(steps)
    if verbose and len(steps) > 1:
        diag = float(np.hypot(*states[0].img_wh)) or 1.0
        print(f"[refine] ego-motion on the static scene: "
              f"{np.median(steps[1:, 0]) / diag * 100:.3f}%/frame translation, "
              f"{np.median(steps[1:, 1]):.3f} deg rotation, "
              f"{np.median(steps[1:, 2]) * 100:.3f}% scale ({n_ok} frames solved)")
    return C, steps

def smooth_shape(states, ids, gi, win: int = 31, poly: int = 2,
                 freeze: bool = True, freeze_pct: float = 60.0,
                 verbose: bool = True) -> None:
    """Hold a cuboid's SHAPE steady without holding back its motion.

    For the tracks the rigid 3D reconstruction could not model, the corners are
    all that is left, and they shake. Smoothing them directly trades shake for
    lag. But the two live on completely different timescales: an object's
    position races across the frame while its shape RELATIVE TO ITS OWN BOX
    changes only with the viewing angle, which barely moves. So the corners are
    expressed in the box frame, held steady there, and put back on the
    (separately smoothed) box. Position keeps its full bandwidth; the shake is
    gone.

    ``freeze`` replaces that shape with the track's own median instead of a
    smoothed version. These tracks were rejected precisely because their
    geometry is not trustworthy frame by frame, so the median is the best single
    statement available, and it cannot wobble.
    """
    per = _track_index(states)
    n = 0
    for tid in ids:
        fr = per.get(tid)
        if not fr:
            continue
        frames = sorted(f for f in fr
                        if np.asarray(fr[f].kpts).shape == (8, 2)
                        and np.isfinite(np.asarray(fr[f].kpts)).all())
        if len(frames) < max(7, poly + 2):
            continue
        norm = np.stack([_norm_kpts(np.asarray(fr[f].kpts, dtype=np.float64),
                                    _boxes_of(fr[f])) for f in frames])
        if freeze:
            # Freeze at the shape from the frames where the object was seen most
            # COMPLETELY, not at the median over all of them. Occlusion and
            # truncation only ever shrink a cuboid, so the frames whose cuboid
            # fills most of its own box are the ones that saw the whole object;
            # a median over everything blends those with half-views and leaves
            # the object permanently in a shape it never had. (A bus emerging
            # from behind a tree earns its correct box late in the shot -- that
            # is the shape it should keep, forwards and backwards.)
            area = np.array([(n[:, 0].max() - n[:, 0].min()) *
                             (n[:, 1].max() - n[:, 1].min()) for n in norm])
            cut = float(np.percentile(area, freeze_pct))
            sel = norm[area >= cut] if (area >= cut).sum() >= 3 else norm
            sm = np.repeat(np.median(sel, axis=0)[None], len(frames), axis=0)
        else:
            flat = norm.reshape(len(frames), -1)
            sm = np.stack([_smooth(flat[:, j], win, poly)
                           for j in range(flat.shape[1])], axis=1).reshape(norm.shape)
        for i, f in enumerate(frames):
            kp = _denorm_kpts(sm[i], _boxes_of(fr[f]))
            fr[f].kpts = kp
            if gi and kp.shape[0] >= max(gi) + 1:
                fr[f].ground = kp[list(gi)].copy()
        n += 1
    if verbose and n:
        how = "held at its median" if freeze else "smoothed"
        print(f"[refine] shape: {n} unmodelled track(s) {how} in their own box frame")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    ub = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return float(inter / (ua + ub - inter))


def dedupe_tracks(states, solid_ids=None, min_overlap: float = 0.4,
                  min_frac: float = 0.5, min_frames: int = 10,
                  box_iou: float = 0.6, max_sep: float = 1.5,
                  verbose: bool = True):
    """Remove duplicate tracks: two objects cannot occupy the same ground.

    The detector sometimes fires twice on one vehicle, and in the camera view
    that is genuinely hard to tell from a real pair -- one car behind another
    overlaps heavily in the image too. On the GROUND it is not ambiguous: two
    real vehicles are side by side or nose to tail and their footprints barely
    touch, while a double detection stacks two footprints on the same spot. So
    the test is done in the bird's-eye plane, which is exactly what that panel
    is for.

    Where a track has no usable footprint the ground cannot settle it, and the
    pair falls back to plain image overlap at a high threshold. That is the case
    that otherwise leaves several markers stacked on one bus: the keypoint head
    fails on buses at oblique drone angles, so those tracks are 2D-only and have
    no footprint to compare.

    Returns the mapping ``{dropped id: surviving id}``.
    """
    from homography_rt import apply_homography
    from uod.bev import convex_overlap_fraction

    solid = set(solid_ids) if solid_ids is not None else None
    foot: Dict[int, Dict[int, np.ndarray]] = {}
    box: Dict[int, Dict[int, np.ndarray]] = {}
    obs: Dict[int, int] = {}
    for fi, s in enumerate(states):
        unit = float(s.bev_unit or 0.0)
        for v in s.tracks:
            obs[v.id] = obs.get(v.id, 0) + int(_observed(v))
            box.setdefault(v.id, {})[fi] = _boxes_of(v)
            g = np.asarray(v.ground, dtype=np.float64)
            if g.shape != (4, 2) or not np.isfinite(g).all() or unit <= 1e-9:
                continue
            P = apply_homography(g, np.asarray(s.H, dtype=np.float64))
            if np.isfinite(P).all():
                foot.setdefault(v.id, {})[fi] = P / unit

    def _use_ground(a, b):
        if solid is None:
            return a in foot and b in foot
        return a in solid and b in solid

    both: Dict[tuple, list] = {}
    for fi, s in enumerate(states):
        ids = [v.id for v in s.tracks]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                k = (min(a, b), max(a, b))
                if _use_ground(a, b):
                    fa = foot.get(a, {}).get(fi)
                    fb = foot.get(b, {}).get(fi)
                    if fa is None or fb is None:
                        continue
                    if np.hypot(*(fa.mean(axis=0) - fb.mean(axis=0))) > max_sep:
                        hit = False                # too far apart to be one object
                    else:
                        hit = max(convex_overlap_fraction(fa, fb),
                                  convex_overlap_fraction(fb, fa)) > min_overlap
                else:
                    ba, bb = box.get(a, {}).get(fi), box.get(b, {}).get(fi)
                    if ba is None or bb is None:
                        continue
                    hit = _iou(ba, bb) > box_iou
                rec = both.setdefault(k, [0, 0])
                rec[0] += 1
                rec[1] += int(hit)

    drop: Dict[int, int] = {}

    def _root(t):                                  # duplicates can chain
        while t in drop:
            t = drop[t]
        return t

    def _better(x, y):
        """Prefer a track with a usable cuboid, then the longer-observed one."""
        if solid is not None and (x in solid) != (y in solid):
            return x if x in solid else y
        return x if obs.get(x, 0) >= obs.get(y, 0) else y

    for (a, b), (n_both, n_over) in sorted(both.items(), key=lambda kv: -kv[1][1]):
        if n_both < min_frames or n_over < min_frac * n_both:
            continue
        wa, wb = _root(a), _root(b)
        if wa == wb:
            continue
        win = _better(wa, wb)
        drop[wb if win == wa else wa] = win
    drop = {k: _root(k) for k in drop}              # collapse the chains
    if not drop:
        if verbose:
            print("[refine] duplicate check: none")
        return {}

    n_dropped = n_kept = 0
    for s in states:
        present = {v.id for v in s.tracks}
        keep = []
        for v in s.tracks:
            w = drop.get(v.id)
            if w is None:
                keep.append(v)
            elif w in present:
                n_dropped += 1                     # the winner covers this frame
            else:
                v.id = w                           # extends the winner's coverage
                keep.append(v)
                n_kept += 1
        s.tracks = keep
    if verbose:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(drop.items()))
        print(f"[refine] duplicate objects: merged {len(drop)} track(s) "
              f"({pairs}); {n_dropped} duplicate marker(s) removed, "
              f"{n_kept} frame(s) re-labelled")
    return drop

def stabilize_boxes(states, ids, win: int = 15, poly: int = 2,
                    max_dev: float = 1.35, verbose: bool = True) -> None:
    """Keep a 2D box the size its own object actually is.

    For tracks drawn as plain boxes there is no cuboid to make rigid, so a bad
    frame shows directly: the detector occasionally returns one box swallowing
    a NEIGHBOURING vehicle as well -- on the drone clip a bus box grows to
    almost twice its height for a few frames and covers the bus behind it too,
    which reads as two markers on one bus.

    An object's apparent size changes smoothly (it approaches or recedes), so
    the width and height series are median-filtered to kill spikes and then
    smoothed, and the box is rebuilt around its own bottom-centre -- the ground
    contact, which is the part the detector gets right and the part the
    bird's-eye marker is derived from.
    """
    from uod.smoothing import _hampel
    per = _track_index(states)
    n_tracks = n_frames = 0
    for tid in ids:
        fr = per.get(tid)
        if not fr:
            continue
        frames = sorted(fr)
        if len(frames) < max(7, poly + 2):
            continue
        B = np.stack([_boxes_of(fr[f]) for f in frames])
        w = B[:, 2] - B[:, 0]
        h = B[:, 3] - B[:, 1]
        ws = _smooth(_hampel(w, k=max(3, win // 2), nsig=3.0), win, poly)
        hs = _smooth(_hampel(h, k=max(3, win // 2), nsig=3.0), win, poly)
        bad = ((w > max_dev * np.maximum(ws, 1e-6))
               | (h > max_dev * np.maximum(hs, 1e-6)))
        if not bad.any():
            continue
        n_tracks += 1
        for i, f in enumerate(frames):
            if not bad[i]:
                continue
            cx = 0.5 * (B[i, 0] + B[i, 2])
            fr[f].box_xyxy = np.array([cx - ws[i] * 0.5, B[i, 3] - hs[i],
                                       cx + ws[i] * 0.5, B[i, 3]])
            n_frames += 1
    if verbose and n_tracks:
        print(f"[refine] box size: {n_frames} oversized frame(s) on {n_tracks} "
              f"track(s) pulled back to the object's own size")


def drop_impossible(states, ped_frac: float = 0.7, veh_mult: float = 2.5,
                    min_frames: int = 10, smooth_win: int = 11,
                    verbose: bool = True) -> None:
    """Discard tracks that move at a speed their class cannot reach.

    Speed is measured on the ground, in object lengths per frame, so it is
    comparable across a scene without any calibration: the reference is the
    traffic itself. Nothing on foot outruns the cars around it, and no car in
    the same shot travels several times faster than all the others. A track that
    does is not an object -- it is an identity that has hopped between several,
    which no amount of smoothing will make honest.
    """
    world = _world_tracks(states)
    classes = {v.id: v.cls for s in states for v in s.tracks}
    scene = _scene_speed(world, classes)
    if scene <= 0:
        return
    drop = set()
    for tid, pts in world.items():
        if len(pts) < min_frames:
            continue
        f = sorted(pts)
        P = np.stack([pts[i] for i in f])
        P = np.stack([_smooth(P[:, 0], smooth_win, 2),
                      _smooth(P[:, 1], smooth_win, 2)], axis=1)
        d = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(f), 1)
        if len(d) < 5:
            continue
        speed = float(np.median(d))
        on_foot = classes.get(tid) not in (1, 2, 3, 5, 7)
        if speed > (ped_frac if on_foot else veh_mult) * scene:
            drop.add(tid)
    if not drop:
        return
    for s in states:
        s.tracks = [v for v in s.tracks if v.id not in drop]
    if verbose:
        print(f"[refine] dropped {len(drop)} track(s) moving faster than their class "
              f"can (scene traffic {scene:.3f} object lengths/frame)")

# --------------------------------------------------------------------------- #
# Entry points. The ordering is not arbitrary: identity must be settled before
# anything is measured from a track, the 2D geometry must be corrected before
# the ground plane is solved from it, and the 3D reconstruction can only happen
# afterwards, in the gauge that solve establishes.
@dataclass
class TrackSets:
    """What the per-track votes decided, carried between the two phases."""
    solid: set = field(default_factory=set)      # cuboid is usable
    boxonly: set = field(default_factory=set)    # draw 2D only, for its whole life
    ratios: dict = field(default_factory=dict)   # id -> (w, h) size correction
    rebuilt: set = field(default_factory=set)    # id -> reconstructed in 3D


def refine_tracks(pipe, states, gi, cfg: RefineConfig, dedupe: bool = True,
                  verbose: bool = True) -> TrackSets:
    """Phase 1: settle identity and 2D geometry, BEFORE the ground-plane solve.

    Everything here either fixes who an object is or corrects the measurements
    the solve will consume, so it has to run first.
    """
    if not states:
        return TrackSets()
    diag = float(np.hypot(*states[0].img_wh)) or 1.0
    merge_split_tracks(pipe, states, cfg.n(cfg.min_track_s),
                       cfg.n(cfg.coast_tail_s), cfg.n(cfg.merge_gap_s),
                       speed_tol=cfg.merge_speed_tol, edge_frac=cfg.edge_frac,
                       entry_gap=cfg.n(cfg.entry_gap_s), verbose=verbose)
    majority_class(states, verbose=verbose)
    interpolate_gaps(states, gi, max_gap=cfg.n(cfg.max_gap_s),
                     speed_ratio=cfg.speed_ratio,
                     speed_floor=cfg.per_frame(cfg.speed_floor_sps),
                     span=cfg.n(cfg.speed_span_s), min_track=cfg.n(cfg.min_track_s),
                     verbose=verbose)
    drop_impossible(states, ped_frac=cfg.ped_speed_frac,
                    veh_mult=cfg.veh_speed_mult, min_frames=cfg.n(cfg.dup_pair_s),
                    smooth_win=cfg.odd(cfg.still_win_s), verbose=verbose)
    solid, boxonly, ratios = classify_tracks(
        states, cfg.min_fill, cfg.min_ok_frac,
        min_box_px=max(8.0, cfg.min_box_frac * diag), verbose=verbose)
    sets = TrackSets(solid=solid, boxonly=boxonly, ratios=ratios)
    if dedupe:
        # After the cuboid vote, so a pair with trustworthy footprints is judged
        # on the ground and a pair without falls back to image overlap.
        for lost in dedupe_tracks(states, solid_ids=sets.solid,
                                  min_overlap=cfg.dup_overlap,
                                  min_frac=cfg.dup_frac,
                                  min_frames=cfg.n(cfg.dup_pair_s),
                                  box_iou=cfg.dup_box_iou,
                                  max_sep=cfg.dup_max_sep, verbose=verbose):
            sets.solid.discard(lost)
            sets.boxonly.discard(lost)
            sets.ratios.pop(lost, None)
    repair_keypoints(states, sets.solid, gi, verbose=verbose)
    stabilize_boxes(states, sets.solid | sets.boxonly, win=cfg.odd(cfg.box_win_s),
                    max_dev=cfg.box_max_dev, verbose=verbose)
    mute_tracks(states, sets.boxonly, verbose=verbose)
    return sets


def refine_geometry(pipe, states, gi, cfg: RefineConfig, sets: TrackSets,
                    fixed: bool, smooth_win: int = 11, poly: int = 2,
                    use_velocity: bool = True, hold_still: bool = True,
                    rigid_cuboids: bool = True, shape_smooth: bool = True,
                    fill_box: bool = True, verbose: bool = True) -> None:
    """Phase 2: reconstruct the 3D geometry, AFTER the ground plane is solved."""
    if not states:
        return
    rigid_and_heading(pipe, states, fixed, use_velocity=use_velocity,
                      hold_still=hold_still, win=smooth_win, poly=poly,
                      d_min=cfg.move_min_len, straight=cfg.straight_min,
                      min_evidence=cfg.n(cfg.evidence_s),
                      weight=cfg.heading_weight,
                      max_disagree=np.radians(cfg.max_disagree_deg),
                      v_still=cfg.per_frame(cfg.still_speed_lps),
                      still_tol=cfg.still_tol, ladder=cfg.ladder(),
                      still_run=cfg.n(cfg.still_run_s),
                      still_win=cfg.odd(cfg.still_win_s), verbose=verbose)
    if rigid_cuboids:
        vps = estimate_vertical_vp(states, gi, win=cfg.n(cfg.vp_win_s),
                                   verbose=verbose)
        sets.rebuilt = rebuild_cuboids(
            pipe, states, gi, vps, sets.ratios if fill_box else None,
            max_err=cfg.max_reproj, win=smooth_win, poly=poly, fixed=fixed,
            shake_cap=cfg.shake_cap, whole_band=cfg.whole_band,
            height_band=cfg.height_band, verbose=verbose) or set()
    if shape_smooth:
        smooth_shape(states, [t for t in sets.solid if t not in sets.rebuilt], gi,
                     win=cfg.odd(cfg.shape_win_s), poly=poly,
                     freeze_pct=cfg.shape_freeze_pct, verbose=verbose)
    if fill_box:
        # Whatever the reconstruction could not model falls back to the 2D size
        # correction, which is only safe now that nothing downstream depends on
        # those corners being a valid ground-plane projection.
        fill_boxes(states, {k: v for k, v in sets.ratios.items()
                            if k not in sets.rebuilt}, gi, lo=cfg.fill_lo,
                   hi=cfg.fill_hi, dead=cfg.fill_dead, verbose=verbose)
    demote_tracks(pipe, states, sets.boxonly, verbose=verbose)

