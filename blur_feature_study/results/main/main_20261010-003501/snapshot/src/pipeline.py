"""Experiment runner: one row per (pair x blur condition x processing x detector x config)."""
import csv
import datetime as dt
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

from . import features as F
from . import metrics as M
from .imaging import apply_blur, unsharp_mask, wiener_restore
from .io_utils import PROJECT_ROOT
from .psf import make_psf

REQUIRED_COLUMNS = [
    "run_id", "sequence_id", "sequence_type", "pair_id", "blur_type", "blur_strength", "blur_angle",
    "processing_method", "detector", "detector_config", "psnr", "ssim", "edge_precision", "edge_recall",
    "edge_f1", "keypoint_repeatability", "reference_keypoints", "target_keypoints", "good_matches",
    "gt_correct_matches_3px", "gt_correct_matches_5px", "match_precision_3px", "match_precision_5px",
    "ransac_inliers", "ransac_inlier_ratio", "homography_error_mean", "homography_error_median",
    "homography_success_3px", "homography_success_5px", "runtime_ms", "failure_reason",
]
EXTRA_COLUMNS = [
    "phase", "split", "group", "blur_id", "wiener_balance", "detector_params", "is_supplementary",
    "is_placeholder", "gt_quality_flag", "reference_index", "target_index", "image_h", "image_w", "valid_margin_px",
    "edge_n_clean", "edge_n_proc", "rep_n_clean_valid", "rep_n_repeated",
    "keypoint_repeatability_vs_default_clean", "ransac_inliers_gt_correct_3px", "homography_error_max",
    "n_grid_valid", "runtime_processing_ms", "runtime_detect_ms", "runtime_match_ms",
    "runtime_homography_ms", "runtime_reference_detect_ms", "match_detail_file", "config_hash",
    "code_hash", "pipeline_error", "attempt", "timestamp",
]
COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS
KEY_COLUMNS = ["pair_id", "blur_id", "group", "processing_method", "detector", "detector_config", "wiener_balance"]

CLEAN_COND = {"blur_id": "none", "blur_type": "none", "strength": None, "angle": None}


def row_key(r):
    """Identity of a result row; works for dicts built in code and rows read back from CSV."""
    def norm(k, v):
        if v is None:
            return ""
        if k == "wiener_balance":
            if isinstance(v, str) and v.strip().lower() in ("", "nan"):
                return ""
            v = float(v)
            return "" if np.isnan(v) else repr(v)
        return str(v)
    return tuple(norm(k, r.get(k)) for k in KEY_COLUMNS)


# ------------------------------------------------------------------ plans
def _entry(group, method, det, cfg_name, params, balance=None, supplementary=False, placeholder=False):
    return {"group": group, "processing_method": method, "detector": det, "detector_config": cfg_name,
            "params": params, "wiener_balance": balance, "is_supplementary": int(supplementary),
            "is_placeholder": int(placeholder)}


def build_plan(phase, cfg, blur_id, selected=None):
    """Plan entries for one (pair, blur condition). blur_id == 'none' -> clean baseline."""
    plan = []
    for det in ("ORB", "SIFT"):
        dcfg = cfg["detectors"][det]
        default = dict(dcfg["default"])
        if phase == "pilot":
            tuned = dict(default); tuned.update(dcfg["pilot_placeholder"])
            tuned_name, balance, ph = "tuned", cfg["processing"]["wiener"]["pilot_placeholder_balance"], True
        elif phase == "main":
            tuned = dict(selected[det]["params"])
            tuned_name, ph = "tuned", False
            balance = None if blur_id == "none" else float(selected["wiener_balance"][blur_id])
        if blur_id == "none":
            plan.append(_entry("CLEAN", "clean", det, "default", default))
            if phase in ("pilot", "main"):
                plan.append(_entry("CLEAN", "clean", det, tuned_name, tuned, supplementary=True, placeholder=ph))
            continue
        plan.append(_entry("A", "blurred", det, "default", default))
        plan.append(_entry("C", "unsharp_mask", det, "default", default))
        if phase == "validation":
            for p in F.tuning_grid(det, cfg):
                plan.append(_entry("B_candidate", "blurred", det, F.config_name(det, p, default), p))
            for b in cfg["processing"]["wiener"]["balance_candidates"]:
                plan.append(_entry("D_candidate", "wiener_known_psf", det, "default", default, balance=float(b)))
        else:
            plan.append(_entry("B", "blurred", det, tuned_name, tuned, placeholder=ph))
            plan.append(_entry("D", "wiener_known_psf", det, "default", default, balance=balance, placeholder=ph))
    return plan


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, cfg, phase, run_id, out_dir, details_dir, cfg_hash, code_hash, logger=None, attempt=1,
                 gt_flags=None):
        self.cfg, self.phase, self.run_id = cfg, phase, run_id
        self.out_dir, self.details_dir = Path(out_dir), Path(details_dir)
        self.cfg_hash, self.code_hash = cfg_hash, code_hash
        self.log = logger or logging.getLogger(__name__)
        self.attempt = attempt
        self.gt_flags = gt_flags or {}
        self.margin = int(cfg["preprocessing"]["valid_margin_px"])
        self.ref_cache = {}      # (sequence_id, ref_index, det, params) -> Features
        self.clean_cache = {}    # (pair_id, det, params) -> Features
        self.clean_edges = {}    # pair_id -> bool map

    # -------------------------------------------------------------- caches
    def _mask(self, shape):
        return F.border_mask(shape, self.margin)

    def reference_features(self, pair, det, params):
        key = (pair.sequence_id, pair.reference_index, det, F.params_json(params))
        if key not in self.ref_cache:
            self.ref_cache[key] = F.detect(det, params, pair.ref, self._mask(pair.ref.shape))
        return self.ref_cache[key]

    def clean_target_features(self, pair, det, params):
        key = (pair.pair_id, det, F.params_json(params))
        if key not in self.clean_cache:
            self.clean_cache[key] = F.detect(det, params, pair.tgt, self._mask(pair.tgt.shape))
        return self.clean_cache[key]

    def edges_clean(self, pair):
        if pair.pair_id not in self.clean_edges:
            self.clean_edges[pair.pair_id] = M.canny(pair.tgt, self.cfg["evaluation"]["canny"])
        return self.clean_edges[pair.pair_id]

    # -------------------------------------------------------------- one unit
    def run_unit(self, pair, cond, plan, done_keys=frozenset()):
        cfg, m = self.cfg, self.margin
        ev = cfg["evaluation"]
        todo = []
        for e in plan:
            probe = {"pair_id": pair.pair_id, "blur_id": cond["blur_id"], "group": e["group"],
                     "processing_method": e["processing_method"], "detector": e["detector"],
                     "detector_config": e["detector_config"], "wiener_balance": e["wiener_balance"]}
            if row_key(probe) not in done_keys:
                todo.append(e)
        if not todo:
            return []

        psf = None if cond["blur_id"] == "none" else make_psf(cond, cfg)
        blurred = None if psf is None else apply_blur(pair.tgt, psf)
        images, img_metrics, det_cache = {}, {}, {}

        def get_image(method, balance):
            key = (method, balance)
            if key not in images:
                t0 = time.perf_counter()
                if method == "clean":
                    img = pair.tgt
                elif method == "blurred":
                    img = blurred
                elif method == "unsharp_mask":
                    u = cfg["processing"]["unsharp_mask"]
                    img = unsharp_mask(blurred, u["alpha"], u["sigma"], u["truncate"])
                elif method == "wiener_known_psf":
                    img = wiener_restore(blurred, psf, balance)
                else:
                    raise ValueError(method)
                rt = 0.0 if method in ("clean", "blurred") else (time.perf_counter() - t0) * 1000.0
                if img.shape != pair.tgt.shape or img.dtype != np.uint8:
                    raise RuntimeError(f"processed image shape/dtype mismatch: {img.shape} {img.dtype}")
                images[key] = (img, rt)
            return images[key]

        def get_img_metrics(method, balance):
            key = (method, balance)
            if key not in img_metrics:
                img, _ = get_image(method, balance)
                d = {"psnr": M.psnr(pair.tgt, img, m, ev["psnr"]["data_range"]),
                     "ssim": M.ssim(pair.tgt, img, m, ev["ssim"])}
                d.update(M.edge_prf(self.edges_clean(pair), M.canny(img, ev["canny"]), ev["edge_tolerance_px"], m))
                img_metrics[key] = d
            return img_metrics[key]

        rows = []
        for e in todo:
            row = self._base_row(pair, cond, e)
            try:
                row.update(self._evaluate(pair, e, get_image, get_img_metrics, det_cache, row))
            except Exception as exc:  # keep the failure as a row
                row["pipeline_error"] = f"{type(exc).__name__}: {exc}"
                row["failure_reason"] = "pipeline_error"
                self.log.error("pipeline error %s %s %s\n%s", pair.pair_id, cond["blur_id"], e["group"],
                               traceback.format_exc())
            rows.append(row)
        return rows

    def _base_row(self, pair, cond, e):
        return {
            "run_id": self.run_id, "phase": self.phase, "split": pair.split,
            "sequence_id": pair.sequence_id, "sequence_type": pair.sequence_type, "pair_id": pair.pair_id,
            "reference_index": pair.reference_index, "target_index": pair.target_index,
            "blur_id": cond["blur_id"], "blur_type": cond["blur_type"], "blur_strength": cond["strength"],
            "blur_angle": cond["angle"], "group": e["group"], "processing_method": e["processing_method"],
            "wiener_balance": e["wiener_balance"], "detector": e["detector"],
            "detector_config": e["detector_config"], "detector_params": F.params_json(e["params"]),
            "is_supplementary": e["is_supplementary"], "is_placeholder": e["is_placeholder"],
            "gt_quality_flag": self.gt_flags.get(pair.pair_id, ""),
            "image_h": pair.tgt.shape[0], "image_w": pair.tgt.shape[1], "valid_margin_px": self.margin,
            "config_hash": self.cfg_hash, "code_hash": self.code_hash, "pipeline_error": "",
            "attempt": self.attempt, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }

    def _evaluate(self, pair, e, get_image, get_img_metrics, det_cache, row):
        cfg, m = self.cfg, self.margin
        ev = cfg["evaluation"]
        det, params, method, bal = e["detector"], e["params"], e["processing_method"], e["wiener_balance"]
        out = dict(get_img_metrics(method, bal))
        img, proc_rt = get_image(method, bal)

        ref_cached = (pair.sequence_id, pair.reference_index, det, F.params_json(params)) in self.ref_cache
        ref = self.reference_features(pair, det, params)
        dkey = (method, bal, det, F.params_json(params))
        if dkey not in det_cache:
            det_cache[dkey] = F.detect(det, params, img, self._mask(img.shape))
        tgt = det_cache[dkey]

        # keypoint repeatability (same detector config on clean target)
        rc = ev["repeatability"]
        clean = self.clean_target_features(pair, det, params)
        rep, n_rep, n_cv = M.repeatability(clean.xy, tgt.xy, rc["tolerance_px"], rc["deduplicate_positions_decimals"])
        default = cfg["detectors"][det]["default"]
        if F.params_json(params) == F.params_json(default):
            rep_def = rep
        else:
            clean_def = self.clean_target_features(pair, det, default)
            rep_def, _, _ = M.repeatability(clean_def.xy, tgt.xy, rc["tolerance_px"], rc["deduplicate_positions_decimals"])

        # matching clean reference <-> processed target
        mr = F.ratio_match(det, ref, tgt, cfg["matching"]["ratio"], cfg["matching"]["knn_k"])
        ref_xy = ref.xy[mr.query_idx] if len(mr.query_idx) else np.zeros((0, 2))
        tgt_xy = tgt.xy[mr.train_idx] if len(mr.train_idx) else np.zeros((0, 2))
        err = M.gt_errors(pair.H_gt, ref_xy, tgt_xy, float(ev["homography"]["min_abs_w"]))
        n_good = len(err)
        c3 = int((err <= 3.0).sum())
        c5 = int((err <= 5.0).sum())

        H_est, inl, h_reason, h_rt = F.estimate_homography(ref_xy, tgt_xy, cfg["ransac"])
        if len(ref) == 0:
            h_reason = "no_reference_keypoints"
        elif len(tgt) == 0:
            h_reason = "no_target_keypoints"
        hres, eval_reason = M.evaluate_homography(H_est, pair.H_gt, pair.ref.shape, pair.tgt.shape,
                                                  ev["homography"], m)
        reason = h_reason or eval_reason
        n_inl = int(inl.sum())

        out.update({
            "keypoint_repeatability": rep, "rep_n_repeated": n_rep, "rep_n_clean_valid": n_cv,
            "keypoint_repeatability_vs_default_clean": rep_def,
            "reference_keypoints": len(ref), "target_keypoints": len(tgt), "good_matches": n_good,
            "gt_correct_matches_3px": c3, "gt_correct_matches_5px": c5,
            "match_precision_3px": c3 / n_good if n_good else np.nan,
            "match_precision_5px": c5 / n_good if n_good else np.nan,
            "ransac_inliers": n_inl, "ransac_inlier_ratio": n_inl / n_good if n_good else np.nan,
            "ransac_inliers_gt_correct_3px": int((inl & (err <= 3.0)).sum()),
            "failure_reason": reason,
            "runtime_processing_ms": proc_rt, "runtime_detect_ms": tgt.runtime_ms,
            "runtime_match_ms": mr.runtime_ms, "runtime_homography_ms": h_rt,
            "runtime_reference_detect_ms": 0.0 if ref_cached else ref.runtime_ms,
        })
        out.update(hres)
        out["runtime_ms"] = proc_rt + tgt.runtime_ms + mr.runtime_ms + h_rt
        out["match_detail_file"] = self._save_details(pair, e, row, ref, tgt, mr, ref_xy, tgt_xy, err, inl, H_est)
        return out

    def _save_details(self, pair, e, row, ref, tgt, mr, ref_xy, tgt_xy, err, inl, H_est):
        bal = "" if e["wiener_balance"] is None else f"__b{e['wiener_balance']}"
        att = "" if self.attempt == 1 else f"__att{self.attempt}"
        name = (f"{row['blur_id']}__{e['group']}__{e['processing_method']}__{e['detector']}__"
                f"{e['detector_config']}{bal}{att}.npz")
        d = self.details_dir / pair.pair_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        meta = {k: row[k] for k in ("run_id", "pair_id", "sequence_id", "blur_id", "group", "processing_method",
                                    "detector", "detector_config", "detector_params", "wiener_balance")}
        np.savez_compressed(
            path, meta=np.array(json.dumps(meta)),
            H_gt=pair.H_gt, H_est=np.full((3, 3), np.nan) if H_est is None else H_est,
            ref_shape=np.array(pair.ref.shape), tgt_shape=np.array(pair.tgt.shape),
            ref_kp=ref.kp.astype(np.float32), tgt_kp=tgt.kp.astype(np.float32),
            query_idx=mr.query_idx.astype(np.int32), train_idx=mr.train_idx.astype(np.int32),
            dist1=mr.dist1.astype(np.float32), dist2=mr.dist2.astype(np.float32),
            ref_xy=ref_xy, tgt_xy=tgt_xy, gt_err=err, ransac_inlier=inl)
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:  # e.g. pytest temp dirs
            return path.as_posix()


# ------------------------------------------------------------------ CSV helpers
class CsvAppender:
    def __init__(self, path):
        self.path = Path(path)
        self.new = not self.path.exists()

    def append(self, rows):
        if not rows:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
            if self.new:
                w.writeheader()
                self.new = False
            for r in rows:
                w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
