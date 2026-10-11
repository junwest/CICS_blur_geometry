"""Evaluation of one (reference features, target features) combination, replicating the base
src/pipeline.Runner._evaluate computations exactly (same functions, same order, same failure reasons),
plus the extension diagnostics (spatial distribution). Image metrics are computed separately.
"""
import numpy as np

from src import features as F
from src import metrics as M


def image_metrics(cfg, clean, img):
    ev, m = cfg["evaluation"], int(cfg["preprocessing"]["valid_margin_px"])
    d = {"psnr": M.psnr(clean, img, m, ev["psnr"]["data_range"]), "ssim": M.ssim(clean, img, m, ev["ssim"])}
    d.update(M.edge_prf(M.canny(clean, ev["canny"]), M.canny(img, ev["canny"]), ev["edge_tolerance_px"], m))
    return d


def repeatability(cfg, clean_tgt, tgt):
    rc = cfg["evaluation"]["repeatability"]
    rep, n_rep, n_cv = M.repeatability(clean_tgt.xy, tgt.xy, rc["tolerance_px"], rc["deduplicate_positions_decimals"])
    return {"keypoint_repeatability": rep, "rep_n_repeated": n_rep, "rep_n_clean_valid": n_cv,
            "keypoint_repeatability_vs_default_clean": rep}          # default detector config only


def match_and_estimate(cfg, det, H_gt, ref_shape, tgt_shape, ref, tgt, spatial=None):
    """Base matching / GT / RANSAC / H evaluation. Returns (metrics dict, details dict)."""
    ev, m = cfg["evaluation"], int(cfg["preprocessing"]["valid_margin_px"])
    mr = F.ratio_match(det, ref, tgt, cfg["matching"]["ratio"], cfg["matching"]["knn_k"])
    ref_xy = ref.xy[mr.query_idx] if len(mr.query_idx) else np.zeros((0, 2))
    tgt_xy = tgt.xy[mr.train_idx] if len(mr.train_idx) else np.zeros((0, 2))
    err = M.gt_errors(H_gt, ref_xy, tgt_xy, float(ev["homography"]["min_abs_w"]))
    n_good = len(err)
    c3 = int((err <= 3.0).sum())
    c5 = int((err <= 5.0).sum())
    H_est, inl, h_reason, h_rt = F.estimate_homography(ref_xy, tgt_xy, cfg["ransac"])
    if len(ref) == 0:
        h_reason = "no_reference_keypoints"
    elif len(tgt) == 0:
        h_reason = "no_target_keypoints"
    hres, eval_reason = M.evaluate_homography(H_est, H_gt, ref_shape, tgt_shape, ev["homography"], m)
    n_inl = int(inl.sum())
    out = {"reference_keypoints": len(ref), "target_keypoints": len(tgt), "good_matches": n_good,
           "gt_correct_matches_3px": c3, "gt_correct_matches_5px": c5,
           "match_precision_3px": c3 / n_good if n_good else np.nan,
           "match_precision_5px": c5 / n_good if n_good else np.nan,
           "ransac_inliers": n_inl, "ransac_inlier_ratio": n_inl / n_good if n_good else np.nan,
           "ransac_inliers_gt_correct_3px": int((inl & (err <= 3.0)).sum()),
           "failure_reason": h_reason or eval_reason,
           "runtime_match_ms": mr.runtime_ms, "runtime_homography_ms": h_rt}
    out.update(hres)
    if spatial is not None:
        good = err <= 3.0
        cov, occ = spatial.measure(ref_xy[good])
        cov_i, occ_i = spatial.measure(ref_xy[good & inl])
        out.update(gtc_hull_coverage=cov, gtc_cell_occupancy=occ,
                   inlier_gtc_hull_coverage=cov_i, inlier_gtc_cell_occupancy=occ_i,
                   spatial_reason=spatial.reason)
    details = {"query_idx": mr.query_idx.astype(np.int32), "train_idx": mr.train_idx.astype(np.int32),
               "dist1": mr.dist1.astype(np.float32), "dist2": mr.dist2.astype(np.float32),
               "gt_err": err, "ransac_inlier": inl,
               "H_est": np.full((3, 3), np.nan) if H_est is None else H_est}
    return out, details
