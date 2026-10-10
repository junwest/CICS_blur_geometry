"""Mechanism / failure analyses that go beyond the 8 metrics.

 1. Motion-blur orientation asymmetry explained by the gradient-orientation energy of each clean target
 2. Edge F1 decomposition (recall-limited under blur; unsharp trades precision for recall)
 3. Detector response: SIFT keypoint starvation vs ORB budget saturation under blur
 4. Success transitions A->D per cell: regressions (A success, D fail) and the D-vs-CLEAN cases
 5. Hard pairs: contribution of clean-failing pairs; per-pair success profiles
 6. Failure-reason mechanics (horizon / singular failures vs match counts)
 7. ORB tuning (B): keypoints / matches / precision trade-off per blur
 8. Illumination vs viewpoint: D-A and C-A effects by sequence type
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUR_ORDER, PROJECT, STATS, TABLES, add_clean_reference, cluster_bootstrap_ci, image_level,  # noqa: E402
                    load_results, main_core, paired_summary, spearman, write_json)

sys.path.insert(0, str(PROJECT))
from src.dataset import load_manifest, load_pair  # noqa: E402
from src.io_utils import load_config  # noqa: E402

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
cfg = load_config()
df = add_clean_reference(main_core(load_results()))
img = image_level(load_results())
seq_of = df.drop_duplicates("pair_id").set_index("pair_id").sequence_id
m = cfg["preprocessing"]["valid_margin_px"]
out = {}

# ---------------------------------------------------------------- 1. gradient orientation energy of clean targets
man = load_manifest(cfg).set_index("pair_id")
grad = []
for pid in sorted(df.pair_id.unique()):
    pair = load_pair(man.loc[pid].to_dict() | {"pair_id": pid}, cfg)
    t = pair.tgt.astype(np.float64)[m:-m, m:-m]
    ix = cv2.Sobel(t, cv2.CV_64F, 1, 0, ksize=3); iy = cv2.Sobel(t, cv2.CV_64F, 0, 1, ksize=3)
    ex, ey = float((ix ** 2).mean()), float((iy ** 2).mean())
    # also: fraction of Canny edge pixels whose gradient is mostly horizontal (|Ix|>|Iy|, i.e. vertical edge lines)
    edges = cv2.Canny(pair.tgt, 80, 160, L2gradient=True)[m:-m, m:-m] > 0
    vert_edge_frac = float((np.abs(ix) > np.abs(iy))[edges].mean()) if edges.any() else np.nan
    grad.append({"pair_id": pid, "grad_energy_x": ex, "grad_energy_y": ey, "log10_ex_over_ey": np.log10(ex / ey),
                 "vertical_edge_fraction": vert_edge_frac, "n_edge_px": int(edges.sum()), "mean_intensity": float(t.mean()),
                 "intensity_std": float(t.std())})
grad = pd.DataFrame(grad).set_index("pair_id")
a = img[img.processing_method == "blurred"].pivot_table(index="pair_id", columns="blur_id", values=["psnr", "edge_f1", "ssim"], observed=True)
grad["dpsnr_l17"] = a[("psnr", "motion_l17_a0")] - a[("psnr", "motion_l17_a90")]
grad["dpsnr_l9"] = a[("psnr", "motion_l9_a0")] - a[("psnr", "motion_l9_a90")]
grad["dedge_l17"] = a[("edge_f1", "motion_l17_a0")] - a[("edge_f1", "motion_l17_a90")]
grad["dssim_l17"] = a[("ssim", "motion_l17_a0")] - a[("ssim", "motion_l17_a90")]
for det in ("ORB", "SIFT"):
    d = df[(df.detector == det) & (df.group == "A")].pivot_table(index="pair_id", columns="blur_id", values=["gt_correct_matches_3px", "keypoint_repeatability"], observed=True)
    grad[f"dgt3_l17_{det}"] = d[("gt_correct_matches_3px", "motion_l17_a0")].astype(float) - d[("gt_correct_matches_3px", "motion_l17_a90")].astype(float)
    grad[f"drep_l17_{det}"] = d[("keypoint_repeatability", "motion_l17_a0")] - d[("keypoint_repeatability", "motion_l17_a90")]
grad.to_csv(TABLES / "T15_gradient_orientation_vs_motion_asymmetry.csv")
rows = []
for y in ["dpsnr_l17", "dpsnr_l9", "dssim_l17", "dedge_l17", "dgt3_l17_ORB", "dgt3_l17_SIFT", "drep_l17_ORB", "drep_l17_SIFT"]:
    for x in ["log10_ex_over_ey", "vertical_edge_fraction"]:
        r, p, n = spearman(grad[x], grad[y]); rows.append({"x": x, "y": y, "rho": r, "p": p, "n": n})
orient = pd.DataFrame(rows)
orient.to_csv(STATS / "S6_orientation_asymmetry_correlations.csv", index=False)
print("== 1. motion-orientation asymmetry (a0 - a90) vs clean-image gradient orientation"); print(orient.round(3).to_string())
print(grad[["log10_ex_over_ey", "vertical_edge_fraction", "dpsnr_l17", "dedge_l17", "dgt3_l17_ORB", "dgt3_l17_SIFT"]].round(3).to_string())

# ---------------------------------------------------------------- 2. Edge F1 decomposition
e = img.groupby(["processing_method", "blur_id"], observed=True)[["edge_precision", "edge_recall", "edge_f1", "edge_n_clean", "edge_n_proc"]].mean()
e["edge_count_ratio_proc_over_clean"] = e.edge_n_proc / e.edge_n_clean
e.to_csv(TABLES / "T16_edge_decomposition.csv")
print("\n== 2. edge decomposition"); print(e.round(3).to_string())
r_rec, _, _ = spearman(img.edge_recall, img.edge_f1); r_prec, _, _ = spearman(img.edge_precision, img.edge_f1)
out["edge_f1_rho_with_recall"] = r_rec; out["edge_f1_rho_with_precision"] = r_prec
print(f"Spearman(edge_f1, recall)={r_rec:.3f}  Spearman(edge_f1, precision)={r_prec:.3f}")

# ---------------------------------------------------------------- 3. detector response: keypoint counts
k = df[df.group.isin(["CLEAN", "A", "C", "D"])].groupby(["detector", "group", "blur_id"], observed=True).agg(
    target_kp=("target_keypoints", "mean"), ref_kp=("reference_keypoints", "mean"), good=("good_matches", "mean"),
    gt3=("gt_correct_matches_3px", "mean"), prec=("match_precision_3px", "mean"), rep=("keypoint_repeatability", "mean"),
    kp_at_budget=("target_keypoints", lambda s: float((s >= 1500).mean()))).reset_index()
k.to_csv(TABLES / "T17_keypoint_counts_by_condition.csv", index=False)
print("\n== 3. keypoint counts (A rows)"); print(k[k.group == "A"].round(3).to_string())
# Does SIFT's loss of keypoints explain its loss of correct matches? per-pair across blur
rows = []
for det in ("ORB", "SIFT"):
    d = df[(df.detector == det) & (df.group == "A")]
    rs = [spearman(g.target_keypoints, g.gt_correct_matches_3px)[0] for _, g in d.groupby("pair_id")]
    rows.append({"detector": det, "view": "per_pair_across_blur_A", "rho_kp_vs_gt3_median": float(np.nanmedian(rs))})
    r, p, n = spearman(d.target_keypoints, d.gt_correct_matches_3px); rows.append({"detector": det, "view": "pooled_A", "rho_kp_vs_gt3": r, "p": p, "n": n})
    r, p, n = spearman(d.target_keypoints, d.match_precision_3px); rows.append({"detector": det, "view": "pooled_A", "rho_kp_vs_precision": r, "p": p, "n": n})
kpr = pd.DataFrame(rows); kpr.to_csv(STATS / "S7_keypoint_count_relations.csv", index=False); print(kpr.round(3).to_string())

# ---------------------------------------------------------------- 4. success transitions and regressions
rows = []
for det in ("ORB", "SIFT"):
    d = df[df.detector == det]
    a_ = d[d.group == "A"].set_index(["pair_id", "blur_id"])
    for t_ in ("B", "C", "D"):
        if det == "SIFT" and t_ == "B":
            continue
        b_ = d[d.group == t_].set_index(["pair_id", "blur_id"]).loc[a_.index]
        lost = a_.index[(a_.homography_success_3px == 1) & (b_.homography_success_3px == 0)]
        for idx in lost:
            rows.append({"detector": det, "contrast": f"{t_}-A", "pair_id": idx[0], "blur_id": idx[1],
                         "A_err": a_.loc[idx, "homography_error_mean"], "T_err": b_.loc[idx, "homography_error_mean"],
                         "T_reason": b_.loc[idx, "failure_reason"], "A_gt3": a_.loc[idx, "gt_correct_matches_3px"], "T_gt3": b_.loc[idx, "gt_correct_matches_3px"],
                         "A_good": a_.loc[idx, "good_matches"], "T_good": b_.loc[idx, "good_matches"], "A_prec": a_.loc[idx, "match_precision_3px"], "T_prec": b_.loc[idx, "match_precision_3px"]})
reg = pd.DataFrame(rows); reg.to_csv(TABLES / "T18_success_regressions_treat_fails_where_A_succeeded.csv", index=False)
print("\n== 4. regressions (A success@3 -> treatment fail)"); print(reg.round(2).to_string())
# D vs CLEAN per cell: D success where CLEAN failed
rows = []
for det in ("ORB", "SIFT"):
    d = df[df.detector == det]
    cl = d[d.group == "CLEAN"].set_index("pair_id")
    dd = d[d.group == "D"]
    for r in dd.itertuples():
        c = cl.loc[r.pair_id]
        if c.homography_success_3px == 0 and r.homography_success_3px == 1:
            rows.append({"detector": det, "pair_id": r.pair_id, "blur_id": r.blur_id, "clean_err": c.homography_error_mean, "clean_reason": c.failure_reason,
                         "clean_good": c.good_matches, "clean_gt3": c.gt_correct_matches_3px, "D_err": r.homography_error_mean, "D_good": r.good_matches, "D_gt3": r.gt_correct_matches_3px, "D_prec": r.match_precision_3px})
dvc = pd.DataFrame(rows); dvc.to_csv(TABLES / "T19_D_success_where_CLEAN_failed.csv", index=False)
print("\n== 4b. D succeeds where CLEAN failed"); print(dvc.round(2).to_string())
# distribution of H error among successes: is D's error distribution as tight as CLEAN's?
rows = []
for det in ("ORB", "SIFT"):
    for g in ("CLEAN", "A", "B", "C", "D"):
        x = df[(df.detector == det) & (df.group == g)].homography_error_mean.dropna()
        rows.append({"detector": det, "group": g, "n_defined": len(x), "p25": x.quantile(.25), "median": x.median(), "p75": x.quantile(.75), "p90": x.quantile(.9), "share_le_1px": float((x <= 1).mean()), "share_le_3px": float((x <= 3).mean())})
herr = pd.DataFrame(rows); herr.to_csv(TABLES / "T20_h_error_distribution_defined_only.csv", index=False)
print("\n== 4c. H error distribution (defined rows only)"); print(herr.round(3).to_string())

# ---------------------------------------------------------------- 5. hard pairs / per-pair profiles
prof = df[df.group != "CLEAN"].groupby(["detector", "pair_id", "group"], observed=True).agg(success=("homography_success_3px", "mean"), gt3=("gt_correct_matches_3px", "mean")).reset_index()
prof = prof.pivot_table(index=["detector", "pair_id"], columns="group", values=["success", "gt3"], observed=True)
cl = df[df.group == "CLEAN"].set_index(["detector", "pair_id"])[["homography_success_3px", "homography_success_5px", "gt_correct_matches_3px", "sequence_type", "gt_quality_flag"]]
prof[("clean", "success3")] = cl.homography_success_3px; prof[("clean", "success5")] = cl.homography_success_5px; prof[("clean", "gt3")] = cl.gt_correct_matches_3px
prof[("meta", "type")] = cl.sequence_type; prof[("meta", "gt_flag")] = cl.gt_quality_flag
prof.to_csv(TABLES / "T21_per_pair_profiles.csv")
print("\n== 5. per-pair success profile (mean over 8 blur)"); print(prof.round(2).to_string())
# share of A-failures attributable to pairs that already fail clean (@5px)
rows = []
for det in ("ORB", "SIFT"):
    d = df[(df.detector == det)]
    cf = set(d[(d.group == "CLEAN") & (d.homography_success_5px == 0)].pair_id)
    for g in ("A", "C", "D"):
        x = d[d.group == g]
        fails = x[x.homography_success_3px == 0]
        rows.append({"detector": det, "group": g, "n_fail": len(fails), "n_fail_in_clean_fail_pairs": int(fails.pair_id.isin(cf).sum()),
                     "share_fail_from_clean_fail_pairs": float(fails.pair_id.isin(cf).mean()) if len(fails) else np.nan, "n_clean_fail_pairs": len(cf)})
hp = pd.DataFrame(rows); hp.to_csv(TABLES / "T22_failures_attributable_to_clean_fail_pairs.csv", index=False); print(hp.to_string())

# ---------------------------------------------------------------- 6. failure-reason mechanics
fr = df[df.group != "CLEAN"].copy(); fr["reason"] = fr.failure_reason.replace("", "success")
mech = fr.groupby(["detector", "reason"]).agg(n=("pair_id", "size"), good_med=("good_matches", "median"), gt3_med=("gt_correct_matches_3px", "median"),
                                              prec_med=("match_precision_3px", "median"), inl_med=("ransac_inliers", "median"), rep_med=("keypoint_repeatability", "median")).reset_index()
mech.to_csv(TABLES / "T23_failure_reason_mechanics.csv", index=False); print("\n== 6. failure reason mechanics"); print(mech.round(3).to_string())
# how many 'successes' rely on < 15 GT-correct matches?
rows = []
for det in ("ORB", "SIFT"):
    for g in ("A", "B", "C", "D"):
        x = df[(df.detector == det) & (df.group == g)]
        rows.append({"detector": det, "group": g, "success_n": int(x.homography_success_3px.sum()), "success_with_gt3_lt15": int(((x.homography_success_3px == 1) & (x.gt_correct_matches_3px < 15)).sum()),
                     "fail_with_gt3_ge30": int(((x.homography_success_3px == 0) & (x.gt_correct_matches_3px >= 30)).sum()), "fail_n": int((x.homography_success_3px == 0).sum())})
thin = pd.DataFrame(rows); thin.to_csv(TABLES / "T24_success_on_thin_evidence.csv", index=False); print(thin.to_string())

# ---------------------------------------------------------------- 7. ORB tuning trade-off per blur
rows = []
d = df[df.detector == "ORB"]
a_ = d[d.group == "A"].set_index(["pair_id", "blur_id"]); b_ = d[d.group == "B"].set_index(["pair_id", "blur_id"]).loc[a_.index]
for blur in BLUR_ORDER:
    sel = a_.index.get_level_values(1) == blur
    cl = seq_of.loc[a_.index.get_level_values(0)[sel]].values
    r = {"blur": blur}
    for mcol in ("target_keypoints", "good_matches", "gt_correct_matches_3px", "match_precision_3px", "keypoint_repeatability_vs_default_clean", "keypoint_repeatability", "ransac_inlier_ratio"):
        s = paired_summary((b_[mcol].astype(float) - a_[mcol].astype(float)).values[sel], cl, n_boot=2000)
        r[mcol] = f"{s['mean']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
    rows.append(r)
tune = pd.DataFrame(rows); tune.to_csv(TABLES / "T25_orb_tuning_tradeoff_per_blur.csv", index=False); print("\n== 7. ORB tuned - default per blur"); print(tune.to_string())

# ---------------------------------------------------------------- 8. by sequence type
rows = []
for det in ("ORB", "SIFT"):
    for stype in ("illumination", "viewpoint"):
        d = df[(df.detector == det) & (df.sequence_type == stype)]
        a_ = d[d.group == "A"].groupby("pair_id")[["homography_success_3px", "gt_correct_matches_3px", "log_gt_correct_ratio"]].mean()
        for t_ in ("B", "C", "D"):
            if det == "SIFT" and t_ == "B":
                continue
            b_ = d[d.group == t_].groupby("pair_id")[["homography_success_3px", "gt_correct_matches_3px", "log_gt_correct_ratio"]].mean().loc[a_.index]
            cl = seq_of.loc[a_.index].values
            for mcol in ("homography_success_3px", "gt_correct_matches_3px", "log_gt_correct_ratio"):
                s = paired_summary((b_[mcol] - a_[mcol]).values, cl, n_boot=2000)
                rows.append({"detector": det, "sequence_type": stype, "contrast": f"{t_}-A", "metric": mcol, "n_pairs": s["n"], "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n_pos": s["n_pos"], "n_neg": s["n_neg"]})
st = pd.DataFrame(rows); st.to_csv(STATS / "S8_effects_by_sequence_type.csv", index=False); print("\n== 8. effects by sequence type (pair-level means, n=12 per type)"); print(st.round(3).to_string())

# A-level degradation by type: clean-normalised
deg = df[df.group == "A"].groupby(["detector", "sequence_type", "blur_id"], observed=True).agg(success=("homography_success_3px", "mean"), log_ratio=("log_gt_correct_ratio", "mean"), rep=("keypoint_repeatability", "mean")).reset_index()
deg.to_csv(TABLES / "T26_degradation_by_type_blur.csv", index=False)
write_json(STATS / "S9_mechanisms_misc.json", out)
