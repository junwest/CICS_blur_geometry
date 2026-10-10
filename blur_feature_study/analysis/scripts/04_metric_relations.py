"""Relations between image quality (PSNR/SSIM), structure preservation (Edge F1, Repeatability) and
geometric matching (GT-correct@3, precision, H error, success). Three dependence-aware views:

  pooled    : Spearman over all processed rows (A, C, D; B only for ORB) -- descriptive only (non-independent)
  within    : contrasts inside the same (pair, blur, detector) cell: does the sign of d(metric) agree with the
              sign of d(target) between two processing versions of the SAME image?  (C-A, D-A, D-C [,B-A])
  per-pair  : across the 8 blur conditions of the same pair (A rows): per-pair Spearman, distribution over pairs
  models    : logistic success ~ z(metric) (+ detector), sequence-cluster-robust SE; within-cell OLS for
              log GT-correct ratio with (pair x blur x detector) fixed effects.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUR_ORDER, STATS, TABLES, VAL_RUN, add_clean_reference, image_level, load_results,  # noqa: E402
                    main_core, spearman, write_json)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
df = add_clean_reference(main_core(load_results()))
img = image_level(load_results())
PRED = ["psnr", "ssim", "edge_f1", "keypoint_repeatability"]
TARG = ["gt_correct_matches_3px", "log_gt_correct_ratio", "match_precision_3px", "homography_error_mean",
        "homography_success_3px"]
proc = df[df.group.isin(["A", "C", "D"])].copy()             # same three groups for both detectors
proc_b = df[df.group.isin(["A", "B", "C", "D"]) & ~((df.detector == "SIFT") & (df.group == "B"))].copy()
# H error with failures as worst rank (for rank correlations)
for d in (proc, proc_b):
    d["h_err_rank_fail_worst"] = d.homography_error_mean.fillna(np.inf)

results = {}

# ---------------------------------------------------------------- 1. image level: PSNR/SSIM vs Edge F1
rows = []
for a, b in (("psnr", "edge_f1"), ("ssim", "edge_f1"), ("psnr", "edge_recall"), ("psnr", "ssim")):
    r, p, n = spearman(img[a], img[b]); rows.append({"view": "pooled_576_images", "x": a, "y": b, "rho": r, "p": p, "n": n})
    # within (pair, blur): contrasts C-A, D-A, D-C
    agree = tot = 0
    for _, g in img.groupby(["pair_id", "blur_id"], observed=True):
        g = g.set_index("processing_method")
        for t, s in (("unsharp_mask", "blurred"), ("wiener_known_psf", "blurred"), ("wiener_known_psf", "unsharp_mask")):
            dx, dy = g.loc[t, a] - g.loc[s, a], g.loc[t, b] - g.loc[s, b]
            if dx != 0 and dy != 0:
                tot += 1; agree += int(np.sign(dx) == np.sign(dy))
    rows.append({"view": "within_image_contrasts", "x": a, "y": b, "sign_agreement": agree / tot, "n": tot})
    # per pair across 8 blur conditions (blurred only)
    rs = [spearman(g[a], g[b])[0] for _, g in img[img.processing_method == "blurred"].groupby("pair_id")]
    rows.append({"view": "per_pair_across_blur_A", "x": a, "y": b, "rho_median": float(np.nanmedian(rs)),
                 "rho_min": float(np.nanmin(rs)), "rho_max": float(np.nanmax(rs)), "n": len(rs)})
    # across blur conditions (means over pairs)
    m = img[img.processing_method == "blurred"].groupby("blur_id", observed=True)[[a, b]].mean()
    r, p, n = spearman(m[a], m[b]); rows.append({"view": "blur_condition_means_A", "x": a, "y": b, "rho": r, "p": p, "n": n})
img_rel = pd.DataFrame(rows)
img_rel.to_csv(STATS / "S2_image_level_relations.csv", index=False)
print("== image-level relations"); print(img_rel.round(3).to_string())

# C-A sign table: PSNR vs edge_f1 by blur (which direction each moves)
piv = img.pivot_table(index=["pair_id", "blur_id"], columns="processing_method", values=["psnr", "edge_f1", "ssim"], observed=True)
ca = pd.DataFrame({"dpsnr": piv[("psnr", "unsharp_mask")] - piv[("psnr", "blurred")],
                   "dssim": piv[("ssim", "unsharp_mask")] - piv[("ssim", "blurred")],
                   "dedge": piv[("edge_f1", "unsharp_mask")] - piv[("edge_f1", "blurred")]}).reset_index()
ca["pattern"] = np.select([(ca.dpsnr > 0) & (ca.dedge > 0), (ca.dpsnr < 0) & (ca.dedge > 0), (ca.dpsnr > 0) & (ca.dedge < 0), (ca.dpsnr < 0) & (ca.dedge < 0)],
                          ["psnr+ edge+", "psnr- edge+", "psnr+ edge-", "psnr- edge-"], "tie")
t = pd.crosstab(ca.blur_id, ca.pattern)
t.to_csv(TABLES / "T11_unsharp_psnr_vs_edge_sign_patterns.csv")
print("\n== Unsharp (C-A): sign pattern of dPSNR vs dEdgeF1 per blur (24 pairs each)"); print(t.to_string())
results["unsharp_sign_patterns"] = t.to_dict()

# ---------------------------------------------------------------- 2. detector level: structure vs matching
rows = []
for det in ("ORB", "SIFT"):
    d = proc[proc.detector == det]
    for x in PRED:
        for y in ["gt_correct_matches_3px", "log_gt_correct_ratio", "match_precision_3px", "h_err_rank_fail_worst", "homography_success_3px"]:
            yy = -d[y] if y == "h_err_rank_fail_worst" else d[y]      # orient so that larger = better
            r, p, n = spearman(d[x], yy)
            rows.append({"detector": det, "view": "pooled_rows", "x": x, "y": y, "rho": r, "p": p, "n": n})
        # within cell contrasts
        for y in ["gt_correct_matches_3px", "match_precision_3px", "homography_error_mean", "homography_success_3px"]:
            agree = tot = 0
            for _, g in d.groupby(["pair_id", "blur_id"], observed=True):
                g = g.set_index("group")
                for t_, s_ in (("C", "A"), ("D", "A"), ("D", "C")):
                    if t_ not in g.index or s_ not in g.index:
                        continue
                    dx, dy = g.loc[t_, x] - g.loc[s_, x], g.loc[t_, y] - g.loc[s_, y]
                    if y == "homography_error_mean":
                        dy = -dy
                    if pd.notna(dx) and pd.notna(dy) and dx != 0 and dy != 0:
                        tot += 1; agree += int(np.sign(dx) == np.sign(dy))
            rows.append({"detector": det, "view": "within_cell_contrasts", "x": x, "y": y, "sign_agreement": agree / tot if tot else np.nan, "n": tot})
        # per pair across blur (A rows)
        for y in ["gt_correct_matches_3px", "match_precision_3px"]:
            rs = [spearman(g[x], g[y])[0] for _, g in d[d.group == "A"].groupby("pair_id")]
            rows.append({"detector": det, "view": "per_pair_across_blur_A", "x": x, "y": y,
                         "rho_median": float(np.nanmedian(rs)), "rho_q1": float(np.nanpercentile(rs, 25)),
                         "rho_q3": float(np.nanpercentile(rs, 75)), "n": len(rs)})
det_rel = pd.DataFrame(rows)
det_rel.to_csv(STATS / "S3_detector_level_relations.csv", index=False)
print("\n== detector-level relations (pooled Spearman; rows A/C/D)")
print(det_rel[det_rel.view == "pooled_rows"].pivot_table(index=["detector", "x"], columns="y", values="rho").round(3).to_string())
print("\n== within-cell sign agreement P(sign d_metric == sign d_target) over C-A, D-A, D-C contrasts")
print(det_rel[det_rel.view == "within_cell_contrasts"].pivot_table(index=["detector", "x"], columns="y", values="sign_agreement").round(3).to_string())
print("\n== per-pair Spearman across 8 blur conditions (A rows): median [q1, q3]")
print(det_rel[det_rel.view == "per_pair_across_blur_A"].round(3).to_string())

# ---------------------------------------------------------------- 3. models
# 3a. logistic: success@3 ~ z(metric) + detector, cluster-robust by sequence (rows A/C/D)
mrows = []
Xz = proc.copy()
for x in PRED:
    Xz[x + "_z"] = (Xz[x] - Xz[x].mean()) / Xz[x].std()
Xz["sift"] = (Xz.detector == "SIFT").astype(float)
groups = Xz.sequence_id.astype("category").cat.codes.values
for x in PRED:
    X = sm.add_constant(Xz[[x + "_z", "sift"]].astype(float))
    m = sm.GLM(Xz.homography_success_3px.astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    pred = m.predict(X)
    auc = stats.mannwhitneyu(pred[Xz.homography_success_3px == 1], pred[Xz.homography_success_3px == 0]).statistic / ((Xz.homography_success_3px == 1).sum() * (Xz.homography_success_3px == 0).sum())
    mrows.append({"model": "logit_success3", "predictors": x, "coef_z": m.params[x + "_z"], "se_cluster": m.bse[x + "_z"], "p": m.pvalues[x + "_z"],
                  "mcfadden_r2": 1 - m.llf / m.llnull, "auc": float(auc), "n": int(m.nobs)})
X = sm.add_constant(Xz[[p + "_z" for p in PRED] + ["sift"]].astype(float))
m = sm.GLM(Xz.homography_success_3px.astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
for x in PRED:
    mrows.append({"model": "logit_success3_joint", "predictors": "all4", "term": x, "coef_z": m.params[x + "_z"], "se_cluster": m.bse[x + "_z"], "p": m.pvalues[x + "_z"], "mcfadden_r2": 1 - m.llf / m.llnull, "n": int(m.nobs)})
# per detector separately
for det in ("ORB", "SIFT"):
    sel = Xz.detector == det
    g = groups[sel.values]
    for x in PRED:
        X = sm.add_constant(Xz.loc[sel, [x + "_z"]].astype(float))
        mm = sm.GLM(Xz.loc[sel, "homography_success_3px"].astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g})
        mrows.append({"model": f"logit_success3_{det}", "predictors": x, "coef_z": mm.params[x + "_z"], "se_cluster": mm.bse[x + "_z"], "p": mm.pvalues[x + "_z"], "mcfadden_r2": 1 - mm.llf / mm.llnull, "n": int(mm.nobs)})

# 3b. within-cell OLS (fixed effects for pair x blur x detector), target = log GT-correct ratio
def within(dat, cols):
    g = dat.groupby(["pair_id", "blur_id", "detector"], observed=True)[cols]
    return dat[cols] - g.transform("mean")
W = within(proc, PRED + ["log_gt_correct_ratio", "match_precision_3px"])
for y in ("log_gt_correct_ratio", "match_precision_3px"):
    ok = W[PRED + [y]].notna().all(axis=1)
    for x in PRED:
        mm = sm.OLS(W.loc[ok, y], W.loc[ok, [x]]).fit(cov_type="cluster", cov_kwds={"groups": groups[ok.values]})
        mrows.append({"model": f"within_cell_OLS_{y}", "predictors": x, "coef": mm.params[x], "se_cluster": mm.bse[x], "p": mm.pvalues[x], "r2_within": mm.rsquared, "n": int(mm.nobs)})
    mm = sm.OLS(W.loc[ok, y], W.loc[ok, PRED]).fit(cov_type="cluster", cov_kwds={"groups": groups[ok.values]})
    for x in PRED:
        mrows.append({"model": f"within_cell_OLS_{y}_joint", "predictors": "all4", "term": x, "coef": mm.params[x], "se_cluster": mm.bse[x], "p": mm.pvalues[x], "r2_within": mm.rsquared, "n": int(mm.nobs)})
models = pd.DataFrame(mrows)
models.to_csv(STATS / "S4_models.csv", index=False)
print("\n== models"); print(models.round(4).to_string())

# ---------------------------------------------------------------- 4. dissociation counts
rows = []
for det in ("ORB", "SIFT"):
    d = proc_b[proc_b.detector == det]
    for t_, s_ in (("C", "A"), ("D", "A"), ("B", "A"), ("D", "C")):
        if t_ == "B" and det == "SIFT":
            continue
        a = d[d.group == s_].set_index(["pair_id", "blur_id"]); b = d[d.group == t_].set_index(["pair_id", "blur_id"]).loc[a.index]
        dp, de, dr = b.psnr - a.psnr, b.edge_f1 - a.edge_f1, b.keypoint_repeatability - a.keypoint_repeatability
        dg = b.gt_correct_matches_3px.astype(float) - a.gt_correct_matches_3px.astype(float)
        ds = b.homography_success_3px.astype(float) - a.homography_success_3px.astype(float)
        rows.append({"detector": det, "contrast": f"{t_}-{s_}", "n": len(a),
                     "psnr_up": int((dp > 0).sum()), "psnr_up_gt_down": int(((dp > 0) & (dg < 0)).sum()), "psnr_down_gt_up": int(((dp < 0) & (dg > 0)).sum()),
                     "psnr_up_success_lost": int(((dp > 0) & (ds < 0)).sum()), "psnr_down_success_gained": int(((dp < 0) & (ds > 0)).sum()),
                     "edge_up": int((de > 0).sum()), "edge_up_gt_down": int(((de > 0) & (dg < 0)).sum()), "edge_down_gt_up": int(((de < 0) & (dg > 0)).sum()),
                     "rep_up": int((dr > 0).sum()), "rep_up_gt_down": int(((dr > 0) & (dg < 0)).sum()), "rep_down_gt_up": int(((dr < 0) & (dg > 0)).sum()),
                     "gt_up": int((dg > 0).sum()), "gt_down": int((dg < 0).sum()), "success_gained": int((ds > 0).sum()), "success_lost": int((ds < 0).sum())})
diss = pd.DataFrame(rows)
diss.to_csv(TABLES / "T12_dissociation_counts.csv", index=False)
print("\n== dissociation counts per contrast (192 cells)"); print(diss.to_string())

# ---------------------------------------------------------------- 5. validation Wiener balance: PSNR-best vs success-best
sw = pd.read_csv(VAL_RUN / "selection_wiener.csv")
val = load_results(VAL_RUN)
vd = val[val.group == "D_candidate"]
agg = vd.groupby(["detector", "blur_id", "wiener_balance"], observed=True).agg(success3=("homography_success_3px", "mean"), gt3=("gt_correct_matches_3px", "mean"), rep=("keypoint_repeatability", "mean")).reset_index()
best = []
for (det, b), g in agg.groupby(["detector", "blur_id"], observed=True):
    psnr_best = sw[(sw.blur_id == b) & (sw.selected == 1)].wiener_balance.iloc[0]
    s_best = g.sort_values(["success3", "gt3"], ascending=False).wiener_balance.iloc[0]
    g3_best = g.sort_values("gt3", ascending=False).wiener_balance.iloc[0]
    best.append({"detector": det, "blur_id": b, "psnr_best": psnr_best, "success_best": s_best, "gt3_best": g3_best,
                 "success_at_psnr_best": float(g[g.wiener_balance == psnr_best].success3.iloc[0]), "success_max": float(g.success3.max()),
                 "gt3_at_psnr_best": float(g[g.wiener_balance == psnr_best].gt3.iloc[0]), "gt3_max": float(g.gt3.max())})
vb = pd.DataFrame(best)
vb.to_csv(TABLES / "T13_validation_wiener_psnr_vs_success_best.csv", index=False)
print("\n== validation: PSNR-selected balance vs success/GT-correct-best balance (8 validation pairs)"); print(vb.to_string())
results["val_wiener_agree_success"] = int((vb.psnr_best == vb.success_best).sum()); results["val_wiener_agree_gt3"] = int((vb.psnr_best == vb.gt3_best).sum())
results["val_wiener_n"] = int(len(vb))

# ---------------------------------------------------------------- 6. keypoint count vs accuracy
rows = []
for det in ("ORB", "SIFT"):
    d = proc_b[proc_b.detector == det]
    r1 = spearman(d.target_keypoints, d.gt_correct_matches_3px); r2 = spearman(d.target_keypoints, d.match_precision_3px)
    rows.append({"detector": det, "rho_kp_vs_gt3": r1[0], "rho_kp_vs_precision": r2[0], "n": r1[2]})
    a = d[d.group == "A"].set_index(["pair_id", "blur_id"])
    for t_ in (["B", "C", "D"] if det == "ORB" else ["C", "D"]):
        b = d[d.group == t_].set_index(["pair_id", "blur_id"]).loc[a.index]
        more_kp = (b.target_keypoints > a.target_keypoints)
        rows.append({"detector": det, "contrast": f"{t_}-A", "cells_more_kp": int(more_kp.sum()),
                     "more_kp_but_fewer_correct": int((more_kp & (b.gt_correct_matches_3px < a.gt_correct_matches_3px)).sum()),
                     "more_kp_but_lower_precision": int((more_kp & (b.match_precision_3px < a.match_precision_3px)).sum()),
                     "more_kp_and_success_lost": int((more_kp & (b.homography_success_3px < a.homography_success_3px)).sum())})
kp = pd.DataFrame(rows)
kp.to_csv(TABLES / "T14_keypoint_count_vs_accuracy.csv", index=False)
print("\n== keypoint count vs accuracy"); print(kp.to_string())
write_json(STATS / "S5_relations_misc.json", results)
