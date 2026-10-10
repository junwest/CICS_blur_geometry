"""Paired effects of processing / tuning on the same (pair, blur, detector) cell.

Comparisons: B-A (ORB only; SIFT B == A), C-A, D-A, D-C, D-B(ORB), and A-CLEAN, C-CLEAN, D-CLEAN.
Per blur condition: n = 24 pairs (12 sequences). Pooled: pair-level mean over the 8 blur conditions, n = 24.
Inference: Wilcoxon signed-rank on pair differences (ignores sequence clustering) + sequence-cluster bootstrap CI
(respects clustering). Holm correction across the 8 blur conditions within each (detector, comparison, metric).
Success rates: exact McNemar on discordant pairs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUR_ORDER, STATS, TABLES, add_clean_reference, cluster_bootstrap_ci, holm,  # noqa: E402
                    load_results, main_core, mcnemar_exact, paired_summary)

df = add_clean_reference(main_core(load_results()))
seq_of = df.drop_duplicates("pair_id").set_index("pair_id").sequence_id
clean_fail5 = {d: set(df[(df.group == "CLEAN") & (df.detector == d) & (df.homography_success_5px == 0)].pair_id)
               for d in ("ORB", "SIFT")}
flagged = set(df[df.gt_quality_flag == 1].pair_id)

CONT_METRICS = ["psnr", "ssim", "edge_f1", "edge_recall", "edge_precision", "keypoint_repeatability",
                "keypoint_repeatability_vs_default_clean", "target_keypoints", "good_matches",
                "gt_correct_matches_3px", "gt_correct_matches_5px", "match_precision_3px", "ransac_inlier_ratio",
                "homography_error_mean", "log_gt_correct_ratio"]
BIN_METRICS = ["homography_success_3px", "homography_success_5px"]
COMPARISONS = [("B-A", "B", "A"), ("C-A", "C", "A"), ("D-A", "D", "A"), ("D-C", "D", "C"), ("D-B", "D", "B"),
               ("A-CLEAN", "A", "CLEAN"), ("C-CLEAN", "C", "CLEAN"), ("D-CLEAN", "D", "CLEAN")]


def sub(det, group, subset=None):
    d = df[(df.detector == det) & (df.group == group)]
    if subset is not None:
        d = d[d.pair_id.isin(subset)]
    return d


def diffs_for(det, treat, base, blur, subset=None):
    """DataFrame indexed by pair_id: metric differences (treat - base) and binary outcomes."""
    t, b = sub(det, treat, subset), sub(det, base, subset)
    if blur != "pooled":
        t = t[t.blur_id == blur] if treat != "CLEAN" else t
        b = b[b.blur_id == blur] if base != "CLEAN" else b
        t, b = t.set_index("pair_id"), b.set_index("pair_id")
    else:
        t = t.groupby("pair_id")[CONT_METRICS + BIN_METRICS].mean(numeric_only=True)
        b = b.groupby("pair_id")[CONT_METRICS + BIN_METRICS].mean(numeric_only=True)
    idx = t.index.intersection(b.index)
    out = {m: t.loc[idx, m].astype(float) - b.loc[idx, m].astype(float) for m in CONT_METRICS}
    for m in BIN_METRICS:
        out[m + "__treat"] = t.loc[idx, m].astype(float)
        out[m + "__base"] = b.loc[idx, m].astype(float)
    return pd.DataFrame(out, index=idx)


def run(subset_name, subset):
    rows = []
    for det in ("ORB", "SIFT"):
        comps = [c for c in COMPARISONS if not (det == "SIFT" and "B" in (c[1], c[2]))]
        for name, treat, base in comps:
            for blur in BLUR_ORDER + ["pooled"]:
                dd = diffs_for(det, treat, base, blur, subset)
                cl = seq_of.loc[dd.index].values
                for m in CONT_METRICS:
                    s = paired_summary(dd[m].values, cl)
                    s.update(detector=det, comparison=name, blur=blur, metric=m, subset=subset_name)
                    rows.append(s)
                for m in BIN_METRICS:
                    a, b = dd[m + "__base"].values, dd[m + "__treat"].values
                    s = {"detector": det, "comparison": name, "blur": blur, "metric": m, "subset": subset_name,
                         "n": len(a), "rate_base": float(np.mean(a)), "rate_treat": float(np.mean(b)),
                         "mean": float(np.mean(b - a))}
                    lo, hi = cluster_bootstrap_ci(b - a, cl)
                    s.update(ci_lo=lo, ci_hi=hi)
                    if blur != "pooled":
                        s.update(mcnemar_exact(a, b))
                    rows.append(s)
    res = pd.DataFrame(rows)
    res["holm_p"] = np.nan
    for _, g in res[res.blur != "pooled"].groupby(["detector", "comparison", "metric"]):
        p = g["wilcoxon_p"] if g["wilcoxon_p"].notna().any() else g["mcnemar_p"]
        res.loc[g.index, "holm_p"] = holm(p.values)
    return res


all_res = run("all_pairs", None)
sens = pd.concat([
    run("excl_gt_flagged", set(df.pair_id) - flagged),
    run("excl_clean_fail5_ORB", set(df.pair_id) - clean_fail5["ORB"]),
    run("excl_clean_fail5_SIFT", set(df.pair_id) - clean_fail5["SIFT"]),
])
all_res.to_csv(STATS / "S1_paired_effects_all_pairs.csv", index=False)
sens.to_csv(STATS / "S1b_paired_effects_sensitivity.csv", index=False)
pd.Series({f"{k}_clean_fail5": sorted(v) for k, v in clean_fail5.items()}).to_json(STATS / "clean_fail5_pairs.json", indent=2)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
key = ["psnr", "edge_f1", "keypoint_repeatability", "gt_correct_matches_3px", "match_precision_3px",
       "homography_error_mean", "homography_success_3px", "homography_success_5px"]
pooled = all_res[(all_res.blur == "pooled") & (all_res.metric.isin(key))]


def fmt(r):
    s = f"{r['mean']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
    if not np.isnan(r.get("wilcoxon_p", np.nan)):
        s += f" p={r['wilcoxon_p']:.2g}"
    return s


ci = pooled.assign(txt=pooled.apply(fmt, axis=1))
t10 = ci.pivot_table(index=["detector", "comparison"], columns="metric", values="txt", aggfunc="first")[key]
t10.to_csv(TABLES / "T10_pooled_paired_effects_summary.csv")
print("== pooled (pair-level means over 8 blur conditions): mean diff [sequence-cluster bootstrap 95% CI] Wilcoxon p (n=24)")
for c in key:
    print(f"\n-- {c}")
    print(t10[c].to_string())

print("\n== per-blur success@3px: base -> treat (+gained/-lost) McNemar p")
s3 = all_res[(all_res.metric == "homography_success_3px") & (all_res.blur != "pooled") & (all_res.comparison.isin(["B-A", "C-A", "D-A"]))]
s3 = s3.assign(txt=s3.apply(lambda r: f"{r['rate_base']:.2f}->{r['rate_treat']:.2f} (+{int(r['gained'])}/-{int(r['lost'])}) p={r['mcnemar_p']:.2g}", axis=1))
print(s3.pivot_table(index=["detector", "comparison"], columns="blur", values="txt", aggfunc="first")[BLUR_ORDER].T.to_string())

print("\n== per-blur GT-correct@3: mean diff [CI] pos/neg/zero Holm-p")
g3 = all_res[(all_res.metric == "gt_correct_matches_3px") & (all_res.blur != "pooled") & (all_res.comparison.isin(["B-A", "C-A", "D-A"]))]
g3 = g3.assign(txt=g3.apply(lambda r: f"{r['mean']:+.1f} [{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}] {int(r['n_pos'])}/{int(r['n_neg'])}/{int(r['n_zero'])} hp={r['holm_p']:.2g}", axis=1))
print(g3.pivot_table(index=["detector", "comparison"], columns="blur", values="txt", aggfunc="first")[BLUR_ORDER].T.to_string())

print("\n== C-A (unsharp) per blur, image-level metrics (identical for both detectors) and ORB repeatability")
for m in ("psnr", "ssim", "edge_f1", "edge_recall", "edge_precision", "keypoint_repeatability"):
    x = all_res[(all_res.metric == m) & (all_res.blur != "pooled") & (all_res.comparison == "C-A") & (all_res.detector == "ORB")]
    print(m); print(x.set_index("blur").loc[BLUR_ORDER, ["mean", "ci_lo", "ci_hi", "n_pos", "n_neg", "holm_p"]].round(3).to_string())

print("\n== sensitivity: pooled B-A / C-A / D-A on success@3 and GT-correct@3")
for s in ("all_pairs", "excl_gt_flagged", "excl_clean_fail5_ORB", "excl_clean_fail5_SIFT"):
    src = all_res if s == "all_pairs" else sens[sens.subset == s]
    x = src[(src.blur == "pooled") & (src.metric.isin(["homography_success_3px", "gt_correct_matches_3px"])) & (src.comparison.isin(["B-A", "C-A", "D-A"]))]
    print(f"-- {s}"); print(x.pivot_table(index=["detector", "comparison"], columns="metric", values=["mean", "n"]).round(3).to_string())
