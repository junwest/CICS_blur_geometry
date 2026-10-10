"""Shared loading / statistics helpers for the analysis scripts.

Dependence structure handled here:
  * 24 test pairs nested in 12 sequences (2 targets per sequence) -> cluster = sequence_id
  * 8 blur conditions x processing groups are repeated measures of the same pair
  * image-quality metrics (PSNR/SSIM/Edge F1) are identical across detectors -> dedupe to image level
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT / "analysis"
TABLES = ANALYSIS / "tables"
STATS = ANALYSIS / "stats"
FIGS = ANALYSIS / "figures"
for d in (TABLES, STATS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

MAIN_RUN = PROJECT / "results" / "main" / "main_20261010-003501"
VAL_RUN = PROJECT / "results" / "validation" / "validation_20261010-003108"
SEED = 42

BLUR_ORDER = ["gaussian_s1.5", "gaussian_s3.0", "disk_r3", "disk_r5",
              "motion_l9_a0", "motion_l9_a90", "motion_l17_a0", "motion_l17_a90"]
BLUR_LABEL = {"gaussian_s1.5": "Gauss σ1.5", "gaussian_s3.0": "Gauss σ3.0", "disk_r3": "Disk r3",
              "disk_r5": "Disk r5", "motion_l9_a0": "Motion 9px 0°", "motion_l9_a90": "Motion 9px 90°",
              "motion_l17_a0": "Motion 17px 0°", "motion_l17_a90": "Motion 17px 90°"}
GROUP_ORDER = ["CLEAN", "A", "B", "C", "D"]
GROUP_LABEL = {"CLEAN": "Clean", "A": "A: Blurred+Default", "B": "B: Blurred+Tuned",
               "C": "C: Unsharp+Default", "D": "D: Wiener(oracle PSF)+Default"}
METRICS8 = ["psnr", "ssim", "edge_f1", "keypoint_repeatability", "gt_correct_matches_3px",
            "match_precision_3px", "homography_error_mean", "homography_success_3px"]
IMAGE_METRICS = ["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1"]
NUM_COLS = ["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1", "keypoint_repeatability",
            "keypoint_repeatability_vs_default_clean", "match_precision_3px", "match_precision_5px",
            "ransac_inlier_ratio", "homography_error_mean", "homography_error_median", "homography_error_max",
            "wiener_balance", "blur_strength", "blur_angle", "runtime_ms", "runtime_processing_ms",
            "runtime_detect_ms", "runtime_match_ms", "runtime_homography_ms"]
INT_COLS = ["reference_keypoints", "target_keypoints", "good_matches", "gt_correct_matches_3px",
            "gt_correct_matches_5px", "ransac_inliers", "ransac_inliers_gt_correct_3px", "homography_success_3px",
            "homography_success_5px", "n_grid_valid", "edge_n_clean", "edge_n_proc", "rep_n_clean_valid",
            "rep_n_repeated", "gt_quality_flag", "is_supplementary", "is_placeholder"]


def load_results(run=MAIN_RUN):
    df = pd.read_csv(run / "raw_results.csv", keep_default_na=False, dtype=str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c].replace("", "nan"), errors="coerce")
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c].replace("", "nan"), errors="coerce").astype("Int64")
    df["blur_id"] = pd.Categorical(df["blur_id"], ["none"] + BLUR_ORDER, ordered=True)
    df["group"] = pd.Categorical(df["group"], GROUP_ORDER + ["B_candidate", "D_candidate"], ordered=True)
    df["h_failed"] = df["homography_error_mean"].isna()
    df["blur_family"] = df["blur_type"].map({"none": "none", "gaussian": "gaussian", "defocus_disk": "defocus",
                                             "motion_linear": "motion"})
    return df


def main_core(df):
    """Rows used for group comparisons: CLEAN default + A/B/C/D (no supplementary CLEAN+tuned)."""
    return df[(df.is_supplementary == 0)].copy()


def image_level(df):
    """One row per processed image (image metrics do not depend on the detector)."""
    d = df[df.detector == "ORB"]
    d = d[d.group.isin(["A", "C", "D"])]
    return d[["pair_id", "sequence_id", "sequence_type", "blur_id", "blur_family", "group", "processing_method",
              "wiener_balance", "gt_quality_flag"] + IMAGE_METRICS + ["edge_n_clean", "edge_n_proc"]].copy()


def add_clean_reference(df):
    """Attach the CLEAN-default value of detector-level metrics for the same pair & detector."""
    cl = df[(df.group == "CLEAN") & (df.detector_config == "default")]
    cl = cl.set_index(["pair_id", "detector"])[["gt_correct_matches_3px", "good_matches", "target_keypoints",
                                                "homography_error_mean", "homography_success_3px",
                                                "homography_success_5px", "match_precision_3px"]]
    cl.columns = ["clean_" + c for c in cl.columns]
    out = df.merge(cl, left_on=["pair_id", "detector"], right_index=True, how="left")
    out["gt_correct_ratio"] = out["gt_correct_matches_3px"].astype(float) / out["clean_gt_correct_matches_3px"].astype(float)
    out["log_gt_correct_ratio"] = np.log((out["gt_correct_matches_3px"].astype(float) + 1)
                                         / (out["clean_gt_correct_matches_3px"].astype(float) + 1))
    return out


# ----------------------------------------------------------------------------- statistics
def cluster_bootstrap_ci(values, clusters, n_boot=5000, seed=SEED, alpha=0.05):
    """Percentile CI of the MEAN of `values`, resampling whole clusters (sequences) with replacement.
    Vectorised: the mean of a resample = sum(cluster sums) / sum(cluster sizes)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    csum = np.bincount(inv, weights=values, minlength=k)
    cnt = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(n_boot, k))
    boots = csum[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return float(np.nanpercentile(boots, 100 * alpha / 2)), float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))


def paired_summary(diff, clusters, n_boot=5000):
    """Summary of paired differences (one value per pair)."""
    d = np.asarray(diff, dtype=float)
    ok = np.isfinite(d)
    d_ok, c_ok = d[ok], np.asarray(clusters)[ok]
    out = {"n": int(ok.sum()), "mean": float(np.mean(d_ok)) if ok.any() else np.nan,
           "median": float(np.median(d_ok)) if ok.any() else np.nan,
           "n_pos": int((d_ok > 0).sum()), "n_neg": int((d_ok < 0).sum()), "n_zero": int((d_ok == 0).sum())}
    if ok.sum() >= 2:
        lo, hi = cluster_bootstrap_ci(d_ok, c_ok, n_boot=n_boot)
        out["ci_lo"], out["ci_hi"] = lo, hi
        nz = d_ok[d_ok != 0]
        if len(nz) >= 5:
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
            except ValueError:
                out["wilcoxon_p"] = np.nan
        else:
            out["wilcoxon_p"] = np.nan
        # standardised effect size (paired): mean / sd of differences
        sd = np.std(d_ok, ddof=1)
        out["dz"] = float(out["mean"] / sd) if sd > 0 else np.nan
    else:
        out.update(ci_lo=np.nan, ci_hi=np.nan, wilcoxon_p=np.nan, dz=np.nan)
    return out


def mcnemar_exact(a, b):
    """Exact McNemar test for paired binary outcomes a (baseline) and b (treatment)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    n01 = int(((a == 0) & (b == 1)).sum())   # gained
    n10 = int(((a == 1) & (b == 0)).sum())   # lost
    n = n01 + n10
    p = float(stats.binomtest(n01, n, 0.5).pvalue) if n > 0 else np.nan
    return {"gained": n01, "lost": n10, "mcnemar_p": p}


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN preserved)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pv[i])
        running = max(running, val)
        adj[i] = running
    out[ok] = adj
    return out


def write_json(path, obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
