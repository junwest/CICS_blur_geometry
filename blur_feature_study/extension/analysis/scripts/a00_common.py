"""Shared loading / statistics for the extension analysis (run with analysis/.venv).

Runs analysed are listed in extension/analysis/runs.json (fixed run ids; results are never overwritten).
Statistics reuse the base helpers in analysis/scripts/common.py (cluster bootstrap by sequence, paired summary,
exact McNemar, Holm), imported read-only.
Unit of inference: pair (n = 24) after averaging over the 8 blur conditions; CI = sequence cluster bootstrap
(12 clusters, 5000 resamples, seed 42). Cell-level gained/lost counts are descriptive (cells are not independent).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path(__file__).resolve().parents[2]
PROJECT = EXT.parent
RESULTS = EXT / "results"
TABLES = EXT / "analysis" / "tables"
FIGS = EXT / "analysis" / "figures"
STATS = EXT / "analysis" / "stats"
RUNS = json.loads((EXT / "analysis" / "runs.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("base_analysis_common", PROJECT / "analysis" / "scripts" / "common.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
cluster_bootstrap_ci, paired_summary, mcnemar_exact, holm, spearman = (
    _base.cluster_bootstrap_ci, _base.paired_summary, _base.mcnemar_exact, _base.holm, _base.spearman)
BLUR_ORDER, BLUR_LABEL = _base.BLUR_ORDER, _base.BLUR_LABEL

METHODS = ["unsharp", "wiener", "nafnet"]
METHOD_LABEL = {"none": "B (무처리)", "unsharp": "Unsharp", "wiener": "Wiener (Known-PSF)", "nafnet": "NAFNet",
                "clean": "Clean"}
DETECTORS = ["ORB", "SIFT"]
CONTRASTS = [("BR", "BB"), ("RB", "BB"), ("RR", "RB"), ("RR", "BR")]
CONTRASTS_ALL = CONTRASTS + [("RR", "BB")]
# reference categorical palette (dataviz skill, light mode), fixed order; markers are a secondary encoding
C4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST_OK = {"", "projection_error_gt_3px", "projection_error_gt_5px"}


def run_dir(key):
    return RESULTS / RUNS[key]


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True)


def load_main():
    df = read_csv(run_dir("main") / "raw_results.csv")
    df["failure_reason"] = df["failure_reason"].fillna("")
    df["h_valid"] = df["homography_error_mean"].notna()
    df["est_failed"] = ~df["failure_reason"].isin(EST_OK)
    df["log10_herr"] = np.log10(df["homography_error_mean"])
    df["nafnet_output_flag"] = df["nafnet_output_flag"].astype(str).str.lower().eq("true")
    return df


def sensitivity_sets(df):
    """Pairs for the sensitivity analysis: GT-flagged pairs, Clean-fail pairs (@3 px, per detector)."""
    cl = df[df.stage == "CLEAN"]
    gt_flagged = sorted(df[df.gt_quality_flag == 1].pair_id.unique())
    clean_fail = {d: sorted(cl[(cl.detector == d) & (cl.homography_success_3px == 0)].pair_id) for d in DETECTORS}
    return gt_flagged, clean_fail


def cell_table(df, det, meth, x, y):
    """Paired cells (pair, blur) for stage x vs stage y of one method/detector."""
    def pick(stage):
        m = "none" if stage == "BB" else meth
        s = df[(df.detector == det) & (df.stage == stage) & (df.method == m)]
        return s.set_index(["pair_id", "blur_id"])
    a, b = pick(x), pick(y)
    keep = ["sequence_id", "homography_success_3px", "h_valid", "log10_herr", "gt_correct_matches_3px",
            "match_precision_3px", "gtc_hull_coverage", "gtc_cell_occupancy", "target_keypoints",
            "keypoint_repeatability", "ransac_inliers", "ransac_inliers_gt_correct_3px", "nafnet_output_flag",
            "gt_quality_flag"]
    j = a[keep].join(b[keep], lsuffix="_x", rsuffix="_y", how="inner")
    j["sequence_id"] = j["sequence_id_x"]
    return j.reset_index()


def contrast_stats(df, det, meth, x, y, exclude_pairs=(), exclude_flagged_cells=False):
    j = cell_table(df, det, meth, x, y)
    j = j[~j.pair_id.isin(exclude_pairs)]
    if exclude_flagged_cells:
        j = j[~(j.nafnet_output_flag_x | j.nafnet_output_flag_y)]
    out = {"detector": det, "method": meth, "contrast": f"{x}-{y}", "n_cells": len(j)}
    j["d_succ"] = j.homography_success_3px_x - j.homography_success_3px_y
    both = j.h_valid_x & j.h_valid_y
    j["d_log"] = np.where(both, j.log10_herr_x - j.log10_herr_y, np.nan)
    j["d_gtc"] = j.gt_correct_matches_3px_x - j.gt_correct_matches_3px_y
    j["d_loggtc"] = np.log((j.gt_correct_matches_3px_x + 1) / (j.gt_correct_matches_3px_y + 1))
    j["d_cov"] = j.gtc_hull_coverage_x - j.gtc_hull_coverage_y
    j["d_occ"] = j.gtc_cell_occupancy_x - j.gtc_cell_occupancy_y
    j["d_kp"] = j.target_keypoints_x - j.target_keypoints_y
    j["d_prec"] = j.match_precision_3px_x - j.match_precision_3px_y
    mc = mcnemar_exact(j.homography_success_3px_y, j.homography_success_3px_x)
    out.update(succ_x=float(j.homography_success_3px_x.mean()), succ_y=float(j.homography_success_3px_y.mean()),
               gained=mc["gained"], lost=mc["lost"], mcnemar_p_descriptive=mc["mcnemar_p"],
               n_common_valid_cells=int(both.sum()))
    pl = j.groupby("pair_id").agg(seq=("sequence_id", "first"), d_succ=("d_succ", "mean"), d_log=("d_log", "mean"),
                                  d_gtc=("d_gtc", "mean"), d_loggtc=("d_loggtc", "mean"), d_cov=("d_cov", "mean"),
                                  d_occ=("d_occ", "mean"), d_kp=("d_kp", "mean"), d_prec=("d_prec", "mean"))
    for c in ["d_succ", "d_log", "d_gtc", "d_loggtc", "d_cov", "d_occ", "d_kp", "d_prec"]:
        s = paired_summary(pl[c].values, pl.seq.values)
        for k in ("n", "mean", "median", "ci_lo", "ci_hi", "wilcoxon_p", "n_pos", "n_neg", "n_zero"):
            out[f"{c}_{k}"] = s.get(k, np.nan)
    return out, j


def fmt_ci(m, lo, hi, scale=1.0, nd=2):
    if not np.isfinite(m):
        return "–"
    return f"{m * scale:+.{nd}f} [{lo * scale:+.{nd}f}, {hi * scale:+.{nd}f}]"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
    return plt


def save_new(fig, name):
    p = FIGS / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    return p
