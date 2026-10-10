"""Descriptive tables (Test split, n = 24 pairs per cell)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUR_ORDER, IMAGE_METRICS, TABLES, add_clean_reference, image_level, load_results,  # noqa: E402
                    main_core)

df = add_clean_reference(main_core(load_results()))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

det_metrics = ["keypoint_repeatability", "reference_keypoints", "target_keypoints", "good_matches",
               "gt_correct_matches_3px", "gt_correct_matches_5px", "match_precision_3px", "ransac_inliers",
               "ransac_inlier_ratio", "homography_error_mean", "homography_error_median", "homography_success_3px",
               "homography_success_5px", "runtime_ms", "runtime_processing_ms", "runtime_detect_ms"]

# T1 full cell table: detector x group x blur (mean, plus median for error, plus failure count)
def cell(g):
    out = {m: g[m].astype(float).mean() for m in IMAGE_METRICS + det_metrics}
    out["homography_error_median_of_means"] = g.homography_error_mean.median()
    out["n_h_failed"] = int(g.h_failed.sum())
    out["n"] = len(g)
    out["gt_correct_ratio_median"] = g.gt_correct_ratio.median()
    return pd.Series(out)

t1 = df.groupby(["detector", "group", "blur_id"], observed=True).apply(cell, include_groups=False).reset_index()
t1.to_csv(TABLES / "T1_cell_means_detector_group_blur.csv", index=False)

# T2 pooled over blur: detector x group
t2 = df[df.group != "CLEAN"].groupby(["detector", "group"], observed=True).apply(cell, include_groups=False).reset_index()
t2c = df[df.group == "CLEAN"].groupby(["detector", "group"], observed=True).apply(cell, include_groups=False).reset_index()
t2 = pd.concat([t2c, t2]).reset_index(drop=True)
t2.to_csv(TABLES / "T2_pooled_detector_group.csv", index=False)
print("== T2 pooled over 8 blur conditions (CLEAN: 24 rows; others: 192 rows)")
print(t2[["detector", "group", "psnr", "ssim", "edge_f1", "keypoint_repeatability", "target_keypoints", "good_matches",
          "gt_correct_matches_3px", "match_precision_3px", "homography_error_median_of_means", "homography_success_3px",
          "homography_success_5px", "n_h_failed", "runtime_ms"]].round(3).to_string())

# T3 by sequence type
t3 = df.groupby(["detector", "group", "sequence_type"], observed=True).apply(cell, include_groups=False).reset_index()
t3.to_csv(TABLES / "T3_detector_group_sequence_type.csv", index=False)
print("\n== T3 success@3 by sequence type")
print(t3.pivot_table(index=["detector", "group"], columns="sequence_type", values="homography_success_3px", observed=True).round(3).to_string())

# T4 image-level degradation by blur family/condition (A only = raw blur)
img = image_level(load_results())
t4 = img.groupby(["processing_method", "blur_id"], observed=True)[IMAGE_METRICS].agg(["mean", "std"]).round(3)
t4.to_csv(TABLES / "T4_image_metrics_by_processing_blur.csv")
print("\n== T4 image metrics (mean) by processing and blur")
print(img.groupby(["processing_method", "blur_id"], observed=True)[["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1"]].mean().round(3).to_string())

# T5 failure reasons: detector x group x blur
fr = df.copy()
fr["failure_reason"] = fr.failure_reason.replace("", "success@3px")
t5 = pd.crosstab([fr.detector, fr.group, fr.blur_id], fr.failure_reason)
t5.to_csv(TABLES / "T5_failure_reasons_detector_group_blur.csv")
t5b = pd.crosstab([fr.detector, fr.group], fr.failure_reason)
t5b.to_csv(TABLES / "T5b_failure_reasons_detector_group.csv")
print("\n== T5b failure reasons")
print(t5b.to_string())

# T6 per-pair success profile (for heatmaps): pair x blur, per detector/group
t6 = df.pivot_table(index=["detector", "group", "pair_id"], columns="blur_id", values="homography_success_3px", observed=True)
t6.to_csv(TABLES / "T6_success3_pair_by_blur.csv")
t6e = df.pivot_table(index=["detector", "group", "pair_id"], columns="blur_id", values="gt_correct_matches_3px", observed=True)
t6e.to_csv(TABLES / "T6b_gtcorrect3_pair_by_blur.csv")

# T7 clean baseline per pair
cl = df[df.group == "CLEAN"][["pair_id", "sequence_type", "detector", "target_keypoints", "good_matches",
                              "gt_correct_matches_3px", "match_precision_3px", "homography_error_mean",
                              "homography_success_3px", "homography_success_5px", "failure_reason", "gt_quality_flag"]]
cl.to_csv(TABLES / "T7_clean_baseline_per_pair.csv", index=False)
print("\n== T7 clean baseline failures (success@3 == 0)")
print(cl[cl.homography_success_3px == 0].to_string())

# T8 runtime
t8 = df.groupby(["detector", "group"], observed=True)[["runtime_ms", "runtime_processing_ms", "runtime_detect_ms", "runtime_match_ms", "runtime_homography_ms"]].median().round(1)
t8.to_csv(TABLES / "T8_runtime_median_ms.csv")
print("\n== T8 runtime (median ms)")
print(t8.to_string())

# T9 motion orientation asymmetry (image level, A)
a = img[img.processing_method == "blurred"]
piv = a.pivot_table(index="pair_id", columns="blur_id", values=["psnr", "edge_f1"], observed=True)
t9 = pd.DataFrame({
    "dpsnr_l9_a0_minus_a90": piv[("psnr", "motion_l9_a0")] - piv[("psnr", "motion_l9_a90")],
    "dpsnr_l17_a0_minus_a90": piv[("psnr", "motion_l17_a0")] - piv[("psnr", "motion_l17_a90")],
    "dedge_l17_a0_minus_a90": piv[("edge_f1", "motion_l17_a0")] - piv[("edge_f1", "motion_l17_a90")],
})
t9.to_csv(TABLES / "T9_motion_orientation_asymmetry_per_pair.csv")
print("\n== T9 motion orientation asymmetry (a0 - a90), per pair summary")
print(t9.describe().round(3).to_string())
