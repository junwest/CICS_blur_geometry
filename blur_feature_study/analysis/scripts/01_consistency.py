"""Independent consistency check: README / config / selected params / code snapshot / raw CSV / NPZ.
Does not re-run the experiment; re-derives a few numbers from raw artefacts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MAIN_RUN, PROJECT, STATS, VAL_RUN, load_results, write_json  # noqa: E402

sys.path.insert(0, str(PROJECT))
from src.metrics import gt_errors, project, valid_grid  # noqa: E402

checks = {}
df = load_results()
cfg_live = yaml.safe_load(open(PROJECT / "config" / "experiment_config.yaml", encoding="utf-8"))
cfg_snap = yaml.safe_load(open(MAIN_RUN / "snapshot" / "experiment_config.yaml", encoding="utf-8"))
checks["config_snapshot_equals_live_config"] = cfg_live == cfg_snap

# code snapshot vs live src (per file)
diff_files = []
for p in sorted((MAIN_RUN / "snapshot" / "src").glob("*.py")):
    live = PROJECT / "src" / p.name
    if not live.exists() or hashlib.sha256(live.read_bytes()).hexdigest() != hashlib.sha256(p.read_bytes()).hexdigest():
        diff_files.append(p.name)
new_files = sorted(set(q.name for q in (PROJECT / "src").glob("*.py")) - set(q.name for q in (MAIN_RUN / "snapshot" / "src").glob("*.py")))
checks["src_files_changed_since_main"] = diff_files
checks["src_files_added_since_main"] = new_files

# selected params actually used
sel = yaml.safe_load(open(PROJECT / "config" / "selected_params_validation_20261010-003108.yaml", encoding="utf-8"))
b = df[df.group == "B"]
checks["B_rows_use_selected_params"] = bool(all(json.loads(r.detector_params) == sel[r.detector]["params"] for r in b.itertuples()))
checks["D_rows_balance_0.005"] = bool((df[df.group == "D"].wiener_balance == 0.005).all())
checks["SIFT_tuned_equals_default"] = sel["SIFT"]["params"] == cfg_live["detectors"]["SIFT"]["default"]
a = df[(df.group == "A") & (df.detector == "SIFT")].set_index(["pair_id", "blur_id"])
bs = df[(df.group == "B") & (df.detector == "SIFT")].set_index(["pair_id", "blur_id"]).loc[a.index]
cmp_cols = ["target_keypoints", "good_matches", "gt_correct_matches_3px", "homography_error_mean", "failure_reason"]
checks["SIFT_B_rows_identical_to_A"] = bool((a[cmp_cols].fillna(-1).astype(str).values == bs[cmp_cols].fillna(-1).astype(str).values).all())

# design counts
checks["n_rows"] = int(len(df))
checks["n_pairs"] = int(df.pair_id.nunique())
checks["n_sequences"] = int(df.sequence_id.nunique())
checks["pairs_per_sequence"] = df.groupby("sequence_id").pair_id.nunique().to_dict()
checks["rows_per_group_detector"] = {f"{g}/{d}": int(n) for (g, d), n in df.groupby(["group", "detector"], observed=True).size().items()}
checks["gt_flagged_test_pairs"] = sorted(df[df.gt_quality_flag == 1].pair_id.unique().tolist())

# image metrics identical across detectors / shared images
key = ["pair_id", "blur_id", "processing_method", "wiener_balance"]
g = df.assign(wiener_balance=df.wiener_balance.fillna(-1)).groupby(key, observed=True)[["psnr", "ssim", "edge_f1"]].nunique()
checks["image_metrics_identical_across_detectors"] = bool((g <= 1).all().all())

# NPZ re-derivation on a random sample of 120 rows: gt-correct counts, keypoints inside valid mask,
# H_est grid error, RANSAC inlier count
rng = np.random.default_rng(42)
sample = df.sample(120, random_state=42)
margin = cfg_live["preprocessing"]["valid_margin_px"]
bad = {"gt_count": 0, "kp_outside_mask": 0, "h_error": 0, "inliers": 0, "ref_kp_count": 0}
worst_kp = 0.0
for r in sample.itertuples():
    d = np.load(PROJECT / r.match_detail_file)
    e = gt_errors(d["H_gt"], d["ref_xy"], d["tgt_xy"])
    bad["gt_count"] += int((e <= 3).sum()) != r.gt_correct_matches_3px
    bad["inliers"] += int(d["ransac_inlier"].sum()) != r.ransac_inliers
    bad["ref_kp_count"] += len(d["ref_kp"]) != r.reference_keypoints
    kp = d["tgt_kp"][:, :2]
    hh, ww = d["tgt_shape"]
    if len(kp):
        out = ((kp[:, 0] < margin - 1) | (kp[:, 0] > ww - margin) | (kp[:, 1] < margin - 1) | (kp[:, 1] > hh - margin)).sum()
        bad["kp_outside_mask"] += int(out > 0)
    if np.all(np.isfinite(d["H_est"])) and not np.isnan(r.homography_error_mean):
        grid, proj_gt = valid_grid(d["H_gt"], tuple(d["ref_shape"]), tuple(d["tgt_shape"]), margin, 10)
        proj_est, _ = project(d["H_est"], grid)
        err = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
        bad["h_error"] += abs(err - r.homography_error_mean) > 1e-6
checks["npz_sample_mismatches"] = bad

# CLEAN rows trivial image metrics
cl = df[df.group == "CLEAN"]
checks["clean_rows_trivial"] = bool(np.isinf(cl.psnr).all() and (cl.edge_f1 == 1).all() and (cl.keypoint_repeatability == 1).all())

# validation selection reproduction (from validation raw csv)
val = load_results(VAL_RUN)
vb = val[(val.group == "B_candidate")]
tab = vb.groupby(["detector", "detector_config"], observed=True).agg(s3=("homography_success_3px", "mean"), n=("pair_id", "size")).reset_index()
best = tab.sort_values(["detector", "s3"], ascending=[True, False]).groupby("detector").head(1)
checks["validation_best_by_success3"] = {r.detector: (r.detector_config, round(float(r.s3), 4), int(r.n)) for r in best.itertuples()}
vd = val[val.group == "D_candidate"].drop_duplicates(["pair_id", "blur_id", "wiener_balance"])
wt = vd.groupby(["blur_id", "wiener_balance"], observed=True).psnr.mean().reset_index()
checks["validation_wiener_psnr_best"] = {str(b): float(g.sort_values("psnr", ascending=False).wiener_balance.iloc[0]) for b, g in wt.groupby("blur_id", observed=True)}

# keypoint budget saturation
checks["share_rows_target_kp_at_budget"] = {d: float((df[(df.detector == d) & (df.group != "CLEAN")].target_keypoints >= 1500).mean()) for d in ("ORB", "SIFT")}
checks["clean_target_keypoints_min_max"] = {d: [int(cl[cl.detector == d].target_keypoints.min()), int(cl[cl.detector == d].target_keypoints.max())] for d in ("ORB", "SIFT")}

write_json(STATS / "consistency_checks.json", checks)
print(json.dumps(checks, indent=2, default=str))
