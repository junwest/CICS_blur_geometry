"""Phase 3: apply the PRE-REGISTERED selection rules (config: selection_rules) to a validation run.

  python -m src.select_params --run results/validation/<run_id>
Outputs (new files only): <run>/selection_detectors.csv, <run>/selection_wiener.csv,
<run>/selected_params.yaml and config/selected_params_<run_id>.yaml
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .io_utils import PROJECT_ROOT, load_config, write_yaml_new


def detector_table(val, cfg, det, exclude=()):
    b = val[(val.group == "B_candidate") & (val.detector == det) & ~val.pair_id.isin(exclude)]
    default = cfg["detectors"][det]["default"]
    grid = F.tuning_grid(det, cfg)
    rows = []
    for gi, p in enumerate(grid):
        name = F.config_name(det, p, default)
        s = b[b.detector_config == name]
        err = s.homography_error_mean.fillna(np.inf)
        rows.append({
            "detector": det, "detector_config": name, "grid_index": gi, "is_default": int(name == "default"),
            "n_cases": len(s), "success_3px_mean": s.homography_success_3px.mean(),
            "median_error_failures_inf": float(np.median(err)), "gt_correct_3px_mean": s.gt_correct_matches_3px.mean(),
            "success_5px_mean": s.homography_success_5px.mean(), "match_precision_3px_mean": s.match_precision_3px.mean(),
            "repeatability_mean": s.keypoint_repeatability.mean(), "target_keypoints_mean": s.target_keypoints.mean(),
            "params": F.params_json(p),
        })
    t = pd.DataFrame(rows)
    assert t.n_cases.nunique() == 1 and t.n_cases.iloc[0] > 0, "unbalanced validation candidates"
    # rule order: success@3 desc, median error asc, correct@3 desc, default first, grid order
    t = t.sort_values(["success_3px_mean", "median_error_failures_inf", "gt_correct_3px_mean", "is_default", "grid_index"],
                      ascending=[False, True, False, False, True]).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    top = t.iloc[0]
    t["decided_by"] = ""
    if len(t) > 1:
        second = t.iloc[1]
        for crit, col in (("1:success_3px", "success_3px_mean"), ("2:median_error", "median_error_failures_inf"),
                          ("3:gt_correct_3px", "gt_correct_3px_mean")):
            if top[col] != second[col]:
                t.loc[0, "decided_by"] = crit
                break
        else:
            t.loc[0, "decided_by"] = "4:default/grid order"
    return t


def wiener_table(val):
    d = val[val.group == "D_candidate"].drop_duplicates(["pair_id", "blur_id", "wiener_balance"])
    t = d.groupby(["blur_id", "wiener_balance"]).agg(n_pairs=("pair_id", "nunique"), psnr_mean=("psnr", "mean"),
                                                     ssim_mean=("ssim", "mean"), edge_f1_mean=("edge_f1", "mean")).reset_index()
    s = val[val.group == "D_candidate"].groupby(["blur_id", "wiener_balance"]).agg(
        success_3px_mean_both_detectors=("homography_success_3px", "mean"),
        repeatability_mean_both_detectors=("keypoint_repeatability", "mean")).reset_index()
    t = t.merge(s, on=["blur_id", "wiener_balance"])
    t = t.sort_values(["blur_id", "psnr_mean", "ssim_mean", "wiener_balance"], ascending=[True, False, False, False])
    t["selected"] = 0
    t.loc[t.groupby("blur_id").head(1).index, "selected"] = 1
    return t.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    cfg = load_config(run / "snapshot" / "experiment_config.yaml")
    val = pd.read_csv(run / "raw_results.csv")
    assert set(val.split) == {"validation"}, "selection must use validation rows only"
    assert (val.pipeline_error.fillna("") == "").all(), "validation run has pipeline errors"
    flagged = sorted(val.loc[val.gt_quality_flag == 1, "pair_id"].unique())

    det_tabs, sel = [], {"source_validation_run": run.name,
                         "selection_rules": cfg["selection_rules"], "gt_flagged_validation_pairs": flagged}
    for det in ("ORB", "SIFT"):
        t = detector_table(val, cfg, det)
        t_rob = detector_table(val, cfg, det, exclude=flagged)
        det_tabs += [t.assign(subset="all_validation_pairs"), t_rob.assign(subset="excluding_gt_flagged")]
        top = t.iloc[0]
        p = dict(cfg["detectors"][det]["default"])
        p.update(json.loads(top.params))
        sel[det] = {"selected_config": top.detector_config, "decided_by": top.decided_by,
                    "params": p, "validation_success_3px": float(top.success_3px_mean),
                    "robustness_excluding_gt_flagged": {"selected_config": t_rob.iloc[0].detector_config,
                                                        "same_as_main_choice": bool(t_rob.iloc[0].detector_config == top.detector_config)}}
        print(det, "->", top.detector_config, "decided by", top.decided_by,
              "| excluding flagged ->", t_rob.iloc[0].detector_config)
    wt = wiener_table(val)
    sel["wiener_balance"] = {r.blur_id: float(r.wiener_balance) for r in wt[wt.selected == 1].itertuples()}
    print("wiener:", sel["wiener_balance"])

    pd.concat(det_tabs).to_csv(run / "selection_detectors.csv", index=False)
    wt.to_csv(run / "selection_wiener.csv", index=False)
    write_yaml_new(run / "selected_params.yaml", sel)
    dst = PROJECT_ROOT / "config" / f"selected_params_{run.name}.yaml"
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copy2(run / "selected_params.yaml", dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
