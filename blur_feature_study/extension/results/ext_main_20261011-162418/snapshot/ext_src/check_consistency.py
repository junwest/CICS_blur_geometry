"""Consistency of the extension run with the base main run (implementation check, not a result).

  python -m extension.src.check_consistency --run ext_main_<ts>

new CLEAN            <-> base CLEAN (clean, default)
new BB               <-> base A (blurred, default)
new RR / unsharp     <-> base C (unsharp_mask, default)
new RR / wiener      <-> base D (wiener_known_psf, default, selected balance)
All non-time metric columns must be identical (floats compared exactly after round-trip CSV parsing;
NaN == NaN). Also summarises the bitwise detectAndCompute checks and the compute() bookkeeping.
Writes <run>/consistency.json and <run>/consistency.md (never overwrites).
"""
import argparse

import numpy as np
import pandas as pd

from .common import BASE_MAIN_RUN, RESULTS, write_json

KEYS = ["pair_id", "blur_id", "detector"]
COLS = ["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1", "edge_n_clean", "edge_n_proc",
        "keypoint_repeatability", "rep_n_clean_valid", "rep_n_repeated", "keypoint_repeatability_vs_default_clean",
        "reference_keypoints", "target_keypoints", "good_matches", "gt_correct_matches_3px", "gt_correct_matches_5px",
        "match_precision_3px", "match_precision_5px", "ransac_inliers", "ransac_inlier_ratio",
        "ransac_inliers_gt_correct_3px", "homography_error_mean", "homography_error_median", "homography_error_max",
        "homography_success_3px", "homography_success_5px", "n_grid_valid", "failure_reason", "gt_quality_flag",
        "image_h", "image_w"]
MAP = [("CLEAN", "clean", "CLEAN"), ("BB", "none", "A"), ("RR", "unsharp", "C"), ("RR", "wiener", "D")]


def equal(a, b):
    if a.dtype.kind in "iufb" and b.dtype.kind in "iufb":
        a = a.to_numpy(float)
        b = b.to_numpy(float)
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a.fillna("").astype(str).to_numpy() == b.fillna("").astype(str).to_numpy()


def truthy(s):
    return s.astype(str).str.lower().eq("true")


def compare(new, base):
    res = {}
    for stage, meth, grp in MAP:
        n = new[(new.stage == stage) & (new.method == meth)].copy()
        b = base[(base.group == grp) & (base.detector_config == "default")].copy()
        b["blur_id"] = b["blur_id"].fillna("none")
        mg = n.merge(b, on=KEYS, how="outer", suffixes=("_new", "_base"), indicator=True)
        entry = {"n_new": len(n), "n_base": len(b), "unmatched": int((mg._merge != "both").sum()), "mismatch": {}}
        mg = mg[mg._merge == "both"]
        for c in COLS:
            eq = equal(mg[f"{c}_new"], mg[f"{c}_base"])
            if not eq.all():
                bad = mg.loc[~eq, KEYS + [f"{c}_new", f"{c}_base"]].head(5)
                entry["mismatch"][c] = {"n": int((~eq).sum()), "examples": bad.astype(str).to_dict("records")}
        entry["all_equal"] = entry["unmatched"] == 0 and not entry["mismatch"] and len(n) == len(b)
        res[f"{stage}_{meth}_vs_{grp}"] = entry
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    d = RESULTS / args.run
    new = pd.read_csv(d / "raw_results.csv", float_precision="round_trip", keep_default_na=True)
    base = pd.read_csv(BASE_MAIN_RUN / "raw_results.csv", float_precision="round_trip")
    if (new.pair_id.nunique() < 24):
        base = base[base.pair_id.isin(new.pair_id.unique())]
    res = {"base_run": BASE_MAIN_RUN.name, "run": args.run, "comparisons": compare(new, base)}
    same = new[new.detection_image == new.descriptor_image]
    res["detectAndCompute_bitwise"] = {
        "n_checked": int(len(same)),
        "kp_equal": int(truthy(same.dac_kp_equal).sum()), "desc_equal": int(truthy(same.dac_desc_equal).sum()),
        "by_stage": {f"{k[0]}/{k[1]}": f"{int(truthy(g.dac_kp_equal).sum())}/{len(g)} kp, "
                                       f"{int(truthy(g.dac_desc_equal).sum())}/{len(g)} desc"
                     for k, g in same.groupby(["stage", "detector"])}}
    st = new[new.stage != "CLEAN"]
    res["compute_bookkeeping"] = {
        "rows": int(len(st)), "rows_with_removed_keypoints": int((st.n_removed_by_compute > 0).sum()),
        "total_removed": int(st.n_removed_by_compute.sum()), "rows_reordered": int(truthy(st.reordered_by_compute).sum()),
        "rows_attrs_changed": int((st.attrs_changed_by_compute > 0).sum()),
        "sift_rows_sentinel_effective": int(truthy(st[st.detector == "SIFT"].sift_sentinel_effective).sum()),
        "sift_rows": int((st.detector == "SIFT").sum())}
    res["all_consistent"] = bool(all(v["all_equal"] for v in res["comparisons"].values())
                                 and res["detectAndCompute_bitwise"]["kp_equal"] == len(same)
                                 and res["detectAndCompute_bitwise"]["desc_equal"] == len(same))
    write_json(d / "consistency.json", res)
    lines = [f"# Consistency check: {args.run} vs base {BASE_MAIN_RUN.name}", "",
             f"**all_consistent = {res['all_consistent']}**", "", "| comparison | n new | n base | unmatched | mismatching columns |",
             "|---|---|---|---|---|"]
    for k, v in res["comparisons"].items():
        mm = ", ".join("%s (%d)" % (c, x["n"]) for c, x in v["mismatch"].items()) or "none"
        lines.append(f"| {k} | {v['n_new']} | {v['n_base']} | {v['unmatched']} | {mm} |")
    b = res["detectAndCompute_bitwise"]
    lines += ["", f"Separate detect()+compute() vs detectAndCompute (same image): keypoints equal {b['kp_equal']}/{b['n_checked']}, "
                  f"descriptors equal {b['desc_equal']}/{b['n_checked']}.", ""]
    lines += [f"- {k}: {v}" for k, v in b["by_stage"].items()]
    cb = res["compute_bookkeeping"]
    lines += ["", "compute() bookkeeping over all stage rows: " + ", ".join(f"{k} = {v}" for k, v in cb.items())]
    p = d / "consistency.md"
    if p.exists():
        raise FileExistsError(p)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
