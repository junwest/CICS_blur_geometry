"""Result verification (Phase 2 pilot checks and Phase 5 integrity checks).

  python -m src.verify_results --run results/pilot/<run_id> [--sample-images 40] [--qc]
Writes verification_<timestamp>.json / .md into the run directory (never overwrites).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.metrics import structural_similarity

from . import metrics as M
from .dataset import load_manifest, load_pair
from .imaging import apply_blur, unsharp_mask, wiener_float, wiener_restore
from .io_utils import PROJECT_ROOT, load_config, write_json_new, write_text_new
from .pipeline import CLEAN_COND, KEY_COLUMNS, REQUIRED_COLUMNS, build_plan, row_key
from .psf import make_psf, psf_ok, psf_report

NUM = ["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1", "keypoint_repeatability",
       "match_precision_3px", "match_precision_5px", "ransac_inlier_ratio", "homography_error_mean",
       "homography_error_median", "homography_error_max", "wiener_balance", "blur_strength", "blur_angle",
       "keypoint_repeatability_vs_default_clean", "runtime_ms"]
INT = ["reference_keypoints", "target_keypoints", "good_matches", "gt_correct_matches_3px",
       "gt_correct_matches_5px", "ransac_inliers", "homography_success_3px", "homography_success_5px",
       "n_grid_valid", "image_h", "image_w", "attempt", "is_supplementary", "is_placeholder"]


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail=""):
        self.items.append({"check": name, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" :: {detail}" if detail else ""))

    @property
    def all_pass(self):
        return all(i["pass"] for i in self.items)


def load_results(run_dir):
    df = pd.read_csv(run_dir / "raw_results.csv", keep_default_na=False, dtype=str)
    for c in NUM:
        df[c] = pd.to_numeric(df[c].replace("", "nan"), errors="coerce")
    for c in INT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def expected_keys(phase, cfg, pairs, selected):
    keys = set()
    for p in pairs:
        for cond in [CLEAN_COND] + list(cfg["blur"]["conditions"]):
            for e in build_plan(phase, cfg, cond["blur_id"], selected):
                keys.add(row_key({"pair_id": p, "blur_id": cond["blur_id"], "group": e["group"],
                                  "processing_method": e["processing_method"], "detector": e["detector"],
                                  "detector_config": e["detector_config"], "wiener_balance": e["wiener_balance"]}))
    return keys


def regenerate(pair, row, cfg, cache):
    key = (row["pair_id"], row["blur_id"], row["processing_method"], row["wiener_balance"])
    if key in cache:
        return cache[key]
    if row["blur_id"] == "none":
        img = pair.tgt
    else:
        cond = next(c for c in cfg["blur"]["conditions"] if c["blur_id"] == row["blur_id"])
        k = make_psf(cond, cfg)
        b = apply_blur(pair.tgt, k)
        u = cfg["processing"]["unsharp_mask"]
        img = {"blurred": lambda: b,
               "unsharp_mask": lambda: unsharp_mask(b, u["alpha"], u["sigma"], u["truncate"]),
               "wiener_known_psf": lambda: wiener_restore(b, k, float(row["wiener_balance"]))}[row["processing_method"]]()
    cache[key] = img
    return img


def close(a, b, tol=1e-6):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a))


def verify(run_dir, sample_images=40, make_qc=False):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "snapshot" / "experiment_config.yaml")
    metas = sorted(run_dir.glob("run_meta_attempt*.json"))
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    phase = meta["phase"]
    df = load_results(run_dir)
    manifest = load_manifest(cfg).set_index("pair_id")
    m = int(cfg["preprocessing"]["valid_margin_px"])
    ev = cfg["evaluation"]
    C = Checks()
    print(f"== verifying {run_dir.name} ({phase}), {len(df)} rows")

    # ---------------------------------------------------------------- structure
    C.add("required columns present", set(REQUIRED_COLUMNS) <= set(df.columns))
    selected = None
    if phase == "main":
        selected = yaml.safe_load(open(meta["selected_params"], encoding="utf-8"))
    exp = expected_keys(phase, cfg, meta["pairs"], selected)
    got = [row_key(r) for r in df.to_dict("records")]
    C.add("row count == expected", len(df) == len(exp) == meta["expected_rows"],
          f"rows={len(df)} expected={len(exp)} meta={meta['expected_rows']}")
    C.add("no duplicate keys", len(set(got)) == len(got), f"duplicates={len(got) - len(set(got))}")
    missing, extra = exp - set(got), set(got) - exp
    C.add("no missing / unexpected conditions", not missing and not extra,
          f"missing={len(missing)} unexpected={len(extra)}")
    C.add("no pipeline errors", (df.pipeline_error == "").all(), f"errors={(df.pipeline_error != '').sum()}")
    per = df.assign(_wb=df.wiener_balance.fillna(-1)).groupby(
        ["blur_id", "group", "processing_method", "detector", "detector_config", "_wb"]).size()
    C.add("balanced design (same #pairs per condition)", per.nunique() == 1 and per.iloc[0] == len(meta["pairs"]),
          f"rows per condition: {sorted(set(per.tolist()))}")

    # ---------------------------------------------------------------- split hygiene
    splits = set(manifest.loc[meta["pairs"], "split"])
    want = {"pilot": {"validation"}, "validation": {"validation"}, "main": {"test"}}[phase]
    C.add("pairs come from the correct split", splits == want, f"{splits}")
    if phase == "main":
        b = df[df.group == "B"]
        okp = all(json.loads(r.detector_params) == selected[r.detector]["params"] for r in b.itertuples())
        okw = all(close(r.wiener_balance, float(selected["wiener_balance"][r.blur_id]))
                  for r in df[df.group == "D"].itertuples())
        C.add("main uses the validation-selected detector params and Wiener balances", okp and okw)
        vrun = selected.get("source_validation_run", "")
        C.add("selected params come from a validation run", str(vrun).startswith("validation_"), vrun)

    # ---------------------------------------------------------------- same rules everywhere
    C.add("single config_hash / code_hash", df.config_hash.nunique() == 1 and df.code_hash.nunique() == 1,
          f"config={df.config_hash.unique().tolist()} code={df.code_hash.unique().tolist()}")
    C.add("same valid margin in all rows", set(df.valid_margin_px) == {str(m)})
    C.add("feature budget 1500 in all rows", all(json.loads(p)["nfeatures"] == 1500 for p in df.detector_params.unique()))

    # ---------------------------------------------------------------- PSF
    reps = {c["blur_id"]: psf_report(make_psf(c, cfg)) for c in cfg["blur"]["conditions"]}
    C.add("PSF: sum=1, non-negative, odd, centred, symmetric (8 conditions)",
          all(psf_ok(make_psf(c, cfg)) for c in cfg["blur"]["conditions"]),
          "; ".join(f"{k}: {v['shape']} sum={v['sum']:.12f}" for k, v in reps.items()))

    # ---------------------------------------------------------------- geometry / sizes
    ok_size = all((r.image_h, r.image_w) == (manifest.loc[r.pair_id, "target_h"], manifest.loc[r.pair_id, "target_w"])
                  for r in df.drop_duplicates("pair_id").itertuples())
    C.add("processed image size == resized clean target size", ok_size)

    # ---------------------------------------------------------------- value ranges
    blurred = df[df.group != "CLEAN"]
    rng_ok = all(((blurred[c].dropna() >= 0) & (blurred[c].dropna() <= 1)).all()
                 for c in ("edge_precision", "edge_recall", "edge_f1", "keypoint_repeatability", "ssim",
                           "match_precision_3px", "match_precision_5px", "ransac_inlier_ratio"))
    C.add("ratios in [0, 1]", rng_ok)
    nan_counts = {c: int(df[c].isna().sum()) for c in ("psnr", "ssim", "edge_f1", "keypoint_repeatability")}
    C.add("PSNR/SSIM/EdgeF1/Repeatability never NaN", sum(nan_counts.values()) == 0, f"{nan_counts}")
    cl = df[df.group == "CLEAN"]
    C.add("CLEAN baseline: PSNR=inf, SSIM=1, EdgeF1=1, Rep=1",
          bool(np.isinf(cl.psnr).all() and np.allclose(cl.ssim, 1) and (cl.edge_f1 == 1).all()
               and (cl.keypoint_repeatability == 1).all()))
    C.add("finite PSNR for all processed rows", bool(np.isfinite(blurred.psnr).all()))
    C.add("count ordering: correct@3 <= correct@5 <= good; inliers <= good",
          bool(((df.gt_correct_matches_3px <= df.gt_correct_matches_5px) & (df.gt_correct_matches_5px <= df.good_matches)
                & (df.ransac_inliers <= df.good_matches)).all()))
    z = df[df.good_matches == 0]
    C.add("zero matches -> precision NaN, correct=0",
          bool(z.match_precision_3px.isna().all() and (z.gt_correct_matches_3px == 0).all()), f"rows with 0 matches={len(z)}")
    fail = df[df.homography_error_mean.isna()]
    C.add("NaN projection error <=> success=0 with failure_reason",
          bool((fail.homography_success_3px == 0).all() and (fail.homography_success_5px == 0).all()
               and (fail.failure_reason != "").all()), f"failed H rows={len(fail)}")
    C.add("success@5 >= success@3", bool((df.homography_success_5px >= df.homography_success_3px).all()))
    s3 = df[df.homography_success_3px == 1]
    C.add("success@3 rows have mean error <= 3 and empty failure_reason",
          bool((s3.homography_error_mean <= 3).all() and (s3.failure_reason == "").all()))

    # shared processed image -> identical image metrics across detectors / groups
    img_cols = ["psnr", "ssim", "edge_precision", "edge_recall", "edge_f1"]
    g = df.assign(_wb=df.wiener_balance.fillna(-1)).groupby(["pair_id", "blur_id", "processing_method", "_wb"])[img_cols]
    C.add("image metrics identical for rows sharing a processed image", bool((g.nunique(dropna=False) <= 1).all().all()))

    # ---------------------------------------------------------------- re-computation from match details
    n_bad_gt = n_bad_h = n_bad_idx = n_bad_hgt = 0
    hc = ev["homography"]
    pair_cache = {}
    for r in df.itertuples():
        d = np.load(PROJECT_ROOT / r.match_detail_file)
        Hgt_manifest = np.array(json.loads(manifest.loc[r.pair_id, "H_resized"]))
        n_bad_hgt += not np.allclose(d["H_gt"], Hgt_manifest, rtol=1e-12, atol=1e-12)
        q, t = d["query_idx"], d["train_idx"]
        n_bad_idx += not (np.allclose(d["ref_xy"], d["ref_kp"][q, :2], atol=1e-3)
                          and np.allclose(d["tgt_xy"], d["tgt_kp"][t, :2], atol=1e-3)
                          and len(d["ref_kp"]) == r.reference_keypoints and len(d["tgt_kp"]) == r.target_keypoints)
        e = M.gt_errors(d["H_gt"], d["ref_xy"], d["tgt_xy"])
        n_bad_gt += not (int((e <= 3).sum()) == r.gt_correct_matches_3px and int((e <= 5).sum()) == r.gt_correct_matches_5px
                         and len(e) == r.good_matches and int(d["ransac_inlier"].sum()) == r.ransac_inliers)
        H_est = None if not np.all(np.isfinite(d["H_est"])) else d["H_est"]
        shp = tuple(d["ref_shape"]), tuple(d["tgt_shape"])
        out, reason = M.evaluate_homography(H_est, d["H_gt"], shp[0], shp[1], hc, m)
        if r.failure_reason in ("insufficient_matches", "no_reference_keypoints", "no_target_keypoints"):
            reason = r.failure_reason
        n_bad_h += not (close(out["homography_error_mean"], r.homography_error_mean)
                        and close(out["homography_error_median"], r.homography_error_median)
                        and out["homography_success_3px"] == r.homography_success_3px
                        and out["homography_success_5px"] == r.homography_success_5px
                        and reason == r.failure_reason and out["n_grid_valid"] == r.n_grid_valid)
    C.add("H_gt in match details == manifest H_resized", n_bad_hgt == 0, f"mismatches={n_bad_hgt}")
    C.add("match detail indices/coords consistent with keypoints", n_bad_idx == 0, f"mismatches={n_bad_idx}")
    C.add("GT-correct counts re-computed from match details", n_bad_gt == 0, f"mismatches={n_bad_gt}")
    C.add("homography error/success/failure re-computed from H_est", n_bad_h == 0, f"mismatches={n_bad_h}")

    # ---------------------------------------------------------------- image metrics re-computed independently
    img_rows = df.drop_duplicates(["pair_id", "blur_id", "processing_method", "wiener_balance"])
    img_rows = img_rows[img_rows.blur_id != "none"]
    if sample_images and len(img_rows) > sample_images:
        img_rows = img_rows.sample(sample_images, random_state=42)
    n_bad_img, max_dpsnr, max_dssim, n_bad_edge = 0, 0.0, 0.0, 0
    cache = {}
    for r in img_rows.to_dict("records"):
        if r["pair_id"] not in pair_cache:
            pair_cache[r["pair_id"]] = load_pair(manifest.loc[r["pair_id"]].to_dict() | {"pair_id": r["pair_id"]}, cfg)
        pair = pair_cache[r["pair_id"]]
        img = regenerate(pair, r, cfg, cache)
        a, b = M.crop_valid(pair.tgt, m), M.crop_valid(img, m)
        p_cv = cv2.PSNR(a, b)                                   # independent implementation (8-bit, R=255)
        s_sk = structural_similarity(a.astype(float), b.astype(float), data_range=255, gaussian_weights=True,
                                     sigma=1.5, use_sample_covariance=False)
        dp, ds = abs(p_cv - r["psnr"]), abs(s_sk - r["ssim"])
        max_dpsnr, max_dssim = max(max_dpsnr, dp), max(max_dssim, ds)
        n_bad_img += not (dp < 1e-6 and ds < 1e-9)
        e = M.edge_prf(M.canny(pair.tgt, ev["canny"]), M.canny(img, ev["canny"]), ev["edge_tolerance_px"], m)
        n_bad_edge += not close(e["edge_f1"], r["edge_f1"], 1e-12)
    C.add(f"PSNR (cv2.PSNR) / SSIM re-computed from regenerated images (n={len(img_rows)})", n_bad_img == 0,
          f"max |dPSNR|={max_dpsnr:.2e} dB, max |dSSIM|={max_dssim:.2e}")
    C.add("Edge F1 re-computed from regenerated images", n_bad_edge == 0, f"mismatches={n_bad_edge}")

    # ---------------------------------------------------------------- Wiener exactness on a real image
    r0 = img_rows[img_rows.processing_method == "wiener_known_psf"].head(1).to_dict("records")
    if r0:
        r0 = r0[0]
        pair = pair_cache[r0["pair_id"]]
        cond = next(c for c in cfg["blur"]["conditions"] if c["blur_id"] == r0["blur_id"])
        k = make_psf(cond, cfg)
        f = pair.tgt.astype(float)
        w1 = wiener_float(f / 255, k, float(r0["wiener_balance"]))
        w2 = wiener_float(f[::-1, ::-1] / 255, k, float(r0["wiener_balance"]))[::-1, ::-1]
        C.add("Wiener on real image is rotation-equivariant (no shift)", np.abs(w1 - w2).max() < 1e-9,
              f"{r0['pair_id']} {r0['blur_id']} maxdiff={np.abs(w1 - w2).max():.1e}")

    # ---------------------------------------------------------------- phase specific sanity
    if phase in ("pilot", "main"):
        a = df[(df.group == "A") & (df.detector == "ORB")].set_index(["pair_id", "blur_id"]).psnr
        mono = all(a[(p, s)] < a[(p, w)] for p in meta["pairs"]
                   for s, w in (("gaussian_s3.0", "gaussian_s1.5"), ("disk_r5", "disk_r3"),
                                ("motion_l17_a0", "motion_l9_a0"), ("motion_l17_a90", "motion_l9_a90")))
        C.add("sanity: blurred PSNR decreases with blur strength", mono)
    if phase == "pilot":
        cs = df[(df.group == "CLEAN") & (df.detector_config == "default")]
        C.add("sanity: clean baseline (default detectors) succeeds @3px on pilot pairs",
              bool((cs.homography_success_3px == 1).all()),
              "; ".join(f"{r.pair_id}/{r.detector}: err={r.homography_error_mean:.2f}" for r in cs.itertuples()))

    # ---------------------------------------------------------------- QC images
    qc_files = []
    if make_qc:
        qc_dir = PROJECT_ROOT / cfg["paths"]["qc_root"] / run_dir.name
        qc_dir.mkdir(parents=True, exist_ok=False)
        from . import qc
        dsel = df[df.group.isin(["D", "D_candidate"])].drop_duplicates("blur_id").set_index("blur_id").wiener_balance
        balances = {b: float(dsel.get(b, cfg["processing"]["wiener"]["pilot_placeholder_balance"]))
                    for b in [c["blur_id"] for c in cfg["blur"]["conditions"]]}
        for pid in meta["pairs"][: (2 if phase == "pilot" else 1)]:
            pair = pair_cache.get(pid) or load_pair(manifest.loc[pid].to_dict() | {"pair_id": pid}, cfg)
            qc.processing_panel(pair, cfg, balances, qc_dir / f"{pid}__processing_panel.png")
            cond = next(c for c in cfg["blur"]["conditions"] if c["blur_id"] == "disk_r5")
            ims = qc.processed_images(pair, cond, cfg, balances["disk_r5"])
            for k_, im in ims.items():
                qc.edge_overlay(pair.tgt, im, ev["canny"], qc_dir / f"{pid}__disk_r5__{k_}__edges.png")
            for r in df[(df.pair_id == pid) & (df.blur_id.isin(["none", "disk_r5"])) & (df.detector_config == "default")
                        & (df.group.isin(["CLEAN", "A", "C", "D", "D_candidate"]))].itertuples():
                if r.group == "D_candidate" and not close(r.wiener_balance, balances["disk_r5"]):
                    continue
                img = pair.tgt if r.blur_id == "none" else ims[r.processing_method]
                name = f"{pid}__{r.blur_id}__{r.processing_method}__{r.detector}__matches.png"
                qc.match_overlay(pair.ref, img, PROJECT_ROOT / r.match_detail_file, qc_dir / name)
                d = np.load(PROJECT_ROOT / r.match_detail_file)
                clean_d = np.load(PROJECT_ROOT / df[(df.pair_id == pid) & (df.blur_id == "none") & (df.detector == r.detector)
                                                    & (df.detector_config == "default")].iloc[0].match_detail_file)
                qc.keypoint_overlay(clean_d["tgt_kp"][:, :2], d["tgt_kp"][:, :2], img,
                                    qc_dir / f"{pid}__{r.blur_id}__{r.processing_method}__{r.detector}__keypoints.png")
        qc_files = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in qc_dir.glob("*.png"))
        C.add("QC images written", len(qc_files) > 0, f"{len(qc_files)} files in {qc_dir.relative_to(PROJECT_ROOT).as_posix()}")

    # ---------------------------------------------------------------- summary tables
    summary = {
        "rows_by_group_detector": df.groupby(["group", "detector"]).size().to_dict(),
        "rows_by_blur": df.groupby("blur_id").size().to_dict(),
        "failure_reasons": df.failure_reason.replace("", "(success@3px)").value_counts().to_dict(),
        "gt_flagged_pairs": sorted(df[df.gt_quality_flag == "1"].pair_id.unique().tolist()),
    }
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    res = {"run": run_dir.name, "phase": phase, "timestamp": stamp, "all_pass": C.all_pass, "checks": C.items,
           "summary": {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in summary.items()},
           "qc_files": qc_files}
    write_json_new(run_dir / f"verification_{stamp}.json", res)
    md = [f"# Verification: {run_dir.name} ({phase})", "", f"- timestamp: {stamp}", f"- rows: {len(df)}",
          f"- ALL PASS: **{C.all_pass}**", "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {i['check']} | {'PASS' if i['pass'] else 'FAIL'} | {i['detail'].replace('|', '/')} |" for i in C.items]
    md += ["", "## failure_reason counts", ""] + [f"- {k}: {v}" for k, v in summary["failure_reasons"].items()]
    write_text_new(run_dir / f"verification_{stamp}.md", "\n".join(md) + "\n")
    print("ALL PASS" if C.all_pass else "SOME CHECKS FAILED", "->", run_dir / f"verification_{stamp}.md")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample-images", type=int, default=40, help="0 = all processed images")
    ap.add_argument("--qc", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = PROJECT_ROOT / run
    res = verify(run, a.sample_images, a.qc)
    raise SystemExit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
