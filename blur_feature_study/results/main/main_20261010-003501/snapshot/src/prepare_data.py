"""Phase 1: sequence selection, dataset_manifest.csv and data/homography checks.

Usage (from blur_feature_study/):  python -m src.prepare_data
"""
import datetime as dt
import json

import cv2
import numpy as np
import pandas as pd

from .dataset import build_manifest, load_gray_resized, load_pair
from .features import border_mask
from .io_utils import PROJECT_ROOT, imread_unicode, load_config, load_homography, write_json_new
from .metrics import gt_errors

MANIFEST_COMPARE = ["pair_id", "split", "sequence_id", "sequence_type", "target_index", "reference_sha256",
                    "target_sha256", "homography_sha256", "H_resized"]


def _sift_matches(a, b, margin):
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(a, border_mask(a.shape, margin))
    kb, db = sift.detectAndCompute(b, border_mask(b.shape, margin))
    if da is None or db is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = [p[0] for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    pa = np.array([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.array([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pa, pb


def verify_homography_direction(df, cfg):
    """Empirical check on clean pairs: SIFT matches are geometrically consistent with
    H (reference->target) and not with inv(H); also compares the pixel-centre scale
    matrix with a plain diagonal scale."""
    margin = cfg["preprocessing"]["valid_margin_px"]
    rows = []
    for _, r in df.iterrows():
        pair = load_pair(r, cfg)
        pa, pb = _sift_matches(pair.ref, pair.tgt, margin)
        e_fwd = gt_errors(pair.H_gt, pa, pb)
        e_inv = gt_errors(np.linalg.inv(pair.H_gt), pa, pb)
        e_id = gt_errors(np.eye(3), pa, pb)
        # alternative (diagonal) scale matrices for the resize
        _, S_r, hw_r = load_gray_resized(PROJECT_ROOT / r["reference_file"], cfg["preprocessing"]["max_long_side"])
        _, S_t, hw_t = load_gray_resized(PROJECT_ROOT / r["target_file"], cfg["preprocessing"]["max_long_side"])
        D_r = np.diag([S_r[0, 0], S_r[1, 1], 1.0]); D_t = np.diag([S_t[0, 0], S_t[1, 1], 1.0])
        H0 = load_homography(PROJECT_ROOT / r["homography_file"])
        H_diag = D_t @ H0 @ np.linalg.inv(D_r)
        e_diag = gt_errors(H_diag / H_diag[2, 2], pa, pb)
        near = e_fwd < 5
        rows.append({
            "pair_id": r["pair_id"], "split": r["split"], "sequence_type": r["sequence_type"],
            "n_matches": len(pa),
            "n_correct3_H": int((e_fwd <= 3).sum()), "n_correct3_invH": int((e_inv <= 3).sum()),
            "n_correct3_identity": int((e_id <= 3).sum()),
            "median_err_near_H_pixelcenter": float(np.median(e_fwd[near])) if near.any() else np.nan,
            "median_err_near_H_diagscale": float(np.median(e_diag[near])) if near.any() else np.nan,
            "H_minus_I_max": float(np.abs(pair.H_gt - np.eye(3)).max()),
            "scale_ref": float(S_r[0, 0]), "scale_tgt": float(S_t[0, 0]),
        })
    out = pd.DataFrame(rows)
    out["direction_ok"] = (out["n_correct3_H"] >= 10) & (out["n_correct3_H"] >= out["n_correct3_invH"])
    return out


def check_pixel_convention(df):
    """HPatches H at ORIGINAL resolution: as-is (pixel-centre, like OpenCV keypoints) vs
    corner->centre conversion T(-.5) H T(.5). Uses all 5 targets of the selected viewpoint sequences."""
    T = lambda s: np.array([[1, 0, s], [0, 1, s], [0, 0, 1.0]])
    root = PROJECT_ROOT / load_config()["paths"]["hpatches_root"]
    rows = []
    for seq in sorted(df.loc[df.sequence_type == "viewpoint", "sequence_id"].unique()):
        a = cv2.cvtColor(imread_unicode(root / seq / "1.ppm"), cv2.COLOR_BGR2GRAY)
        for t in range(2, 7):
            b = cv2.cvtColor(imread_unicode(root / seq / f"{t}.ppm"), cv2.COLOR_BGR2GRAY)
            pa, pb = _sift_matches(a, b, 0)
            H = load_homography(root / seq / f"H_1_{t}")
            Hc = T(-0.5) @ H @ T(0.5)
            e0, e1 = gt_errors(H, pa, pb), gt_errors(Hc / Hc[2, 2], pa, pb)
            near = (e0 < 3) | (e1 < 3)
            if near.sum() >= 20:
                rows.append({"sequence_id": seq, "target_index": t, "n_near": int(near.sum()),
                             "median_err_as_is": float(np.median(e0[near])),
                             "median_err_corner_to_centre": float(np.median(e1[near]))})
    return pd.DataFrame(rows)


GT_FLAG_RULE = ("flag if (clean SIFT matches with error<=3px under H_gt) < 10 OR "
                "median error of near matches (<5px) > 1.5px  [resized resolution, clean images only]")


def gt_quality_flags(chk):
    flag = (chk["n_correct3_H"] < 10) | (chk["median_err_near_H_pixelcenter"] > 1.5)
    return pd.DataFrame({"pair_id": chk["pair_id"], "gt_quality_flag": flag.astype(int),
                         "n_correct3_H_clean": chk["n_correct3_H"],
                         "median_err_near_clean": chk["median_err_near_H_pixelcenter"],
                         "rule": GT_FLAG_RULE})


def main():
    cfg = load_config()
    mpath = PROJECT_ROOT / cfg["paths"]["manifest"]
    df = build_manifest(cfg)
    if mpath.exists():
        old = pd.read_csv(mpath)
        same = (len(old) == len(df) and all(
            (old[c].astype(str).values == df[c].astype(str).values).all() for c in MANIFEST_COMPARE))
        if not same:
            raise RuntimeError("existing manifest differs from the regenerated one; refusing to overwrite")
        print("manifest exists and is identical:", mpath)
    else:
        df.to_csv(mpath, index=False)
        print("wrote", mpath)
    print(df.groupby(["split", "sequence_type"]).agg(pairs=("pair_id", "count"),
                                                    sequences=("sequence_id", "nunique")))
    assert not (set(df[df.split == "validation"].sequence_id) & set(df[df.split == "test"].sequence_id))

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    chk = verify_homography_direction(df, cfg)
    cpath = PROJECT_ROOT / "data" / f"homography_checks_{stamp}.csv"
    chk.to_csv(cpath, index=False)
    print(chk.to_string())
    conv = check_pixel_convention(df)
    conv_path = PROJECT_ROOT / "data" / f"pixel_convention_check_{stamp}.csv"
    conv.to_csv(conv_path, index=False)
    flags = gt_quality_flags(chk)
    fpath = PROJECT_ROOT / "data" / "gt_quality_flags.csv"
    if fpath.exists():
        old = pd.read_csv(fpath)
        if not (old["gt_quality_flag"].values == flags["gt_quality_flag"].values).all():
            raise RuntimeError("gt_quality_flags.csv exists with different flags; refusing to overwrite")
    else:
        flags.to_csv(fpath, index=False)
    print(flags.to_string())
    summary = {
        "timestamp": stamp, "n_pairs": int(len(df)),
        "n_sequences": int(df.sequence_id.nunique()),
        "direction_ok_all": bool(chk.direction_ok.all()),
        "pairs_direction_not_ok": chk.loc[~chk.direction_ok, "pair_id"].tolist(),
        "median_err_near_pixelcenter": float(chk.median_err_near_H_pixelcenter.median()),
        "median_err_near_diagscale": float(chk.median_err_near_H_diagscale.median()),
        "pixel_convention_pairs": int(len(conv)),
        "pixel_convention_as_is_better": int((conv.median_err_as_is < conv.median_err_corner_to_centre).sum()),
        "pixel_convention_mean_as_is": float(conv.median_err_as_is.mean()),
        "pixel_convention_mean_corner_to_centre": float(conv.median_err_corner_to_centre.mean()),
        "gt_flag_rule": GT_FLAG_RULE,
        "gt_flagged_pairs": flags.loc[flags.gt_quality_flag == 1, "pair_id"].tolist(),
        "checks_csv": cpath.relative_to(PROJECT_ROOT).as_posix(),
        "pixel_convention_csv": conv_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    write_json_new(PROJECT_ROOT / "data" / f"homography_checks_{stamp}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
