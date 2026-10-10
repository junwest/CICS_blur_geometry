"""Visual verification examples: regenerate processed images for selected cells and draw
(1) crops clean/blurred/unsharp/Wiener, (2) Canny overlays, (3) keypoint survival, (4) match overlays A vs D.
Uses the experiment's own src.qc helpers so what is drawn is exactly what was measured."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGS, PROJECT, add_clean_reference, load_results, main_core  # noqa: E402

sys.path.insert(0, str(PROJECT))
from src import qc  # noqa: E402
from src.dataset import load_manifest, load_pair  # noqa: E402
from src.io_utils import imwrite_unicode, load_config  # noqa: E402
from src.metrics import canny  # noqa: E402

cfg = load_config()
df = add_clean_reference(main_core(load_results()))
man = load_manifest(cfg).set_index("pair_id")
out = FIGS / "visual_examples"
out.mkdir(exist_ok=True)
sel = {"disk_r5": 0.005, "motion_l17_a0": 0.005, "motion_l17_a90": 0.005, "gaussian_s3.0": 0.005, "gaussian_s1.5": 0.005}


def label(img, text):
    o = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(o, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return o


def row_of(pid, blur, det, group):
    r = df[(df.pair_id == pid) & (df.blur_id == blur) & (df.detector == det) & (df.group == group)].iloc[0]
    return r


def example(pid, blur, crop=260, cx=None, cy=None):
    pair = load_pair(man.loc[pid].to_dict() | {"pair_id": pid}, cfg)
    cond = next(c for c in cfg["blur"]["conditions"] if c["blur_id"] == blur)
    ims = qc.processed_images(pair, cond, cfg, sel[blur])
    h, w = pair.tgt.shape
    y0 = (h - crop) // 2 if cy is None else cy; x0 = (w - crop) // 2 if cx is None else cx
    tiles, edge_tiles = [], []
    stat = {}
    for key, g in (("clean", "CLEAN"), ("blurred", "A"), ("unsharp_mask", "C"), ("wiener_known_psf", "D")):
        r = row_of(pid, "none" if g == "CLEAN" else blur, "SIFT", g)
        txt = f"{g}: PSNR {r.psnr:.1f} EdgeF1 {r.edge_f1:.2f}" if g != "CLEAN" else "clean target"
        tiles.append(label(ims[key][y0:y0 + crop, x0:x0 + crop], txt))
        # edge overlay crop
        from scipy.ndimage import distance_transform_edt
        ec, ep = canny(pair.tgt, cfg["evaluation"]["canny"]), canny(ims[key], cfg["evaluation"]["canny"])
        dc = distance_transform_edt(~ec) if ec.any() else np.full(ec.shape, np.inf)
        dp = distance_transform_edt(~ep) if ep.any() else np.full(ep.shape, np.inf)
        vis = cv2.cvtColor((pair.tgt * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[ep & (dc <= 2)] = (0, 200, 0); vis[ep & (dc > 2)] = (0, 0, 255); vis[ec & (dp > 2)] = (255, 80, 0)
        edge_tiles.append(label(vis[y0:y0 + crop, x0:x0 + crop], f"{g} edges P={r.edge_precision:.2f} R={r.edge_recall:.2f}"))
        for det in ("ORB", "SIFT"):
            rr = row_of(pid, "none" if g == "CLEAN" else blur, det, g)
            stat[(g, det)] = rr
    imwrite_unicode(out / f"{pid}__{blur}__crops.png", np.vstack([np.hstack(tiles), np.hstack(edge_tiles)]))
    # keypoint overlays (crop) for both detectors: green = clean keypoint repeated, red = lost
    kp_rows = []
    for det in ("ORB", "SIFT"):
        tiles = []
        clean_d = np.load(PROJECT / stat[("CLEAN", det)].match_detail_file)
        for key, g in (("blurred", "A"), ("unsharp_mask", "C"), ("wiener_known_psf", "D")):
            d = np.load(PROJECT / stat[(g, det)].match_detail_file)
            from scipy.spatial import cKDTree
            cxy, pxy = clean_d["tgt_kp"][:, :2], d["tgt_kp"][:, :2]
            vis = cv2.cvtColor(ims[key], cv2.COLOR_GRAY2BGR)
            dist = cKDTree(pxy).query(cxy)[0] if len(pxy) else np.full(len(cxy), np.inf)
            for (x, y), dd in zip(cxy, dist):
                if y0 <= y < y0 + crop and x0 <= x < x0 + crop:
                    cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 200, 0) if dd <= 3 else (0, 0, 255), 1, cv2.LINE_AA)
            r = stat[(g, det)]
            tiles.append(label(vis[y0:y0 + crop, x0:x0 + crop], f"{det} {g} rep={r.keypoint_repeatability:.2f} kp={int(r.target_keypoints)} gt3={int(r.gt_correct_matches_3px)}"))
        kp_rows.append(np.hstack(tiles))
    imwrite_unicode(out / f"{pid}__{blur}__keypoints.png", np.vstack(kp_rows))
    # match overlays for A and D (SIFT and ORB)
    for det in ("ORB", "SIFT"):
        for g, key in (("A", "blurred"), ("D", "wiener_known_psf")):
            r = stat[(g, det)]
            qc.match_overlay(pair.ref, ims[key], PROJECT / r.match_detail_file, out / f"{pid}__{blur}__{det}__{g}__matches.png")
    return {k: {"gt3": int(v.gt_correct_matches_3px), "good": int(v.good_matches), "err": v.homography_error_mean, "rep": v.keypoint_repeatability, "psnr": v.psnr, "edge_f1": v.edge_f1, "reason": v.failure_reason} for k, v in stat.items()}


summary = {}
# (1) unsharp dissociation example: motion 17 px, PSNR down / edge up / matching flat
summary["v_busstop__1-2/motion_l17_a90"] = example("v_busstop__1-2", "motion_l17_a90")
# (2) strong restoration example: disk r5 on a viewpoint pair where A fails and D succeeds
summary["v_calder__1-5/disk_r5"] = example("v_calder__1-5", "disk_r5")
# (3) mild blur: gaussian 1.5 on an illumination pair (ceiling case)
summary["i_fruits__1-2/gaussian_s1.5"] = example("i_fruits__1-2", "gaussian_s1.5")
# (4) hard pair failing even clean: v_underground__1-5 (ORB clean error 470 px)
pair = load_pair(man.loc["v_underground__1-5"].to_dict() | {"pair_id": "v_underground__1-5"}, cfg)
for det in ("ORB", "SIFT"):
    r = row_of("v_underground__1-5", "none", det, "CLEAN")
    qc.match_overlay(pair.ref, pair.tgt, PROJECT / r.match_detail_file, out / f"v_underground__1-5__none__{det}__CLEAN__matches.png")
    summary[f"v_underground__1-5/clean/{det}"] = {"gt3": int(r.gt_correct_matches_3px), "good": int(r.good_matches), "err": r.homography_error_mean, "reason": r.failure_reason}
pd.DataFrame({k: {kk if isinstance(kk, str) else f"{kk[0]}_{kk[1]}": vv for kk, vv in v.items()} for k, v in summary.items()}).to_json(out / "examples_summary.json", indent=2, default_handler=str)
print(pd.DataFrame({k: v for k, v in summary.items() if "clean" in k}).T.to_string())
for k, v in summary.items():
    if "clean" in k:
        continue
    print(k)
    for kk, vv in v.items():
        print("   ", kk, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in vv.items()})
