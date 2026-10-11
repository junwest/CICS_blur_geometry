"""Exp1 (main): stage decomposition of the TARGET features; reference fixed to Clean.

  python -m extension.src.run_main --nafnet-run ext_nafnet_<ts> [--limit-pairs N]   (from blur_feature_study/, .venv)

Per test pair (24) x blur (8) x detector (ORB, SIFT; base 'default' config):
  B = apply_blur(clean target)      R_m = Unsharp(B) | Wiener(B, exact PSF, base balance) | NAFNet(B) (cached PNG)
  BB = detect(B) + compute(B)       shared baseline, once per cell
  BR = detect(B) + compute(R_m)     RB = detect(R_m) + compute(B)     RR = detect(R_m) + compute(R_m)
  CLEAN row per (pair, detector) = base pipeline (detectAndCompute on the clean target).
Every target image is also processed with detectAndCompute and compared bitwise with the separate
detect()+compute() path (dac_kp_equal / dac_desc_equal).
Outputs (new run directory): raw_results.csv, image_metrics.csv, details/<pair>/*.npz, run_meta.json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import stagewise as S
from .common import (RESULTS, blur_cond, environment_info, load_base, load_ext_config, new_run_dir, pixel_sha,
                     read_png_gray, rel, test_pairs, write_json, PROJECT_ROOT)
from .evaluate import image_metrics, match_and_estimate, repeatability
from .spatial import SpatialFrame
from src import features as F
from src.dataset import load_pair
from src.imaging import apply_blur, unsharp_mask, wiener_restore
from src.psf import make_psf

METHODS = ["unsharp", "wiener", "nafnet"]
DETECTORS = ["ORB", "SIFT"]


class NafnetCache:
    """NAFNet outputs of a nafnet_infer run; an output is used only if the regenerated input hash matches."""

    def __init__(self, run):
        self.dir = RESULTS / run
        self.man = pd.read_csv(self.dir / "nafnet_manifest.csv").set_index("image_id")
        self.meta = json.loads((self.dir / "run_meta.json").read_text(encoding="utf-8"))
        self.flags = self.man.chan_spread_max >= 255          # DECISIONS.md: output-only divergence rule

    def get(self, image_id, regenerated_input):
        r = self.man.loc[image_id]
        if pixel_sha(regenerated_input) != r.input_pixel_sha256:
            raise RuntimeError(f"NAFNet input hash mismatch for {image_id}: cached output not reusable")
        out = read_png_gray(PROJECT_ROOT / r.output_path)
        if out.shape != regenerated_input.shape or pixel_sha(out) != r.output_pixel_sha256:
            raise RuntimeError(f"NAFNet output shape/hash mismatch for {image_id}")
        return out, float(r.t_total_ms), str(r.device), bool(self.flags.loc[image_id])


def processed_images(cfg, sel, B, psf, cond, naf, pair):
    """{method: (image, processing_ms, device, nafnet_flag)}; 'none' = B."""
    u = cfg["processing"]["unsharp_mask"]
    out = {"none": (B, 0.0, "", False)}
    t0 = time.perf_counter(); U = unsharp_mask(B, u["alpha"], u["sigma"], u["truncate"])
    out["unsharp"] = (U, (time.perf_counter() - t0) * 1000.0, "cpu", False)
    bal = float(sel["wiener_balance"][cond["blur_id"]])
    t0 = time.perf_counter(); W = wiener_restore(B, psf, bal)
    out["wiener"] = (W, (time.perf_counter() - t0) * 1000.0, "cpu", False)
    out["nafnet"] = naf.get(f"tgt__{pair.pair_id}__{cond['blur_id']}", B)
    for k, (img, *_rest) in out.items():
        if img.shape != pair.tgt.shape or img.dtype != np.uint8:
            raise RuntimeError(f"{k}: processed image shape/dtype mismatch {img.shape} {img.dtype}")
    return out, bal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nafnet-run", required=True)
    ap.add_argument("--limit-pairs", type=int, default=0, help="smoke test only (separate run prefix)")
    args = ap.parse_args(argv)
    cfg, sel, man, flags = load_base()
    ext = load_ext_config()
    naf = NafnetCache(args.nafnet_run)
    run_id, out = new_run_dir("ext_smoke_main" if args.limit_pairs else "ext_main")
    det_dir = out / "details"
    m = int(cfg["preprocessing"]["valid_margin_px"])
    pairs = test_pairs(man)
    if args.limit_pairs:
        pairs = pairs.iloc[: args.limit_pairs]
    rows, img_rows = [], []
    ref_cache = {}
    t_start = time.time()
    for _, prow in pairs.iterrows():
        pair = load_pair(prow, cfg)
        mask_t = F.border_mask(pair.tgt.shape, m)
        sp = SpatialFrame(pair.H_gt, pair.ref.shape, pair.tgt.shape, m)
        pdir = det_dir / pair.pair_id
        pdir.mkdir(parents=True, exist_ok=False)
        base = {"pair_id": pair.pair_id, "sequence_id": pair.sequence_id, "sequence_type": pair.sequence_type,
                "reference_index": pair.reference_index, "target_index": pair.target_index,
                "gt_quality_flag": flags.get(pair.pair_id, 0), "image_h": pair.tgt.shape[0], "image_w": pair.tgt.shape[1],
                "overlap_area": sp.area, "n_eligible_cells": int(sp.eligible.sum())}
        clean_feats, refs = {}, {}
        for det in DETECTORS:
            params = cfg["detectors"][det]["default"]
            key = (pair.sequence_id, pair.reference_index, det)
            if key not in ref_cache:
                ref_cache[key] = F.detect(det, params, pair.ref, F.border_mask(pair.ref.shape, m))
            ref = refs[det] = ref_cache[key]
            clean = clean_feats[det] = F.detect(det, params, pair.tgt, mask_t)
            # separate-path check on the clean target as well
            kC, _ = S.detect_only(det, params, pair.tgt, mask_t)
            fC, idsC, infoC = S.describe(det, params, pair.tgt, kC)
            kp_eq, desc_eq = S.same_features(clean, fC)
            met, dt = match_and_estimate(cfg, det, pair.H_gt, pair.ref.shape, pair.tgt.shape, ref, clean, sp)
            row = dict(base, blur_id="none", blur_type="none", blur_strength=np.nan, blur_angle=np.nan, detector=det,
                       detector_config="default", method="clean", stage="CLEAN", detection_image="clean",
                       descriptor_image="clean", wiener_balance=np.nan, nafnet_output_flag=False,
                       **image_metrics(cfg, pair.tgt, pair.tgt), **repeatability(cfg, clean, clean), **met,
                       n_detected=len(clean), n_after_compute=len(clean), n_removed_by_compute=0,
                       reordered_by_compute=False, attrs_changed_by_compute=0,
                       sift_sentinel_effective=infoC["sift_sentinel_effective"],
                       dac_kp_equal=kp_eq, dac_desc_equal=desc_eq, processing_device="",
                       runtime_processing_ms=0.0, runtime_detect_ms=clean.runtime_ms, runtime_describe_ms=0.0,
                       runtime_detect_and_compute_ms=clean.runtime_ms)
            row["runtime_ms"] = row["runtime_detect_ms"] + met["runtime_match_ms"] + met["runtime_homography_ms"]
            rows.append(row)
            np.savez_compressed(pdir / f"clean__{det}.npz", H_gt=pair.H_gt, ref_shape=np.array(pair.ref.shape),
                                tgt_shape=np.array(pair.tgt.shape), ref_kp=ref.kp, clean_kp=clean.kp,
                                **{k: v for k, v in dt.items()})
        for cond in cfg["blur"]["conditions"]:
            psf = make_psf(cond, cfg)
            B = apply_blur(pair.tgt, psf)
            imgs, bal = processed_images(cfg, sel, B, psf, cond, naf, pair)
            cbase = dict(base, blur_id=cond["blur_id"], blur_type=cond["blur_type"], blur_strength=cond["strength"],
                         blur_angle=cond["angle"])
            imetrics = {}
            for meth, (img, pms, dev, nflag) in imgs.items():
                imetrics[meth] = image_metrics(cfg, pair.tgt, img)
                img_rows.append(dict(cbase, method=meth, processing_ms=pms, processing_device=dev,
                                     wiener_balance=bal if meth == "wiener" else np.nan,
                                     nafnet_output_flag=nflag, image_pixel_sha256=pixel_sha(img), **imetrics[meth]))
            for det in DETECTORS:
                params = cfg["detectors"][det]["default"]
                ref, clean = refs[det], clean_feats[det]
                npz = {}
                kps, dtimes, dacs = {}, {}, {}
                for meth, (img, *_r) in imgs.items():
                    kps[meth], dtimes[meth] = S.detect_only(det, params, img, mask_t)
                    dacs[meth] = F.detect(det, params, img, mask_t)
                    npz[f"kp_{meth}"] = S.kp_array(kps[meth])

                def stage_row(meth, stage, det_src, desc_src):
                    feats, ids, info = S.describe(det, params, imgs[desc_src][0], kps[det_src])
                    met, dt = match_and_estimate(cfg, det, pair.H_gt, pair.ref.shape, pair.tgt.shape, ref, feats, sp)
                    same = det_src == desc_src
                    kp_eq, desc_eq = S.same_features(dacs[det_src], feats) if same else (np.nan, np.nan)
                    proc_src = "none" if stage == "BB" else meth
                    pms, dev, nflag = imgs[proc_src][1], imgs[proc_src][2], imgs[proc_src][3]
                    im = imetrics[det_src] if same else {}
                    r = dict(cbase, detector=det, detector_config="default", method=meth, stage=stage,
                             detection_image="B" if det_src == "none" else "R",
                             descriptor_image="B" if desc_src == "none" else "R",
                             wiener_balance=bal if meth == "wiener" else np.nan, nafnet_output_flag=nflag,
                             **im, **repeatability(cfg, clean, feats), **met,
                             n_detected=info["n_in"], n_after_compute=info["n_out"],
                             n_removed_by_compute=len(info["removed_ids"]), reordered_by_compute=info["reordered"],
                             attrs_changed_by_compute=info["attrs_changed"],
                             sift_sentinel_effective=info["sift_sentinel_effective"],
                             dac_kp_equal=kp_eq, dac_desc_equal=desc_eq, processing_device=dev,
                             runtime_processing_ms=pms, runtime_detect_ms=dtimes[det_src],
                             runtime_describe_ms=feats.runtime_ms,
                             runtime_detect_and_compute_ms=dacs[det_src].runtime_ms if same else np.nan)
                    r["runtime_ms"] = (pms + dtimes[det_src] + feats.runtime_ms + met["runtime_match_ms"]
                                       + met["runtime_homography_ms"])
                    tag = f"{stage}_{meth}"
                    npz[f"ids_{tag}"] = ids.astype(np.int32)
                    if info["attrs_changed"]:
                        npz[f"kp_{tag}"] = feats.kp
                    for k, v in dt.items():
                        npz[f"{k}_{tag}"] = v
                    return r

                rows.append(stage_row("none", "BB", "none", "none"))
                for meth in METHODS:
                    rows.append(stage_row(meth, "BR", "none", meth))
                    rows.append(stage_row(meth, "RB", meth, "none"))
                    rows.append(stage_row(meth, "RR", meth, meth))
                np.savez_compressed(pdir / f"{cond['blur_id']}__{det}.npz", **npz)
        print(f"{pair.pair_id} done ({time.time() - t_start:.0f}s, {len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", run_id)
    df.to_csv(out / "raw_results.csv", index=False)
    im = pd.DataFrame(img_rows)
    im.insert(0, "run_id", run_id)
    im.to_csv(out / "image_metrics.csv", index=False)
    write_json(out / "run_meta.json", {
        "run_id": run_id, "nafnet_run": args.nafnet_run, "nafnet_device": naf.meta.get("device_name"),
        "n_rows": len(df), "n_image_rows": len(im), "limit_pairs": args.limit_pairs,
        "elapsed_s": round(time.time() - t_start, 1), "environment": environment_info(),
        "timing_scope": {
            "runtime_processing_ms": "Unsharp/Wiener: in-process CPU (single process, cv2/numpy threads as configured); "
                                     "NAFNet: GPU t_total_ms from the NAFNet manifest (uint8 in memory -> uint8 out, "
                                     "excludes model load, warm-up and PNG I/O). Not comparable across devices.",
            "runtime_detect_ms": "detector.detect() on the detection image (stage rows); detectAndCompute for CLEAN",
            "runtime_describe_ms": "detector.compute() on the descriptor image",
            "runtime_detect_and_compute_ms": "detectAndCompute on the same image (BB/RR/CLEAN; base-comparable)",
            "runtime_ms": "processing + detect + describe + match + homography (reference detection excluded, as in base)"},
        "counts": df.groupby("stage").size().to_dict()})
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
