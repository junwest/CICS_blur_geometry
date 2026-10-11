"""Exp2: both images blurred (representative motion-blur conditions only); full pipeline
(detectAndCompute on each image, as in the base study).

  python -m extension.src.run_bothblur --nafnet-run ext_nafnet_<ts>   (from blur_feature_study/, .venv)

kernel configs (extension_config.yaml both_blur): same = (ref motion_l17_a0, tgt motion_l17_a0),
diff = (ref motion_l17_a90, tgt motion_l17_a0). Per method (unsharp, wiener, nafnet) the processing
configs ref_only / tgt_only / both; 'none' (both images blurred, unprocessed) is shared by all methods.
Wiener restores each image with its own exact generation kernel and the base balance for that blur id.
Rows = 24 pairs x 2 kernel configs x 2 detectors x (1 + 3 x 3) = 960.
"""
import argparse
import time

import numpy as np
import pandas as pd

from .common import keep_awake, blur_cond, environment_info, load_base, load_ext_config, new_run_dir, test_pairs, write_json
from .evaluate import image_metrics, match_and_estimate
from .run_main import DETECTORS, METHODS, NafnetCache
from .spatial import SpatialFrame
from src import features as F
from src.dataset import load_pair
from src.imaging import apply_blur, unsharp_mask, wiener_restore
from src.psf import make_psf

PROC_CONFIGS = ["ref_only", "tgt_only", "both"]


def process(cfg, sel, meth, img, blur_id, naf, image_id):
    """(image, ms, device, nafnet_flag)"""
    if meth == "unsharp":
        u = cfg["processing"]["unsharp_mask"]
        t0 = time.perf_counter(); r = unsharp_mask(img, u["alpha"], u["sigma"], u["truncate"])
        return r, (time.perf_counter() - t0) * 1000.0, "cpu", False
    if meth == "wiener":
        psf = make_psf(blur_cond(cfg, blur_id), cfg)
        t0 = time.perf_counter(); r = wiener_restore(img, psf, float(sel["wiener_balance"][blur_id]))
        return r, (time.perf_counter() - t0) * 1000.0, "cpu", False
    if meth == "nafnet":
        return naf.get(image_id, img)
    raise ValueError(meth)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nafnet-run", required=True)
    args = ap.parse_args(argv)
    awake = keep_awake()
    cfg, sel, man, flags = load_base()
    ext = load_ext_config()
    naf = NafnetCache(args.nafnet_run)
    run_id, out = new_run_dir("ext_bothblur")
    m = int(cfg["preprocessing"]["valid_margin_px"])
    kcfg = ext["both_blur"]["kernel_configs"]
    rows = []
    feat_cache = {}
    t_start = time.time()
    for _, prow in test_pairs(man).iterrows():
        pair = load_pair(prow, cfg)
        sp = SpatialFrame(pair.H_gt, pair.ref.shape, pair.tgt.shape, m)
        mref, mtgt = F.border_mask(pair.ref.shape, m), F.border_mask(pair.tgt.shape, m)
        for kname, kc in kcfg.items():
            rb, tb = kc["reference"], kc["target"]
            Bref = apply_blur(pair.ref, make_psf(blur_cond(cfg, rb), cfg))
            Btgt = apply_blur(pair.tgt, make_psf(blur_cond(cfg, tb), cfg))
            ref_imgs = {"B": (Bref, 0.0, "", False)}
            tgt_imgs = {"B": (Btgt, 0.0, "", False)}
            for meth in METHODS:
                ref_imgs[meth] = process(cfg, sel, meth, Bref, rb, naf, f"ref__{pair.sequence_id}__{rb}")
                tgt_imgs[meth] = process(cfg, sel, meth, Btgt, tb, naf, f"tgt__{pair.pair_id}__{tb}")
            ref_im = {k: image_metrics(cfg, pair.ref, v[0]) for k, v in ref_imgs.items()}
            tgt_im = {k: image_metrics(cfg, pair.tgt, v[0]) for k, v in tgt_imgs.items()}
            combos = [("none", "none", "B", "B")]
            for meth in METHODS:
                combos += [(meth, "ref_only", meth, "B"), (meth, "tgt_only", "B", meth), (meth, "both", meth, meth)]
            for det in DETECTORS:
                params = cfg["detectors"][det]["default"]
                for meth, pc, rk, tk in combos:
                    fkey_r = ("ref", pair.sequence_id, rb, rk, det)          # reference image shared within a sequence
                    if fkey_r not in feat_cache:
                        feat_cache[fkey_r] = F.detect(det, params, ref_imgs[rk][0], mref)
                    fr = feat_cache[fkey_r]
                    fkey_t = ("tgt", pair.pair_id, tb, tk, det)
                    if fkey_t not in feat_cache:
                        feat_cache[fkey_t] = F.detect(det, params, tgt_imgs[tk][0], mtgt)
                    ft = feat_cache[fkey_t]
                    met, _ = match_and_estimate(cfg, det, pair.H_gt, pair.ref.shape, pair.tgt.shape, fr, ft, sp)
                    r = {"pair_id": pair.pair_id, "sequence_id": pair.sequence_id, "sequence_type": pair.sequence_type,
                         "gt_quality_flag": flags.get(pair.pair_id, 0), "kernel_config": kname, "ref_blur_id": rb,
                         "tgt_blur_id": tb, "method": meth, "processing_config": pc, "ref_image": rk, "tgt_image": tk,
                         "detector": det, "detector_config": "default",
                         "ref_nafnet_output_flag": ref_imgs[rk][3], "tgt_nafnet_output_flag": tgt_imgs[tk][3]}
                    r.update({f"ref_{k}": v for k, v in ref_im[rk].items()})
                    r.update({f"tgt_{k}": v for k, v in tgt_im[tk].items()})
                    r.update(met)
                    r.update(processing_device=",".join(sorted({d for d in (ref_imgs[rk][2], tgt_imgs[tk][2]) if d})),
                             runtime_processing_ref_ms=ref_imgs[rk][1], runtime_processing_tgt_ms=tgt_imgs[tk][1],
                             runtime_detect_ref_ms=fr.runtime_ms, runtime_detect_tgt_ms=ft.runtime_ms)
                    r["runtime_ms"] = (r["runtime_processing_ref_ms"] + r["runtime_processing_tgt_ms"] + fr.runtime_ms
                                       + ft.runtime_ms + met["runtime_match_ms"] + met["runtime_homography_ms"])
                    rows.append(r)
        print(f"{pair.pair_id} done ({time.time() - t_start:.0f}s, {len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", run_id)
    df.to_csv(out / "raw_results.csv", index=False)
    write_json(out / "run_meta.json", {
        "run_id": run_id, "keep_awake": awake, "nafnet_run": args.nafnet_run, "n_rows": len(df), "elapsed_s": round(time.time() - t_start, 1),
        "environment": environment_info(), "both_blur": ext["both_blur"],
        "timing_scope": "runtime_ms = processing of both images (Unsharp/Wiener CPU in-process; NAFNet GPU t_total_ms "
                        "from the manifest) + detectAndCompute on both images + match + homography. Feature times are "
                        "measured at first computation (images shared between rows are computed once).",
        "counts": {"/".join(k): int(v) for k, v in
                   df.groupby(["kernel_config", "method", "processing_config"]).size().items()}})
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
