"""Exp3: Wiener kernel mismatch (dependence on the exact kernel; NOT a kernel / balance search).

  python -m extension.src.run_mismatch   (from blur_feature_study/, .venv)

The blurred target is unchanged (true blur motion_l17_a0, base generation). The kernel GIVEN to the Wiener
filter is one of the pre-registered supplied kernels (exact, len13, len21, ang05, ang15); balance fixed at
the base value 0.005. Full pipeline (detectAndCompute), clean reference. Rows = 24 x 5 x 2 = 240.
The 'exact' rows must reproduce base group D for motion_l17_a0 (checked here, written to consistency.json).
"""
import time

import numpy as np
import pandas as pd

from .check_consistency import COLS, equal
from .common import keep_awake, BASE_MAIN_RUN, blur_cond, environment_info, load_base, load_ext_config, new_run_dir, test_pairs, write_json
from .evaluate import image_metrics, match_and_estimate, repeatability
from .kernels import supplied_kernel
from .run_main import DETECTORS
from .spatial import SpatialFrame
from src import features as F
from src.dataset import load_pair
from src.imaging import apply_blur, wiener_restore
from src.psf import make_psf


def kernel_info(k, k_true):
    yy, xx = np.mgrid[:k.shape[0], :k.shape[1]]
    c = (k.shape[0] - 1) / 2
    info = {"kernel_h": k.shape[0], "kernel_w": k.shape[1], "kernel_sum": float(k.sum()),
            "kernel_centroid_dx": float((k * (xx - c)).sum()), "kernel_centroid_dy": float((k * (yy - c)).sum())}
    if k.shape == k_true.shape:
        info["kernel_max_abs_diff_vs_true"] = float(np.abs(k - k_true).max())
    return info


def main():
    awake = keep_awake()
    cfg, sel, man, flags = load_base()
    ext = load_ext_config()
    km = ext["kernel_mismatch"]
    true_id = km["true_blur"]
    bal = float(km["balance"])
    if bal != float(sel["wiener_balance"][true_id]):
        raise RuntimeError("pre-registered balance differs from the base selected value")
    run_id, out = new_run_dir("ext_mismatch")
    m = int(cfg["preprocessing"]["valid_margin_px"])
    psf_true = make_psf(blur_cond(cfg, true_id), cfg)
    kernels = {name: supplied_kernel(v["length"], v["angle_deg"]) for name, v in km["supplied_kernels"].items()}
    if not np.array_equal(kernels["exact"], psf_true):
        raise RuntimeError("'exact' supplied kernel differs from the generation PSF")
    for name, k in kernels.items():
        np.save(out / f"kernel_{name}.npy", k)
    rows, ref_cache = [], {}
    t_start = time.time()
    for _, prow in test_pairs(man).iterrows():
        pair = load_pair(prow, cfg)
        sp = SpatialFrame(pair.H_gt, pair.ref.shape, pair.tgt.shape, m)
        mask_t = F.border_mask(pair.tgt.shape, m)
        B = apply_blur(pair.tgt, psf_true)
        clean_t = {det: F.detect(det, cfg["detectors"][det]["default"], pair.tgt, mask_t) for det in DETECTORS}
        for name, k in kernels.items():
            t0 = time.perf_counter(); R = wiener_restore(B, k, bal); pms = (time.perf_counter() - t0) * 1000.0
            im = image_metrics(cfg, pair.tgt, R)
            for det in DETECTORS:
                params = cfg["detectors"][det]["default"]
                key = (pair.sequence_id, pair.reference_index, det)
                if key not in ref_cache:
                    ref_cache[key] = F.detect(det, params, pair.ref, F.border_mask(pair.ref.shape, m))
                ft = F.detect(det, params, R, mask_t)
                met, _ = match_and_estimate(cfg, det, pair.H_gt, pair.ref.shape, pair.tgt.shape, ref_cache[key], ft, sp)
                r = {"pair_id": pair.pair_id, "sequence_id": pair.sequence_id, "sequence_type": pair.sequence_type,
                     "gt_quality_flag": flags.get(pair.pair_id, 0), "blur_id": true_id, "supplied_kernel": name,
                     "supplied_length": km["supplied_kernels"][name]["length"],
                     "supplied_angle_deg": km["supplied_kernels"][name]["angle_deg"], "wiener_balance": bal,
                     "detector": det, "detector_config": "default", "image_h": pair.tgt.shape[0], "image_w": pair.tgt.shape[1],
                     **kernel_info(k, psf_true), **im, **repeatability(cfg, clean_t[det], ft), **met,
                     "processing_device": "cpu", "runtime_processing_ms": pms, "runtime_detect_ms": ft.runtime_ms}
                r["runtime_ms"] = pms + ft.runtime_ms + met["runtime_match_ms"] + met["runtime_homography_ms"]
                rows.append(r)
        print(f"{pair.pair_id} done ({time.time() - t_start:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", run_id)
    df.to_csv(out / "raw_results.csv", index=False)
    # consistency: exact kernel == base D rows for the true blur
    new = pd.read_csv(out / "raw_results.csv", float_precision="round_trip")
    base = pd.read_csv(BASE_MAIN_RUN / "raw_results.csv", float_precision="round_trip")
    b = base[(base.group == "D") & (base.blur_id == true_id) & (base.detector_config == "default")]
    n = new[new.supplied_kernel == "exact"]
    mg = n.merge(b, on=["pair_id", "detector"], suffixes=("_new", "_base"))
    mism = {c: int((~equal(mg[f"{c}_new"], mg[f"{c}_base"])).sum()) for c in COLS}
    cons = {"n_new": len(n), "n_base": len(b), "n_merged": len(mg), "mismatch_counts": mism,
            "all_equal": len(mg) == len(n) == len(b) and not any(mism.values())}
    write_json(out / "consistency.json", cons)
    write_json(out / "run_meta.json", {
        "run_id": run_id, "keep_awake": awake, "n_rows": len(df), "elapsed_s": round(time.time() - t_start, 1),
        "environment": environment_info(), "kernel_mismatch": km,
        "kernels": {k: kernel_info(v, psf_true) for k, v in kernels.items()},
        "timing_scope": "runtime_ms = Wiener (CPU, in-process) + detectAndCompute + match + homography",
        "exact_vs_base_D_all_equal": cons["all_equal"]})
    print("exact vs base D:", cons["all_equal"], {k: v for k, v in mism.items() if v})
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
