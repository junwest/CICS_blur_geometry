"""NAFNet output checks (size / coordinate alignment / output-only divergence flag).

  python -m extension.src.nafnet_checks --nafnet-run ext_nafnet_<ts>   (from blur_feature_study/, .venv)

Per NAFNet output (240 images of the nafnet run):
  * input hash re-check against regenerated base images, output size == input size;
  * output-only indicators (NO Clean / matching information): PSNR(output, its own input), fraction of
    pixels with |output - input| > 64, inter-channel spread (from the manifest; R=G=B input);
    divergence flag = chan_spread_max >= 255 (some pixel with output channels at opposite extremes
    although the three input channels are identical) -- rule fixed in DECISIONS.md before any matching run;
  * global translation vs the clean image (gradient LS estimator copied from tests/test_imaging.py::_shift,
    margin 24): NAFNet(B) vs clean, with the controls Wiener(B) vs clean and B vs clean (the estimator has a
    content-dependent bias for heavy directional blur, ~0.07 px in the base tests), and NAFNet(clean) vs clean.
Writes a new run directory ext_nafchk_<ts>/ with nafnet_checks.csv and summary.json.
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd

from .common import keep_awake, blur_cond, environment_info, load_base, new_run_dir, pixel_sha, test_pairs, write_json, PROJECT_ROOT
from .run_main import NafnetCache
from src.dataset import load_pair
from src.imaging import apply_blur, wiener_restore
from src.psf import make_psf


def _shift(a, b, m=24):
    """Copied verbatim from tests/test_imaging.py::_shift (gradient least-squares translation of b relative to a,
    b(x) ~ a(x - s); iterative Gauss-Newton with cubic warping)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    gy, gx = np.gradient(a)
    sl = (slice(m, -m), slice(m, -m))
    gx, gy = gx[sl], gy[sl]
    A = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
    s = np.zeros(2)
    for _ in range(10):                      # iterative (Gauss-Newton) refinement
        bw = cv2.warpAffine(b, np.float64([[1, 0, s[0]], [0, 1, s[1]]]), b.shape[::-1],
                            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        d = (bw - a)[sl]
        step = np.linalg.solve(A, np.array([(gx * d).sum(), (gy * d).sum()]))
        s = s - step
        if np.hypot(*step) < 1e-4:
            break
    return s[0], s[1]


def out_vs_in(o, i):
    d = np.abs(o.astype(np.float64) - i.astype(np.float64))
    mse = (d ** 2).mean()
    return {"psnr_out_vs_in": float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
            "frac_absdiff_gt64": float((d > 64).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nafnet-run", required=True)
    args = ap.parse_args(argv)
    awake = keep_awake()
    cfg, sel, man, flags = load_base()
    naf = NafnetCache(args.nafnet_run)
    run_id, out = new_run_dir("ext_nafchk")
    rows = []
    t0 = time.time()
    for _, prow in test_pairs(man).iterrows():
        pair = load_pair(prow, cfg)
        iid = f"cleantgt__{pair.pair_id}"
        N0, *_ = naf.get(iid, pair.tgt)
        r = {"image_id": iid, "role": "clean_target", "pair_id": pair.pair_id, "sequence_id": pair.sequence_id,
             "blur_id": "none", "size_equal": N0.shape == pair.tgt.shape, **out_vs_in(N0, pair.tgt)}
        r["shift_nafnet_dx"], r["shift_nafnet_dy"] = _shift(pair.tgt, N0)
        rows.append(r)
        for cond in cfg["blur"]["conditions"]:
            psf = make_psf(cond, cfg)
            B = apply_blur(pair.tgt, psf)
            iid = f"tgt__{pair.pair_id}__{cond['blur_id']}"
            N, *_ = naf.get(iid, B)
            W = wiener_restore(B, psf, float(sel["wiener_balance"][cond["blur_id"]]))
            r = {"image_id": iid, "role": "blurred_target", "pair_id": pair.pair_id, "sequence_id": pair.sequence_id,
                 "blur_id": cond["blur_id"], "size_equal": N.shape == B.shape, **out_vs_in(N, B)}
            for tag, img in (("nafnet", N), ("wiener", W), ("blurred", B)):
                try:
                    r[f"shift_{tag}_dx"], r[f"shift_{tag}_dy"] = _shift(pair.tgt, img)
                except np.linalg.LinAlgError as e:
                    r[f"shift_{tag}_dx"] = r[f"shift_{tag}_dy"] = np.nan
                    r[f"shift_{tag}_error"] = str(e)
            rows.append(r)
    # blurred references (Exp2 inputs): hash/size/output-only indicators
    for seq in sorted(test_pairs(man).sequence_id.unique()):
        prow = test_pairs(man)[test_pairs(man).sequence_id == seq].iloc[0]
        pair = load_pair(prow, cfg)
        for bid in ("motion_l17_a0", "motion_l17_a90"):
            B = apply_blur(pair.ref, make_psf(blur_cond(cfg, bid), cfg))
            iid = f"ref__{seq}__{bid}"
            N, *_ = naf.get(iid, B)
            rows.append({"image_id": iid, "role": "blurred_reference", "pair_id": "", "sequence_id": seq, "blur_id": bid,
                         "size_equal": N.shape == B.shape, **out_vs_in(N, B)})
    df = pd.DataFrame(rows)
    mf = naf.man[["chan_spread_max", "chan_spread_mean", "pad_h", "pad_w", "h", "w"]]
    df = df.merge(mf, left_on="image_id", right_index=True, how="left")
    df["divergence_flag"] = df.chan_spread_max >= 255
    df.to_csv(out / "nafnet_checks.csv", index=False)
    bt = df[df.role == "blurred_target"]
    summ = {"run_id": run_id, "nafnet_run": args.nafnet_run, "n_images": len(df),
            "all_sizes_equal": bool(df.size_equal.all()),
            "divergence_rule": "chan_spread_max >= 255 (output-only; R=G=B input)",
            "n_divergent_by_role": df.groupby("role").divergence_flag.sum().astype(int).to_dict(),
            "n_divergent_blurred_target_by_blur": bt.groupby("blur_id").divergence_flag.sum().astype(int).to_dict(),
            "divergent_images": df[df.divergence_flag].image_id.tolist(),
            "psnr_out_vs_in_range_divergent": [float(df[df.divergence_flag].psnr_out_vs_in.min()),
                                               float(df[df.divergence_flag].psnr_out_vs_in.max())],
            "psnr_out_vs_in_range_not_divergent": [float(df[~df.divergence_flag].psnr_out_vs_in.min()),
                                                   float(df[~df.divergence_flag].psnr_out_vs_in.max())],
            "elapsed_s": round(time.time() - t0, 1), "environment": environment_info()}
    ok = bt[~bt.divergence_flag]
    for tag in ("nafnet", "wiener", "blurred"):
        summ[f"shift_{tag}_vs_clean_non_divergent_blurred_targets"] = {
            "median_abs_dx": float(ok[f"shift_{tag}_dx"].abs().median()), "median_abs_dy": float(ok[f"shift_{tag}_dy"].abs().median()),
            "max_abs_dx": float(ok[f"shift_{tag}_dx"].abs().max()), "max_abs_dy": float(ok[f"shift_{tag}_dy"].abs().max()),
            "by_blur_median_dx_dy": {b: [float(g[f"shift_{tag}_dx"].median()), float(g[f"shift_{tag}_dy"].median())]
                                     for b, g in ok.groupby("blur_id")}}
    ct = df[(df.role == "clean_target") & ~df.divergence_flag]
    summ["shift_nafnet_clean_vs_clean_non_divergent"] = {
        "n": len(ct), "max_abs_dx": float(ct.shift_nafnet_dx.abs().max()), "max_abs_dy": float(ct.shift_nafnet_dy.abs().max())}
    write_json(out / "summary.json", summ)
    print({k: v for k, v in summ.items() if k != "environment"})
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
