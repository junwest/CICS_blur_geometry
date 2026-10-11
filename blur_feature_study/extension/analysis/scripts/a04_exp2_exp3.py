"""Exp2 (both images blurred) and Exp3 (Wiener kernel mismatch) + NAFNet output-check summary.

  analysis\\.venv\\Scripts\\python.exe extension/analysis/scripts/a04_exp2_exp3.py    (from blur_feature_study/)

Exp2: one cell per (pair, kernel config, detector, method, processing config); paired contrasts vs 'none'
(both blurred, unprocessed) over the 24 pairs (sequence cluster bootstrap). Context rows from Exp1 for
motion_l17_a0 with a Clean reference (BB and RR) are listed separately (different reference image).
Exp3: paired contrasts vs the exact kernel and vs BB (no processing) for the same 24 pairs.
Outputs: tables/T11_exp2.csv, T11b_exp2_contrasts.csv, T12_exp3.csv, T12b_exp3_contrasts.csv, T13_nafnet_checks.csv,
figures/F7_exp2.png, F8_exp3.png
"""
import numpy as np
import pandas as pd

from a00_common import (C4, DETECTORS, EST_OK, INK, INK2, MARKERS, METHOD_LABEL, METHODS, TABLES, load_main,
                        mcnemar_exact, paired_summary, read_csv, run_dir, save_new, setup_mpl)

PCS = ["none", "ref_only", "tgt_only", "both"]
PC_LABEL = {"none": "양쪽 무처리", "ref_only": "기준만 처리", "tgt_only": "대상만 처리", "both": "양쪽 처리"}


def prep(d):
    d["failure_reason"] = d["failure_reason"].fillna("")
    d["h_valid"] = d.homography_error_mean.notna()
    d["est_failed"] = ~d.failure_reason.isin(EST_OK)
    d["log10_herr"] = np.log10(d.homography_error_mean)
    return d


def paired(a, b, key):
    """a = treatment rows, b = baseline rows (one per key), both DataFrames."""
    j = a.set_index(key).join(b.set_index(key), lsuffix="_a", rsuffix="_b", how="inner")
    seq = j["sequence_id_a"].values
    ds = (j.homography_success_3px_a - j.homography_success_3px_b).values
    both = j.h_valid_a & j.h_valid_b
    dl = np.where(both, j.log10_herr_a - j.log10_herr_b, np.nan)
    dg = (j.gt_correct_matches_3px_a - j.gt_correct_matches_3px_b).values
    out = {"n": len(j), "succ_a": j.homography_success_3px_a.mean(), "succ_b": j.homography_success_3px_b.mean()}
    mc = mcnemar_exact(j.homography_success_3px_b, j.homography_success_3px_a)
    out.update(gained=mc["gained"], lost=mc["lost"], mcnemar_p_descriptive=mc["mcnemar_p"])
    for name, v in (("d_succ", ds), ("d_log", dl), ("d_gtc", dg)):
        s = paired_summary(v, seq)
        out.update({f"{name}_{k}": s.get(k, np.nan) for k in ("n", "mean", "ci_lo", "ci_hi")})
    return out


def summary(g):
    return {"n": len(g), "success3_n": int(g.homography_success_3px.sum()), "success3_rate": g.homography_success_3px.mean(),
            "est_fail_rate": g.est_failed.mean(), "herr_median_valid": g.homography_error_mean.median(),
            "gtc_mean": g.gt_correct_matches_3px.mean(), "gtc_hull_coverage_mean": g.gtc_hull_coverage.mean()}


def main():
    e2 = prep(read_csv(run_dir("bothblur") / "raw_results.csv"))
    e3 = prep(read_csv(run_dir("mismatch") / "raw_results.csv"))
    e1 = load_main()
    for c in ("ref_nafnet_output_flag", "tgt_nafnet_output_flag"):
        e2[c] = e2[c].astype(str).str.lower().eq("true")

    # ---------------- Exp2
    rows = []
    for kc in ("same", "diff"):
        for det in DETECTORS:
            g0 = e2[(e2.kernel_config == kc) & (e2.detector == det) & (e2.method == "none")]
            rows.append({"kernel_config": kc, "detector": det, "method": "none", "processing_config": "none", **summary(g0),
                         "ref_psnr_mean": g0.ref_psnr.mean(), "tgt_psnr_mean": g0.tgt_psnr.mean()})
            for meth in METHODS:
                for pc in PCS[1:]:
                    g = e2[(e2.kernel_config == kc) & (e2.detector == det) & (e2.method == meth) & (e2.processing_config == pc)]
                    rows.append({"kernel_config": kc, "detector": det, "method": meth, "processing_config": pc, **summary(g),
                                 "ref_psnr_mean": g.ref_psnr.mean(), "tgt_psnr_mean": g.tgt_psnr.mean(),
                                 "n_nafnet_flagged": int((g.ref_nafnet_output_flag | g.tgt_nafnet_output_flag).sum())})
    for det in DETECTORS:
        for stage, meth in (("BB", "none"), ("RR", "unsharp"), ("RR", "wiener"), ("RR", "nafnet")):
            g = e1[(e1.detector == det) & (e1.stage == stage) & (e1.method == meth) & (e1.blur_id == "motion_l17_a0")]
            rows.append({"kernel_config": "exp1_context_clean_ref", "detector": det, "method": meth,
                         "processing_config": f"{stage} (기준 Clean, 대상 motion_l17_a0)", **summary(g)})
    t11 = pd.DataFrame(rows)
    t11.to_csv(TABLES / "T11_exp2.csv", index=False)
    key = ["pair_id"]
    crow = []
    for kc in ("same", "diff"):
        for det in DETECTORS:
            base = e2[(e2.kernel_config == kc) & (e2.detector == det) & (e2.method == "none")]
            for meth in METHODS:
                sub = e2[(e2.kernel_config == kc) & (e2.detector == det) & (e2.method == meth)]
                for pc in PCS[1:]:
                    crow.append({"kernel_config": kc, "detector": det, "method": meth, "contrast": f"{pc}-none",
                                 **paired(sub[sub.processing_config == pc], base, key)})
                crow.append({"kernel_config": kc, "detector": det, "method": meth, "contrast": "both-tgt_only",
                             **paired(sub[sub.processing_config == "both"], sub[sub.processing_config == "tgt_only"], key)})
    t11b = pd.DataFrame(crow)
    t11b.to_csv(TABLES / "T11b_exp2_contrasts.csv", index=False)

    # ---------------- Exp3
    kord = ["exact", "len13", "len21", "ang05", "ang15"]
    rows = []
    for det in DETECTORS:
        bb = e1[(e1.detector == det) & (e1.stage == "BB") & (e1.blur_id == "motion_l17_a0")]
        rows.append({"detector": det, "supplied_kernel": "BB (무처리, Exp1)", **summary(bb), "psnr_mean": bb.psnr.mean(),
                     "ssim_mean": bb.ssim.mean(), "edge_f1_mean": bb.edge_f1.mean()})
        for k in kord:
            g = e3[(e3.detector == det) & (e3.supplied_kernel == k)]
            rows.append({"detector": det, "supplied_kernel": k, **summary(g), "psnr_mean": g.psnr.mean(),
                         "ssim_mean": g.ssim.mean(), "edge_f1_mean": g.edge_f1.mean(),
                         "edge_precision_mean": g.edge_precision.mean(), "edge_recall_mean": g.edge_recall.mean()})
    t12 = pd.DataFrame(rows)
    t12.to_csv(TABLES / "T12_exp3.csv", index=False)
    crow = []
    for det in DETECTORS:
        ex = e3[(e3.detector == det) & (e3.supplied_kernel == "exact")]
        bb = e1[(e1.detector == det) & (e1.stage == "BB") & (e1.blur_id == "motion_l17_a0")][
            ["pair_id", "sequence_id", "homography_success_3px", "h_valid", "log10_herr", "gt_correct_matches_3px"]]
        for k in kord[1:]:
            g = e3[(e3.detector == det) & (e3.supplied_kernel == k)]
            crow.append({"detector": det, "supplied_kernel": k, "vs": "exact", **paired(g, ex, key)})
        for k in kord:
            g = e3[(e3.detector == det) & (e3.supplied_kernel == k)]
            crow.append({"detector": det, "supplied_kernel": k, "vs": "BB", **paired(g, bb, key)})
    t12b = pd.DataFrame(crow)
    t12b.to_csv(TABLES / "T12b_exp3_contrasts.csv", index=False)

    # ---------------- NAFNet checks summary + Exp1 outcome by flag (descriptive)
    nc = read_csv(run_dir("nafnet_checks") / "nafnet_checks.csv")
    rows = []
    for role, g in nc.groupby("role"):
        for b, gb in g.groupby("blur_id"):
            rows.append({"role": role, "blur_id": b, "n": len(gb), "n_divergent": int(gb.divergence_flag.sum()),
                         "psnr_out_vs_in_median": gb.psnr_out_vs_in.median(),
                         "shift_nafnet_median_dx_nondiv": gb[~gb.divergence_flag].get("shift_nafnet_dx", pd.Series(dtype=float)).median(),
                         "shift_nafnet_median_dy_nondiv": gb[~gb.divergence_flag].get("shift_nafnet_dy", pd.Series(dtype=float)).median(),
                         "shift_nafnet_max_abs_nondiv": np.nanmax(np.abs(gb[~gb.divergence_flag][["shift_nafnet_dx", "shift_nafnet_dy"]].values))
                         if "shift_nafnet_dx" in gb and (~gb.divergence_flag).any() and gb.shift_nafnet_dx.notna().any() else np.nan,
                         "shift_wiener_max_abs": np.nanmax(np.abs(gb[["shift_wiener_dx", "shift_wiener_dy"]].values))
                         if "shift_wiener_dx" in gb and gb.shift_wiener_dx.notna().any() else np.nan,
                         "shift_blurred_max_abs": np.nanmax(np.abs(gb[["shift_blurred_dx", "shift_blurred_dy"]].values))
                         if "shift_blurred_dx" in gb and gb.shift_blurred_dx.notna().any() else np.nan})
    t13 = pd.DataFrame(rows)
    t13.to_csv(TABLES / "T13_nafnet_checks.csv", index=False)
    rr = e1[(e1.stage == "RR") & (e1.method == "nafnet")]
    bbx = e1[e1.stage == "BB"].set_index(["pair_id", "blur_id", "detector"])
    rr = rr.join(bbx[["homography_success_3px", "gt_correct_matches_3px"]], on=["pair_id", "blur_id", "detector"], rsuffix="_BB")
    flagtab = rr.groupby(["detector", "nafnet_output_flag"]).agg(
        n=("pair_id", "size"), success_RR=("homography_success_3px", "mean"), success_BB=("homography_success_3px_BB", "mean"),
        gtc_RR=("gt_correct_matches_3px", "mean"), gtc_BB=("gt_correct_matches_3px_BB", "mean")).reset_index()
    flagtab.to_csv(TABLES / "T13b_nafnet_flag_outcomes.csv", index=False)

    # ---------------- F7 Exp2
    plt = setup_mpl()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharey=True)
    for i, det in enumerate(DETECTORS):
        for k, kc in enumerate(("same", "diff")):
            ax = axes[i, k]
            n0 = t11[(t11.kernel_config == kc) & (t11.detector == det) & (t11.method == "none")].success3_rate.iloc[0] * 100
            for mi, meth in enumerate(METHODS):
                y = [n0] + [t11[(t11.kernel_config == kc) & (t11.detector == det) & (t11.method == meth)
                                & (t11.processing_config == pc)].success3_rate.iloc[0] * 100 for pc in PCS[1:]]
                ax.plot(range(4), y, marker=MARKERS[mi + 1], color=C4[mi], ms=7, lw=1.8, mec="#fcfcfb", mew=1,
                        label=METHOD_LABEL[meth] if (i == 0 and k == 0) else None)
            ctx = t11[(t11.kernel_config == "exp1_context_clean_ref") & (t11.detector == det)]
            bbv = ctx[ctx.method == "none"].success3_rate.iloc[0] * 100
            ax.axhline(bbv, color=INK2, lw=1, ls="--")
            ax.text(3.45, bbv + 1.5, f"Exp1 BB (기준 Clean)\n{bbv:.0f}%", fontsize=7, color=INK2, va="bottom")
            ax.set_xticks(range(4)); ax.set_xticklabels([PC_LABEL[p] for p in PCS])
            ax.set_xlim(-0.3, 4.1); ax.set_ylim(-3, 103); ax.grid(axis="x", visible=False)
            ttl = "같은 커널 (기준 17px 0°, 대상 17px 0°)" if kc == "same" else "다른 커널 (기준 17px 90°, 대상 17px 0°)"
            ax.set_title(f"{det}: {ttl}", color=INK, fontsize=9)
            if k == 0:
                ax.set_ylabel("Success@3px (% of 24 pairs)")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_new(fig, "F7_exp2.png")

    # ---------------- F8 Exp3
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    xs = ["len13", "exact", "len21", "exact", "ang05", "ang15"]
    xl = ["길이 13", "길이 17\n(정확)", "길이 21", "0°\n(정확)", "+5°", "+15°"]
    for ax, (col, lab, sc) in zip(axes, [("success3_rate", "Success@3px (%)", 100), ("gtc_mean", "GT-correct 매칭 수 (평균)", 1),
                                         ("psnr_mean", "PSNR (dB, 평균)", 1)]):
        for di, det in enumerate(DETECTORS):
            if col == "psnr_mean" and di == 1:
                continue
            g = t12[t12.detector == det].set_index("supplied_kernel")
            y = [g.loc[k, col] * sc for k in xs]
            lbl = det if col != "psnr_mean" else "영상 지표 (검출기 무관)"
            colr = C4[di] if col != "psnr_mean" else INK2
            ax.plot(range(3), y[:3], marker=MARKERS[di], color=colr, ms=7, lw=1.8, mec="#fcfcfb", mew=1, label=lbl)
            ax.plot(range(3, 6), y[3:], marker=MARKERS[di], color=colr, ms=7, lw=1.8, mec="#fcfcfb", mew=1)
            bbv = g.loc["BB (무처리, Exp1)", col] * sc
            ax.axhline(bbv, color=colr, lw=1, ls="--")
            ax.text(5.35, bbv, "BB", fontsize=7, color=colr, va="center")
        ax.set_xticks(range(6)); ax.set_xticklabels(xl, fontsize=8)
        ax.set_xlabel("길이 오류 (방향 0°)          방향 오류 (길이 17)", fontsize=8, color=INK2)
        ax.axvline(2.5, color=GRIDC, lw=1)
        ax.set_title(lab, color=INK, fontsize=9.5); ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Exp3: Wiener에 주는 커널만 부정확 (실제 블러 motion 17px 0°, balance 0.005 고정; 점선 = 무처리 BB)", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_new(fig, "F8_exp3.png")

    pd.set_option("display.width", 250)
    print(t11.round(3).to_string())
    print(t11b[["kernel_config", "detector", "method", "contrast", "n", "succ_a", "succ_b", "gained", "lost", "d_succ_mean",
                "d_succ_ci_lo", "d_succ_ci_hi", "d_log_mean", "d_log_ci_lo", "d_log_ci_hi", "d_log_n", "d_gtc_mean"]].round(3).to_string())
    print(t12.round(3).to_string())
    print(t12b[["detector", "supplied_kernel", "vs", "n", "succ_a", "succ_b", "gained", "lost", "d_succ_mean", "d_succ_ci_lo",
                "d_succ_ci_hi", "d_log_mean", "d_log_ci_lo", "d_log_ci_hi", "d_log_n", "d_gtc_mean"]].round(3).to_string())
    print(t13.round(3).to_string())
    print(flagtab.round(3).to_string())


GRIDC = "#e4e3df"

if __name__ == "__main__":
    main()
