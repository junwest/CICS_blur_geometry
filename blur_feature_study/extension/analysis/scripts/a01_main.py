"""Exp1 main analysis: overall table, BB/BR/RB/RR paired contrasts, sensitivity, image metrics.

  analysis\\.venv\\Scripts\\python.exe extension/analysis/scripts/a01_main.py      (from blur_feature_study/)

Outputs: tables/T1_overall.{csv,md}, T1b_success_by_blur.csv, T2_contrasts.{csv,md}, T3_sensitivity.csv,
T4_image_metrics.csv, stats/main_stats.json, figures/F1_stage_contrasts.png, F2_success_by_blur.png
"""
import json

import numpy as np
import pandas as pd

from a00_common import (BLUR_LABEL, BLUR_ORDER, C4, CONTRASTS, CONTRASTS_ALL, DETECTORS, FIGS, INK, INK2, MARKERS,
                        METHOD_LABEL, METHODS, STATS, TABLES, contrast_stats, fmt_ci, holm, load_main, read_csv,
                        run_dir, save_new, sensitivity_sets, setup_mpl)


def overall(df, im):
    rows = []
    conds = [("CLEAN", "clean"), ("BB", "none")] + [(s, m) for m in METHODS for s in ("BR", "RB", "RR")]
    for det in DETECTORS:
        for stage, meth in conds:
            g = df[(df.detector == det) & (df.stage == stage) & (df.method == meth)]
            r = {"detector": det, "stage": stage, "method": meth, "n_cells": len(g),
                 "success3_n": int(g.homography_success_3px.sum()), "success3_rate": g.homography_success_3px.mean(),
                 "success5_rate": g.homography_success_5px.mean(), "est_fail_rate": g.est_failed.mean(),
                 "h_invalid_or_failed_rate": (~g.h_valid).mean(), "n_valid_h": int(g.h_valid.sum()),
                 "herr_median_valid": g.homography_error_mean.median(),
                 "gtc_mean": g.gt_correct_matches_3px.mean(), "gtc_median": g.gt_correct_matches_3px.median(),
                 "match_precision_mean": g.match_precision_3px.mean(), "target_kp_mean": g.target_keypoints.mean(),
                 "repeatability_mean": g.keypoint_repeatability.mean(),
                 "inlier_gt_incorrect_mean": (g.ransac_inliers - g.ransac_inliers_gt_correct_3px).mean(),
                 "gtc_hull_coverage_mean": g.gtc_hull_coverage.mean(), "gtc_cell_occupancy_mean": g.gtc_cell_occupancy.mean(),
                 "runtime_processing_ms_median": g.runtime_processing_ms.median(),
                 "processing_device": ",".join(sorted(set(g.processing_device.dropna().astype(str)) - {""})),
                 "runtime_total_ms_median": g.runtime_ms.median()}
            if stage in ("BB", "RR"):
                imm = im[im.method == meth]
                for c in ("psnr", "ssim", "edge_precision", "edge_recall", "edge_f1"):
                    r[f"img_{c}_mean"] = imm[c].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def overall_md(t):
    lines = ["| 검출기 | 조건 | Success@3 | 추정 실패율 | H 오차 중앙값 (유효 H) | GT-correct 평균 | 매칭 정밀도 | "
             "hull coverage | 4×4 점유율 | PSNR | SSIM | Edge F1 | 처리 시간 중앙값 ms (장치) |", "|" + "---|" * 13]
    for _, r in t.iterrows():
        name = "Clean" if r.stage == "CLEAN" else ("BB (무처리)" if r.stage == "BB" else f"{r.stage} {METHOD_LABEL[r.method]}")
        img = (f"{r.img_psnr_mean:.2f} | {r.img_ssim_mean:.3f} | {r.img_edge_f1_mean:.3f}" if r.stage in ("BB", "RR")
               else ("∞ | 1 | 1" if r.stage == "CLEAN" else "– | – | –"))
        dev = f" ({r.processing_device})" if r.processing_device else ""
        lines.append(f"| {r.detector} | {name} | {r.success3_n}/{r.n_cells} ({100 * r.success3_rate:.1f}%) | "
                     f"{100 * r.est_fail_rate:.1f}% | {r.herr_median_valid:.2f} (n={r.n_valid_h}) | {r.gtc_mean:.1f} | "
                     f"{r.match_precision_mean:.3f} | {r.gtc_hull_coverage_mean:.3f} | {r.gtc_cell_occupancy_mean:.3f} | "
                     f"{img} | {r.runtime_processing_ms_median:.1f}{dev} |")
    return "\n".join(lines)


def main():
    df = load_main()
    im = read_csv(run_dir("main") / "image_metrics.csv")
    im["nafnet_output_flag"] = im.nafnet_output_flag.astype(str).str.lower().eq("true")
    gt_flagged, clean_fail = sensitivity_sets(df)

    # ---------------- T1 overall
    t1 = overall(df, im)
    t1.to_csv(TABLES / "T1_overall.csv", index=False)
    (TABLES / "T1_overall.md").write_text(overall_md(t1) + "\n", encoding="utf-8")
    sb = (df[df.stage != "CLEAN"].groupby(["detector", "stage", "method", "blur_id"]).homography_success_3px.sum()
          .unstack("blur_id")[BLUR_ORDER])
    sb.to_csv(TABLES / "T1b_success_by_blur.csv")

    # ---------------- T2 contrasts (pair-level, cluster bootstrap)
    rows = []
    for det in DETECTORS:
        for meth in METHODS:
            for x, y in CONTRASTS_ALL:
                s, _ = contrast_stats(df, det, meth, x, y)
                rows.append(s)
    t2 = pd.DataFrame(rows)
    for c in ("d_succ", "d_log", "d_gtc"):
        t2[f"{c}_wilcoxon_p_holm"] = holm(t2[f"{c}_wilcoxon_p"].values)
    t2.to_csv(TABLES / "T2_contrasts.csv", index=False)
    md = ["| 검출기 | 방법 | 대비 | ΔSuccess@3 (pp, 쌍 평균 [95% CI]) | gained/lost 셀 | Δlog10 H 오차 (공통 유효 셀; n쌍) | "
          "ΔGT-correct 수 | Δhull coverage |", "|" + "---|" * 8]
    for _, r in t2.iterrows():
        md.append(f"| {r.detector} | {METHOD_LABEL[r.method]} | {r.contrast} | "
                  f"{fmt_ci(r.d_succ_mean, r.d_succ_ci_lo, r.d_succ_ci_hi, 100, 1)} | {r.gained}/{r.lost} | "
                  f"{fmt_ci(r.d_log_mean, r.d_log_ci_lo, r.d_log_ci_hi, 1, 3)} (n={r.d_log_n}) | "
                  f"{fmt_ci(r.d_gtc_mean, r.d_gtc_ci_lo, r.d_gtc_ci_hi, 1, 1)} | "
                  f"{fmt_ci(r.d_cov_mean, r.d_cov_ci_lo, r.d_cov_ci_hi, 1, 3)} |")
    (TABLES / "T2_contrasts.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- T3 sensitivity
    srows = []
    for det in DETECTORS:
        for meth in METHODS:
            for x, y in CONTRASTS_ALL:
                variants = [("all", (), False), ("excl_gt_flagged", tuple(gt_flagged), False),
                            ("excl_clean_fail", tuple(clean_fail[det]), False),
                            ("excl_gt_flagged_and_clean_fail", tuple(set(gt_flagged) | set(clean_fail[det])), False)]
                if meth == "nafnet":
                    variants.append(("excl_nafnet_divergent_cells", (), True))
                for name, ex, fl in variants:
                    s, _ = contrast_stats(df, det, meth, x, y, exclude_pairs=ex, exclude_flagged_cells=fl)
                    s["variant"] = name
                    srows.append(s)
    t3 = pd.DataFrame(srows)
    t3.to_csv(TABLES / "T3_sensitivity.csv", index=False)

    # ---------------- T4 image metrics
    irows = []
    for meth in ["none"] + METHODS:
        for blur in BLUR_ORDER + ["pooled"]:
            g = im[im.method == meth] if blur == "pooled" else im[(im.method == meth) & (im.blur_id == blur)]
            r = {"method": meth, "blur_id": blur, "n": len(g), "n_nafnet_divergent": int(g.nafnet_output_flag.sum())}
            for c in ("psnr", "ssim", "edge_precision", "edge_recall", "edge_f1"):
                r[c] = g[c].mean()
                if meth == "nafnet":
                    r[f"{c}_excl_divergent"] = g[~g.nafnet_output_flag][c].mean()
            irows.append(r)
    t4 = pd.DataFrame(irows)
    t4.to_csv(TABLES / "T4_image_metrics.csv", index=False)

    stats = {"gt_flagged_pairs": gt_flagged, "clean_fail_pairs_3px": clean_fail,
             "n_rows": len(df), "rows_by_stage": df.groupby("stage").size().to_dict(),
             "nafnet_divergent_target_cells_per_detector": int(df[(df.method == "nafnet") & (df.stage == "RR") &
                                                                   (df.detector == "ORB")].nafnet_output_flag.sum())}
    (STATS / "main_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------- F1 paired-difference figure
    plt = setup_mpl()
    outcomes = [("d_succ", "ΔSuccess@3px (pp)", 100), ("d_log", "Δlog10 H 오차\n(공통 유효 셀; 음수 = 개선)", 1),
                ("d_gtc", "ΔGT-correct 매칭 수", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for i, det in enumerate(DETECTORS):
        for k, (c, lab, sc) in enumerate(outcomes):
            ax = axes[i, k]
            ax.axhline(0, color=INK2, lw=1)
            for mi, meth in enumerate(METHODS):
                for ci, (x, y) in enumerate(CONTRASTS):
                    r = t2[(t2.detector == det) & (t2.method == meth) & (t2.contrast == f"{x}-{y}")].iloc[0]
                    xp = mi + (ci - 1.5) * 0.18
                    m, lo, hi = r[f"{c}_mean"] * sc, r[f"{c}_ci_lo"] * sc, r[f"{c}_ci_hi"] * sc
                    ax.errorbar(xp, m, yerr=[[m - lo], [hi - m]], fmt=MARKERS[ci], color=C4[ci], ms=7, lw=1.6,
                                capsize=0, mec=SURF_EDGE, mew=1.5, label=f"{x}-{y}" if (mi == 0 and i == 0 and k == 0) else None)
            ax.set_xticks(range(3))
            ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(lab, fontsize=9.5, color=INK)
            if k == 0:
                ax.set_ylabel(det, fontsize=11, color=INK)
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               title="대비 (첫 글자 = 검출 영상, 둘째 = 기술자 영상; 점 = 24쌍 평균, 막대 = 시퀀스 클러스터 부트스트랩 95% CI)",
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_new(fig, "F1_stage_contrasts.png")

    # ---------------- F2 success by blur (BB, RR x 3, Clean reference)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    series = [("BB", "none", "#8a8984", "o", "BB (무처리)")] + [("RR", m, C4[i], MARKERS[i + 1], f"RR {METHOD_LABEL[m]}")
                                                             for i, m in enumerate(METHODS)]
    for ax, det in zip(axes, DETECTORS):
        cl = df[(df.detector == det) & (df.stage == "CLEAN")].homography_success_3px.mean()
        ax.axhline(100 * cl, color=INK, lw=1, ls="--")
        ax.text(len(BLUR_ORDER) - 0.5, 100 * cl, f"Clean {100 * cl:.0f}%", ha="right", va="bottom", color=INK2, fontsize=8)
        for stage, meth, col, mk, lab in series:
            g = df[(df.detector == det) & (df.stage == stage) & (df.method == meth)]
            y = [100 * g[g.blur_id == b].homography_success_3px.mean() for b in BLUR_ORDER]
            ax.plot(range(len(BLUR_ORDER)), y, marker=mk, color=col, ms=6, lw=1.6, label=lab, mec=SURF_EDGE, mew=1)
        ax.set_xticks(range(len(BLUR_ORDER)))
        ax.set_xticklabels([BLUR_LABEL[b] for b in BLUR_ORDER], rotation=30, ha="right")
        ax.set_title(det, color=INK)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Success@3px (% of 24 pairs)")
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    save_new(fig, "F2_success_by_blur.png")
    print(overall_md(t1))
    print("\n".join(md))


SURF_EDGE = "#fcfcfb"

if __name__ == "__main__":
    main()
