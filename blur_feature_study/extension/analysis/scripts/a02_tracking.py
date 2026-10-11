"""Tracking diagnostics (uses Clean + GT; NOT matching performance).

  analysis\\.venv\\Scripts\\python.exe extension/analysis/scripts/a02_tracking.py     (from blur_feature_study/)

Outputs: tables/T5_states.csv, T6_links.csv, T6b_links_by_blur.csv, T6c_paired_displacement.csv,
T7_fixed_landmark_h.csv, T8_oracle_h.csv, T9_transitions.csv, stats/tracking_stats.json,
figures/F3_states.png, F4_displacement.png, F5_fixed_landmark_vs_pipeline.png
Pooled fractions = sum of counts / sum of reference-set sizes over all cells; CIs = pair-level means with
sequence cluster bootstrap (95 %, 5000, seed 42).
"""
import json

import numpy as np
import pandas as pd

from a00_common import (BLUR_ORDER, C4, DETECTORS, INK, INK2, MARKERS, METHOD_LABEL, METHODS, STATS, TABLES,
                        cluster_bootstrap_ci, load_main, paired_summary, read_csv, run_dir, save_new, setup_mpl)

STATE_GROUPS = {"정답 매칭 (연결된 위치)": ["correct_via_linked"],
                "정답 매칭 (다른 키포인트)": ["correct_via_other", "lost_but_correct"],
                "재검출, 정답 매칭 아님": ["redetected_no_match", "redetected_wrong_match"],
                "재검출 안 됨 (3 px)": ["not_redetected"]}
DIMG = ["B", "unsharp", "wiener", "nafnet"]
DLAB = {"B": "B (무처리)", "unsharp": "Unsharp", "wiener": "Wiener", "nafnet": "NAFNet"}
DCOL = {"B": "#8a8984", "unsharp": C4[0], "wiener": C4[1], "nafnet": C4[2]}


def pair_ci(g, num, den):
    pl = g.groupby("pair_id").agg(seq=("sequence_id", "first"), n=(num, "sum"), d=(den, "sum"))
    v = (pl.n / pl.d).values
    lo, hi = cluster_bootstrap_ci(v[np.isfinite(v)], pl.seq.values[np.isfinite(v)])
    return float(np.nanmean(v)), lo, hi


def main():
    tr = run_dir("tracking")
    st = read_csv(tr / "tracking_states.csv")
    lk = read_csv(tr / "tracking_links.csv")
    lm = read_csv(tr / "fixed_landmark_h.csv")
    tt = read_csv(tr / "tracking_transitions.csv")
    disp = pd.read_csv(tr / "displacements.csv.gz")
    df = load_main()
    out = {}

    # ---------------- T5 states
    conds = [("BB", "none")] + [(s, m) for m in METHODS for s in ("BR", "RB", "RR")]
    rows = []
    for det in DETECTORS:
        for stage, meth in conds:
            g = st[(st.detector == det) & (st.stage == stage) & (st.method == meth)]
            n = g.n_reference_set.sum()
            r = {"detector": det, "stage": stage, "method": meth, "n_cells": len(g), "reference_elements": int(n)}
            for s in ["correct_via_linked", "correct_via_other", "lost_but_correct", "not_redetected",
                      "redetected_no_match", "redetected_wrong_match"]:
                r[f"frac_{s}"] = g[s].sum() / n
            r["frac_correct_in_set"] = g.n_correct_in_set.sum() / n
            m_, lo, hi = pair_ci(g, "n_correct_in_set", "n_reference_set")
            r.update(pair_mean_frac_correct=m_, ci_lo=lo, ci_hi=hi)
            r["unique_linked_correct_rate"] = g.n_unique_linked_correct.sum() / g.n_unique_linked.sum()
            r["ambiguous_linked_correct_rate"] = g.n_ambiguous_linked_correct.sum() / max(g.n_ambiguous_linked.sum(), 1)
            r["outside_set_correct_mean"] = g.n_outside_set_correct.mean()
            r["outside_set_correct_linked_mean"] = g.n_outside_set_correct_linked_to_clean.mean()
            r["outside_set_correct_unlinked_mean"] = g.n_outside_set_correct_unlinked.mean()
            r["outside_share_of_all_correct"] = g.n_outside_set_correct.sum() / max(g.n_gt_correct_total.sum(), 1)
            r["gt_correct_total_mean"] = g.n_gt_correct_total.mean()
            r["ransac_inliers_gt_incorrect_mean"] = g.n_ransac_inliers_gt_incorrect.mean()
            r["ransac_inlier_gt_incorrect_share"] = g.n_ransac_inliers_gt_incorrect.sum() / max(g.n_ransac_inliers.sum(), 1)
            rows.append(r)
    t5 = pd.DataFrame(rows)
    t5.to_csv(TABLES / "T5_states.csv", index=False)

    # ---------------- T6 links / displacement (per detection image)
    rows = []
    for det in DETECTORS:
        for D in DIMG:
            g = lk[(lk.detector == det) & (lk.detection_image == D)]
            dd = disp[(disp.detector == det) & (disp.detection_image == D)]
            n = g.n_reference_set.sum()
            r = {"detector": det, "detection_image": D, "n_cells": len(g),
                 "frac_ref_unique": g.n_ref_unique.sum() / n, "frac_ref_ambiguous": g.n_ref_ambiguous.sum() / n,
                 "frac_ref_lost": g.n_ref_lost.sum() / n,
                 "refloc_linked_rate_mean": g.refloc_linked_rate.mean(), "chance_linked_rate_mean": g.chance_linked_rate.mean(),
                 "refloc_unique_rate_mean": g.refloc_unique_rate.mean(), "chance_unique_rate_mean": g.chance_unique_rate.mean(),
                 "condition_keypoints_mean": g.n_condition_keypoints.mean(),
                 "frac_condition_locations_unlinked_mean": g.frac_condition_locations_unlinked_to_clean.mean(),
                 "n_unique_links": len(dd), "mean_dx": dd.dx.mean(), "mean_dy": dd.dy.mean(), "std_dx": dd.dx.std(),
                 "std_dy": dd.dy.std(), "median_absd": dd.absd.median(), "q25_absd": dd.absd.quantile(0.25),
                 "q75_absd": dd.absd.quantile(0.75), "q90_absd": dd.absd.quantile(0.9), "frac_absd_le1": (dd.absd <= 1).mean(),
                 "frac_absd_le0.5": (dd.absd <= 0.5).mean(), "median_size_ratio": dd.size_ratio.median(),
                 "frac_same_octave": (dd.octave_clean == dd.octave_cond).mean(),
                 "median_abs_angle_diff": dd.abs_angle_diff.median()}
            rows.append(r)
    t6 = pd.DataFrame(rows)
    t6.to_csv(TABLES / "T6_links.csv", index=False)
    rows = []
    for det in DETECTORS:
        for D in DIMG:
            for b in BLUR_ORDER:
                g = lk[(lk.detector == det) & (lk.detection_image == D) & (lk.blur_id == b)]
                dd = disp[(disp.detector == det) & (disp.detection_image == D) & (disp.blur_id == b)]
                n = g.n_reference_set.sum()
                rows.append({"detector": det, "detection_image": D, "blur_id": b, "frac_ref_unique": g.n_ref_unique.sum() / n,
                             "frac_ref_ambiguous": g.n_ref_ambiguous.sum() / n, "frac_ref_lost": g.n_ref_lost.sum() / n,
                             "chance_linked_rate_mean": g.chance_linked_rate.mean(), "n_unique_links": len(dd),
                             "mean_dx": dd.dx.mean(), "mean_dy": dd.dy.mean(), "std_dx": dd.dx.std(), "std_dy": dd.dy.std(),
                             "median_absd": dd.absd.median(), "frac_absd_le1": (dd.absd <= 1).mean()})
    pd.DataFrame(rows).to_csv(TABLES / "T6b_links_by_blur.csv", index=False)

    # ---------------- T6c paired displacement on the common subset (uniquely linked in both B and R detection)
    rows = []
    for det in DETECTORS:
        for meth in METHODS:
            g = lm[(lm.detector == det) & (lm.method == meth) & (lm.n_landmarks > 0)].copy()
            g["d"] = g.mean_absd_R - g.mean_absd_B
            pl = g.groupby("pair_id").agg(seq=("sequence_id", "first"), d=("d", "mean"))
            s = paired_summary(pl.d.values, pl.seq.values)
            rows.append({"detector": det, "method": meth, "n_cells_with_common": len(g), "n_elements": int(g.n_landmarks.sum()),
                         "n_R_closer": int(g.n_R_closer.sum()), "n_R_farther": int(g.n_R_farther.sum()),
                         "n_equal": int(g.n_equal.sum()), "frac_R_closer": g.n_R_closer.sum() / g.n_landmarks.sum(),
                         "pair_mean_delta_mean_absd": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n_pairs": s["n"],
                         "mean_absd_B_cellavg": g.mean_absd_B.mean(), "mean_absd_R_cellavg": g.mean_absd_R.mean()})
    t6c = pd.DataFrame(rows)
    t6c.to_csv(TABLES / "T6c_paired_displacement.csv", index=False)

    # ---------------- T7 fixed-landmark H vs pipeline
    rows = []
    for det in DETECTORS:
        for meth in METHODS:
            g = lm[(lm.detector == det) & (lm.method == meth)].copy()
            a = g[g.lm_analyzable]
            r = {"detector": det, "method": meth, "n_cells": len(g), "n_analyzable": len(a),
                 "frac_analyzable": len(a) / len(g), "landmarks_median_analyzable": a.n_landmarks.median(),
                 "reason_counts_unanalyzable": g[~g.lm_analyzable].lm_h_reason_B.value_counts().to_dict()}
            for pos in ("clean", "B", "R"):
                r[f"success3_{pos}"] = a[f"lm_h_success_3px_{pos}"].mean()
                r[f"herr_median_{pos}"] = a[f"lm_h_err_mean_{pos}"].median()
            a = a.merge(df[(df.detector == det) & (df.stage == "BB")][["pair_id", "blur_id", "log10_herr", "homography_success_3px"]]
                        .rename(columns={"log10_herr": "pipe_log_BB", "homography_success_3px": "pipe_s_BB"}), on=["pair_id", "blur_id"])
            a = a.merge(df[(df.detector == det) & (df.stage == "RR") & (df.method == meth)][["pair_id", "blur_id", "log10_herr", "homography_success_3px"]]
                        .rename(columns={"log10_herr": "pipe_log_RR", "homography_success_3px": "pipe_s_RR"}), on=["pair_id", "blur_id"])
            a["d_lm"] = np.log10(a.lm_h_err_mean_R) - np.log10(a.lm_h_err_mean_B)
            a["d_pipe"] = a.pipe_log_RR - a.pipe_log_BB
            for c in ("d_lm", "d_pipe"):
                pl = a.groupby("pair_id").agg(seq=("sequence_id", "first"), v=(c, "mean"))
                s = paired_summary(pl.v.values, pl.seq.values)
                r.update({f"{c}_mean": s["mean"], f"{c}_ci_lo": s["ci_lo"], f"{c}_ci_hi": s["ci_hi"], f"{c}_n_pairs": s["n"]})
            r["pipe_success_BB_on_analyzable"] = a.pipe_s_BB.mean()
            r["pipe_success_RR_on_analyzable"] = a.pipe_s_RR.mean()
            rows.append(r)
    t7 = pd.DataFrame(rows)
    t7.to_csv(TABLES / "T7_fixed_landmark_h.csv", index=False)

    # ---------------- T8 oracle-inlier H
    rows = []
    for det in DETECTORS:
        for stage, meth in [("CLEAN", "clean")] + conds:
            g = st[(st.detector == det) & (st.stage == stage) & (st.method == meth)]
            p = df[(df.detector == det) & (df.stage == stage) & (df.method == meth)]
            ok = g.oracle_h_reason.fillna("").isin(["", "projection_error_gt_3px", "projection_error_gt_5px"])
            rows.append({"detector": det, "stage": stage, "method": meth, "n_cells": len(g),
                         "frac_oracle_analyzable_ge8": (g.n_oracle_points >= 8).mean(),
                         "oracle_success3": g.oracle_h_success_3px.mean(), "pipeline_success3": p.homography_success_3px.mean(),
                         "oracle_herr_median_valid": g[ok].oracle_h_err_mean.median(),
                         "cells_oracle_success_pipeline_fail": int(((g.set_index(["pair_id", "blur_id"]).oracle_h_success_3px == 1)
                                                                    & (p.set_index(["pair_id", "blur_id"]).homography_success_3px == 0)).sum()),
                         "cells_oracle_fail": int((g.oracle_h_success_3px == 0).sum())})
    t8 = pd.DataFrame(rows)
    t8.to_csv(TABLES / "T8_oracle_h.csv", index=False)

    # ---------------- T9 transitions of reference elements
    rows = []
    stn = st[st.stage == "BB"][["pair_id", "blur_id", "detector", "n_reference_set"]]
    tt2 = tt.merge(stn, on=["pair_id", "blur_id", "detector"])
    for det in DETECTORS:
        for meth in METHODS:
            for c in ["BR-BB", "RB-BB", "RR-RB", "RR-BR", "RR-BB"]:
                g = tt2[(tt2.detector == det) & (tt2.method == meth) & (tt2.contrast == c)].copy()
                g["net"] = (g.gained - g.lost) / g.n_reference_set
                pl = g.groupby("pair_id").agg(seq=("sequence_id", "first"), v=("net", "mean"))
                s = paired_summary(pl.v.values, pl.seq.values)
                n = g.n_reference_set.sum()
                rows.append({"detector": det, "method": meth, "contrast": c, "frac_gained": g.gained.sum() / n,
                             "frac_lost": g.lost.sum() / n, "frac_kept": g.kept.sum() / n, "frac_never": g.never.sum() / n,
                             "pair_mean_net": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"]})
    t9 = pd.DataFrame(rows)
    t9.to_csv(TABLES / "T9_transitions.csv", index=False)

    out["fixed_landmark_unanalyzable_reasons"] = {f"{r.detector}/{r.method}": r.reason_counts_unanalyzable for r in t7.itertuples()}
    (STATS / "tracking_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------- F3 state composition
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    labels = ["BB"] + [f"{s}\n{METHOD_LABEL[m].split(' ')[0]}" for m in METHODS for s in ("BR", "RB", "RR")]
    xs = [0] + [1 + mi * 3.4 + si for mi in range(3) for si in range(3)]
    for ax, det in zip(axes, DETECTORS):
        bottom = np.zeros(len(conds))
        for gi, (gname, cols) in enumerate(STATE_GROUPS.items()):
            vals = np.array([t5[(t5.detector == det) & (t5.stage == s) & (t5.method == m)][[f"frac_{c}" for c in cols]].sum(axis=1).iloc[0]
                             for s, m in conds]) * 100
            ax.bar(xs, vals, bottom=bottom, color=C4[gi], width=0.82, edgecolor="#fcfcfb", linewidth=2,
                   label=gname if det == "ORB" else None)
            bottom += vals
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{det}: Clean 기준 집합(GT ≤ 3 px)의 상태", color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("기준 집합 대비 비율 (%, 전체 셀 합산)")
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_new(fig, "F3_states.png")

    # ---------------- F4 displacement ECDF + link status vs chance
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    grid = np.linspace(0, 3, 301)
    for i, det in enumerate(DETECTORS):
        ax = axes[0, i]
        for D in DIMG:
            v = np.sort(disp[(disp.detector == det) & (disp.detection_image == D)].absd.values)
            ax.plot(grid, np.searchsorted(v, grid, side="right") / len(v), color=DCOL[D], lw=2, label=f"{DLAB[D]} (n={len(v):,})")
        ax.set_xlim(0, 3); ax.set_ylim(0, 1)
        ax.set_xlabel("Clean 대비 검출 위치 변화 |d| (px; unique 연결만, 3 px에서 절단)")
        ax.set_ylabel("누적 비율")
        ax.set_title(f"{det}: 위치 변화 분포 (8개 블러 합산)", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, i]
        x = np.arange(len(DIMG))
        tg = t6[t6.detector == det].set_index("detection_image").loc[DIMG]
        b1 = tg.frac_ref_unique.values * 100
        b2 = tg.frac_ref_ambiguous.values * 100
        ax.bar(x, b1, color="#52514e", width=0.6, edgecolor="#fcfcfb", linewidth=2, label="unique 연결")
        ax.bar(x, b2, bottom=b1, color="#b9b8b2", width=0.6, edgecolor="#fcfcfb", linewidth=2, label="ambiguous 연결")
        ax.scatter(x, tg.chance_linked_rate_mean.values * 100, marker="_", s=900, color="#e34948", lw=2.5, zorder=5,
                   label="chance (±20 px 이동 위치의 연결률)")
        for xi, (u, a_) in enumerate(zip(b1, b2)):
            ax.text(xi, u + a_ + 1.5, f"{u + a_:.0f}%", ha="center", color=INK2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([DLAB[d] for d in DIMG])
        ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
        ax.set_ylabel("기준 집합 중 3 px 안 재검출 비율 (%)")
        ax.set_title(f"{det}: 재검출(연결) 상태", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_new(fig, "F4_displacement.png")

    # ---------------- F5 fixed-landmark vs pipeline
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, det in zip(axes, DETECTORS):
        ax.axhline(0, color=INK2, lw=1)
        for mi, meth in enumerate(METHODS):
            r = t7[(t7.detector == det) & (t7.method == meth)].iloc[0]
            for k, (c, lab) in enumerate([("d_lm", "고정 랜드마크 H: R 위치 - B 위치"), ("d_pipe", "전체 파이프라인: RR - BB")]):
                xp = mi + (k - 0.5) * 0.25
                m_, lo, hi = r[f"{c}_mean"], r[f"{c}_ci_lo"], r[f"{c}_ci_hi"]
                ax.errorbar(xp, m_, yerr=[[m_ - lo], [hi - m_]], fmt=MARKERS[k], color=C4[k], ms=7, lw=1.6,
                            mec="#fcfcfb", mew=1.5, label=lab if (mi == 0 and det == "ORB") else None)
        ax.set_xticks(range(3))
        tk = []
        for m in METHODS:
            r = t7[(t7.detector == det) & (t7.method == m)].iloc[0]
            tk.append(f"{METHOD_LABEL[m]}\n(분석 가능 셀 {r.frac_analyzable * 100:.0f}%, {int(r.d_lm_n_pairs)}쌍)")
        ax.set_xticklabels(tk, fontsize=8)
        ax.set_title(det if det == "SIFT" else "ORB (분석 가능 셀이 적어 해석하지 않음; y축 별도)", color=INK, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δlog10 H 오차 (음수 = 개선)\n분석 가능 셀의 쌍 평균 ± 95% CI")
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_new(fig, "F5_fixed_landmark_vs_pipeline.png")

    pd.set_option("display.width", 250)
    print(t5[["detector", "stage", "method", "frac_correct_in_set", "frac_correct_via_linked", "frac_correct_via_other",
              "frac_lost_but_correct", "frac_not_redetected", "frac_redetected_no_match", "frac_redetected_wrong_match",
              "outside_set_correct_mean", "outside_set_correct_unlinked_mean", "outside_share_of_all_correct",
              "ransac_inlier_gt_incorrect_share"]].round(3).to_string())
    print(t6.round(3).to_string())
    print(t6c.round(3).to_string())
    print(t7.drop(columns=["reason_counts_unanalyzable"]).round(3).to_string())
    print(t8.round(3).to_string())
    print(t9.round(3).to_string())


if __name__ == "__main__":
    main()
