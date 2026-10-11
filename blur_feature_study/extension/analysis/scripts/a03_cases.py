"""Cases: spatial distribution of correspondences together with H error (BB vs RR).

  analysis\\.venv\\Scripts\\python.exe extension/analysis/scripts/a03_cases.py      (from blur_feature_study/)

SELECTION RULE (fixed in this docstring before any case figure was drawn):
  per detector, candidate cells = all (pair, blur, method) of Exp1, methods pooled.
  * percentile cases: among cells where BOTH BB and RR have a valid H (common-valid subset), D = log10 err(RR)
    - log10 err(BB); take the cell nearest the 10th / 50th / 90th percentile of D (ties: pair_id, blur order,
    method order). 10th = improvement side, 50th = no change, 90th = worsening side.
  * one gain transition (BB Success@3 = 0 -> RR = 1) and one loss transition (1 -> 0): among those cells the one
    with the lower-median change in GT-correct matches (ties as above); "none" if no such cell exists.
  NAFNet-divergent cells are not excluded (flagged in the panel title).
Panels show the REFERENCE image (Clean) with the overlap polygon and the 4x4 eligible cells; dots = GT-correct
ratio-test matches (filled = also RANSAC inlier), crosses = RANSAC inliers that are GT-incorrect.
Output: figures/F6_cases_ORB.png, F6_cases_SIFT.png, tables/T10_cases.csv (with each case's percentile position).
"""
import sys

import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

from a00_common import (BLUR_LABEL, BLUR_ORDER, C4, DETECTORS, INK, INK2, METHOD_LABEL, METHODS, PROJECT, TABLES,
                        load_main, run_dir, save_new, setup_mpl)

sys.path.insert(0, str(PROJECT))
from src.io_utils import load_config  # noqa: E402
from src.dataset import load_manifest, load_pair  # noqa: E402
from extension.src.spatial import SpatialFrame  # noqa: E402

BORD = {b: i for i, b in enumerate(BLUR_ORDER)}
MORD = {m: i for i, m in enumerate(METHODS)}


def select(df, det):
    bb = df[(df.detector == det) & (df.stage == "BB")].set_index(["pair_id", "blur_id"])
    rows = []
    for meth in METHODS:
        rr = df[(df.detector == det) & (df.stage == "RR") & (df.method == meth)].set_index(["pair_id", "blur_id"])
        j = rr[["log10_herr", "homography_success_3px", "gt_correct_matches_3px", "nafnet_output_flag",
                "homography_error_mean"]].join(bb[["log10_herr", "homography_success_3px", "gt_correct_matches_3px",
                                                   "homography_error_mean"]], lsuffix="_RR", rsuffix="_BB")
        j["method"] = meth
        rows.append(j.reset_index())
    c = pd.concat(rows, ignore_index=True)
    c["D"] = c.log10_herr_RR - c.log10_herr_BB
    c["dgtc"] = c.gt_correct_matches_3px_RR - c.gt_correct_matches_3px_BB
    c["o"] = list(zip(c.pair_id, c.blur_id.map(BORD), c.method.map(MORD)))
    cv = c[c.D.notna()]
    picks = []
    for q in (10, 50, 90):
        t = np.percentile(cv.D, q)
        s = cv.assign(dist=(cv.D - t).abs()).sort_values(["dist", "o"]).iloc[0]
        picks.append((f"Δlog10 오차 {q}번째 백분위", s, q))
    for name, a, b in (("gain (BB 실패 → RR 성공)", 0, 1), ("loss (BB 성공 → RR 실패)", 1, 0)):
        g = c[(c.homography_success_3px_BB == a) & (c.homography_success_3px_RR == b)].sort_values(["dgtc", "o"])
        if len(g):
            picks.append((f"{name}; n={len(g)}", g.iloc[(len(g) - 1) // 2], None))
        else:
            picks.append((f"{name}: 해당 셀 없음", None, None))
    for i, (lab, s, q) in enumerate(picks):
        if s is not None and np.isfinite(s.D):
            picks[i] = (lab, s, float((cv.D < s.D).mean() * 100))
    return picks, len(cv), len(c)


def panel(ax, img, sp, ref_kp, z, tag, row, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.55)
    if sp.area > 0:
        p = np.vstack([sp.overlap, sp.overlap[:1]])
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=1.2)
    for i in range(sp.n):
        for j in range(sp.n):
            if sp.eligible[i, j]:
                ax.add_patch(Rectangle(
                    (sp.x0 + j * sp.cw, sp.y0 + i * sp.ch), sp.cw, sp.ch, fill=False, ec=INK2, lw=0.5, ls=":"))
    q, err, inl = z[f"query_idx_{tag}"], z[f"gt_err_{tag}"], z[f"ransac_inlier_{tag}"]
    xy = ref_kp[q, :2]
    good = err <= 3
    ax.scatter(*xy[good & ~inl].T, s=10, facecolors="none", edgecolors=C4[0], lw=0.8)
    ax.scatter(*xy[good & inl].T, s=10, color=C4[0], lw=0)
    ax.scatter(*xy[~good & inl].T, s=14, marker="x", color=C4[1], lw=0.9)
    err_s = "실패" if not np.isfinite(row.homography_error_mean) else f"{row.homography_error_mean:.2f} px"
    ax.set_title(f"{title}\nH 오차 {err_s} · GT-correct {int(row.gt_correct_matches_3px)}\n"
                 f"coverage {row.gtc_hull_coverage:.2f} · 점유 {row.gtc_cell_occupancy:.2f}", fontsize=7.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def main():
    df = load_main()
    cfg = load_config()
    man = load_manifest(cfg).set_index("pair_id")
    src = run_dir("main") / "details"
    plt = setup_mpl()
    trows = []
    for det in DETECTORS:
        picks, n_cv, n_all = select(df, det)
        ratios = []
        for lab, s, pct in picks:
            if s is None:
                ratios.append(0.25)
            else:
                r_ = man.loc[s.pair_id]
                ratios.append(float(r_.reference_h) / float(r_.reference_w))
        fig, axes = plt.subplots(len(picks), 2, figsize=(9, sum(2.75 * x + 0.8 for x in ratios)),
                                 gridspec_kw={"height_ratios": [2.75 * x + 0.8 for x in ratios]})
        for r, (lab, s, pct) in enumerate(picks):
            if s is None:
                for ax in axes[r]:
                    ax.axis("off")
                axes[r, 0].text(0.5, 0.5, lab, ha="center", va="center", transform=axes[r, 0].transAxes)
                trows.append({"detector": det, "case": lab})
                continue
            pair = load_pair(man.loc[[s.pair_id]].reset_index().iloc[0], cfg)
            sp = SpatialFrame(pair.H_gt, pair.ref.shape, pair.tgt.shape, int(cfg["preprocessing"]["valid_margin_px"]))
            cz = np.load(src / s.pair_id / f"clean__{det}.npz")
            z = np.load(src / s.pair_id / f"{s.blur_id}__{det}.npz")
            for k, (stage, meth) in enumerate((("BB", "none"), ("RR", s.method))):
                row = df[(df.detector == det) & (df.stage == stage) & (df.method == meth) & (df.pair_id == s.pair_id)
                         & (df.blur_id == s.blur_id)].iloc[0]
                flag = " [NAFNet 발산 출력]" if (stage == "RR" and bool(row.nafnet_output_flag)) else ""
                title = ("BB (무처리)" if stage == "BB" else f"RR ({METHOD_LABEL[meth]})") + flag
                panel(axes[r, k], pair.ref, sp, cz["ref_kp"], z, f"{stage}_{meth}", row, title)
            pos = f" · 공통 유효 셀 중 {pct:.0f}% 위치" if pct is not None else ""
            axes[r, 0].set_ylabel(f"{lab}\n{s.pair_id}\n{BLUR_LABEL[s.blur_id]}, {METHOD_LABEL[s.method]}{pos}",
                                  fontsize=7.5, color=INK, rotation=0, ha="right", va="center", labelpad=8)
            trows.append({"detector": det, "case": lab, "pair_id": s.pair_id, "blur_id": s.blur_id, "method": s.method,
                          "herr_BB": s.homography_error_mean_BB, "herr_RR": s.homography_error_mean_RR, "D_log10": s.D,
                          "percentile_position": pct, "gtc_BB": s.gt_correct_matches_3px_BB,
                          "gtc_RR": s.gt_correct_matches_3px_RR, "nafnet_flag": bool(s.nafnet_output_flag),
                          "n_common_valid_cells": n_cv, "n_cells": n_all})
        fig.suptitle(f"{det}: 대응점 공간 분포와 H 오차 (기준 영상 좌표; 점 = GT-correct, 채움 = RANSAC inlier, × = GT-incorrect inlier;\n"
                     f"실선 = overlap, 점선 = 점유율 계산 셀). 선택 규칙은 a03_cases.py docstring", fontsize=8.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_new(fig, f"F6_cases_{det}.png")
    pd.DataFrame(trows).to_csv(TABLES / "T10_cases.csv", index=False)
    print(pd.DataFrame(trows).to_string())


if __name__ == "__main__":
    main()
