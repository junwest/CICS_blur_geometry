"""EXPLORATORY associations of the H-error change (RR - BB) with candidate correlates (no causal claim).

  analysis\\.venv\\Scripts\\python.exe extension/analysis/scripts/a05_associations.py   (from blur_feature_study/)

Per (detector, method), common-valid cells (both BB and RR have a valid H):
  y  = Δlog10 H error (RR - BB)
  x1 = Δlog((GT-correct + 1)) (correspondence count)
  x2 = Δ hull coverage of GT-correct matches, x3 = Δ 4x4 occupancy (spatial distribution)
  x4 = Δ mean |d| on the common landmark subset (uniquely linked in both B and R detection; position change
       relative to Clean, censored at 3 px; only cells with >= 1 common element)
Spearman rho at cell level (cells are NOT independent: 8 blurs x 24 pairs) and at pair level (pair means, n <= 24).
Also the correlations among the correlates (they are not separable here).
Output: tables/T14_associations.csv
"""
import numpy as np
import pandas as pd

from a00_common import DETECTORS, METHODS, TABLES, cell_table, load_main, read_csv, run_dir, spearman


def main():
    df = load_main()
    lm = read_csv(run_dir("tracking") / "fixed_landmark_h.csv")
    rows = []
    for det in DETECTORS:
        for meth in METHODS:
            j = cell_table(df, det, meth, "RR", "BB")
            j = j[j.h_valid_x & j.h_valid_y].copy()
            j["y"] = j.log10_herr_x - j.log10_herr_y
            j["x_gtc"] = np.log((j.gt_correct_matches_3px_x + 1) / (j.gt_correct_matches_3px_y + 1))
            j["x_cov"] = j.gtc_hull_coverage_x - j.gtc_hull_coverage_y
            j["x_occ"] = j.gtc_cell_occupancy_x - j.gtc_cell_occupancy_y
            l = lm[(lm.detector == det) & (lm.method == meth) & (lm.n_landmarks > 0)][["pair_id", "blur_id", "mean_absd_R", "mean_absd_B", "n_landmarks"]]
            j = j.merge(l, on=["pair_id", "blur_id"], how="left")
            j["x_pos"] = j.mean_absd_R - j.mean_absd_B
            pl = j.groupby("pair_id")[["y", "x_gtc", "x_cov", "x_occ", "x_pos"]].mean()
            r = {"detector": det, "method": meth, "n_cells": len(j), "n_cells_with_pos": int(j.x_pos.notna().sum()),
                 "n_pairs": len(pl)}
            for x in ("x_gtc", "x_cov", "x_occ", "x_pos"):
                rc, pc, nc = spearman(j[x], j.y)
                rp, pp, npp = spearman(pl[x], pl.y)
                r.update({f"rho_cell_{x}": rc, f"n_cell_{x}": nc, f"rho_pair_{x}": rp, f"p_pair_{x}": pp, f"n_pair_{x}": npp})
            for a, b in (("x_gtc", "x_cov"), ("x_gtc", "x_occ"), ("x_gtc", "x_pos"), ("x_cov", "x_pos")):
                r[f"rho_cell_{a}_vs_{b}"] = spearman(j[a], j[b])[0]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "T14_associations.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).T.to_string())


if __name__ == "__main__":
    main()
