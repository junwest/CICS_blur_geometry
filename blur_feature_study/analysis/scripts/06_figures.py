"""Exploratory / verification figures (PNG). Light theme, palette = dataviz reference palette
(documented as validated; node was unavailable so the validator was not re-run here).
Categorical slots in FIXED order: A=blue, C=orange, D=aqua, B=yellow; CLEAN = neutral ink (reference, not a series).
Detectors are facets, never colours. One y-axis per panel."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BLUR_LABEL, BLUR_ORDER, FIGS, STATS, TABLES, add_clean_reference, image_level, load_results, main_core  # noqa: E402

C = {"A": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "B": "#eda100", "CLEAN": "#52514e"}
LAB = {"A": "A blurred + default", "B": "B blurred + tuned", "C": "C unsharp + default", "D": "D Wiener (oracle PSF)", "CLEAN": "clean (reference)"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
div_cmap = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "legend.frameon": False, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold"})
BL = [BLUR_LABEL[b] for b in BLUR_ORDER]
df = add_clean_reference(main_core(load_results()))
img = image_level(load_results())
eff = pd.read_csv(STATS / "S1_paired_effects_all_pairs.csv")
NOTES = []


def save(fig, name, note):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    NOTES.append((name, note))
    print("saved", name)


def xblur(ax):
    ax.set_xticks(range(8)); ax.set_xticklabels(BL, rotation=35, ha="right")


# ------------------------------------------------------------------ F01 degradation / recovery profile
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
for j, det in enumerate(("ORB", "SIFT")):
    for i, (metric, lab) in enumerate((("homography_success_3px", "Homography success@3px (mean over 24 pairs)"), ("gt_correct_matches_3px", "GT-correct matches@3px (mean)"))):
        ax = axes[i, j]
        d = df[df.detector == det]
        cl = d[d.group == "CLEAN"][metric].astype(float).mean()
        ax.axhline(cl, color=C["CLEAN"], ls="--", lw=1.2, label=LAB["CLEAN"])
        for g in ("A", "B", "C", "D"):
            if det == "SIFT" and g == "B":
                continue
            y = d[d.group == g].groupby("blur_id", observed=True)[metric].mean().loc[BLUR_ORDER].values.astype(float)
            ax.plot(range(8), y, "-o", color=C[g], lw=2, ms=6, label=LAB[g], mec=SURF, mew=0.8)
        ax.set_title(f"{det}: {lab}" if i == 0 else f"{det}: {lab}")
        if i == 1:
            xblur(ax)
        if i == 0:
            ax.set_ylim(0, 1.02)
        if j == 0 and i == 0:
            ax.legend(loc="lower left", fontsize=8)
save(fig, "F01_profile_by_blur", "Success@3px (top) and GT-correct@3px (bottom) per blur condition for each group; dashed = clean baseline. Facets = detector. Test split, 24 pairs per point.")

# ------------------------------------------------------------------ F02 paired effects forest (success and GT-correct)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for j, det in enumerate(("ORB", "SIFT")):
    for i, (metric, lab, xl) in enumerate((("homography_success_3px", "d success@3px (treat - A)", "difference in success rate"),
                                           ("gt_correct_matches_3px", "d GT-correct@3px (treat - A)", "difference in correct matches"))):
        ax = axes[i, j]
        comps = ["B-A", "C-A", "D-A"] if det == "ORB" else ["C-A", "D-A"]
        for k, comp in enumerate(comps):
            e = eff[(eff.detector == det) & (eff.comparison == comp) & (eff.metric == metric) & (eff.blur != "pooled")].set_index("blur").loc[BLUR_ORDER]
            off = (k - (len(comps) - 1) / 2) * 0.22
            yy = np.arange(8) + off
            ax.errorbar(e["mean"], yy, xerr=[e["mean"] - e.ci_lo, e.ci_hi - e["mean"]], fmt="o", color=C[comp[0]], ms=6, lw=1.6, capsize=2, label=f"{comp}")
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(8)); ax.set_yticklabels(BL); ax.invert_yaxis()
        ax.set_title(f"{det}: {lab}"); ax.set_xlabel(xl)
        if i == 0 and j == 0:
            ax.legend(loc="lower right", fontsize=8)
save(fig, "F02_paired_effects_forest", "Paired within-cell differences (treatment minus A) per blur condition with sequence-cluster bootstrap 95% CI. Top: success@3px; bottom: GT-correct@3px. B-A only for ORB (SIFT tuned == default).")

# ------------------------------------------------------------------ F03 unsharp dissociation across metrics
mets = [("psnr", "dPSNR [dB]"), ("ssim", "dSSIM"), ("edge_f1", "dEdge F1"), ("keypoint_repeatability", "dRepeatability"), ("gt_correct_matches_3px", "dGT-correct@3"), ("homography_success_3px", "dSuccess@3")]
fig, axes = plt.subplots(2, 6, figsize=(17, 6))
for i, det in enumerate(("ORB", "SIFT")):
    for j, (m, lab) in enumerate(mets):
        ax = axes[i, j]
        for comp in ("C-A", "D-A"):
            e = eff[(eff.detector == det) & (eff.comparison == comp) & (eff.metric == m) & (eff.blur != "pooled")].set_index("blur").loc[BLUR_ORDER]
            off = -0.17 if comp == "C-A" else 0.17
            ax.errorbar(np.arange(8) + off, e["mean"], yerr=[e["mean"] - e.ci_lo, e.ci_hi - e["mean"]], fmt="o", color=C[comp[0]], ms=5, lw=1.4, capsize=2, label=comp)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f"{det}: {lab}", fontsize=9)
        xblur(ax) if i == 1 else ax.set_xticks(range(8), [""] * 8)
        if m in ("psnr", "ssim", "edge_f1") and i == 1:
            ax.set_title(f"{lab} (image-level)", fontsize=9)
axes[0, 0].legend(fontsize=8)
save(fig, "F03_processing_effects_all_metrics", "C-A (unsharp, orange) and D-A (Wiener, aqua) per blur for six metrics; CI = sequence-cluster bootstrap. Shows the unsharp dissociation: for motion blur PSNR falls while Edge F1 / repeatability rise and matching is flat.")

# ------------------------------------------------------------------ F04 PSNR vs Edge F1 (image level)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
fam_marker = {"gaussian": "o", "defocus": "s", "motion": "^"}
for ax, (xm, xl) in zip(axes, (("psnr", "PSNR [dB]"), ("ssim", "SSIM"))):
    for g, pm in (("A", "blurred"), ("C", "unsharp_mask"), ("D", "wiener_known_psf")):
        d = img[img.processing_method == pm]
        for fam, mk in fam_marker.items():
            dd = d[d.blur_family == fam]
            ax.scatter(dd[xm], dd.edge_f1, s=22, marker=mk, color=C[g], alpha=0.75, edgecolor=SURF, linewidth=0.4,
                       label=f"{LAB[g]}" if fam == "gaussian" else None)
    ax.set_xlabel(xl); ax.set_ylabel("Edge F1 (tol 2px)")
    ax.set_title(f"{xl} vs Edge F1, 576 processed images")
h, l = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D  # noqa: E402
h += [Line2D([], [], marker=m, color=INK2, ls="", ms=6) for m in fam_marker.values()]; l += list(fam_marker)
axes[0].legend(h, l, fontsize=8, loc="upper left")
save(fig, "F04_psnr_ssim_vs_edgef1", "Image-level scatter: PSNR (left) / SSIM (right) vs Edge F1 for all 576 processed test images; colour = processing, marker = blur family. Motion-blurred images (triangles) sit at low PSNR but high Edge F1.")

# ------------------------------------------------------------------ F05 metrics vs matching (log GT-correct ratio)
proc = df[df.group.isin(["A", "C", "D"])]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
for i, det in enumerate(("ORB", "SIFT")):
    d = proc[proc.detector == det]
    for j, (xm, xl) in enumerate((("psnr", "PSNR [dB]"), ("ssim", "SSIM"), ("edge_f1", "Edge F1"), ("keypoint_repeatability", "Keypoint repeatability"))):
        ax = axes[i, j]
        for g in ("A", "C", "D"):
            dd = d[d.group == g]
            ax.scatter(dd[xm], dd.log_gt_correct_ratio, s=16, color=C[g], alpha=0.7, edgecolor=SURF, linewidth=0.3, label=LAB[g])
        from scipy.stats import spearmanr  # noqa: E402
        ok = d[[xm, "log_gt_correct_ratio"]].notna().all(axis=1)
        rho = spearmanr(d.loc[ok, xm], d.loc[ok, "log_gt_correct_ratio"]).statistic
        ax.set_title(f"{det}: {xl} (rho = {rho:.2f})", fontsize=9)
        ax.set_xlabel(xl)
        if j == 0:
            ax.set_ylabel("log (GT-correct+1) / (clean GT-correct+1)")
axes[0, 0].legend(fontsize=8, loc="lower right")
save(fig, "F05_metrics_vs_log_gt_ratio", "Each metric vs clean-normalised log GT-correct ratio, rows = detector, 576 rows (A/C/D) per detector. Repeatability shows the tightest relation; Edge F1 the loosest.")

# ------------------------------------------------------------------ F06 predictor comparison
rel = pd.read_csv(STATS / "S3_detector_level_relations.csv")
mod = pd.read_csv(STATS / "S4_models.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
order = ["psnr", "ssim", "edge_f1", "keypoint_repeatability"]; names = ["PSNR", "SSIM", "Edge F1", "Repeat."]
x = np.arange(4)
for ax, det, off in ((axes[0], "ORB", -0.18), (axes[0], "SIFT", 0.18)):
    v = rel[(rel.detector == det) & (rel.view == "pooled_rows") & (rel.y == "log_gt_correct_ratio")].set_index("x").loc[order, "rho"]
    ax.bar(x + off, v, width=0.34, color=C["A"] if det == "ORB" else C["C"], label=det)
axes[0].set_title("Pooled Spearman rho vs log GT-correct ratio", fontsize=9); axes[0].set_xticks(x, names); axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1)
for ax, det, off in ((axes[1], "ORB", -0.18), (axes[1], "SIFT", 0.18)):
    v = rel[(rel.detector == det) & (rel.view == "per_pair_across_blur_A") & (rel.y == "gt_correct_matches_3px")].set_index("x").loc[order]
    ax.bar(x + off, v.rho_median, width=0.34, color=C["A"] if det == "ORB" else C["C"], label=det)
    ax.errorbar(x + off, v.rho_median, yerr=[v.rho_median - v.rho_q1, v.rho_q3 - v.rho_median], fmt="none", ecolor=INK2, lw=1)
axes[1].set_title("Per-pair rho across 8 blur conditions (A; median, IQR)", fontsize=9); axes[1].set_xticks(x, names); axes[1].set_ylim(-0.1, 1)
v = mod[mod.model == "within_cell_OLS_log_gt_correct_ratio"].set_index("predictors").loc[order, "r2_within"]
axes[2].bar(x, v, width=0.5, color=C["D"])
axes[2].set_title("Within-cell R2 (pair x blur x detector FE)", fontsize=9); axes[2].set_xticks(x, names); axes[2].set_ylim(0, 1)
for ax in axes:
    ax.set_ylabel("")
save(fig, "F06_predictor_comparison", "How well each metric tracks matching performance under three dependence-aware views. Left: pooled rank correlation; middle: per-pair correlation across blur conditions; right: within-cell fixed-effects R2 (both detectors pooled).")

# ------------------------------------------------------------------ F07 heatmaps of log GT-correct ratio per pair x blur (A vs D)
fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
order_pairs = df[df.group == "CLEAN"].drop_duplicates("pair_id").sort_values(["sequence_type", "pair_id"]).pair_id.tolist()
for i, det in enumerate(("ORB", "SIFT")):
    for j, g in enumerate(("A", "D")):
        d = df[(df.detector == det) & (df.group == g)].pivot_table(index="pair_id", columns="blur_id", values="log_gt_correct_ratio", observed=True).loc[order_pairs, BLUR_ORDER]
        s = df[(df.detector == det) & (df.group == g)].pivot_table(index="pair_id", columns="blur_id", values="homography_success_3px", observed=True).loc[order_pairs, BLUR_ORDER]
        ax = axes[i, j]
        im = ax.imshow(np.exp(d.values), cmap=seq_cmap, vmin=0, vmax=1, aspect="auto")
        for r in range(d.shape[0]):
            for c in range(d.shape[1]):
                if s.values[r, c] == 0:
                    ax.plot(c, r, "x", color="#e34948", ms=5, mew=1.2)
        ax.set_xticks(range(8), BL, rotation=35, ha="right"); ax.set_yticks(range(len(order_pairs)), order_pairs if j == 0 else [""] * len(order_pairs), fontsize=7)
        ax.set_title(f"{det} - group {g}: retained GT-correct fraction (x = H fail@3px)", fontsize=9); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.5, label="(correct+1)/(clean correct+1): 1 = clean level")
save(fig, "F07_heatmap_pair_blur", "Per-pair x blur map of the retained fraction of GT-correct matches (processed / clean) for A (left) and D (right); red x = homography failure@3px. Dark = close to clean level, light = almost nothing retained. i_ = illumination, v_ = viewpoint.")

# ------------------------------------------------------------------ F08 motion-orientation asymmetry
g = pd.read_csv(TABLES / "T15_gradient_orientation_vs_motion_asymmetry.csv").set_index("pair_id")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (y, lab) in zip(axes, (("dpsnr_l17", "dPSNR (0deg - 90deg) [dB], motion 17px, blurred"), ("dedge_l17", "dEdge F1 (0deg - 90deg), motion 17px"), ("dgt3_l17_SIFT", "dGT-correct SIFT (0deg - 90deg), motion 17px, group A"))):
    ax.scatter(g.log10_ex_over_ey, g[y], s=28, color=C["A"], edgecolor=SURF)
    from scipy.stats import spearmanr  # noqa: E402
    r = spearmanr(g.log10_ex_over_ey, g[y]).statistic
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("log10(Ex/Ey): >0 = vertical edges dominate"); ax.set_title(f"{lab}\nSpearman rho = {r:.2f}", fontsize=9)
save(fig, "F08_orientation_asymmetry", "Clean-image gradient orientation (x) vs the 0deg-minus-90deg difference under 17px motion blur for PSNR, Edge F1 and SIFT GT-correct matches (24 pairs). Image-quality asymmetry is a scene property; the matching asymmetry is not predicted by it.")

# ------------------------------------------------------------------ F09 ORB tuning trade-off
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (m, lab) in zip(axes, (("target_keypoints", "d target keypoints"), ("good_matches", "d ratio-test matches"), ("gt_correct_matches_3px", "d GT-correct@3"), ("match_precision_3px", "d match precision@3"))):
    e = eff[(eff.detector == "ORB") & (eff.comparison == "B-A") & (eff.metric == m) & (eff.blur != "pooled")].set_index("blur").loc[BLUR_ORDER]
    ax.errorbar(range(8), e["mean"], yerr=[e["mean"] - e.ci_lo, e.ci_hi - e["mean"]], fmt="o", color=C["B"], ms=6, lw=1.6, capsize=2)
    ax.axhline(0, color=MUTED, lw=1); ax.set_title(f"ORB tuned - default: {lab}", fontsize=9); xblur(ax)
save(fig, "F09_orb_tuning_tradeoff", "ORB tuned (scaleFactor 1.2, nlevels 12, fastThreshold 20) minus default, per blur: keypoints, ratio-test matches, GT-correct matches and match precision, with cluster-bootstrap CI.")

# ------------------------------------------------------------------ F10 H error ECDF per group
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, det in zip(axes, ("ORB", "SIFT")):
    for g in ("CLEAN", "A", "B", "C", "D"):
        if det == "SIFT" and g == "B":
            continue
        x = df[(df.detector == det) & (df.group == g)]
        e = np.sort(x.homography_error_mean.fillna(np.inf).values)
        n = len(e); y = np.arange(1, n + 1) / n
        xs = np.clip(e, 0.05, 100)
        ax.step(xs, y, where="post", color=C[g], lw=2 if g != "CLEAN" else 1.4, ls="--" if g == "CLEAN" else "-", label=f"{LAB[g]} (undefined: {int(np.isinf(e).sum())}/{n})")
    ax.set_xscale("log"); ax.axvline(3, color=MUTED, lw=1, ls=":"); ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("mean grid projection error vs H_gt [px] (undefined -> right end)"); ax.set_title(f"{det}: ECDF of homography error"); ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("fraction of cells")
save(fig, "F10_h_error_ecdf", "Empirical CDF of the mean grid projection error per group (failures / undefined placed at the right end). Dotted lines: 3 px and 5 px success thresholds.")

# ------------------------------------------------------------------ F11 edge precision / recall decomposition
e = pd.read_csv(TABLES / "T16_edge_decomposition.csv")
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for b in BLUR_ORDER:
    pts = {pm: e[(e.blur_id == b) & (e.processing_method == pm)].iloc[0] for pm in ("blurred", "unsharp_mask", "wiener_known_psf")}
    ax.annotate("", xy=(pts["unsharp_mask"].edge_recall, pts["unsharp_mask"].edge_precision), xytext=(pts["blurred"].edge_recall, pts["blurred"].edge_precision), arrowprops=dict(arrowstyle="->", color=C["C"], lw=1.4))
    ax.annotate("", xy=(pts["wiener_known_psf"].edge_recall, pts["wiener_known_psf"].edge_precision), xytext=(pts["blurred"].edge_recall, pts["blurred"].edge_precision), arrowprops=dict(arrowstyle="->", color=C["D"], lw=1.4))
    ax.scatter(pts["blurred"].edge_recall, pts["blurred"].edge_precision, color=C["A"], s=36, zorder=3, edgecolor=SURF)
    ax.text(pts["blurred"].edge_recall + 0.01, pts["blurred"].edge_precision - 0.012, BLUR_LABEL[b], fontsize=7, color=INK2)
ax.set_xlabel("edge recall (clean edges recovered within 2 px)"); ax.set_ylabel("edge precision (processed edges near a clean edge)")
ax.set_title("Edge F1 decomposition: blurred (dot) -> unsharp (orange) / Wiener (aqua)"); ax.set_ylim(0.8, 1.005); ax.set_xlim(0, 0.9)
save(fig, "F11_edge_precision_recall", "Mean edge precision vs recall per blur condition; arrows from the blurred image to unsharp (orange) and Wiener (aqua). Blur removes recall, not precision; unsharp buys recall with lower precision; Wiener restores recall at ~0.99 precision.")

# ------------------------------------------------------------------ F12 effects by sequence type
st = pd.read_csv(STATS / "S8_effects_by_sequence_type.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, metric, lab in zip(axes, ("homography_success_3px", "log_gt_correct_ratio"), ("d success@3px", "d log GT-correct ratio")):
    k = 0; ticks = []
    for det in ("ORB", "SIFT"):
        for comp in (["B-A", "C-A", "D-A"] if det == "ORB" else ["C-A", "D-A"]):
            for stype, mk in (("illumination", "o"), ("viewpoint", "s")):
                r = st[(st.detector == det) & (st.contrast == comp) & (st.metric == metric) & (st.sequence_type == stype)].iloc[0]
                yy = k + (-0.17 if stype == "illumination" else 0.17)
                ax.errorbar(r["mean"], yy, xerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], fmt=mk, color=C[comp[0]], ms=6, capsize=2, lw=1.4, mfc=C[comp[0]] if stype == "illumination" else SURF, mec=C[comp[0]])
            ticks.append(f"{det} {comp}"); k += 1
    ax.set_yticks(range(k), ticks); ax.invert_yaxis(); ax.axvline(0, color=MUTED, lw=1); ax.set_title(f"{lab}: filled = illumination (n=12), open = viewpoint (n=12)", fontsize=9)
save(fig, "F12_effects_by_sequence_type", "Pooled treatment effects split by sequence type (12 pairs each); CI = cluster bootstrap over 6 sequences per type, so intervals are wide.")

# ------------------------------------------------------------------ F13 keypoint counts under blur (detector response)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, det in zip(axes, ("ORB", "SIFT")):
    d = df[df.detector == det]
    cl = d[d.group == "CLEAN"].target_keypoints.astype(float).mean()
    ax.axhline(cl, color=C["CLEAN"], ls="--", lw=1.2, label="clean target")
    ax.axhline(1500, color=MUTED, lw=0.8, ls=":", label="budget 1500")
    for g in ("A", "C", "D"):
        y = d[d.group == g].groupby("blur_id", observed=True).target_keypoints.mean().loc[BLUR_ORDER].values.astype(float)
        ax.plot(range(8), y, "-o", color=C[g], lw=2, ms=6, mec=SURF, label=LAB[g])
    ax.set_title(f"{det}: mean target keypoints"); xblur(ax); ax.set_ylim(0, 1600)
axes[0].legend(fontsize=8, loc="lower left")
save(fig, "F13_keypoint_counts", "Mean number of detected target keypoints per blur condition and group. ORB saturates at the 1500 budget for mild blur; SIFT loses keypoints monotonically (contrast threshold).")

# ------------------------------------------------------------------ F14 slopegraph GT-correct A -> C -> D for a strong blur (per pair)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for ax, det in zip(axes, ("ORB", "SIFT")):
    d = df[(df.detector == det) & (df.blur_id == "motion_l17_a0")].pivot_table(index="pair_id", columns="group", values="gt_correct_matches_3px", observed=True)
    cl = df[(df.detector == det) & (df.group == "CLEAN")].set_index("pair_id").gt_correct_matches_3px
    for pid, r in d.iterrows():
        ys = [float(r["A"]), float(r["C"]), float(r["D"]), float(cl[pid])]
        ax.plot([0, 1, 2, 3], np.array(ys) + 1, "-", color=INK2 if pid.startswith("i_") else C["A"], alpha=0.55, lw=1.2)
    ax.set_yscale("log"); ax.set_xticks([0, 1, 2, 3], ["A blurred", "C unsharp", "D Wiener", "clean"]); ax.set_title(f"{det}: GT-correct+1 per pair, motion 17px 0deg")
    ax.plot([], [], color=INK2, label="illumination pairs"); ax.plot([], [], color=C["A"], label="viewpoint pairs"); ax.legend(fontsize=8)
axes[0].set_ylabel("GT-correct matches@3px + 1 (log)")
save(fig, "F14_slopegraph_motion17", "Per-pair trajectories of GT-correct matches from blurred (A) through unsharp (C) and Wiener (D) to the clean image, for the hardest condition (motion 17 px, 0deg).")

with open(FIGS / "FIGURES.md", "w", encoding="utf-8") as f:
    f.write("# Figure index (auto-generated by 06_figures.py)\n\nSee analysis_report.md section 7 for per-figure interpretation and caveats.\n\n")
    for n, note in NOTES:
        f.write(f"- **{n}.png** — {note}\n")
print("done")
