"""Tracking of the Clean reference set through the stage conditions (DIAGNOSTIC: uses Clean and GT information;
never mixed with matching performance). Rules: extension_config.yaml `tracking`.

  python -m extension.src.tracking --run ext_main_<ts>      (from blur_feature_study/, .venv)

Reference set (per pair, detector): Clean pipeline ratio-test matches with GT error <= 3 px. Element r =
(reference keypoint q_r, clean target keypoint t_r at location c_r).
Linking (per detection image D of a cell: B or R_m; detect() keypoints, locations deduplicated at 3 decimals):
  lost      : no D location within 3 px of c_r
  unique    : exactly one D location within 3 px AND that location has exactly one clean location within 3 px
  ambiguous : any other case with >= 1 candidate (no displacement is computed)
  Displacements are censored at 3 px: nothing beyond the window is observed. "Clean 대비 검출 위치 변화" only.
States per stage condition (partition of the reference set):
  correct_via_linked : q_r's ratio-test match has GT error <= 3 px and its target keypoint is a D location within 3 px of c_r
  correct_via_other  : correct, target keypoint elsewhere, c_r linked (unique/ambiguous)
  lost_but_correct   : correct although c_r is lost (target keypoint > 3 px from c_r)
  not_redetected     : not correct and c_r lost
  redetected_no_match / redetected_wrong_match : c_r linked, q_r without match / matched with error > 3 px
Outside-set correct matches: GT-correct matches whose reference keypoint is not in the reference set, split by
whether the target keypoint location lies within 3 px of any clean target location.
Chance baseline: the same linking rule for the reference-set clean locations shifted by (+-20,0), (0,+-20) px
(shifted points kept only inside the target valid region).
Fixed-landmark H (per cell and method): elements uniquely linked in BOTH the B- and the R_m-detection, >= 8;
cv2.findHomography(method=0) with the same correspondence identity and count for target = clean / B / R positions.
Oracle-inlier H (per stage row): cv2.findHomography(method=0) on all GT-correct matches, >= 8.
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import RESULTS, environment_info, keep_awake, load_base, load_ext_config, new_run_dir, write_json
from .stagewise import unpack_sift_octave
from src import metrics as M

METHODS = ["unsharp", "wiener", "nafnet"]
DET_IMAGES = ["none"] + METHODS
SHIFTS = [(20, 0), (-20, 0), (0, 20), (0, -20)]
STATES = ["correct_via_linked", "correct_via_other", "lost_but_correct", "not_redetected",
          "redetected_no_match", "redetected_wrong_match"]


def stage_list():
    out = [("BB", "none", "none")]
    for m in METHODS:
        out += [("BR", m, "none"), ("RB", m, m), ("RR", m, m)]
    return out                                   # (stage, method, detection image)


def locations(kp, decimals):
    """Deduplicated locations: (xy of the first keypoint at each location, loc index per keypoint, first kp index)."""
    if len(kp) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    _, first, inv = np.unique(np.round(kp[:, :2], decimals), axis=0, return_index=True, return_inverse=True)
    return kp[first, :2], np.asarray(inv).ravel(), first


def link(Q, L, r):
    """status per query (0 lost, 1 unique, 2 ambiguous), candidate lists, unique L index (-1), L linked to any Q."""
    n = len(Q)
    status = np.zeros(n, dtype=int)
    uniq = np.full(n, -1)
    if n == 0 or len(L) == 0:
        return status, [[] for _ in range(n)], uniq, np.zeros(len(L), dtype=bool)
    cand = cKDTree(L).query_ball_point(Q, r)
    nq = np.array([len(c) for c in cKDTree(Q).query_ball_point(L, r)])
    for i, c in enumerate(cand):
        if not c:
            continue
        if len(c) == 1 and nq[c[0]] == 1:
            status[i], uniq[i] = 1, c[0]
        else:
            status[i] = 2
    return status, cand, uniq, nq > 0


def octave_index(det, octave):
    return unpack_sift_octave(octave)[0] if det == "SIFT" else int(octave)


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def fit_h(src, dst, H_gt, ref_shape, tgt_shape, hc, margin, min_n):
    if len(src) < min_n:
        return {"err_mean": np.nan, "success_3px": 0, "reason": f"insufficient_points_lt{min_n}"}
    H, _ = cv2.findHomography(np.asarray(src, np.float64).reshape(-1, 1, 2), np.asarray(dst, np.float64).reshape(-1, 1, 2), 0)
    hres, reason = M.evaluate_homography(H, H_gt, ref_shape, tgt_shape, hc, margin)
    return {"err_mean": hres["homography_error_mean"], "success_3px": hres["homography_success_3px"],
            "reason": reason}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    keep_awake()
    cfg, sel, man, flags = load_base()
    tc = load_ext_config()["tracking"]
    R = float(tc["link_radius_px"]) + 1e-9
    DEC = int(tc["location_dedup_decimals"])
    MIN_LM = int(tc["fixed_landmark_h"]["min_landmarks"])
    hc = cfg["evaluation"]["homography"]
    m = int(cfg["preprocessing"]["valid_margin_px"])
    src = RESULTS / args.run
    raw = pd.read_csv(src / "raw_results.csv", float_precision="round_trip")
    run_id, out = new_run_dir("ext_tracking")
    states_rows, link_rows, lm_rows, tr_rows, disp_rows = [], [], [], [], []
    t0 = time.time()
    cells = raw[raw.stage == "BB"][["pair_id", "sequence_id", "blur_id", "detector", "gt_quality_flag"]].drop_duplicates()
    for (pid, det), cgrp in cells.groupby(["pair_id", "detector"], sort=False):
        cz = np.load(src / "details" / pid / f"clean__{det}.npz")
        H_gt, rs, ts = cz["H_gt"], tuple(cz["ref_shape"]), tuple(cz["tgt_shape"])
        ref_kp, clean_kp = cz["ref_kp"], cz["clean_kp"]
        cq, ct, cerr = cz["query_idx"], cz["train_idx"], cz["gt_err"]
        sel_r = cerr <= 3.0
        q_r, t_r = cq[sel_r].astype(int), ct[sel_r].astype(int)
        n_ref = len(q_r)
        in_set = np.zeros(len(ref_kp), dtype=bool)
        in_set[q_r] = True
        Cxy, Cinv, _ = locations(clean_kp, DEC)
        cloc_r = Cinv[t_r]
        uref = np.unique(cloc_r)
        h, w = ts
        seq = cgrp.sequence_id.iloc[0]
        gtf = int(cgrp.gt_quality_flag.iloc[0])
        key0 = {"pair_id": pid, "sequence_id": seq, "detector": det, "gt_quality_flag": gtf}
        # CLEAN reference: oracle H of the clean matches
        o = fit_h(ref_kp[q_r, :2], clean_kp[t_r, :2], H_gt, rs, ts, hc, m, MIN_LM)
        states_rows.append(dict(key0, blur_id="none", method="clean", stage="CLEAN", n_reference_set=n_ref,
                                n_gt_correct_total=n_ref, oracle_h_err_mean=o["err_mean"],
                                oracle_h_success_3px=o["success_3px"], oracle_h_reason=o["reason"],
                                n_oracle_points=n_ref))
        for blur in cgrp.blur_id:
            z = np.load(src / "details" / pid / f"{blur}__{det}.npz")
            key = dict(key0, blur_id=blur)
            L = {}
            for D in DET_IMAGES:
                kpD = z[f"kp_{D}"]
                Lxy, Linv, Lfirst = locations(kpD, DEC)
                st_all, cand_all, uq_all, l_linked = link(Cxy, Lxy, R)
                st_r, uq_r = st_all[cloc_r], uq_all[cloc_r]
                L[D] = dict(kp=kpD, Lxy=Lxy, Linv=Linv, st=st_r, uq=uq_r, cand=[set(cand_all[c]) for c in cloc_r],
                            l_linked=l_linked)
                # chance baseline on the (deduplicated) reference-set clean locations
                ch_l, ch_u, ch_n = 0, 0, 0
                for dx, dy in SHIFTS:
                    Qs = Cxy[uref] + np.array([dx, dy], float)
                    ok = (Qs[:, 0] >= m) & (Qs[:, 0] <= w - 1 - m) & (Qs[:, 1] >= m) & (Qs[:, 1] <= h - 1 - m)
                    s, *_ = link(Qs[ok], Lxy, R)
                    ch_l += int((s > 0).sum()); ch_u += int((s == 1).sum()); ch_n += int(ok.sum())
                st_u = st_all[uref]
                lr = dict(key, detection_image="B" if D == "none" else D, n_clean_locations=len(Cxy),
                          n_reference_set=n_ref, n_reference_locations=len(uref),
                          n_ref_unique=int((st_r == 1).sum()), n_ref_ambiguous=int((st_r == 2).sum()),
                          n_ref_lost=int((st_r == 0).sum()),
                          refloc_linked_rate=float((st_u > 0).mean()) if len(uref) else np.nan,
                          refloc_unique_rate=float((st_u == 1).mean()) if len(uref) else np.nan,
                          chance_linked_rate=ch_l / ch_n if ch_n else np.nan,
                          chance_unique_rate=ch_u / ch_n if ch_n else np.nan, chance_n_points=ch_n,
                          n_condition_keypoints=len(kpD), n_condition_locations=len(Lxy),
                          frac_condition_locations_unlinked_to_clean=float((~l_linked).mean()) if len(Lxy) else np.nan)
                u = np.where(st_r == 1)[0]
                if len(u):
                    li = uq_r[u]
                    dxy = Lxy[li] - clean_kp[t_r[u], :2]
                    dd = np.hypot(dxy[:, 0], dxy[:, 1])
                    fk = kpD[Lfirst[li]]
                    ratio = fk[:, 2] / clean_kp[t_r[u], 2]
                    oc = np.array([octave_index(det, o_) for o_ in clean_kp[t_r[u], 5]])
                    od = np.array([octave_index(det, o_) for o_ in fk[:, 5]])
                    ang = np.array([np.abs(wrap180(kpD[Linv == l_, 3] - clean_kp[t_r[e], 3])).min() for l_, e in zip(li, u)])
                    lr.update(n_displacement=len(u), mean_dx=float(dxy[:, 0].mean()), mean_dy=float(dxy[:, 1].mean()),
                              std_dx=float(dxy[:, 0].std(ddof=1)) if len(u) > 1 else np.nan,
                              std_dy=float(dxy[:, 1].std(ddof=1)) if len(u) > 1 else np.nan,
                              mean_absd=float(dd.mean()), median_absd=float(np.median(dd)),
                              q25_absd=float(np.quantile(dd, 0.25)), q75_absd=float(np.quantile(dd, 0.75)),
                              q90_absd=float(np.quantile(dd, 0.9)), frac_absd_le1=float((dd <= 1.0).mean()),
                              median_size_ratio=float(np.median(ratio)),
                              mean_abs_log2_size_ratio=float(np.abs(np.log2(ratio)).mean()),
                              frac_same_octave=float((oc == od).mean()), median_abs_angle_diff=float(np.median(ang)))
                    disp_rows.append(pd.DataFrame({"pair_id": pid, "blur_id": blur, "detector": det,
                                                   "detection_image": lr["detection_image"], "element": u,
                                                   "dx": dxy[:, 0], "dy": dxy[:, 1], "absd": dd, "size_ratio": ratio,
                                                   "octave_clean": oc, "octave_cond": od, "abs_angle_diff": ang}))
                link_rows.append(lr)
            correct_by_stage = {}
            for stage, meth, D in stage_list():
                tag = f"{stage}_{meth}"
                ld = L[D]
                ids = z[f"ids_{tag}"].astype(int)
                kp_stage = z[f"kp_{tag}"] if f"kp_{tag}" in z.files else ld["kp"][ids]
                q, t, err, inl = z[f"query_idx_{tag}"].astype(int), z[f"train_idx_{tag}"].astype(int), z[f"gt_err_{tag}"], z[f"ransac_inlier_{tag}"]
                good = err <= 3.0
                mt = np.full(len(ref_kp), -1)
                me = np.full(len(ref_kp), np.inf)
                mt[q], me[q] = t, err
                matched = mt[q_r] >= 0
                corr = matched & (me[q_r] <= 3.0)
                lost = ld["st"] == 0
                # target keypoint location (detection-image loc index) of each element's match
                tloc = np.full(n_ref, -1)
                tloc[matched] = ld["Linv"][ids[mt[q_r][matched]]]
                vialinked = np.array([tloc[e] in ld["cand"][e] for e in range(n_ref)], dtype=bool)
                s = {"correct_via_linked": int((corr & ~lost & vialinked).sum()),
                     "correct_via_other": int((corr & ~lost & ~vialinked).sum()),
                     "lost_but_correct": int((corr & lost).sum()),
                     "not_redetected": int((~corr & lost).sum()),
                     "redetected_no_match": int((~corr & ~lost & ~matched).sum()),
                     "redetected_wrong_match": int((~corr & ~lost & matched).sum())}
                assert sum(s.values()) == n_ref
                outside = good & ~in_set[q]
                o_loc = ld["Linv"][ids[t[outside]]]
                n_out_linked = int(ld["l_linked"][o_loc].sum()) if len(o_loc) else 0
                n_good = int(good.sum())
                assert int(corr.sum()) + int(outside.sum()) == n_good
                o = fit_h(ref_kp[q[good], :2], kp_stage[t[good], :2], H_gt, rs, ts, hc, m, MIN_LM)
                st_r = ld["st"]
                states_rows.append(dict(key, method=meth, stage=stage, n_reference_set=n_ref, **s,
                                        n_correct_in_set=int(corr.sum()),
                                        n_unique_linked=int((st_r == 1).sum()), n_unique_linked_correct=int((corr & (st_r == 1)).sum()),
                                        n_ambiguous_linked=int((st_r == 2).sum()),
                                        n_ambiguous_linked_correct=int((corr & (st_r == 2)).sum()),
                                        n_outside_set_correct=int(outside.sum()), n_outside_set_correct_linked_to_clean=n_out_linked,
                                        n_outside_set_correct_unlinked=int(outside.sum()) - n_out_linked,
                                        n_gt_correct_total=n_good, n_ransac_inliers=int(inl.sum()),
                                        n_ransac_inliers_gt_incorrect=int((inl & ~good).sum()),
                                        n_oracle_points=n_good, oracle_h_err_mean=o["err_mean"],
                                        oracle_h_success_3px=o["success_3px"], oracle_h_reason=o["reason"]))
                correct_by_stage[tag] = corr
            for meth in METHODS:
                for a, b in [("BB_none", f"BR_{meth}"), ("BB_none", f"RB_{meth}"), (f"RB_{meth}", f"RR_{meth}"),
                             (f"BR_{meth}", f"RR_{meth}"), ("BB_none", f"RR_{meth}")]:
                    ca, cb = correct_by_stage[a], correct_by_stage[b]
                    tr_rows.append(dict(key, method=meth, contrast=f"{b.split('_')[0]}-{a.split('_')[0]}",
                                        gained=int((~ca & cb).sum()), lost=int((ca & ~cb).sum()),
                                        kept=int((ca & cb).sum()), never=int((~ca & ~cb).sum())))
                # fixed-landmark H + paired displacement on the common subset
                lB, lR = L["none"], L[meth]
                lm = (lB["st"] == 1) & (lR["st"] == 1)
                e = np.where(lm)[0]
                rowlm = dict(key, method=meth, n_reference_set=n_ref, n_landmarks=len(e),
                             n_unique_B=int((lB["st"] == 1).sum()), n_unique_R=int((lR["st"] == 1).sum()))
                if len(e):
                    pB = lB["Lxy"][lB["uq"][e]]
                    pR = lR["Lxy"][lR["uq"][e]]
                    pc = clean_kp[t_r[e], :2]
                    dB, dR = np.hypot(*(pB - pc).T), np.hypot(*(pR - pc).T)
                    rowlm.update(mean_absd_B=float(dB.mean()), mean_absd_R=float(dR.mean()),
                                 median_absd_B=float(np.median(dB)), median_absd_R=float(np.median(dR)),
                                 n_R_closer=int((dR < dB).sum()), n_R_farther=int((dR > dB).sum()),
                                 n_equal=int((dR == dB).sum()))
                    pr = ref_kp[q_r[e], :2]
                    for tag_, pts in (("clean", pc), ("B", pB), ("R", pR)):
                        f = fit_h(pr, pts, H_gt, rs, ts, hc, m, MIN_LM)
                        rowlm.update({f"lm_h_err_mean_{tag_}": f["err_mean"], f"lm_h_success_3px_{tag_}": f["success_3px"],
                                      f"lm_h_reason_{tag_}": f["reason"]})
                else:
                    for tag_ in ("clean", "B", "R"):
                        rowlm.update({f"lm_h_err_mean_{tag_}": np.nan, f"lm_h_success_3px_{tag_}": 0,
                                      f"lm_h_reason_{tag_}": f"insufficient_points_lt{MIN_LM}"})
                rowlm["lm_analyzable"] = len(e) >= MIN_LM
                lm_rows.append(rowlm)
        print(f"{pid} {det} done ({time.time() - t0:.0f}s)", flush=True)
    st_df = pd.DataFrame(states_rows); st_df.to_csv(out / "tracking_states.csv", index=False)
    pd.DataFrame(link_rows).to_csv(out / "tracking_links.csv", index=False)
    pd.DataFrame(lm_rows).to_csv(out / "fixed_landmark_h.csv", index=False)
    pd.DataFrame(tr_rows).to_csv(out / "tracking_transitions.csv", index=False)
    pd.concat(disp_rows, ignore_index=True).to_csv(out / "displacements.csv.gz", index=False, compression="gzip")
    # cross-check: total GT-correct per stage row == raw_results gt_correct_matches_3px
    mg = st_df.merge(raw[["pair_id", "blur_id", "detector", "method", "stage", "gt_correct_matches_3px"]],
                     on=["pair_id", "blur_id", "detector", "method", "stage"], how="left")
    xcheck = bool((mg.n_gt_correct_total == mg.gt_correct_matches_3px).all())
    write_json(out / "run_meta.json", {"run_id": run_id, "source_run": args.run, "tracking_config": tc,
                                       "n_state_rows": len(st_df), "gt_correct_crosscheck_with_raw_results": xcheck,
                                       "elapsed_s": round(time.time() - t0, 1), "environment": environment_info()})
    print("crosscheck:", xcheck)
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
