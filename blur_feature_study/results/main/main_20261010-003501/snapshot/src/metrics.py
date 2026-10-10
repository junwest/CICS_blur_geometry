"""The eight evaluation metrics (+ helpers).

Image metrics (PSNR, SSIM, Edge F1) compare clean target vs processed target in the
same pixel grid, restricted to the common valid region (margin from every border).
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity


def crop_valid(img, m):
    return img[m:img.shape[0] - m, m:img.shape[1] - m]


def valid_region_mask(shape, m):
    v = np.zeros(shape, dtype=bool)
    v[m:shape[0] - m, m:shape[1] - m] = True
    return v


# ---------------------------------------------------------------- 1, 2: PSNR / SSIM
def psnr(clean, proc, margin, data_range=255.0):
    a = crop_valid(clean, margin).astype(np.float64)
    b = crop_valid(proc, margin).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("PSNR: shape mismatch")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(clean, proc, margin, sc):
    a = crop_valid(clean, margin).astype(np.float64)
    b = crop_valid(proc, margin).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("SSIM: shape mismatch")
    return float(structural_similarity(a, b, data_range=float(sc["data_range"]),
                                       gaussian_weights=bool(sc["gaussian_weights"]),
                                       sigma=float(sc["sigma"]),
                                       use_sample_covariance=bool(sc["use_sample_covariance"]),
                                       K1=float(sc["K1"]), K2=float(sc["K2"])))


# ---------------------------------------------------------------- 3: Edge F1
def canny(img, cc):
    return cv2.Canny(img, int(cc["threshold1"]), int(cc["threshold2"]),
                     apertureSize=int(cc["apertureSize"]), L2gradient=bool(cc["L2gradient"])) > 0


def _dist_to_edges(edges):
    if not edges.any():
        return np.full(edges.shape, np.inf)
    return distance_transform_edt(~edges)


def edge_prf(edges_clean, edges_proc, tol, margin):
    """Tolerance-based edge precision / recall / F1 (no one-to-one constraint).
    Precision: processed edge pixels within `tol` px of a clean edge pixel.
    Recall:    clean edge pixels within `tol` px of a processed edge pixel.
    Only edge pixels inside the valid region are counted; distances are computed on
    the full maps so an edge just outside the region can still be a match.
    Conventions: no clean edges -> P, R, F1 = NaN; no processed edges -> P = NaN, R = 0, F1 = 0."""
    valid = valid_region_mask(edges_clean.shape, margin)
    pe = edges_proc & valid
    ce = edges_clean & valid
    n_p, n_c = int(pe.sum()), int(ce.sum())
    out = {"edge_n_clean": n_c, "edge_n_proc": n_p}
    if n_c == 0:
        out.update(edge_precision=np.nan, edge_recall=np.nan, edge_f1=np.nan)
        return out
    rec = float((_dist_to_edges(edges_proc)[ce] <= tol).sum() / n_c)
    if n_p == 0:
        out.update(edge_precision=np.nan, edge_recall=rec, edge_f1=0.0)
        return out
    prec = float((_dist_to_edges(edges_clean)[pe] <= tol).sum() / n_p)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    out.update(edge_precision=prec, edge_recall=rec, edge_f1=float(f1))
    return out


# ---------------------------------------------------------------- 4: Keypoint repeatability
def unique_xy(xy, decimals=3):
    if len(xy) == 0:
        return np.zeros((0, 2))
    return np.unique(np.round(np.asarray(xy, dtype=np.float64), decimals), axis=0)


def one_to_one_count(a, b, tol):
    """Maximum number of one-to-one pairs (a_i, b_j) with ||a_i - b_j|| <= tol.
    Hungarian assignment (min total distance among max-cardinality solutions) is
    solved independently on each connected component of the tolerance graph."""
    if len(a) == 0 or len(b) == 0:
        return 0
    tb = cKDTree(b)
    rows, cols, ds = [], [], []
    for i, js in enumerate(tb.query_ball_point(a, r=tol + 1e-9)):
        for j in js:
            d = float(np.hypot(*(a[i] - b[j])))
            if d <= tol + 1e-9:
                rows.append(i); cols.append(j); ds.append(d)
    if not rows:
        return 0
    rows, cols, ds = np.array(rows), np.array(cols), np.array(ds)
    na, nb = len(a), len(b)
    g = coo_matrix((np.ones(len(rows)), (rows, cols + na)), shape=(na + nb, na + nb))
    _, labels = connected_components(g, directed=False)
    big = 1e6
    total = 0
    comp_of_edge = labels[rows]
    for c in np.unique(comp_of_edge):
        sel = comp_of_edge == c
        r, cc, d = rows[sel], cols[sel], ds[sel]
        ur, ri = np.unique(r, return_inverse=True)
        uc, ci = np.unique(cc, return_inverse=True)
        cost = np.full((len(ur), len(uc)), big)
        cost[ri, ci] = d
        ai, bi = linear_sum_assignment(cost)
        total += int((cost[ai, bi] < big).sum())
    return total


def repeatability(clean_xy, proc_xy, tol, decimals=3):
    """Location-based repeatability = N_repeated / N_clean_valid (same image frame)."""
    c = unique_xy(clean_xy, decimals)
    p = unique_xy(proc_xy, decimals)
    if len(c) == 0:
        return np.nan, 0, 0
    n_rep = one_to_one_count(c, p, tol)
    return n_rep / len(c), n_rep, len(c)


# ---------------------------------------------------------------- 5, 6: GT-correct matches
def project(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = ph[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = ph[:, :2] / w[:, None]
    return xy, w


def gt_errors(H_gt, ref_xy, tgt_xy, min_abs_w=1e-8):
    """e_i = || x_target_i - project(H_gt, x_reference_i) ||_2 (inf if not projectable)."""
    if len(ref_xy) == 0:
        return np.zeros(0)
    proj, w = project(H_gt, ref_xy)
    e = np.linalg.norm(np.asarray(tgt_xy) - proj, axis=1)
    e[~(w > min_abs_w) | ~np.isfinite(e)] = np.inf
    return e


# ---------------------------------------------------------------- 7, 8: Homography
def valid_grid(H_gt, ref_shape, tgt_shape, margin, n, min_abs_w=1e-8):
    """n x n grid over the reference valid region, kept if its H_gt projection lies
    inside the target valid region."""
    hr, wr = ref_shape
    ht, wt = tgt_shape
    xs = np.linspace(margin, wr - 1 - margin, n)
    ys = np.linspace(margin, hr - 1 - margin, n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    proj, w = project(H_gt, pts)
    ok = ((w > min_abs_w) & np.all(np.isfinite(proj), axis=1)
          & (proj[:, 0] >= margin) & (proj[:, 0] <= wt - 1 - margin)
          & (proj[:, 1] >= margin) & (proj[:, 1] <= ht - 1 - margin))
    return pts[ok], proj[ok]


def homography_validity(H, grid, min_abs_w=1e-8):
    if H is None:
        return "homography_estimation_failed"
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return "invalid_homography_nonfinite"
    nrm = np.linalg.norm(H)
    if nrm == 0 or abs(np.linalg.det(H / nrm)) < 1e-12:
        return "invalid_homography_singular"
    if len(grid):
        _, w = project(H, grid)
        if not (np.all(w > min_abs_w) or np.all(w < -min_abs_w)):
            return "invalid_homography_horizon_in_overlap"
    return ""


def evaluate_homography(H_est, H_gt, ref_shape, tgt_shape, hc, margin):
    """Mean/median distance between H_est and H_gt projections of the valid grid."""
    grid, proj_gt = valid_grid(H_gt, ref_shape, tgt_shape, margin, int(hc["grid_n"]), float(hc["min_abs_w"]))
    out = {"n_grid_valid": int(len(grid)), "homography_error_mean": np.nan,
           "homography_error_median": np.nan, "homography_error_max": np.nan}
    for t in hc["success_thresholds_px"]:
        out[f"homography_success_{int(t)}px"] = 0
    reason = homography_validity(H_est, grid, float(hc["min_abs_w"]))
    if reason:
        return out, reason
    if len(grid) < int(hc["min_valid_grid_points"]):
        return out, "insufficient_grid_points"
    proj_est, _ = project(H_est, grid)
    e = np.linalg.norm(proj_est - proj_gt, axis=1)
    if not np.all(np.isfinite(e)):
        return out, "invalid_homography_nonfinite_projection"
    out.update(homography_error_mean=float(e.mean()), homography_error_median=float(np.median(e)),
               homography_error_max=float(e.max()))
    for t in hc["success_thresholds_px"]:
        out[f"homography_success_{int(t)}px"] = int(e.mean() <= t)
    ts = sorted(hc["success_thresholds_px"])
    if e.mean() <= ts[0]:
        return out, ""
    exceeded = [t for t in ts if e.mean() > t]
    return out, f"projection_error_gt_{int(max(exceeded))}px"
