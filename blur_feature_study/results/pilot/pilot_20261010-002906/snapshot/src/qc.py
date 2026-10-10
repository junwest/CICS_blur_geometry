"""Minimal QC images for implementation checking (not paper figures)."""
from pathlib import Path

import cv2
import numpy as np

from .imaging import apply_blur, unsharp_mask, wiener_restore
from .io_utils import imwrite_unicode
from .metrics import canny, project
from .psf import make_psf


def processed_images(pair, cond, cfg, balance):
    k = make_psf(cond, cfg)
    b = apply_blur(pair.tgt, k)
    u = cfg["processing"]["unsharp_mask"]
    return {"clean": pair.tgt, "blurred": b,
            "unsharp_mask": unsharp_mask(b, u["alpha"], u["sigma"], u["truncate"]),
            "wiener_known_psf": wiener_restore(b, k, balance)}


def _label(img, text):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def processing_panel(pair, cfg, balances, out_path, crop=220):
    """rows = blur conditions, cols = clean / blurred / unsharp / wiener (centre crop)."""
    h, w = pair.tgt.shape
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    rows = []
    for cond in cfg["blur"]["conditions"]:
        ims = processed_images(pair, cond, cfg, balances[cond["blur_id"]])
        tiles = [_label(ims[k][y0:y0 + crop, x0:x0 + crop], f"{cond['blur_id']}:{k[:7]}")
                 for k in ("clean", "blurred", "unsharp_mask", "wiener_known_psf")]
        rows.append(np.hstack(tiles))
    imwrite_unicode(out_path, np.vstack(rows))


def edge_overlay(clean, proc, cc, out_path, tol=2.0):
    """green: processed edge near a clean edge (TP), red: spurious processed edge (FP),
    blue: clean edge missed (FN)."""
    from scipy.ndimage import distance_transform_edt
    ec, ep = canny(clean, cc), canny(proc, cc)
    dc = distance_transform_edt(~ec) if ec.any() else np.full(ec.shape, np.inf)
    dp = distance_transform_edt(~ep) if ep.any() else np.full(ep.shape, np.inf)
    vis = cv2.cvtColor((clean * 0.4).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[ep & (dc <= tol)] = (0, 255, 0)
    vis[ep & (dc > tol)] = (0, 0, 255)
    vis[ec & (dp > tol)] = (255, 0, 0)
    imwrite_unicode(out_path, vis)


def match_overlay(ref, tgt, npz_path, out_path, thr=3.0, max_lines=300):
    """ref | target. Lines: green = GT-correct (<= thr px), red = incorrect.
    Target: cyan = reference border projected by H_gt, yellow = by H_est."""
    d = np.load(npz_path)
    h1, w1 = ref.shape; h2, w2 = tgt.shape
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:] = cv2.cvtColor(tgt, cv2.COLOR_GRAY2BGR)
    rx, tx, err = d["ref_xy"], d["tgt_xy"], d["gt_err"]
    idx = np.arange(len(err))
    if len(idx) > max_lines:
        idx = np.random.default_rng(0).choice(idx, max_lines, replace=False)
    for i in idx:
        c = (0, 200, 0) if err[i] <= thr else (0, 0, 255)
        p1 = (int(round(rx[i, 0])), int(round(rx[i, 1])))
        p2 = (int(round(tx[i, 0])) + w1, int(round(tx[i, 1])))
        cv2.line(canvas, p1, p2, c, 1, cv2.LINE_AA)
    border = np.array([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]], np.float64)
    for H, col in ((d["H_gt"], (255, 255, 0)), (d["H_est"], (0, 255, 255))):
        if np.all(np.isfinite(H)):
            p, wv = project(H, border)
            if np.all(wv > 0) and np.all(np.abs(p) < 1e5):
                pts = (p + [w1, 0]).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
    n_ok = int((err <= thr).sum())
    cv2.putText(canvas, f"matches={len(err)} gt_correct@{thr:g}px={n_ok}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(out_path, canvas)


def keypoint_overlay(clean_xy, proc_xy, img, out_path, tol=3.0):
    """green: clean keypoint repeated in processed image (within tol), red: not repeated."""
    from scipy.spatial import cKDTree
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(proc_xy):
        dist, _ = cKDTree(proc_xy).query(clean_xy)
    else:
        dist = np.full(len(clean_xy), np.inf)
    for (x, y), dd in zip(clean_xy, dist):
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 200, 0) if dd <= tol else (0, 0, 255), 1)
    imwrite_unicode(out_path, vis)
