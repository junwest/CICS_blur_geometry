"""Spatial distribution of correspondences in the REFERENCE frame (extension_config.yaml `spatial_distribution`).

overlap  = reference valid rectangle  intersected with  H_gt^-1(target valid rectangle)   (convex polygon)
hull_coverage  = area(convex hull of points  intersected with overlap) / area(overlap)
cell_occupancy = 4x4 cells over the reference valid rectangle; a cell is eligible if >= 50 % of its 10x10
                 sample points lie inside the overlap; occupancy = eligible cells with >= 1 point / eligible cells
Valid rectangle in pixel-centre coordinates: [m, W-1-m] x [m, H-1-m] (same as src.metrics.valid_grid).
"""
import cv2
import numpy as np

from src.metrics import project


def valid_rect(shape, margin):
    h, w = shape
    return np.array([[margin, margin], [w - 1 - margin, margin], [w - 1 - margin, h - 1 - margin],
                     [margin, h - 1 - margin]], dtype=np.float64)


def _ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly if signed >= 0 else poly[::-1].copy()


def _intersect(p1, p2):
    area, inter = cv2.intersectConvexConvex(_ccw(p1).astype(np.float32), _ccw(p2).astype(np.float32))
    if inter is None or area <= 0:
        return 0.0, np.zeros((0, 2))
    return float(area), inter.reshape(-1, 2).astype(np.float64)


class SpatialFrame:
    def __init__(self, H_gt, ref_shape, tgt_shape, margin, n_cells=4, n_samples=10, min_eligible_frac=0.5):
        self.margin = margin
        self.ref_rect = valid_rect(ref_shape, margin)
        q, w = project(np.linalg.inv(H_gt), valid_rect(tgt_shape, margin))
        self.reason = ""
        if not (np.all(w > 0) and np.all(np.isfinite(q))):
            self.reason = "target_rect_not_projectable"
            self.overlap, self.area = np.zeros((0, 2)), 0.0
        else:
            self.area, self.overlap = _intersect(self.ref_rect, q)
            if self.area <= 0:
                self.reason = "empty_overlap"
        h, w_ = ref_shape
        self.x0, self.y0 = float(margin), float(margin)
        self.cw = (w_ - 1 - 2 * margin) / n_cells
        self.ch = (h - 1 - 2 * margin) / n_cells
        self.n = n_cells
        elig = np.zeros((n_cells, n_cells), dtype=bool)
        if self.area > 0:
            poly = self.overlap.astype(np.float32).reshape(-1, 1, 2)
            fr = (np.arange(n_samples) + 0.5) / n_samples
            for i in range(n_cells):
                for j in range(n_cells):
                    xs = self.x0 + (j + fr) * self.cw
                    ys = self.y0 + (i + fr) * self.ch
                    inside = sum(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for y in ys for x in xs)
                    elig[i, j] = inside >= min_eligible_frac * n_samples * n_samples
        self.eligible = elig

    def measure(self, pts):
        """(hull_coverage, cell_occupancy) of reference-frame points (N,2)."""
        if self.area <= 0:
            return np.nan, np.nan
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        cov = 0.0
        if len(np.unique(np.round(pts, 6), axis=0)) >= 3:
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
            if len(hull) >= 3:
                cov = _intersect(hull, self.overlap)[0] / self.area
        n_el = int(self.eligible.sum())
        if n_el == 0:
            return cov, np.nan
        occ = np.zeros_like(self.eligible)
        if len(pts):
            j = np.clip(np.floor((pts[:, 0] - self.x0) / self.cw).astype(int), 0, self.n - 1)
            i = np.clip(np.floor((pts[:, 1] - self.y0) / self.ch).astype(int), 0, self.n - 1)
            occ[i, j] = True
        return cov, float((occ & self.eligible).sum() / n_el)
