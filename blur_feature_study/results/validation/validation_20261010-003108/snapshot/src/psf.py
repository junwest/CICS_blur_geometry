"""Point spread functions. All kernels are odd-sized, non-negative, sum to 1,
centred on the central pixel and point-symmetric (k == k[::-1, ::-1])."""
import numpy as np


def gaussian_psf(sigma, truncate=4.0):
    r = int(np.ceil(truncate * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def disk_psf(radius, supersample=33):
    """Anti-aliased uniform disk (defocus). Each pixel weight = fraction of the
    pixel area inside the circle of `radius` centred on the central pixel centre."""
    r = int(np.ceil(radius))
    size = 2 * r + 1
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    centers = np.arange(-r, r + 1, dtype=np.float64)
    coords = (centers[:, None] + sub[None, :]).ravel()
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    inside = (xx ** 2 + yy ** 2) <= radius ** 2
    k = inside.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return k / k.sum()


def motion_psf(length, angle_deg):
    """Uniform linear motion of `length` pixels centred on the kernel centre.
    0 deg = horizontal (x), 90 deg = vertical (y). Only axis-aligned angles are
    supported so that no interpolation (and no sub-pixel centre shift) is needed."""
    length = int(length)
    if length % 2 != 1:
        raise ValueError("motion length must be odd for an exactly centred kernel")
    k = np.zeros((length, length), dtype=np.float64)
    c = length // 2
    if int(angle_deg) % 180 == 0:
        k[c, :] = 1.0
    elif int(angle_deg) % 180 == 90:
        k[:, c] = 1.0
    else:
        raise NotImplementedError("only 0 and 90 degree motion are used in this study")
    return k / k.sum()


def make_psf(cond, cfg):
    t = cond["blur_type"]
    if t == "gaussian":
        return gaussian_psf(float(cond["strength"]), cfg["blur"]["gaussian_truncate"])
    if t == "defocus_disk":
        return disk_psf(float(cond["strength"]), cfg["blur"]["disk_supersample"])
    if t == "motion_linear":
        return motion_psf(int(cond["strength"]), int(cond["angle"]))
    raise ValueError(f"unknown blur type {t}")


def psf_report(k):
    """Numbers used by the automatic PSF checks."""
    h, w = k.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    s = k.sum()
    return {
        "shape": [h, w],
        "sum": float(s),
        "min": float(k.min()),
        "centroid_dx": float((k * xx).sum() / s - cx),
        "centroid_dy": float((k * yy).sum() / s - cy),
        "odd": bool(h % 2 == 1 and w % 2 == 1),
        "point_symmetric_maxdiff": float(np.abs(k - k[::-1, ::-1]).max()),
        "radius": int(max(h, w) // 2),
    }


def psf_ok(k, tol=1e-9):
    r = psf_report(k)
    return (abs(r["sum"] - 1.0) < tol and r["min"] >= 0 and r["odd"]
            and abs(r["centroid_dx"]) < tol and abs(r["centroid_dy"]) < tol
            and r["point_symmetric_maxdiff"] < tol)
