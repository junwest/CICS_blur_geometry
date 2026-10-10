import numpy as np
import pytest
from skimage.metrics import structural_similarity

from src import metrics as M

HC = {"grid_n": 10, "min_valid_grid_points": 10, "success_thresholds_px": [3, 5], "min_abs_w": 1e-8}
SSIM_CFG = {"data_range": 255, "gaussian_weights": True, "sigma": 1.5, "use_sample_covariance": False,
            "K1": 0.01, "K2": 0.03}


def T(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


# ------------------------------------------------------------------ PSNR / SSIM
def test_psnr_identical_and_known(texture):
    assert M.psnr(texture, texture, 24) == float("inf")
    b = texture.astype(np.int16) + 5
    assert abs(M.psnr(texture, np.clip(b, 0, 255).astype(np.uint8), 24) - 20 * np.log10(255 / 5)) < 0.05


def test_psnr_ignores_border(texture):
    b = texture.copy(); b[:10, :] = 0; b[:, -10:] = 255
    assert M.psnr(texture, b, 24) == float("inf")


def test_ssim(texture):
    assert M.ssim(texture, texture, 24, SSIM_CFG) == pytest.approx(1.0)
    noisy = np.clip(texture + np.random.default_rng(0).normal(0, 10, texture.shape), 0, 255).astype(np.uint8)
    s = M.ssim(texture, noisy, 24, SSIM_CFG)
    ref = structural_similarity(texture[24:-24, 24:-24].astype(float), noisy[24:-24, 24:-24].astype(float),
                                data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert s == pytest.approx(ref) and s < 1


# ------------------------------------------------------------------ Edge F1
def _line_map(shape=(100, 100), col=50):
    e = np.zeros(shape, bool); e[10:90, col] = True
    return e


def test_edge_identical():
    e = _line_map()
    r = M.edge_prf(e, e, 2.0, 5)
    assert r["edge_precision"] == r["edge_recall"] == r["edge_f1"] == 1.0


def test_edge_tolerance():
    c = _line_map(col=50)
    assert M.edge_prf(c, _line_map(col=52), 2.0, 5)["edge_f1"] == 1.0    # 2 px -> inside tolerance
    assert M.edge_prf(c, _line_map(col=53), 2.0, 5)["edge_f1"] == 0.0    # 3 px -> outside


def test_spurious_edges_reduce_precision_only():
    c = _line_map(col=50)
    p = c.copy(); p[10:90, 20] = True                                      # false edge
    r = M.edge_prf(c, p, 2.0, 5)
    assert r["edge_recall"] == 1.0 and r["edge_precision"] == pytest.approx(0.5)


def test_edge_empty_conventions():
    c = _line_map(); z = np.zeros_like(c)
    r = M.edge_prf(c, z, 2.0, 5)
    assert np.isnan(r["edge_precision"]) and r["edge_recall"] == 0 and r["edge_f1"] == 0
    r = M.edge_prf(z, c, 2.0, 5)
    assert np.isnan(r["edge_f1"])


# ------------------------------------------------------------------ Repeatability
def test_repeatability_basic():
    a = np.array([[10.0, 10], [50, 50], [80, 20]])
    assert M.repeatability(a, a, 3.0)[0] == 1.0
    assert M.repeatability(a, a + [2.9, 0], 3.0)[0] == 1.0
    assert M.repeatability(a, a + [3.1, 0], 3.0)[0] == 0.0
    assert M.repeatability(a, np.zeros((0, 2)), 3.0)[0] == 0.0


def test_repeatability_one_to_one():
    clean = np.array([[10.0, 10], [11, 10]])
    proc = np.array([[10.5, 10]])
    rep, n_rep, n_c = M.repeatability(clean, proc, 3.0)
    assert (rep, n_rep, n_c) == (0.5, 1, 2)


def test_repeatability_optimal_not_greedy():
    a = np.array([[0.0, 0], [2.5, 0]])
    b = np.array([[2.0, 0], [-2.5, 0]])
    assert M.one_to_one_count(a, b, 3.0) == 2


def test_repeatability_dedup():
    a = np.array([[10.0, 10], [10.0, 10], [30, 30]])    # SIFT multi-orientation duplicate
    rep, n_rep, n_c = M.repeatability(a, np.array([[10.0, 10]]), 3.0)
    assert n_c == 2 and rep == 0.5


# ------------------------------------------------------------------ GT errors
def test_gt_errors():
    H = T(5, -3)
    ref = np.array([[10.0, 10], [20, 20]])
    tgt = np.array([[15.0, 7], [25, 21]])
    e = M.gt_errors(H, ref, tgt)
    assert e[0] == pytest.approx(0) and e[1] == pytest.approx(4)
    Hbad = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1.0]])   # w < 0 -> not projectable
    assert np.isinf(M.gt_errors(Hbad, ref, tgt)).all()


# ------------------------------------------------------------------ Homography
@pytest.mark.parametrize("d,s3,s5,reason", [(0, 1, 1, ""), (4, 0, 1, "projection_error_gt_3px"),
                                            (6, 0, 0, "projection_error_gt_5px")])
def test_homography_eval_thresholds(d, s3, s5, reason):
    Hgt = np.array([[1.02, 0.01, 7], [-0.01, 0.98, -4], [1e-5, 2e-5, 1]])
    Hest = T(d, 0) @ Hgt
    out, r = M.evaluate_homography(Hest, Hgt, (300, 400), (300, 400), HC, 24)
    assert out["homography_error_mean"] == pytest.approx(d, abs=1e-6)
    assert out["homography_success_3px"] == s3 and out["homography_success_5px"] == s5 and r == reason
    assert out["n_grid_valid"] >= 10


def test_homography_failures():
    Hgt = np.eye(3)
    out, r = M.evaluate_homography(None, Hgt, (300, 400), (300, 400), HC, 24)
    assert np.isnan(out["homography_error_mean"]) and out["homography_success_3px"] == 0
    assert r == "homography_estimation_failed"
    _, r = M.evaluate_homography(np.full((3, 3), np.nan), Hgt, (300, 400), (300, 400), HC, 24)
    assert r.startswith("invalid_homography")
    _, r = M.evaluate_homography(np.zeros((3, 3)), Hgt, (300, 400), (300, 400), HC, 24)
    assert r.startswith("invalid_homography")
    horizon = np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 2.0]])   # w changes sign inside the image
    _, r = M.evaluate_homography(horizon, Hgt, (300, 400), (300, 400), HC, 24)
    assert r == "invalid_homography_horizon_in_overlap"
    out, r = M.evaluate_homography(np.eye(3), T(5000, 0), (300, 400), (300, 400), HC, 24)
    assert r == "insufficient_grid_points" and out["n_grid_valid"] == 0


def test_grid_restricted_to_overlap():
    pts, proj = M.valid_grid(T(200, 0), (300, 400), (300, 400), 24, 10)
    assert len(pts) > 0 and np.all(proj[:, 0] <= 400 - 1 - 24)
    assert len(pts) < 100
