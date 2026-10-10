import cv2
import numpy as np
import pytest
from scipy import ndimage

from src.imaging import (apply_blur, convolve_reflect, quantize_u8, unsharp_mask, wiener_float,
                         wiener_restore)
from src.metrics import psnr
from src.psf import gaussian_psf, make_psf


def _shift(a, b, m=24):
    """Gradient least-squares translation estimate of b relative to a (b(x) ~ a(x - s)).
    Unbiased for zero-phase (symmetric) blur differences; calibrated in test_shift_estimator.
    (cv2.phaseCorrelate was rejected: it discards magnitude, so quantisation noise at
    frequencies where the OTF is ~0 or negative biases its sub-pixel peak by ~0.1 px.)"""
    a = a.astype(np.float64); b = b.astype(np.float64)
    gy, gx = np.gradient(a)
    sl = (slice(m, -m), slice(m, -m))
    gx, gy = gx[sl], gy[sl]
    A = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
    s = np.zeros(2)
    for _ in range(10):                      # iterative (Gauss-Newton) refinement
        bw = cv2.warpAffine(b, np.float64([[1, 0, s[0]], [0, 1, s[1]]]), b.shape[::-1],
                            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        d = (bw - a)[sl]
        step = np.linalg.solve(A, np.array([(gx * d).sum(), (gy * d).sum()]))
        s = s - step
        if np.hypot(*step) < 1e-4:
            break
    return s[0], s[1]


def test_shift_estimator_calibration(texture):
    M = np.float64([[1, 0, 0.3], [0, 1, -0.2]])
    shifted = cv2.warpAffine(texture.astype(np.float64), M, texture.shape[::-1],
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    dx, dy = _shift(texture, shifted)
    assert abs(dx - 0.3) < 0.03 and abs(dy + 0.2) < 0.03
    assert np.hypot(*_shift(texture, texture)) == 0


def test_wiener_impulse_response_centred(cfg):
    """Exact no-shift test: blur + Wiener (float, no quantisation) is a linear system whose
    interior impulse response must be centred on the impulse and point-symmetric."""
    for cond in cfg["blur"]["conditions"]:
        k = make_psf(cond, cfg)
        for bal in cfg["processing"]["wiener"]["balance_candidates"]:
            img = np.zeros((161, 181)); img[80, 90] = 1.0
            r = wiener_float(convolve_reflect(img, k), k, bal)
            win = r[80 - 40:80 + 41, 90 - 40:90 + 41]
            assert np.abs(win - win[::-1, ::-1]).max() < 1e-9, cond
            yy, xx = np.mgrid[-40:41, -40:41]
            assert abs((win * xx).sum() / win.sum()) < 1e-9 and abs((win * yy).sum() / win.sum()) < 1e-9
            assert win[40, 40] == win.max()


def test_convolution_matches_scipy_reflect():
    rng = np.random.default_rng(0)
    img = rng.random((60, 80)) * 255
    k = rng.random((5, 7)); k /= k.sum()          # asymmetric kernel -> checks flip/orientation
    ours = convolve_reflect(img, k)
    ref = ndimage.convolve(img, k, mode="reflect")
    assert np.abs(ours - ref).max() < 1e-9


def test_impulse_response_is_centred_psf(cfg):
    for cond in cfg["blur"]["conditions"]:
        k = make_psf(cond, cfg)
        img = np.zeros((101, 121)); img[50, 60] = 1.0
        out = convolve_reflect(img, k)
        r = k.shape[0] // 2
        assert np.allclose(out[50 - r:50 + r + 1, 60 - r:60 + r + 1], k, atol=1e-12)
        yy, xx = np.mgrid[0:101, 0:121]
        assert abs((out * xx).sum() / out.sum() - 60) < 1e-9
        assert abs((out * yy).sum() / out.sum() - 50) < 1e-9


def test_blur_output_type_and_no_shift(cfg, texture):
    """Exact, estimator-free no-shift test: a zero-phase operator commutes with 180-degree
    rotation (a translation s would show up as a 2s discrepancy). An image-based LS shift
    estimate is NOT used for blurred images: for heavy directional blur (motion 17 px) it
    shows a content-dependent bias (~0.07 px) even for the unquantised, exactly symmetric blur."""
    for cond in cfg["blur"]["conditions"]:
        k = make_psf(cond, cfg)
        b = apply_blur(texture, k)
        assert b.dtype == np.uint8 and b.shape == texture.shape
        f = convolve_reflect(texture, k)
        rot = convolve_reflect(texture[::-1, ::-1], k)[::-1, ::-1]
        assert np.abs(rot - f).max() < 1e-9, cond
        bal = cfg["processing"]["wiener"]["balance_candidates"][0]
        w = wiener_float(f / 255.0, k, bal)
        w_rot = wiener_float(f[::-1, ::-1] / 255.0, k, bal)[::-1, ::-1]
        assert np.abs(w - w_rot).max() < 1e-9, cond


def test_unsharp_formula_and_clipping(texture):
    b = apply_blur(texture, gaussian_psf(1.5))
    out = unsharp_mask(b, 1.0, 1.2, 4.0)
    i = b.astype(np.float64)
    g = ndimage.convolve(i, gaussian_psf(1.2, 4.0), mode="reflect")
    expect = np.clip(np.rint(i + (i - g)), 0, 255).astype(np.uint8)
    assert np.array_equal(out, expect)
    step = np.zeros((40, 40), np.uint8); step[:, 20:] = 255
    s = unsharp_mask(step, 1.0, 1.2)
    assert s.min() >= 0 and s.max() <= 255 and s.dtype == np.uint8


@pytest.mark.parametrize("balance", [0.005, 0.02, 0.08])
def test_wiener_improves_and_does_not_shift(cfg, texture, balance):
    m = cfg["preprocessing"]["valid_margin_px"]
    for cond in cfg["blur"]["conditions"]:
        k = make_psf(cond, cfg)
        b = apply_blur(texture, k)
        r = wiener_restore(b, k, balance)
        assert r.dtype == np.uint8 and r.shape == texture.shape
        dx, dy = _shift(texture, r)
        assert abs(dx) < 0.05 and abs(dy) < 0.05, (cond, dx, dy)
        assert psnr(texture, r, m) > psnr(texture, b, m), cond


def test_wiener_no_boundary_artifacts(cfg, texture):
    """Mirror extension: error in the outer band must not exceed the interior error much."""
    k = make_psf({"blur_type": "gaussian", "strength": 3.0}, cfg)
    r = wiener_restore(apply_blur(texture, k), k, 0.005)
    err = np.abs(r.astype(float) - texture.astype(float))
    band = np.ones_like(err, bool); band[10:-10, 10:-10] = False
    assert err[band].mean() <= 1.5 * err[~band].mean() + 1.0


def test_wiener_near_inverse_without_quantisation(cfg, texture):
    k = make_psf({"blur_type": "gaussian", "strength": 1.5}, cfg)
    blurred_float = convolve_reflect(texture, k)            # no quantisation noise
    r = wiener_restore(blurred_float, k, 1e-4)
    assert psnr(texture, r, 24) > psnr(texture, quantize_u8(blurred_float), 24) + 5
