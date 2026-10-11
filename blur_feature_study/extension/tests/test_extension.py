import numpy as np
import pytest

from extension.src import kernels, spatial, stagewise as S
from extension.src.tracking import link, locations
from src import features as F
from src.imaging import apply_blur, wiener_restore
from src.psf import gaussian_psf, motion_psf


# ------------------------------------------------------------------ kernels (Exp3)
def test_line_psf_reproduces_motion_psf():
    for L in (9, 13, 17, 21):
        assert np.array_equal(kernels.line_psf(L, 0), motion_psf(L, 0))


@pytest.mark.parametrize("ang", [5, 15])
def test_rotated_kernel_symmetric_normalised(ang):
    k = kernels.supplied_kernel(17, ang)
    assert k.shape[0] % 2 == 1 and k.shape[0] == k.shape[1]
    assert np.array_equal(k, k[::-1, ::-1])
    assert abs(k.sum() - 1) < 1e-12
    yy, xx = np.mgrid[:k.shape[0], :k.shape[1]]
    c = (k.shape[0] - 1) / 2
    assert abs((k * (xx - c)).sum()) < 1e-12 and abs((k * (yy - c)).sum()) < 1e-12
    # end-point lateral offset 8.5*sin(ang) -> rows used = 2*round(offset)+1
    off = 8.5 * np.sin(np.deg2rad(ang))
    assert (k.sum(axis=1) > 0).sum() == 2 * int(np.floor(off + 0.5)) + 1


def test_supplied_exact_is_generation_kernel():
    assert np.array_equal(kernels.supplied_kernel(17, 0), motion_psf(17, 0))


# ------------------------------------------------------------------ stage-wise extraction
DEFAULTS = {"ORB": {"nfeatures": 1500, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "firstLevel": 0,
                    "WTA_K": 2, "scoreType": "HARRIS", "patchSize": 31, "fastThreshold": 20},
            "SIFT": {"nfeatures": 1500, "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10.0,
                     "sigma": 1.6, "enable_precise_upscale": False}}


@pytest.mark.parametrize("det", ["ORB", "SIFT"])
@pytest.mark.parametrize("sigma", [0.0, 3.0])
def test_separate_path_equals_detect_and_compute(texture, det, sigma):
    img = texture if sigma == 0 else apply_blur(texture, gaussian_psf(sigma, 4.0))
    p = DEFAULTS[det]
    mask = F.border_mask(img.shape, 24)
    dac = F.detect(det, p, img, mask)
    kps, _ = S.detect_only(det, p, img, mask)
    feats, ids, info = S.describe(det, p, img, kps)
    assert np.array_equal(ids, np.arange(len(kps)))
    assert info["n_out"] == info["n_in"] and not info["reordered"] and info["attrs_changed"] == 0
    assert S.same_features(dac, feats) == (True, True)


def test_sift_sentinel_needed_and_effective(texture):
    """Heavy blur -> no octave -1 keypoint -> plain compute() differs from detectAndCompute, describe() does not."""
    img = apply_blur(texture, gaussian_psf(4.0, 4.0))
    p = DEFAULTS["SIFT"]
    mask = F.border_mask(img.shape, 24)
    kps, _ = S.detect_only("SIFT", p, img, mask)
    if min(S.unpack_sift_octave(k.octave)[0] for k in kps) < 0:
        pytest.skip("this synthetic image still has octave -1 keypoints")
    dac = F.detect("SIFT", p, img, mask)
    _, plain = F.create_detector("SIFT", p).compute(img, S.with_ids(kps))
    assert not np.array_equal(plain, dac.desc)
    feats, ids, info = S.describe("SIFT", p, img, kps)
    assert info["sift_sentinel_effective"]
    assert S.same_features(dac, feats) == (True, True)


@pytest.mark.parametrize("det", ["ORB", "SIFT"])
def test_cross_image_describe_keeps_keypoints(texture, det):
    psf = motion_psf(9, 0)
    B = apply_blur(texture, psf)
    Rw = wiener_restore(B, psf, 0.005)
    p = DEFAULTS[det]
    kB, _ = S.detect_only(det, p, B, F.border_mask(B.shape, 24))
    fBR, ids, info = S.describe(det, p, Rw, kB)
    assert info["n_out"] == len(kB) and info["attrs_changed"] == 0 and not info["reordered"]
    assert np.array_equal(fBR.kp, S.kp_array(kB))


# ------------------------------------------------------------------ spatial distribution
def test_spatial_identity_full_overlap():
    sp = spatial.SpatialFrame(np.eye(3), (300, 400), (300, 400), 24)
    assert abs(sp.area - (400 - 1 - 48) * (300 - 1 - 48)) < 1e-3
    assert sp.eligible.all()
    corners = spatial.valid_rect((300, 400), 24)
    cov, occ = sp.measure(corners)
    assert abs(cov - 1) < 1e-6 and abs(occ - 4 / 16) < 1e-12
    assert sp.measure(np.zeros((0, 2))) == (0.0, 0.0)
    assert sp.measure(corners[:2])[0] == 0.0


def test_spatial_shifted_overlap():
    H = np.array([[1, 0, -200.0], [0, 1, 0], [0, 0, 1]])        # target = reference shifted left by 200 px
    sp = spatial.SpatialFrame(H, (300, 400), (300, 400), 24)
    # overlap: x in [224, 375], y in [24, 275]
    assert abs(sp.area - (375 - 224) * (275 - 24)) < 1e-3
    assert sp.eligible[:, :2].sum() == 0 and sp.eligible[:, 2:].all()


# ------------------------------------------------------------------ linking rule
def test_link_status_rule():
    Q = np.array([[10.0, 10.0], [50.0, 50.0], [100.0, 100.0], [103.5, 100.0], [200.0, 200.0]])
    Lc = np.array([[11.0, 10.0],                     # unique for Q0
                   [51.0, 50.0], [49.0, 50.5],        # two candidates for Q1 -> ambiguous
                   [101.8, 100.0],                    # within 3 px of Q2 and Q3 -> both ambiguous
                   [300.0, 300.0]])                   # Q4 lost
    st, cand, uq, l_linked = link(Q, Lc, 3.0 + 1e-9)
    assert st.tolist() == [1, 2, 2, 2, 0]
    assert uq[0] == 0 and (uq[1:] == -1).all()
    assert l_linked.tolist() == [True, True, True, True, False]


def test_locations_dedup():
    kp = np.array([[1.0, 2.0, 3, 10, 0, 0], [1.0, 2.0, 3, 50, 0, 0], [5.0, 5.0, 3, 0, 0, 0]])
    xy, inv, first = locations(kp, 3)
    assert len(xy) == 2 and inv[0] == inv[1] != inv[2]
