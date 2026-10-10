import numpy as np
import pytest

from src.psf import disk_psf, gaussian_psf, make_psf, motion_psf, psf_ok, psf_report


def test_all_conditions_valid(cfg):
    assert len(cfg["blur"]["conditions"]) == 8
    for cond in cfg["blur"]["conditions"]:
        k = make_psf(cond, cfg)
        r = psf_report(k)
        assert psf_ok(k), (cond, r)
        assert abs(r["sum"] - 1) < 1e-12
        assert abs(r["centroid_dx"]) < 1e-12 and abs(r["centroid_dy"]) < 1e-12


@pytest.mark.parametrize("sigma,size", [(1.5, 13), (3.0, 25)])
def test_gaussian_size(sigma, size):
    assert gaussian_psf(sigma, 4.0).shape == (size, size)


@pytest.mark.parametrize("r,size", [(3, 7), (5, 11)])
def test_disk(r, size):
    k = disk_psf(r, 33)
    assert k.shape == (size, size)
    c = size // 2
    # uniform inside, zero at the corners, edge pixels partially covered
    assert np.isclose(k[c, c], k[c, c + 1]) and np.isclose(k[c, c], k[c + 1, c + 1])
    assert k[0, 0] == 0
    assert 0 < k[c, 0] < k[c, c]
    # unnormalised area ~ pi r^2
    area = k.sum() / k[c, c]
    assert abs(area - np.pi * r * r) / (np.pi * r * r) < 0.01


@pytest.mark.parametrize("length", [9, 17])
def test_motion_orientation(length):
    c = length // 2
    h = motion_psf(length, 0)
    v = motion_psf(length, 90)
    assert np.count_nonzero(h) == length and np.all(h[c, :] > 0)
    assert np.count_nonzero(v) == length and np.all(v[:, c] > 0)
    assert np.allclose(h.T, v)


def test_motion_rejects_even_or_oblique():
    with pytest.raises(ValueError):
        motion_psf(10, 0)
    with pytest.raises(NotImplementedError):
        motion_psf(9, 45)
