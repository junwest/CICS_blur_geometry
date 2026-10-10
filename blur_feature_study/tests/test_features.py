import json

import cv2
import numpy as np
import pytest

from src import features as F
from src.metrics import gt_errors


@pytest.fixture(scope="module")
def big_texture():
    rng = np.random.default_rng(7)
    img = cv2.GaussianBlur(rng.normal(0, 1, (480, 640)), (0, 0), 1.5)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def test_grids(cfg):
    for det, n in (("ORB", 8), ("SIFT", 4)):
        g = F.tuning_grid(det, cfg)
        assert len(g) == n
        names = [F.config_name(det, p, cfg["detectors"][det]["default"]) for p in g]
        assert names.count("default") == 1 and len(set(names)) == n
        assert all(p["nfeatures"] == 1500 for p in g)


@pytest.mark.parametrize("det", ["ORB", "SIFT"])
def test_detect_mask_budget_determinism(cfg, big_texture, det):
    p = cfg["detectors"][det]["default"]
    mask = F.border_mask(big_texture.shape, 24)
    a = F.detect(det, p, big_texture, mask)
    b = F.detect(det, p, big_texture, mask)
    assert len(a) > 0
    assert np.array_equal(a.kp, b.kp) and np.array_equal(a.desc, b.desc)
    assert a.xy[:, 0].min() >= 24 - 0.5 and a.xy[:, 0].max() <= 640 - 24 + 0.5
    assert len(a) <= 1500 * 1.05


@pytest.mark.parametrize("det", ["ORB", "SIFT"])
def test_match_identity(cfg, big_texture, det):
    p = cfg["detectors"][det]["default"]
    mask = F.border_mask(big_texture.shape, 24)
    a = F.detect(det, p, big_texture, mask)
    mr = F.ratio_match(det, a, a, 0.75)
    assert len(mr.query_idx) > 0.5 * len(a)
    e = gt_errors(np.eye(3), a.xy[mr.query_idx], a.xy[mr.train_idx])
    assert (e <= 3).mean() > 0.99


def test_match_empty():
    empty = F.Features(np.zeros((0, 6)), None, 0.0)
    mr = F.ratio_match("ORB", empty, empty, 0.75)
    assert len(mr.query_idx) == 0


def test_ransac_recovers_h(cfg):
    rng = np.random.default_rng(3)
    H = np.array([[1.05, 0.02, 12], [-0.03, 0.97, -7], [1e-4, -2e-4, 1]])
    src = rng.uniform(0, 600, (200, 2))
    ph = np.hstack([src, np.ones((200, 1))]) @ H.T
    dst = ph[:, :2] / ph[:, 2:]
    dst[:60] = rng.uniform(0, 600, (60, 2))                  # 30 % outliers
    Hest, inl, reason, _ = F.estimate_homography(src, dst, cfg["ransac"])
    assert reason == "" and inl[60:].mean() > 0.95 and inl[:60].mean() < 0.1
    assert np.allclose(Hest / Hest[2, 2], H, atol=1e-6)
    _, _, reason, _ = F.estimate_homography(src[:3], dst[:3], cfg["ransac"])
    assert reason == "insufficient_matches"


def test_params_json_stable(cfg):
    p = cfg["detectors"]["SIFT"]["default"]
    assert json.loads(F.params_json(p)) == p
