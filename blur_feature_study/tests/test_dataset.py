import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import HPATCHES, ROOT, needs_data
from src.dataset import (load_pair, resize_homography, resized_size, scale_matrix,
                         select_sequences)


def test_resized_size():
    assert resized_size(1000, 2000, 960) == (480, 960)
    assert resized_size(2000, 1000, 960) == (960, 480)
    assert resized_size(500, 700, 960) == (500, 700)          # no upscaling
    assert max(resized_size(2056, 3088, 960)) == 960


@pytest.mark.parametrize("new_w", [624, 504, 960])
def test_scale_matrix_matches_cv2_resize(new_w):
    """Blob centroid after cv2.resize(INTER_AREA) must land at S @ p (pixel-centre convention)."""
    h, w = 700, 1200
    x0, y0 = 517.3, 288.6
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 12.0 ** 2))
    nh, nw = resized_size(h, w, new_w)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    sy, sx = np.mgrid[0:nh, 0:nw]
    cx, cy = (small * sx).sum() / small.sum(), (small * sy).sum() / small.sum()
    S = scale_matrix((h, w), (nh, nw))
    px, py, _ = S @ np.array([x0, y0, 1.0])
    assert abs(cx - px) < 0.02 and abs(cy - py) < 0.02


def test_resize_homography_point_consistency():
    rng = np.random.default_rng(1)
    H = np.array([[0.9, 0.05, 30.0], [-0.04, 1.1, -12.0], [1e-4, -5e-5, 1.0]])
    S_r = scale_matrix((1000, 1500), resized_size(1000, 1500, 960))
    S_t = scale_matrix((900, 1400), resized_size(900, 1400, 960))
    Hr = resize_homography(H, S_r, S_t)
    for _ in range(20):
        p = np.array([*rng.uniform(0, 1400, 2), 1.0])
        q = H @ p; q /= q[2]
        q_r = S_t @ q
        p_r = S_r @ p
        q2 = Hr @ p_r; q2 /= q2[2]
        assert np.allclose(q2[:2], q_r[:2], atol=1e-9)


@needs_data
def test_selection_deterministic_and_disjoint(cfg):
    ds = cfg["dataset"]
    a = select_sequences(HPATCHES, 42, ds["n_validation"], ds["n_test"])
    b = select_sequences(HPATCHES, 42, ds["n_validation"], ds["n_test"])
    assert a == b
    val = {s["sequence_id"] for s in a if s["split"] == "validation"}
    test = {s["sequence_id"] for s in a if s["split"] == "test"}
    assert len(val) == 4 and len(test) == 12 and not (val & test)
    for s in a:
        assert s["sequence_id"].startswith("i_" if s["sequence_type"] == "illumination" else "v_")


@needs_data
def test_manifest_consistent(cfg):
    m = pd.read_csv(ROOT / cfg["paths"]["manifest"])
    assert len(m) == 32 and m.pair_id.is_unique
    assert (m.groupby("split").size().to_dict() == {"test": 24, "validation": 8})
    assert set(m.target_index) == {2, 5} and set(m.reference_index) == {1}
    ds = cfg["dataset"]
    sel = select_sequences(HPATCHES, 42, ds["n_validation"], ds["n_test"])
    assert set(m.sequence_id) == {s["sequence_id"] for s in sel}
    for _, r in m.iterrows():
        assert max(r.reference_h, r.reference_w) <= 960 and max(r.target_h, r.target_w) <= 960


@needs_data
def test_load_pair_shapes(cfg):
    m = pd.read_csv(ROOT / cfg["paths"]["manifest"])
    p = load_pair(m.iloc[0], cfg)
    assert p.ref.dtype == np.uint8 and p.tgt.dtype == np.uint8 and p.ref.ndim == 2
    assert abs(p.H_gt[2, 2] - 1) < 1e-12
