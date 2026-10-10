import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_config  # noqa: E402

HPATCHES = ROOT / "data" / "hpatches-sequences-release"
needs_data = pytest.mark.skipif(not HPATCHES.exists(), reason="HPatches data not available")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def texture():
    """Deterministic natural-like test image (smoothed noise + shapes), uint8."""
    import cv2
    rng = np.random.default_rng(42)
    img = rng.normal(0, 1, (240, 320))
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = (img - img.min()) / (img.max() - img.min()) * 160 + 40
    cv2.rectangle(img, (60, 50), (150, 140), 230, -1)
    cv2.circle(img, (230, 160), 40, 20, -1)
    cv2.line(img, (20, 220), (300, 190), 250, 3)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
