import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def cfg():
    from src.io_utils import load_config
    return load_config()


@pytest.fixture(scope="session")
def texture():
    """Deterministic textured uint8 image with structure at several scales."""
    import cv2
    rng = np.random.default_rng(7)
    img = np.zeros((360, 480), np.float64)
    for s in (2, 5, 11, 23):
        n = rng.random((360 // s + 2, 480 // s + 2))
        img += cv2.resize(n, (480, 360), interpolation=cv2.INTER_CUBIC) * s
    img = (img - img.min()) / (img.max() - img.min()) * 255
    for _ in range(40):
        x, y = rng.integers(30, 450), rng.integers(30, 330)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 12, int(y) + 9), float(rng.integers(0, 256)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)
