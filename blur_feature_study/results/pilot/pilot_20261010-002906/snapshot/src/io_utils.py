"""File I/O helpers.

cv2.imread/imwrite cannot handle non-ASCII Windows paths (the project lives under a
Korean user name), so images are read/written through np.fromfile/tofile + imdecode/imencode.
All writers refuse to overwrite existing files.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"cannot decode image: {path}")
    return img


def imwrite_unicode(path, img):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def load_homography(path):
    H = np.loadtxt(str(path), dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError(f"invalid homography file: {path}")
    return H


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "config" / "experiment_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg):
    """Hash of the full configuration (canonical JSON)."""
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def code_hash():
    """Hash over all src/*.py files, stored with every result row."""
    h = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def write_text_new(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_yaml_new(path, obj):
    write_text_new(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def write_json_new(path, obj):
    write_text_new(path, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
