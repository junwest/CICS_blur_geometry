"""Shared helpers for the extension experiments.

The base study code in src/ is imported read-only; nothing in src/, config/ or results/ is modified.
All outputs go to extension/results/<run_id>/ (a new directory per run, never reused).
"""
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import imread_unicode, load_config, config_hash, code_hash as base_code_hash  # noqa: E402
from src.dataset import load_manifest, load_pair  # noqa: E402

import cv2  # noqa: E402

RESULTS = EXT_ROOT / "results"
BASE_MAIN_RUN = PROJECT_ROOT / "results" / "main" / "main_20261010-003501"


def keep_awake():
    """Ask Windows not to sleep while THIS process runs (released automatically at process exit; no system
    setting is changed). A first Exp1 run was suspended ~17.6 h by host sleep (see DECISIONS.md)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        return True
    except Exception:
        return False


def load_ext_config():
    with open(EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_base():
    """Base config, validation-selected params, manifest (test split) and GT flags."""
    cfg = load_config()
    with open(PROJECT_ROOT / "config" / "selected_params_validation_20261010-003108.yaml", "r", encoding="utf-8") as f:
        sel = yaml.safe_load(f)
    man = load_manifest(cfg)
    fl = pd.read_csv(PROJECT_ROOT / "data" / "gt_quality_flags.csv")
    flags = dict(zip(fl.pair_id, fl.gt_quality_flag.astype(int)))
    return cfg, sel, man, flags


def test_pairs(man):
    return man[man.split == "test"].reset_index(drop=True)


def ext_code_hash():
    h = hashlib.sha256()
    for p in sorted((EXT_ROOT / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def pixel_sha(img):
    """Hash of the pixel array (shape + dtype + bytes); independent of the file container."""
    a = np.ascontiguousarray(img)
    h = hashlib.sha256()
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def new_run_dir(prefix):
    run_id = f"{prefix}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    d = RESULTS / run_id
    d.mkdir(parents=True, exist_ok=False)          # never reuse a run directory
    snap = d / "snapshot"
    (snap / "ext_src").mkdir(parents=True)
    for p in (EXT_ROOT / "src").glob("*.py"):
        shutil.copy2(p, snap / "ext_src" / p.name)
    shutil.copy2(EXT_ROOT / "config" / "extension_config.yaml", snap / "extension_config.yaml")
    return run_id, d


def write_png_new(path, img):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(path)
    buf.tofile(str(path))


def read_png_gray(path):
    return imread_unicode(path, cv2.IMREAD_UNCHANGED)


def write_json(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def rel(p):
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def blur_cond(cfg, blur_id):
    for c in cfg["blur"]["conditions"]:
        if c["blur_id"] == blur_id:
            return c
    raise KeyError(blur_id)


def environment_info():
    import platform
    import scipy
    import skimage
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "opencv": cv2.__version__, "cv2_threads": cv2.getNumThreads(), "scipy": scipy.__version__,
            "scikit-image": skimage.__version__, "pandas": pd.__version__,
            "base_code_hash": base_code_hash(), "ext_code_hash": ext_code_hash(),
            "processor": platform.processor()}
    return info
