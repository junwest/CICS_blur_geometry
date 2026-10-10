"""ORB / SIFT detection, ratio-test matching and RANSAC homography estimation."""
import itertools
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

ORB_SCORE = {"HARRIS": cv2.ORB_HARRIS_SCORE, "FAST": cv2.ORB_FAST_SCORE}
NORMS = {"ORB": cv2.NORM_HAMMING, "SIFT": cv2.NORM_L2}


def create_detector(name, p):
    if name == "ORB":
        return cv2.ORB_create(nfeatures=int(p["nfeatures"]), scaleFactor=float(p["scaleFactor"]),
                              nlevels=int(p["nlevels"]), edgeThreshold=int(p["edgeThreshold"]),
                              firstLevel=int(p["firstLevel"]), WTA_K=int(p["WTA_K"]),
                              scoreType=ORB_SCORE[p["scoreType"]], patchSize=int(p["patchSize"]),
                              fastThreshold=int(p["fastThreshold"]))
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=int(p["nfeatures"]), nOctaveLayers=int(p["nOctaveLayers"]),
                               contrastThreshold=float(p["contrastThreshold"]),
                               edgeThreshold=float(p["edgeThreshold"]), sigma=float(p["sigma"]),
                               enable_precise_upscale=bool(p["enable_precise_upscale"]))
    raise ValueError(name)


def config_name(detector, params, default):
    """'default' for the default-equivalent parameter set, otherwise a readable id."""
    if all(params[k] == default[k] for k in default):
        return "default"
    if detector == "ORB":
        return f"orb_sf{params['scaleFactor']}_nl{params['nlevels']}_ft{params['fastThreshold']}"
    return f"sift_ct{params['contrastThreshold']}_s{params['sigma']}"


def tuning_grid(detector, cfg):
    d = cfg["detectors"][detector]
    keys = list(d["tuning_grid"].keys())
    out = []
    for vals in itertools.product(*(d["tuning_grid"][k] for k in keys)):
        p = dict(d["default"])
        p.update(dict(zip(keys, vals)))
        out.append(p)
    return out


def params_json(p):
    return json.dumps(p, sort_keys=True)


def border_mask(shape, margin):
    m = np.zeros(shape, dtype=np.uint8)
    m[margin:shape[0] - margin, margin:shape[1] - margin] = 255
    return m


@dataclass
class Features:
    kp: np.ndarray          # (N, 6) float64: x, y, size, angle, response, octave
    desc: np.ndarray        # (N, D) or None
    runtime_ms: float

    @property
    def xy(self):
        return self.kp[:, :2]

    def __len__(self):
        return 0 if self.kp is None else self.kp.shape[0]


def detect(detector, params, img, mask):
    det = create_detector(detector, params)
    t0 = time.perf_counter()
    kps, desc = det.detectAndCompute(img, mask)
    dt = (time.perf_counter() - t0) * 1000.0
    kp = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kps],
                  dtype=np.float64).reshape(-1, 6)
    if desc is None or len(kps) == 0:
        desc = None
    return Features(kp, desc, dt)


@dataclass
class MatchResult:
    query_idx: np.ndarray   # reference keypoint index
    train_idx: np.ndarray   # target keypoint index
    dist1: np.ndarray
    dist2: np.ndarray
    runtime_ms: float


def ratio_match(detector, ref: Features, tgt: Features, ratio, k=2):
    t0 = time.perf_counter()
    q, t, d1, d2 = [], [], [], []
    if ref.desc is not None and tgt.desc is not None and len(ref) > 0 and len(tgt) > 0:
        bf = cv2.BFMatcher(NORMS[detector], crossCheck=False)
        for pair in bf.knnMatch(ref.desc, tgt.desc, k=k):
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < ratio * n.distance:
                q.append(m.queryIdx); t.append(m.trainIdx)
                d1.append(m.distance); d2.append(n.distance)
    dt = (time.perf_counter() - t0) * 1000.0
    return MatchResult(np.array(q, dtype=np.int64), np.array(t, dtype=np.int64),
                       np.array(d1, dtype=np.float64), np.array(d2, dtype=np.float64), dt)


def estimate_homography(src, dst, rc):
    """Returns (H or None, inlier bool mask, failure reason or '', runtime_ms)."""
    n = len(src)
    inl = np.zeros(n, dtype=bool)
    if n < rc["min_matches"]:
        return None, inl, "insufficient_matches", 0.0
    t0 = time.perf_counter()
    cv2.setRNGSeed(42)
    H, mask = cv2.findHomography(src.reshape(-1, 1, 2).astype(np.float32),
                                 dst.reshape(-1, 1, 2).astype(np.float32),
                                 method=cv2.RANSAC, ransacReprojThreshold=float(rc["reproj_threshold_px"]),
                                 maxIters=int(rc["max_iters"]), confidence=float(rc["confidence"]))
    dt = (time.perf_counter() - t0) * 1000.0
    if H is None or H.size == 0:
        return None, inl, "homography_estimation_failed", dt
    if mask is not None:
        inl = mask.ravel().astype(bool)
    return H.astype(np.float64), inl, "", dt
