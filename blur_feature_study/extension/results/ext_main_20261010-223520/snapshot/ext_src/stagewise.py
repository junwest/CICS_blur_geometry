"""Stage-wise target features: keypoints detected on one image, descriptors computed on another.

detect_only() = detector.detect(img, mask); describe() = detector.compute(img, keypoints) with keypoint
identity tracked through class_id (= detection index). OpenCV may drop, reorder or modify keypoints in
compute(); every call reports what happened (info dict) and the returned Features keep only the
keypoints that compute() returned, in its output order, with their detection indices (`ids`).

SIFT detail (see DECISIONS.md): OpenCV SIFT compute() starts the Gaussian pyramid at
firstOctave = min(0, lowest octave among the provided keypoints), whereas detectAndCompute always starts
at octave -1 (2x upsampled base). Without an octave -1 keypoint (heavily blurred images) compute() would
use a different pyramid and give different descriptors than detectAndCompute for the same keypoints.
describe() therefore appends one sentinel keypoint with octave -1 (class_id SENTINEL_ID) so that compute()
uses the same pyramid convention as detectAndCompute, and removes it afterwards.
"""
import time

import cv2
import numpy as np

from src import features as F

SENTINEL_ID = -2


def unpack_sift_octave(octave):
    o = int(octave) & 255
    return (o if o < 128 else o - 256), (int(octave) >> 8) & 255


def kp_array(kps):
    return np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kps],
                    dtype=np.float64).reshape(-1, 6)


def detect_only(det, params, img, mask):
    d = F.create_detector(det, params)
    t0 = time.perf_counter()
    kps = d.detect(img, mask)
    return list(kps), (time.perf_counter() - t0) * 1000.0


def with_ids(kps):
    return [cv2.KeyPoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, i) for i, k in enumerate(kps)]


def describe(det, params, img, kps):
    """Descriptors of `kps` (detected elsewhere) computed on `img`.
    Returns (Features, ids, info); Features.kp rows correspond to ids (detection indices)."""
    kin = with_ids(kps)
    n_in = len(kin)
    sentinel_effective = False
    if det == "SIFT":
        min_oct = min((unpack_sift_octave(k.octave)[0] for k in kin), default=0)
        sentinel_effective = min_oct >= 0          # plain compute() would have used firstOctave = 0
        h, w = img.shape[:2]
        kin.append(cv2.KeyPoint(w / 2.0, h / 2.0, 3.2, 0.0, 0.0, 255 | (1 << 8), SENTINEL_ID))
    d = F.create_detector(det, params)
    t0 = time.perf_counter()
    kout, desc = d.compute(img, kin)
    ms = (time.perf_counter() - t0) * 1000.0
    ids = np.array([k.class_id for k in kout], dtype=np.int64)
    keep = ids != SENTINEL_ID
    kout = [k for k, m in zip(kout, keep) if m]
    ids = ids[keep]
    if desc is not None:
        desc = desc[keep]
    kp_out = kp_array(kout)
    kp_in = kp_array(kps)
    attrs_changed = int((~np.all(kp_out == kp_in[ids], axis=1)).sum()) if len(ids) else 0
    info = {"n_in": n_in, "n_out": int(len(ids)), "removed_ids": sorted(set(range(n_in)) - set(ids.tolist())),
            "reordered": bool(len(ids) > 1 and np.any(np.diff(ids) <= 0)), "attrs_changed": attrs_changed,
            "sift_sentinel_effective": bool(sentinel_effective)}
    if desc is None or len(ids) == 0:
        desc = None
    return F.Features(kp_out, desc, ms), ids, info


def same_features(a, b):
    """Bitwise equality of keypoint arrays and descriptors."""
    if len(a) != len(b) or not np.array_equal(a.kp, b.kp):
        return False, False
    if a.desc is None or b.desc is None:
        return True, a.desc is None and b.desc is None
    return True, bool(a.desc.shape == b.desc.shape and np.array_equal(a.desc, b.desc))
