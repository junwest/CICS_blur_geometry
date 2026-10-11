"""Probe: why SIFT detect()+compute() descriptors can differ from detectAndCompute (implementation check only).

Hypothesis: OpenCV SIFT compute() (= detectAndCompute(useProvidedKeypoints=True)) sets
firstOctave = min(0, min octave of the provided keypoints), while detectAndCompute without provided
keypoints always uses firstOctave = -1 (2x upsampled base image). If no provided keypoint is in octave -1,
compute() builds the Gaussian pyramid from the non-upsampled image -> slightly different descriptors.
Fix under test: append one sentinel keypoint with octave -1 (class_id -2), drop it after compute().
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.io_utils import load_config  # noqa: E402
from src.dataset import load_manifest, load_pair  # noqa: E402
from src import features as F  # noqa: E402
from src.psf import make_psf  # noqa: E402
from src.imaging import apply_blur, wiener_restore, unsharp_mask  # noqa: E402


def unpack(o):
    oc = o & 255
    oc = oc if oc < 128 else oc - 256
    return oc, (o >> 8) & 255


def kp_arr(kps):
    return np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id] for k in kps]).reshape(-1, 7)


def compute(det_name, p, img, kps, sentinel):
    kin = [cv2.KeyPoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, i) for i, k in enumerate(kps)]
    if sentinel:
        h, w = img.shape
        kin.append(cv2.KeyPoint(w / 2.0, h / 2.0, 3.2, 0.0, 0.0, 255 | (1 << 8), -2))
    kout, d = F.create_detector(det_name, p).compute(img, kin)
    a = kp_arr(kout)
    keep = a[:, 6] != -2
    return a[keep], d[keep]


cfg = load_config()
man = load_manifest(cfg)
test = man[man.split == "test"]
p = cfg["detectors"]["SIFT"]["default"]
uc = cfg["processing"]["unsharp_mask"]
n = {"plain_eq": 0, "sent_eq": 0, "total": 0, "minoct_ge0": 0, "sent_eq_when_minoct_ge0": 0, "plain_eq_when_minoct_lt0": 0,
     "n_lt0": 0}
for pid in list(test.pair_id.unique())[:3]:
    pair = load_pair(test[test.pair_id == pid].iloc[0], cfg)
    mask = F.border_mask(pair.tgt.shape, cfg["preprocessing"]["valid_margin_px"])
    for cond in cfg["blur"]["conditions"]:
        psf = make_psf(cond, cfg)
        B = apply_blur(pair.tgt, psf)
        imgs = {"B": B, "U": unsharp_mask(B, uc["alpha"], uc["sigma"], uc["truncate"]), "W": wiener_restore(B, psf, 0.005)}
        for name, img in imgs.items():
            kA, dA = F.create_detector("SIFT", p).detectAndCompute(img, mask)
            kD = F.create_detector("SIFT", p).detect(img, mask)
            minoct = min(unpack(k.octave)[0] for k in kD)
            a0, d0 = compute("SIFT", p, img, kD, False)
            a1, d1 = compute("SIFT", p, img, kD, True)
            ids_ok = np.array_equal(a1[:, 6], np.arange(len(kD))) and np.array_equal(a1[:, :6], kp_arr(kD)[:, :6])
            pe = dA.shape == d0.shape and np.array_equal(dA, d0)
            se = dA.shape == d1.shape and np.array_equal(dA, d1)
            n["total"] += 1; n["plain_eq"] += pe; n["sent_eq"] += se
            if minoct >= 0:
                n["minoct_ge0"] += 1; n["sent_eq_when_minoct_ge0"] += se
            else:
                n["n_lt0"] += 1; n["plain_eq_when_minoct_lt0"] += pe
            if not pe or not se or not ids_ok:
                print(f"{pid} {cond['blur_id']:16s} {name} n={len(kD)} minoct={minoct} plain_eq={pe} sentinel_eq={se} "
                      f"ids_attrs_ok={ids_ok} max|d_plain-dA|={np.abs(d0 - dA).max() if d0.shape == dA.shape else 'shape'}")
print(n)
