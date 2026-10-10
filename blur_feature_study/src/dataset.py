"""HPatches sequence selection, manifest building and pair loading.

HPatches convention (verified in prepare_data.verify_homography_direction):
H_1_k maps pixel coordinates of image 1 (reference) to image k (target).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .io_utils import PROJECT_ROOT, file_sha256, imread_unicode, load_homography

SEQ_PREFIX = {"illumination": "i_", "viewpoint": "v_"}


def list_sequences(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[:2] in ("i_", "v_"))


def select_sequences(root, seed, n_validation, n_test):
    """Seeded, sequence-level split (no sequence appears in two splits)."""
    rng = np.random.default_rng(seed)
    seqs = list_sequences(root)
    out = []
    for stype in ("illumination", "viewpoint"):
        names = sorted(s for s in seqs if s.startswith(SEQ_PREFIX[stype]))
        perm = rng.permutation(len(names))
        n_v, n_t = n_validation[stype], n_test[stype]
        for rank, idx in enumerate(perm[: n_v + n_t]):
            out.append({
                "sequence_id": names[idx],
                "sequence_type": stype,
                "split": "validation" if rank < n_v else "test",
                "selection_rank": rank,
            })
    return out


def resized_size(h, w, max_side):
    """Long side -> max_side (never upscale). Returns (new_h, new_w)."""
    if max(h, w) <= max_side:
        return h, w
    if w >= h:
        return int(round(h * max_side / w)), max_side
    return max_side, int(round(w * max_side / h))


def scale_matrix(orig_hw, new_hw):
    """Coordinate map of cv2.resize under the pixel-centre convention:
    x_new = sx * (x_old + 0.5) - 0.5."""
    sy = new_hw[0] / orig_hw[0]
    sx = new_hw[1] / orig_hw[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                     [0.0, sy, 0.5 * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def resize_homography(H, S_ref, S_tgt):
    Hr = S_tgt @ H @ np.linalg.inv(S_ref)
    return Hr / Hr[2, 2]


def load_gray_resized(path, max_side):
    """Returns (uint8 gray image, scale matrix S, original (h, w))."""
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    nh, nw = resized_size(h, w, max_side)
    if (nh, nw) != (h, w):
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return gray, scale_matrix((h, w), (nh, nw)), (h, w)


def build_manifest(cfg):
    root = PROJECT_ROOT / cfg["paths"]["hpatches_root"]
    ds = cfg["dataset"]
    max_side = cfg["preprocessing"]["max_long_side"]
    sel = select_sequences(root, cfg["experiment"]["seed"], ds["n_validation"], ds["n_test"])
    ref_i = ds["reference_index"]
    rows = []
    for s in sel:
        seq_dir = root / s["sequence_id"]
        ref_path = seq_dir / f"{ref_i}.ppm"
        ref_img = imread_unicode(ref_path)
        ref_hw = ref_img.shape[:2]
        ref_new = resized_size(*ref_hw, max_side)
        for t in ds["target_indices"]:
            tgt_path = seq_dir / f"{t}.ppm"
            h_path = seq_dir / f"H_{ref_i}_{t}"
            tgt_hw = imread_unicode(tgt_path).shape[:2]
            tgt_new = resized_size(*tgt_hw, max_side)
            H = load_homography(h_path)
            Hr = resize_homography(H, scale_matrix(ref_hw, ref_new), scale_matrix(tgt_hw, tgt_new))
            rows.append({
                "pair_id": f"{s['sequence_id']}__{ref_i}-{t}",
                "split": s["split"],
                "sequence_id": s["sequence_id"],
                "sequence_type": s["sequence_type"],
                "selection_rank": s["selection_rank"],
                "reference_index": ref_i,
                "target_index": t,
                "reference_file": ref_path.relative_to(PROJECT_ROOT).as_posix(),
                "target_file": tgt_path.relative_to(PROJECT_ROOT).as_posix(),
                "homography_file": h_path.relative_to(PROJECT_ROOT).as_posix(),
                "reference_orig_h": ref_hw[0], "reference_orig_w": ref_hw[1],
                "target_orig_h": tgt_hw[0], "target_orig_w": tgt_hw[1],
                "reference_h": ref_new[0], "reference_w": ref_new[1],
                "target_h": tgt_new[0], "target_w": tgt_new[1],
                "H_original": json.dumps(H.tolist()),
                "H_resized": json.dumps(Hr.tolist()),
                "reference_sha256": file_sha256(ref_path),
                "target_sha256": file_sha256(tgt_path),
                "homography_sha256": file_sha256(h_path),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "sequence_type", "selection_rank", "target_index"],
                        ascending=[False, True, True, True]).reset_index(drop=True)
    return df


def load_manifest(cfg):
    return pd.read_csv(PROJECT_ROOT / cfg["paths"]["manifest"])


@dataclass
class PairData:
    pair_id: str
    sequence_id: str
    sequence_type: str
    split: str
    reference_index: int
    target_index: int
    ref: np.ndarray        # uint8 gray, resized
    tgt: np.ndarray        # uint8 gray, resized (clean target)
    H_gt: np.ndarray       # reference -> target, resized coordinates


def load_pair(row, cfg):
    max_side = cfg["preprocessing"]["max_long_side"]
    ref, S_r, ref_hw = load_gray_resized(PROJECT_ROOT / row["reference_file"], max_side)
    tgt, S_t, tgt_hw = load_gray_resized(PROJECT_ROOT / row["target_file"], max_side)
    H = load_homography(PROJECT_ROOT / row["homography_file"])
    H_gt = resize_homography(H, S_r, S_t)
    # consistency with the manifest (guards against silent data changes)
    if not np.allclose(H_gt, np.array(json.loads(row["H_resized"])), rtol=1e-9, atol=1e-9):
        raise ValueError(f"H_resized mismatch with manifest for {row['pair_id']}")
    if ref.shape != (row["reference_h"], row["reference_w"]) or tgt.shape != (row["target_h"], row["target_w"]):
        raise ValueError(f"image size mismatch with manifest for {row['pair_id']}")
    return PairData(row["pair_id"], row["sequence_id"], row["sequence_type"], row["split"],
                    int(row["reference_index"]), int(row["target_index"]), ref, tgt, H_gt)
