"""Integration test: one real validation pair through the full pipeline (twice)."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs_data
from src.dataset import load_pair
from src.metrics import gt_errors
from src.pipeline import CLEAN_COND, COLUMNS, REQUIRED_COLUMNS, Runner, build_plan, row_key

VOLATILE = {"runtime_ms", "runtime_processing_ms", "runtime_detect_ms", "runtime_match_ms",
            "runtime_homography_ms", "runtime_reference_detect_ms", "timestamp", "match_detail_file"}


def _run(cfg, tmp, pair, cond):
    r = Runner(cfg, "pilot", "test_run", tmp, tmp / "results" / "matching_details" / "test_run", "cfg", "code")
    rows = r.run_unit(pair, cond, build_plan("pilot", cfg, cond["blur_id"]))
    return rows


@needs_data
def test_pipeline_unit_end_to_end(cfg, tmp_path):
    m = pd.read_csv(ROOT / cfg["paths"]["manifest"])
    pair = load_pair(m[m.pair_id == "i_veggies__1-2"].iloc[0], cfg)
    cond = cfg["blur"]["conditions"][0]

    rows1 = _run(cfg, tmp_path / "a", pair, cond)
    rows2 = _run(cfg, tmp_path / "b", pair, cond)
    plan = build_plan("pilot", cfg, cond["blur_id"])
    assert len(rows1) == len(plan) == 8
    assert len({row_key(r) for r in rows1}) == len(rows1)
    for r1, r2 in zip(rows1, rows2):                               # determinism
        assert r1["pipeline_error"] == ""
        for k in r1:
            if k in VOLATILE:
                continue
            a, b = r1[k], r2[k]
            if isinstance(a, float) and np.isnan(a):
                assert isinstance(b, float) and np.isnan(b), k
            else:
                assert a == b, k
    for r in rows1:
        assert set(REQUIRED_COLUMNS) <= set(r) and set(r) <= set(COLUMNS)
        assert 0 <= r["edge_f1"] <= 1 and 0 <= r["keypoint_repeatability"] <= 1
        assert r["gt_correct_matches_3px"] <= r["gt_correct_matches_5px"] <= r["good_matches"]
        assert r["ransac_inliers"] <= r["good_matches"]
        # GT-correct count re-computed from the saved match details
        d = np.load(r["match_detail_file"])
        e = gt_errors(d["H_gt"], d["ref_xy"], d["tgt_xy"])
        assert int((e <= 3).sum()) == r["gt_correct_matches_3px"]
        assert len(d["query_idx"]) == r["good_matches"] == len(d["ransac_inlier"])
        assert json.loads(str(d["meta"]))["pair_id"] == pair.pair_id
    # the two groups that share the blurred image share the image metrics
    a = [r for r in rows1 if r["group"] == "A" and r["detector"] == "ORB"][0]
    b = [r for r in rows1 if r["group"] == "B" and r["detector"] == "ORB"][0]
    assert a["psnr"] == b["psnr"] and a["edge_f1"] == b["edge_f1"]


@needs_data
def test_clean_baseline_trivial_metrics(cfg, tmp_path):
    m = pd.read_csv(ROOT / cfg["paths"]["manifest"])
    pair = load_pair(m[m.pair_id == "i_veggies__1-2"].iloc[0], cfg)
    rows = _run(cfg, tmp_path, pair, CLEAN_COND)
    assert len(rows) == 4
    for r in rows:
        assert r["psnr"] == float("inf") and r["ssim"] == pytest.approx(1.0)
        assert r["edge_f1"] == 1.0 and r["keypoint_repeatability"] == 1.0
        assert r["homography_success_3px"] == 1          # easy, clean, well-registered pair
