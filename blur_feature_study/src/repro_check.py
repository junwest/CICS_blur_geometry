"""Cross-run reproducibility: re-run a subset of a finished run in a fresh process and
compare every non-volatile column.

  python -m src.repro_check --run results/main/<run_id> [--n-per-type 1]
Writes results/reproducibility/<new_id>/ (rows, match details, comparison json).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .dataset import load_manifest, load_pair
from .io_utils import PROJECT_ROOT, load_config, write_json_new
from .pipeline import CLEAN_COND, COLUMNS, Runner, build_plan, row_key

VOLATILE = {"run_id", "phase", "timestamp", "match_detail_file", "code_hash", "attempt", "runtime_ms",
            "runtime_processing_ms", "runtime_detect_ms", "runtime_match_ms", "runtime_homography_ms",
            "runtime_reference_detect_ms"}


def _same(a, b):
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return (np.isnan(fa) and np.isnan(fb)) or fa == fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-per-type", type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else PROJECT_ROOT / a.run
    cfg = load_config(run / "snapshot" / "experiment_config.yaml")
    meta = json.loads(sorted(run.glob("run_meta_attempt*.json"))[-1].read_text(encoding="utf-8"))
    selected = yaml.safe_load(open(meta["selected_params"], encoding="utf-8")) if meta["selected_params"] else None
    orig = pd.read_csv(run / "raw_results.csv", keep_default_na=False, dtype=str)
    orig_by_key = {row_key(r): r for r in orig.to_dict("records")}

    man = load_manifest(cfg)
    m = man[man.pair_id.isin(meta["pairs"])]
    pairs = m.sort_values(["sequence_type", "selection_rank", "target_index"]).groupby("sequence_type").head(a.n_per_type)
    rid = f"repro_{run.name}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = PROJECT_ROOT / "results" / "reproducibility" / rid
    out.mkdir(parents=True, exist_ok=False)
    fl = pd.read_csv(PROJECT_ROOT / "data" / "gt_quality_flags.csv")
    runner = Runner(cfg, meta["phase"], rid, out, out / "match_details", orig.config_hash.iloc[0], "repro",
                    gt_flags=dict(zip(fl.pair_id, fl.gt_quality_flag.astype(int))))
    n_rows, mismatches = 0, []
    for _, prow in pairs.iterrows():
        pair = load_pair(prow, cfg)
        for cond in [CLEAN_COND] + list(cfg["blur"]["conditions"]):
            for r in runner.run_unit(pair, cond, build_plan(meta["phase"], cfg, cond["blur_id"], selected)):
                n_rows += 1
                o = orig_by_key[row_key(r)]
                for c in COLUMNS:
                    if c in VOLATILE:
                        continue
                    v = r.get(c)
                    v = "" if v is None else v
                    if not _same(v, o[c]):
                        mismatches.append({"key": row_key(r), "column": c, "rerun": str(v), "original": o[c]})
    res = {"original_run": run.name, "repro_id": rid, "pairs": pairs.pair_id.tolist(), "rows_compared": n_rows,
           "columns_compared": [c for c in COLUMNS if c not in VOLATILE], "n_mismatches": len(mismatches),
           "mismatches": mismatches[:50], "identical": len(mismatches) == 0}
    write_json_new(out / "comparison.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "columns_compared"}, indent=2))


if __name__ == "__main__":
    main()
