"""Run one experiment phase.

  python -m src.run_experiment --phase pilot
  python -m src.run_experiment --phase validation
  python -m src.run_experiment --phase main --selected-params config/selected_params_<val_run_id>.yaml
  python -m src.run_experiment --phase main --resume <run_id> [--selected-params ...]   # re-run missing/failed rows only
Optional subsets: --pairs p1,p2  --blurs none,gaussian_s1.5
"""
import argparse
import csv
import datetime as dt
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from .dataset import load_manifest, load_pair
from .env_report import write_environment
from .io_utils import PROJECT_ROOT, code_hash, config_hash, file_sha256, load_config, write_json_new
from .pipeline import CLEAN_COND, COLUMNS, KEY_COLUMNS, CsvAppender, Runner, build_plan, row_key


def select_pairs(phase, manifest, flags):
    if phase == "pilot":
        # pre-declared: first validation sequence (selection order) of each type that is NOT
        # GT-flagged, target image 2 only
        v = manifest[(manifest.split == "validation") & (manifest.target_index == 2)].copy()
        v = v[v.pair_id.map(lambda p: flags.get(p, 0) == 0)]
        return v.sort_values("selection_rank").groupby("sequence_type", sort=True).head(1)
    split = "validation" if phase == "validation" else "test"
    return manifest[manifest.split == split]


def load_selected(path, cfg):
    with open(path, "r", encoding="utf-8") as f:
        sel = yaml.safe_load(f)
    for det in ("ORB", "SIFT"):
        p = sel[det]["params"]
        assert set(p) == set(cfg["detectors"][det]["default"]), f"{det} params incomplete"
        assert p["nfeatures"] == 1500
    ids = {c["blur_id"] for c in cfg["blur"]["conditions"]}
    assert set(sel["wiener_balance"]) == ids, "wiener balance missing for some blur conditions"
    for b in sel["wiener_balance"].values():
        assert float(b) in [float(x) for x in cfg["processing"]["wiener"]["balance_candidates"]]
    return sel


def setup_logger(run_id):
    log = logging.getLogger(run_id)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(PROJECT_ROOT / "logs" / f"{run_id}.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    return log


def finalize(out_dir, log):
    """raw_results.csv = latest attempt per row key (all attempts kept separately)."""
    allp = out_dir / "raw_results_all_attempts.csv"
    df = pd.read_csv(allp, keep_default_na=False, dtype=str)
    df["_key"] = [row_key(r) for r in df.to_dict("records")]
    df["_att"] = df["attempt"].astype(int)
    final = df.sort_values("_att", kind="stable").groupby("_key", sort=False).tail(1)
    final = final.drop(columns=["_key", "_att"])[COLUMNS]
    fpath = out_dir / "raw_results.csv"
    if fpath.exists():
        fpath.rename(out_dir / f"raw_results.superseded_{dt.datetime.now():%Y%m%d-%H%M%S}.csv")
    final.to_csv(fpath, index=False)
    errs = final[final.pipeline_error != ""]
    epath = out_dir / f"errors_{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
    errs.to_csv(epath, index=False)
    log.info("finalized %s: %d rows (%d with pipeline errors)", fpath, len(final), len(errs))
    return len(final), len(errs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "validation", "main"])
    ap.add_argument("--run-id")
    ap.add_argument("--resume")
    ap.add_argument("--selected-params")
    ap.add_argument("--pairs")
    ap.add_argument("--blurs")
    args = ap.parse_args(argv)

    cfg = load_config()
    cfg_h, code_h = config_hash(cfg), code_hash()
    manifest = load_manifest(cfg)
    fl = pd.read_csv(PROJECT_ROOT / "data" / "gt_quality_flags.csv")
    flags = dict(zip(fl.pair_id, fl.gt_quality_flag.astype(int)))
    pairs = select_pairs(args.phase, manifest, flags)
    if args.pairs:
        pairs = pairs[pairs.pair_id.isin(args.pairs.split(","))]
    conds = [CLEAN_COND] + list(cfg["blur"]["conditions"])
    if args.blurs:
        conds = [c for c in conds if c["blur_id"] in args.blurs.split(",")]

    selected, sel_path = None, None
    if args.phase == "main":
        if not args.selected_params:
            ap.error("--selected-params is required for the main phase")
        sel_path = Path(args.selected_params)
        if not sel_path.is_absolute():
            sel_path = PROJECT_ROOT / sel_path
        selected = load_selected(sel_path, cfg)

    if args.resume:
        run_id = args.resume
        out_dir = PROJECT_ROOT / "results" / args.phase / run_id
        prev = pd.read_csv(out_dir / "raw_results_all_attempts.csv", keep_default_na=False, dtype=str)
        attempt = int(prev["attempt"].astype(int).max()) + 1
        done = {row_key(r) for r in prev.to_dict("records") if r["pipeline_error"] == ""}
    else:
        run_id = args.run_id or f"{args.phase}_{dt.datetime.now():%Y%m%d-%H%M%S}"
        out_dir = PROJECT_ROOT / "results" / args.phase / run_id
        out_dir.mkdir(parents=True, exist_ok=False)          # never reuse a run directory
        attempt, done = 1, set()
        snap = out_dir / "snapshot"
        (snap / "src").mkdir(parents=True)
        for p in (PROJECT_ROOT / "src").glob("*.py"):
            shutil.copy2(p, snap / "src" / p.name)
        shutil.copy2(PROJECT_ROOT / "config" / "experiment_config.yaml", snap / "experiment_config.yaml")
        shutil.copy2(PROJECT_ROOT / cfg["paths"]["manifest"], snap / "dataset_manifest.csv")
        if sel_path:
            shutil.copy2(sel_path, snap / sel_path.name)
        write_environment(out_dir / "environment.txt")

    log = setup_logger(run_id)
    details = PROJECT_ROOT / cfg["paths"]["match_details_root"] / run_id
    runner = Runner(cfg, args.phase, run_id, out_dir, details, cfg_h, code_h, log, attempt, flags)
    app = CsvAppender(out_dir / "raw_results_all_attempts.csv")
    meta = {"run_id": run_id, "phase": args.phase, "attempt": attempt, "argv": sys.argv,
            "start": dt.datetime.now().isoformat(timespec="seconds"), "config_hash": cfg_h, "code_hash": code_h,
            "pairs": pairs.pair_id.tolist(), "blur_ids": [c["blur_id"] for c in conds],
            "selected_params": str(sel_path) if sel_path else None,
            "selected_params_sha256": file_sha256(sel_path) if sel_path else None,
            "expected_rows": sum(len(build_plan(args.phase, cfg, c["blur_id"], selected)) for c in conds) * len(pairs)}
    log.info("run %s phase=%s attempt=%d pairs=%d conds=%d expected_rows=%d", run_id, args.phase, attempt,
             len(pairs), len(conds), meta["expected_rows"])

    t_start = time.time()
    n_rows = 0
    for i, (_, prow) in enumerate(pairs.iterrows()):
        try:
            pair = load_pair(prow, cfg)
        except Exception:
            log.error("cannot load pair %s\n%s", prow["pair_id"], traceback.format_exc())
            continue
        for cond in conds:
            plan = build_plan(args.phase, cfg, cond["blur_id"], selected)
            t0 = time.time()
            try:
                rows = runner.run_unit(pair, cond, plan, done)
            except Exception as exc:  # unit-level failure: record every planned row as failed
                log.error("unit failure %s %s\n%s", pair.pair_id, cond["blur_id"], traceback.format_exc())
                rows = []
                for e in plan:
                    r = runner._base_row(pair, cond, e)
                    r.update(pipeline_error=f"{type(exc).__name__}: {exc}", failure_reason="pipeline_error")
                    rows.append(r)
            app.append(rows)
            n_rows += len(rows)
            log.info("[%d/%d] %s %-15s rows=%d (%.1fs, total %.0fs)", i + 1, len(pairs), pair.pair_id,
                     cond["blur_id"], len(rows), time.time() - t0, time.time() - t_start)
    meta.update(end=dt.datetime.now().isoformat(timespec="seconds"), elapsed_s=round(time.time() - t_start, 1),
                rows_written_this_attempt=n_rows)
    n_final, n_err = finalize(out_dir, log)
    meta.update(final_rows=n_final, final_rows_with_pipeline_error=n_err)
    write_json_new(out_dir / f"run_meta_attempt{attempt}.json", meta)
    log.info("done: %s", out_dir)
    print(f"RUN_DIR={out_dir}")


if __name__ == "__main__":
    main()
