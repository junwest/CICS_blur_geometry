"""Step 0: write the blurred images that the (separate-venv) NAFNet step needs.

  python -m extension.src.prepare_inputs            (run from blur_feature_study/ with .venv)

Images are generated exactly as in the base study (src.dataset.load_pair -> src.imaging.apply_blur)
and stored losslessly (PNG, uint8 gray) together with a pixel hash. Later steps regenerate the same
images in memory and only use a NAFNet output if the regenerated input hash matches.
"""
import time

import pandas as pd

from .common import (blur_cond, environment_info, load_base, load_ext_config, new_run_dir, pixel_sha, rel,
                     test_pairs, write_json, write_png_new)
from src.dataset import load_pair
from src.imaging import apply_blur
from src.psf import make_psf


def main():
    cfg, sel, man, flags = load_base()
    ext = load_ext_config()
    run_id, out = new_run_dir("ext_inputs")
    img_dir = out / "images"
    rows = []
    pairs = test_pairs(man)
    ref_blurs = sorted({v["reference"] for v in ext["both_blur"]["kernel_configs"].values()})
    done_refs = set()
    t0 = time.time()
    for _, prow in pairs.iterrows():
        pair = load_pair(prow, cfg)
        # clean target (NAFNet identity / alignment diagnostic only)
        iid = f"cleantgt__{pair.pair_id}"
        p = img_dir / f"{iid}.png"
        write_png_new(p, pair.tgt)
        rows.append(dict(image_id=iid, role="clean_target", pair_id=pair.pair_id, sequence_id=pair.sequence_id,
                         blur_id="none", path=rel(p), h=pair.tgt.shape[0], w=pair.tgt.shape[1],
                         pixel_sha256=pixel_sha(pair.tgt)))
        for cond in cfg["blur"]["conditions"]:
            B = apply_blur(pair.tgt, make_psf(cond, cfg))
            iid = f"tgt__{pair.pair_id}__{cond['blur_id']}"
            p = img_dir / f"{iid}.png"
            write_png_new(p, B)
            rows.append(dict(image_id=iid, role="blurred_target", pair_id=pair.pair_id, sequence_id=pair.sequence_id,
                             blur_id=cond["blur_id"], path=rel(p), h=B.shape[0], w=B.shape[1],
                             pixel_sha256=pixel_sha(B)))
        if pair.sequence_id not in done_refs:
            done_refs.add(pair.sequence_id)
            for bid in ref_blurs:
                Br = apply_blur(pair.ref, make_psf(blur_cond(cfg, bid), cfg))
                iid = f"ref__{pair.sequence_id}__{bid}"
                p = img_dir / f"{iid}.png"
                write_png_new(p, Br)
                rows.append(dict(image_id=iid, role="blurred_reference", pair_id="", sequence_id=pair.sequence_id,
                                 blur_id=bid, path=rel(p), h=Br.shape[0], w=Br.shape[1],
                                 pixel_sha256=pixel_sha(Br)))
        print(pair.pair_id, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "input_manifest.csv", index=False)
    write_json(out / "run_meta.json", {"run_id": run_id, "n_images": len(df), "elapsed_s": round(time.time() - t0, 1),
                                       "environment": environment_info(),
                                       "counts": df.role.value_counts().to_dict()})
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
