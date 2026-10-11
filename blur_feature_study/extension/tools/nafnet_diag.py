"""Diagnose NAFNet raw outputs (run with .venv_nafnet). Not part of the experiment pipeline.

  .venv_nafnet\\Scripts\\python.exe extension/tools/nafnet_diag.py <inputs_run_dir> <out_dir> <image_id> [<image_id> ...]

For each image: NaN/Inf count, per-channel min/max/mean of the raw float output (before clamp),
fraction of values outside [0,1], channel spread, gray PSNR vs input; saves the raw RGB output
(clamped, uint8) and the gray output as PNG for visual inspection.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extension.src import nafnet_infer as N  # noqa: E402
import yaml  # noqa: E402


def main():
    in_dir, out_dir, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(N.EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    net, _ = N.build_model(ext, dev)
    with open(in_dir / "input_manifest.csv", newline="", encoding="utf-8") as f:
        items = {r["image_id"]: r for r in csv.DictReader(f)}
    for iid in ids:
        g = N.imread_gray(N.PROJECT_ROOT / items[iid]["path"])
        x = N.gray_to_tensor(g).to(dev)
        with torch.no_grad():
            y = net(x)
        a = y.squeeze(0).float().cpu().numpy()            # 3 x H x W raw
        nan, inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
        fin = np.where(np.isfinite(a), a, 0)
        out_of = float(((fin < 0) | (fin > 1)).mean())
        stats = [(float(fin[c].min()), float(fin[c].max()), float(fin[c].mean())) for c in range(3)]
        spread = np.abs(fin - fin[[1, 2, 0]]).max(axis=0)
        gray, d = N.tensor_to_gray(y)
        mse = np.mean((gray.astype(float) - g.astype(float)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
        sat_in = float((g >= 250).mean())
        print(f"{iid}: nan={nan} inf={inf} frac_outside01={out_of:.4f} ch(min,max,mean)={[tuple(round(v, 3) for v in s) for s in stats]} "
              f"raw_spread>0.1 frac={float((spread > 0.1).mean()):.4f} spread_max_u8={d['chan_spread_max']} "
              f"psnr_gray_vs_input={psnr:.2f} input_frac>=250={sat_in:.4f} input_mean={g.mean():.1f}")
        rgb = (np.clip(fin, 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); buf.tofile(str(out_dir / f"{iid}__rgb.png"))
        ok, buf = cv2.imencode(".png", gray); buf.tofile(str(out_dir / f"{iid}__gray.png"))
        ok, buf = cv2.imencode(".png", g); buf.tofile(str(out_dir / f"{iid}__input.png"))


if __name__ == "__main__":
    main()
