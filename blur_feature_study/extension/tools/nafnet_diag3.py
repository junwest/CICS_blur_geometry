"""Is the NAFNet divergence specific to gray-replicated (R=G=B) input? Diagnostic only (.venv_nafnet);
colour images are NEVER used in the experiment (no method receives colour information).

  .venv_nafnet\\Scripts\\python.exe extension/tools/nafnet_diag3.py <pair_id> [<pair_id> ...]

For the clean target of each pair: colour image resized exactly like the base preprocessing
(long side <= 960, INTER_AREA, resize after reading; base converts to gray BEFORE resizing, so the
colour version is resized in colour) -> NAFNet; versus the base gray image replicated to 3 channels.
Also: the colour image with its channels averaged (R=G=B but from the same resized colour image),
and the gray image blurred with Gaussian sigma 1.5 vs the colour image blurred the same way.
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


def run(net, rgb01, dev):
    x = torch.from_numpy(np.ascontiguousarray(rgb01.transpose(2, 0, 1)[None].astype(np.float32))).to(dev)
    with torch.no_grad():
        y = net(x).squeeze(0).double().cpu().numpy().transpose(1, 2, 0)
    out01 = float(((y < 0) | (y > 1)).mean())
    mse = np.mean((np.clip(y, 0, 1) - rgb01) ** 2) * 255 ** 2
    return out01, 10 * np.log10(255 ** 2 / mse), float(y.min()), float(y.max())


def main():
    with open(N.EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda")
    net, _ = N.build_model(ext, dev)
    with open(N.PROJECT_ROOT / "data" / "dataset_manifest.csv", newline="", encoding="utf-8") as f:
        man = {r["pair_id"]: r for r in csv.DictReader(f)}
    for pid in sys.argv[1:]:
        r = man[pid]
        bgr = cv2.imdecode(np.fromfile(str(N.PROJECT_ROOT / r["target_file"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = int(r["target_h"]), int(r["target_w"])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != (h, w):
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        g3 = np.repeat((gray.astype(np.float64) / 255.0)[:, :, None], 3, axis=2)
        avg3 = np.repeat(rgb.mean(axis=2, keepdims=True), 3, axis=2)
        chroma = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean() * 255)
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
        print(f"{pid} {h}x{w} mean|chroma|={chroma:.2f} grey levels")
        for tag, img in [("colour", rgb), ("gray replicated (base)", g3), ("colour channel-mean replicated", avg3),
                         ("colour + gauss1.5", blur(rgb)), ("gray replicated + gauss1.5", blur(g3))]:
            o, p, mn, mx = run(net, img, dev)
            print(f"  {tag:32s} frac_outside01={o:.4f} psnr_vs_input={p:6.2f} min={mn:8.3f} max={mx:8.3f}")


if __name__ == "__main__":
    main()
