"""Sanity check of the official model on the official demo image (in-distribution motion blur), and on its
own (sharp) output. Diagnostic only (.venv_nafnet).

  .venv_nafnet\\Scripts\\python.exe extension/tools/nafnet_diag4.py <out_dir>
"""
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
        return net(x).squeeze(0).double().cpu().numpy().transpose(1, 2, 0)


def report(tag, inp, y):
    out01 = float(((y < 0) | (y > 1)).mean())
    mse = np.mean((np.clip(y, 0, 1) - inp) ** 2) * 255 ** 2
    print(f"{tag:44s} {inp.shape[1]}x{inp.shape[0]} frac_outside01={out01:.4f} psnr_vs_input={10 * np.log10(255 ** 2 / mse):6.2f} "
          f"min={y.min():8.3f} max={y.max():8.3f}")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with open(N.EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda")
    net, _ = N.build_model(ext, dev)
    demo = N.EXT_ROOT / "third_party" / "NAFNet" / "demo"
    bgr = cv2.imdecode(np.fromfile(str(demo / "blurry.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    y = run(net, rgb, dev); report("demo/blurry.jpg (colour)", rgb, y)
    y1 = np.clip(y, 0, 1)
    cv2.imencode(".png", cv2.cvtColor((y1 * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tofile(str(out / "demo_deblur.png"))
    y2 = run(net, y1, dev); report("NAFNet(NAFNet(demo)) (sharp input)", y1, y2)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    g3 = np.repeat(g[:, :, None], 3, axis=2)
    report("demo/blurry.jpg gray replicated", g3, run(net, g3, dev))
    for name in ("sr_img_l.png", "denoise_img.png"):
        b = cv2.imdecode(np.fromfile(str(demo / name), dtype=np.uint8), cv2.IMREAD_COLOR)
        r = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        report(f"demo/{name} (clean-ish colour)", r, run(net, r, dev))


if __name__ == "__main__":
    main()
