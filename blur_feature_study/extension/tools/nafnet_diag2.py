"""Locate the NAFNet output divergence (run with .venv_nafnet). Diagnostic only, not part of the pipeline.

  .venv_nafnet\\Scripts\\python.exe extension/tools/nafnet_diag2.py <inputs_run_dir> <image_id> [<image_id> ...]

1. official NAFNetLocal (fp32, GPU): max |activation| after intro, every NAFBlock, ending.
2. plain NAFNet (global average pooling in SCA, i.e. without the TLC conversion), same weights.
3. official NAFNetLocal in float64 on CPU (rules out fp32 precision as the cause).
4. official NAFNetLocal on the centre 256x256 crop (training patch size; TLC pooling == global there).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extension.src import nafnet_infer as N  # noqa: E402
import yaml  # noqa: E402


def summarize(tag, y, g):
    a = y.squeeze(0).double().cpu().numpy()
    gray = (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2])
    out01 = float(((a < 0) | (a > 1)).mean())
    mse = np.mean((np.clip(gray, 0, 1) * 255 - g.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
    print(f"  {tag:34s} min={a.min():9.3f} max={a.max():9.3f} frac_outside01={out01:.4f} psnr_vs_input={psnr:6.2f}")


def profile(net, x):
    rec = []
    hooks = [net.intro.register_forward_hook(lambda m, i, o: rec.append(("intro", float(o.abs().max()))))]
    for li, enc in enumerate(net.encoders):
        for bi, blk in enumerate(enc):
            hooks.append(blk.register_forward_hook(lambda m, i, o, t=f"enc{li}.{bi}": rec.append((t, float(o.abs().max())))))
    for bi, blk in enumerate(net.middle_blks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, t=f"mid.{bi}": rec.append((t, float(o.abs().max())))))
    for li, dec in enumerate(net.decoders):
        for bi, blk in enumerate(dec):
            hooks.append(blk.register_forward_hook(lambda m, i, o, t=f"dec{li}.{bi}": rec.append((t, float(o.abs().max())))))
    hooks.append(net.ending.register_forward_hook(lambda m, i, o: rec.append(("ending", float(o.abs().max())))))
    with torch.no_grad():
        y = net(x)
    for h in hooks:
        h.remove()
    return y, rec


def main():
    in_dir, ids = Path(sys.argv[1]), sys.argv[2:]
    with open(N.EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gpu = torch.device("cuda")
    net, _ = N.build_model(ext, gpu)
    arch = sys.modules["basicsr.models.archs.NAFNet_arch"]
    ng = dict(ext["nafnet"]["network_g"]); ng.pop("type")
    plain = arch.NAFNet(**ng)
    sd = torch.load(str(N.PROJECT_ROOT / ext["nafnet"]["checkpoint"]), map_location="cpu", weights_only=True)["params"]
    print("plain load:", plain.load_state_dict(sd, strict=True))
    plain.eval().to(gpu)
    net64, _ = N.build_model(ext, torch.device("cpu"))
    net64 = net64.double()
    with open(in_dir / "input_manifest.csv", newline="", encoding="utf-8") as f:
        items = {r["image_id"]: r for r in csv.DictReader(f)}
    for iid in ids:
        g = N.imread_gray(N.PROJECT_ROOT / items[iid]["path"])
        x = N.gray_to_tensor(g)
        print(f"{iid} {g.shape}")
        y, rec = profile(net, x.to(gpu))
        print("  activation max|.| profile:", " ".join(f"{t}={v:.3g}" for t, v in rec))
        summarize("NAFNetLocal fp32 GPU (official)", y, g)
        with torch.no_grad():
            summarize("NAFNet plain (global SCA) fp32 GPU", plain(x.to(gpu)), g)
            summarize("NAFNetLocal float64 CPU", net64(x.double()), g)
            h0, w0 = (g.shape[0] - 256) // 2, (g.shape[1] - 256) // 2
            gc = np.ascontiguousarray(g[h0:h0 + 256, w0:w0 + 256])
            summarize("NAFNetLocal centre 256x256 crop", net(N.gray_to_tensor(gc).to(gpu)), gc)


if __name__ == "__main__":
    main()
