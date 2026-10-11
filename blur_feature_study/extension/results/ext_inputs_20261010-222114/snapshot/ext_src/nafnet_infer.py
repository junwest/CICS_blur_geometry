"""Step 1: NAFNet-GoPro-width32 restoration (official architecture + official pretrained weights).

Runs in the SEPARATE environment .venv_nafnet (torch + numpy + opencv + pyyaml only), so the base
experiment environment (.venv) stays unchanged:

  .venv_nafnet\\Scripts\\python.exe -m extension.src.nafnet_infer --inputs extension/results/<ext_inputs_run>
        [--cpu-check 3] [--repeat-check 5]

Input/output handling (fixed, identical for every image; no colour information is added):
  gray uint8 HxW -> float32 / 255 -> 3 identical channels (R=G=B) -> 1x3xHxW
  -> NAFNetLocal (official test option for GoPro width32; internal zero pad to a multiple of 16
     on the right/bottom and crop back to [:H,:W])
  -> official tensor2img (clamp [0,1], RGB->BGR, *255, round, uint8)
  -> cv2.COLOR_BGR2GRAY (the same gray conversion as the base preprocessing) -> uint8 HxW
Outputs: <run>/images/<image_id>.png, <run>/nafnet_manifest.csv, <run>/run_meta.json
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import platform
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

EXT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXT_ROOT.parent


def load_official_arch(repo):
    """Import the three official arch files unchanged. The basicsr package __init__ files pull in the
    whole training framework (lmdb, torchvision, ...), so only the arch modules are loaded, under their
    original module names, with a stub for basicsr.utils.get_root_logger (imported by arch_util but not
    used by NAFNet)."""
    for name in ("basicsr", "basicsr.models", "basicsr.models.archs", "basicsr.utils"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules["basicsr.utils"].get_root_logger = lambda *a, **k: logging.getLogger("basicsr")
    arch_dir = Path(repo) / "basicsr" / "models" / "archs"
    for name in ("arch_util", "local_arch", "NAFNet_arch"):
        spec = importlib.util.spec_from_file_location(f"basicsr.models.archs.{name}", arch_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["basicsr.models.archs.NAFNet_arch"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pixel_sha(img):
    a = np.ascontiguousarray(img)
    h = hashlib.sha256()
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def imread_gray(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or img.dtype != np.uint8:
        raise IOError(f"expected uint8 gray PNG: {path}")
    return img


def imwrite_new(path, img):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


def build_model(ext, device):
    nc = ext["nafnet"]
    repo = EXT_ROOT / "third_party" / "NAFNet"
    arch = load_official_arch(repo)
    ng = dict(nc["network_g"])
    assert ng.pop("type") == "NAFNetLocal"
    net = arch.NAFNetLocal(**ng)                       # default train_size=(1,3,256,256), fast_imp=False (official)
    ck = PROJECT_ROOT / nc["checkpoint"]
    sha = file_sha256(ck)
    if sha != nc["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint sha256 mismatch: {sha}")
    state = torch.load(str(ck), map_location="cpu", weights_only=True)
    state = state[nc["state_dict_key"]]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}   # as basicsr load_network
    missing, unexpected = net.load_state_dict(state, strict=bool(nc["strict_load"]))
    net.eval().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    return net, {"checkpoint_sha256": sha, "n_params": int(n_params), "missing_keys": list(missing),
                 "unexpected_keys": list(unexpected), "repo_files_sha256": {
                     f: file_sha256(repo / f) for f in nc["arch_files"]}}


def gray_to_tensor(g):
    x = g.astype(np.float32) / 255.0                   # official imfrombytes(float32=True)
    x = np.repeat(x[None, :, :], 3, axis=0)            # R=G=B (no colour information)
    return torch.from_numpy(np.ascontiguousarray(x))[None]


def tensor_to_gray(out):
    """Official tensor2img 3-D path (rgb2bgr=True, uint8), then cv2.COLOR_BGR2GRAY."""
    t = out.squeeze(0).float().detach().cpu().clamp_(0, 1)
    img = t.numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = (img * 255.0).round().astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ch = bgr.astype(np.int16)
    spread = np.max(np.abs(ch - ch[:, :, [1, 2, 0]]), axis=2)    # max inter-channel difference per pixel
    return gray, {"chan_spread_max": int(spread.max()), "chan_spread_mean": float(spread.mean())}


def restore(net, g, device):
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    x = gray_to_tensor(g).to(device)
    sync()
    t1 = time.perf_counter()
    with torch.no_grad():
        y = net(x)
    sync()
    t2 = time.perf_counter()
    gray, diag = tensor_to_gray(y)
    t3 = time.perf_counter()
    if tuple(y.shape[-2:]) != g.shape or gray.shape != g.shape:
        raise RuntimeError(f"output shape {tuple(y.shape)} / {gray.shape} != input {g.shape}")
    diag.update(t_total_ms=(t3 - t0) * 1000, t_forward_ms=(t2 - t1) * 1000, t_h2d_ms=(t1 - t0) * 1000,
                t_d2h_post_ms=(t3 - t2) * 1000, out_h=int(y.shape[-2]), out_w=int(y.shape[-1]))
    return gray, diag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="extension/results/<ext_inputs run>")
    ap.add_argument("--cpu-check", type=int, default=0, help="also run N images on CPU (timing + numeric diff)")
    ap.add_argument("--repeat-check", type=int, default=5, help="re-run N images on the main device (determinism)")
    ap.add_argument("--roles", default="blurred_target,blurred_reference,clean_target")
    args = ap.parse_args(argv)

    with open(EXT_ROOT / "config" / "extension_config.yaml", "r", encoding="utf-8") as f:
        ext = yaml.safe_load(f)
    nc = ext["nafnet"]
    torch.backends.cuda.matmul.allow_tf32 = bool(nc["numerics"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(nc["numerics"]["tf32"])
    torch.backends.cudnn.deterministic = bool(nc["numerics"]["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(nc["numerics"]["cudnn_benchmark"])
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dir = Path(args.inputs)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    with open(in_dir / "input_manifest.csv", newline="", encoding="utf-8") as f:
        items = [r for r in csv.DictReader(f) if r["role"] in args.roles.split(",")]

    run_id = f"ext_nafnet_{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = EXT_ROOT / "results" / run_id
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "snapshot").mkdir()
    (out / "snapshot" / "nafnet_infer.py").write_bytes(Path(__file__).read_bytes())
    (out / "snapshot" / "extension_config.yaml").write_bytes((EXT_ROOT / "config" / "extension_config.yaml").read_bytes())

    t_load0 = time.perf_counter()
    net, model_info = build_model(ext, device)
    t_load = (time.perf_counter() - t_load0) * 1000

    # warm-up (excluded from timing)
    g0 = imread_gray(PROJECT_ROOT / items[0]["path"])
    tw = time.perf_counter(); restore(net, g0, device); t_warm = (time.perf_counter() - tw) * 1000

    rows = []
    outputs = {}
    t_start = time.time()
    for i, it in enumerate(items):
        g = imread_gray(PROJECT_ROOT / it["path"])
        if pixel_sha(g) != it["pixel_sha256"]:
            raise RuntimeError(f"input hash mismatch {it['image_id']}")
        gray, diag = restore(net, g, device)
        p = out / "images" / f"{it['image_id']}.png"
        imwrite_new(p, gray)
        if i < max(args.repeat_check, args.cpu_check):      # keep only what the checks need (low host memory)
            outputs[it["image_id"]] = gray
        rows.append({"image_id": it["image_id"], "role": it["role"], "pair_id": it["pair_id"],
                     "sequence_id": it["sequence_id"], "blur_id": it["blur_id"], "input_path": it["path"],
                     "input_pixel_sha256": it["pixel_sha256"], "output_path": p.relative_to(PROJECT_ROOT).as_posix(),
                     "output_pixel_sha256": pixel_sha(gray), "h": g.shape[0], "w": g.shape[1],
                     "pad_h": (16 - g.shape[0] % 16) % 16, "pad_w": (16 - g.shape[1] % 16) % 16,
                     "device": device.type, **diag})
        if i % 20 == 0:
            print(f"[{i + 1}/{len(items)}] {it['image_id']} {diag['t_total_ms']:.0f} ms ({time.time() - t_start:.0f}s)",
                  flush=True)
    with open(out / "nafnet_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    checks = {}
    # determinism on the main device
    rep = []
    for it in items[: args.repeat_check]:
        g = imread_gray(PROJECT_ROOT / it["path"])
        gray, _ = restore(net, g, device)
        rep.append({"image_id": it["image_id"], "identical": bool(np.array_equal(gray, outputs[it["image_id"]])),
                    "max_abs_diff": int(np.abs(gray.astype(int) - outputs[it["image_id"]].astype(int)).max())})
    checks["repeat_same_device"] = rep
    # CPU vs main device (numeric agreement + CPU timing, for reference only)
    cpu = []
    if args.cpu_check and device.type == "cuda":
        net_cpu, _ = build_model(ext, torch.device("cpu"))
        for it in items[: args.cpu_check]:
            g = imread_gray(PROJECT_ROOT / it["path"])
            try:
                gray_c, d = restore(net_cpu, g, torch.device("cpu"))
                diff = np.abs(gray_c.astype(int) - outputs[it["image_id"]].astype(int))
                cpu.append({"image_id": it["image_id"], "cpu_t_total_ms": d["t_total_ms"], "max_abs_diff": int(diff.max()),
                            "frac_pixels_diff": float((diff > 0).mean()), "h": g.shape[0], "w": g.shape[1]})
            except (RuntimeError, MemoryError) as e:
                cpu.append({"image_id": it["image_id"], "error": f"{type(e).__name__}: {e}"[:300]})
    checks["cpu_vs_device"] = cpu

    meta = {"run_id": run_id, "inputs_run": in_dir.name, "n_images": len(rows), "device": device.type,
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "opencv": cv2.__version__, "python": sys.version.split()[0],
            "numerics": {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                         "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                         "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark, "torch_threads": torch.get_num_threads()},
            "model": model_info, "nafnet_config": nc, "t_model_load_ms": t_load, "t_warmup_ms": t_warm,
            "timing_scope": "t_total_ms = uint8 gray in memory -> tensor -> device -> forward -> host -> uint8 gray "
                            "(torch.cuda.synchronize at both ends); excludes PNG I/O, model load and warm-up",
            "elapsed_s": round(time.time() - t_start, 1), "checks": checks,
            "end": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
