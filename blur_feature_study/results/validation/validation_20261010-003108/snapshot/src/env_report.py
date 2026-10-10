"""Environment description (environment.txt).  Usage: python -m src.env_report [output_path]"""
import ctypes
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path

from .io_utils import PROJECT_ROOT


def _ram_gb():
    if os.name != "nt":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullTotalPhys / 2 ** 30, 2)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def environment_text():
    import cv2, numpy, pandas, scipy, skimage, yaml  # noqa: E401
    cpu = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) if os.name == "nt" else platform.processor()
    lines = [
        f"generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()} ({platform.version()})",
        f"cpu: {cpu} | logical cores: {os.cpu_count()}",
        f"ram_total_gb: {_ram_gb()}",
        f"gpu: {_run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])} (not used: all processing is CPU)",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"opencv: {cv2.__version__} (opencv-contrib-python), cv2 threads: {cv2.getNumThreads()}",
        f"numpy: {numpy.__version__}", f"scipy: {scipy.__version__}", f"scikit-image: {skimage.__version__}",
        f"pandas: {pandas.__version__}", f"pyyaml: {yaml.__version__}",
        "", "pip freeze:", _run([sys.executable, "-m", "pip", "freeze"]),
    ]
    return "\n".join(lines) + "\n"


def write_environment(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(environment_text(), encoding="utf-8")
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "environment.txt"
    if target.exists():
        target = target.with_name(f"environment_{dt.datetime.now():%Y%m%d-%H%M%S}.txt")
    print("wrote", write_environment(target))
