import ctypes
import ctypes.wintypes as wt
import os
import sys


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD), ("ThreadCount", wt.DWORD)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def report(tag):
    pi = PERFORMANCE_INFORMATION(); pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    ps = pi.PageSize
    print(f"{tag:28s} sys_free_commit={ (pi.CommitLimit - pi.CommitTotal) * ps / 2**20:8.0f} MB   proc_commit={pm.PagefileUsage / 2**20:7.0f} MB", flush=True)


if __name__ == "__main__":
    report("start")
    import numpy as np
    report("numpy")
    if len(sys.argv) > 1 and sys.argv[1] == "torch":
        import torch
        report("import torch")
        torch.zeros(1, device="cuda")
        report("cuda init")
        x = torch.randn(1, 32, 656, 960, device="cuda")
        report("alloc 80MB tensor")
        y = torch.nn.functional.conv2d(x, torch.randn(64, 32, 1, 1, device="cuda"))
        report("conv2d 1x1")
        z = torch.nn.functional.conv2d(y, torch.randn(64, 1, 3, 3, device="cuda"), padding=1, groups=64)
        report("depthwise conv")
