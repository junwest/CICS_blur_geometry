"""List imported DLL names of PE files (minimal parser, read-only)."""
import os
import struct
import sys


def imports(path):
    with open(path, "rb") as f:
        d = f.read()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    nsec = struct.unpack_from("<H", d, pe + 6)[0]
    optsz = struct.unpack_from("<H", d, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", d, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    imp_rva = struct.unpack_from("<I", d, dd + 8)[0]
    secs = []
    so = opt + optsz
    for i in range(nsec):
        va, vsz, raw, rawsz = struct.unpack_from("<IIII", d, so + i * 40 + 8)[0:1] + struct.unpack_from("<I", d, so + i * 40 + 8 + 4)[0:1] + struct.unpack_from("<II", d, so + i * 40 + 16)
        vsize = struct.unpack_from("<I", d, so + i * 40 + 8)[0]
        va = struct.unpack_from("<I", d, so + i * 40 + 12)[0]
        rawsz = struct.unpack_from("<I", d, so + i * 40 + 16)[0]
        raw = struct.unpack_from("<I", d, so + i * 40 + 20)[0]
        secs.append((va, max(vsize, rawsz), raw))

    def off(rva):
        for va, sz, raw in secs:
            if va <= rva < va + sz:
                return rva - va + raw
        raise ValueError(rva)
    names = []
    p = off(imp_rva)
    while True:
        name_rva = struct.unpack_from("<I", d, p + 12)[0]
        if name_rva == 0:
            break
        o = off(name_rva)
        names.append(d[o:d.index(b"\0", o)].decode())
        p += 20
    return names


if __name__ == "__main__":
    libdir = sys.argv[1]
    allimp = {}
    for fn in sorted(os.listdir(libdir)):
        if fn.lower().endswith((".dll", ".pyd")):
            try:
                allimp[fn] = imports(os.path.join(libdir, fn))
            except Exception as e:
                allimp[fn] = [f"ERR {e}"]
    present = {f.lower() for f in os.listdir(libdir)}
    sysdir = r"C:\Windows\System32"
    need = set()
    for fn, ims in allimp.items():
        for im in ims:
            l = im.lower()
            if l in present or l.startswith("api-ms-win") or l.startswith("ext-ms-") or os.path.exists(os.path.join(sysdir, im)):
                continue
            need.add(im)
    print("missing (not in lib dir / System32):", sorted(need))
    for fn in ("c10.dll", "torch_cpu.dll", "torch_python.dll"):
        if fn in allimp:
            print(fn, allimp[fn])
