"""Supplied (possibly wrong) kernels for the Wiener kernel-mismatch check.

Only used to build the kernel GIVEN to the Wiener filter. Blur generation always uses the
base src.psf kernels.
"""
import numpy as np

from src.psf import motion_psf


def line_psf(length, angle_deg, samples_per_px=2000):
    """Uniform line segment of `length` px centred at the kernel centre, direction `angle_deg`
    (0 = +x, counter-clockwise towards -y is NOT used; positive angle rotates from +x towards +y,
    i.e. image row index). Pixel weight = length of the segment inside the pixel square,
    estimated by dense mid-point sampling. Symmetrised (k + rot180 k)/2 and normalised.

    At angle 0 with an odd length this reproduces src.psf.motion_psf(length, 0)."""
    L = float(length)
    th = np.deg2rad(angle_deg)
    n = int(np.ceil(L * samples_per_px))
    t = (np.arange(n) + 0.5) / n * L - L / 2.0          # mid-points, symmetric about 0
    x = t * np.cos(th)
    y = t * np.sin(th)
    r = int(np.ceil(max(np.abs(x).max(), np.abs(y).max()) + 0.5))
    size = 2 * r + 1
    ix = np.floor(x + 0.5).astype(int) + r                # pixel j covers [j-0.5, j+0.5)
    iy = np.floor(y + 0.5).astype(int) + r
    k = np.zeros((size, size))
    np.add.at(k, (iy, ix), 1.0)
    k = 0.5 * (k + k[::-1, ::-1])
    k /= k.sum()
    # trim all-zero outer rings (keeps odd size and centring)
    while k.shape[0] > 1 and not k[0].any() and not k[-1].any() and not k[:, 0].any() and not k[:, -1].any():
        k = k[1:-1, 1:-1]
    return k


def supplied_kernel(length, angle_deg):
    if angle_deg == 0 and int(length) % 2 == 1:
        return motion_psf(int(length), 0)
    return line_psf(length, angle_deg)
