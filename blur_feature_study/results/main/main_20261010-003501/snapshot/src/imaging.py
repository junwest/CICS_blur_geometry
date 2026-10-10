"""Blur generation and the two processing methods (unsharp mask, known-PSF Wiener).

Boundary rule (identical everywhere): half-sample symmetric reflection
(cv2.BORDER_REFLECT == numpy 'symmetric' == scipy.ndimage 'reflect').
"""
import cv2
import numpy as np
from skimage.restoration import wiener as sk_wiener

from .psf import gaussian_psf


def quantize_u8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def convolve_reflect(img, kernel):
    """True 2-D convolution (kernel flipped) with half-sample symmetric borders."""
    return cv2.filter2D(np.asarray(img, dtype=np.float64), cv2.CV_64F, cv2.flip(kernel, -1),
                        borderType=cv2.BORDER_REFLECT)


def apply_blur(clean_u8, psf):
    """Blurred image, rounded/clipped to uint8 (8-bit quantisation is the only noise)."""
    return quantize_u8(convolve_reflect(clean_u8, psf))


def unsharp_mask(blur_u8, alpha=1.0, sigma=1.2, truncate=4.0):
    """I_sharp = I_blur + alpha * (I_blur - G_sigma * I_blur), clipped to [0, 255]."""
    i = blur_u8.astype(np.float64)
    g = convolve_reflect(i, gaussian_psf(sigma, truncate))
    return quantize_u8(i + alpha * (i - g))


def wiener_float(y, psf, balance):
    """Known-PSF Wiener-Hunt deconvolution (skimage, Laplacian regulariser) on a float image.

    The image is mirror-extended to 2H x 2W before the FFT. For a point-symmetric
    PSF and reflect-boundary blur, the blur of the mirrored image is exactly a
    circular convolution, so the FFT model matches the degradation and no
    wrap-around boundary ringing / shift is introduced."""
    y = np.asarray(y, dtype=np.float64)
    h, w = y.shape
    ext = np.pad(y, ((0, h), (0, w)), mode="symmetric")
    return sk_wiener(ext, psf, balance, clip=False)[:h, :w]


def wiener_restore(blur_u8, psf, balance):
    """uint8 in -> [0,1] float -> Wiener -> clipped/quantised uint8 out."""
    x = wiener_float(np.asarray(blur_u8, dtype=np.float64) / 255.0, psf, balance)
    return quantize_u8(x * 255.0)
