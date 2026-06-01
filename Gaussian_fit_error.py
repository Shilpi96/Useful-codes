## This code will fit a radio source with gaussian and calculate the error from 50% of the contours of the source.
import numpy as np
import pdb
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
import astropy.units as u

# ------------------------
# 2D Gaussian model
# ------------------------
def gauss2d(coords, A, x0, y0, theta, sx, sy):
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)

    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)

    return (A * np.exp(-0.5 * ((xp/sx)**2 + (yp/sy)**2))).ravel()


# ------------------------
# Gaussian fit (50% contour idea)
# ------------------------
def fit_gaussian(data):

    z = np.array(data, dtype=float)

    A = np.nanmax(z)
    y0, x0 = np.unravel_index(np.nanargmax(z), z.shape)

    sx0 = sy0 = max(2.0, 0.1 * min(z.shape))

    p0 = [A, x0, y0, 0.0, sx0, sy0]

    bounds = (
        [0, 0, 0, -np.pi, 1, 1],
        [3*A, z.shape[1], z.shape[0], np.pi, z.shape[1], z.shape[0]]
    )

    yy, xx = np.indices(z.shape)

    popt, pcov = curve_fit(
        gauss2d,
        (xx.ravel(), yy.ravel()),
        z.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    return popt, z, xx, yy


# ------------------------
# ΔF (background noise)
# ------------------------
def estimate_deltaF(data, model, A):

    # remove source influence (NOT 50% contour!)
    mask = model > 0.1 * A

    background = data[~mask]

    _, _, std = sigma_clipped_stats(background, sigma=3, maxiters=5)

    return std


# ------------------------
# MAIN
# ------------------------
fits_path = "/home/sbhunia/nenufar_images/type_II_2024_07/fits_SB414/SB414/SB414-t0030-image.fits"

with fits.open(fits_path) as hdul:
    data = hdul[0].data

while data.ndim > 2:
    data = data[0]

data = np.squeeze(data)


# ------------------------
# FIT
# ------------------------
popt, z, xx, yy = fit_gaussian(data)

A, x0, y0, theta, sx, sy = popt


# ------------------------
# MODEL
# ------------------------
model = gauss2d((xx.ravel(), yy.ravel()), *popt).reshape(z.shape)


# ------------------------
# MASKS
# ------------------------
fit_mask = model > 0.5 * A        # conside 50% contour region
noise_mask = model > 0.1 * A      # exclude source fully
# ------------------------
# ΔF
# ------------------------
deltaF = estimate_deltaF(z, model, A)
# ------------------------
# CONVERT σ (pixel → arcsec if needed)
# ------------------------
bmaj = hdul[0].header['BMAJ'] * 3600 ## deg to arcsec
bmin = hdul[0].header['BMIN'] * 3600 ## deg to arcsec
h = np.sqrt(bmaj * bmin)
cdelt_x = abs(hdul[0].header['CDELT1']) * 3600
cdelt_y = abs(hdul[0].header['CDELT2']) * 3600 ## CDELTA value in arcsec
sigma_x = sx * cdelt_x
sigma_y = sy * cdelt_y
S0 = A


# ------------------------
# KONTAR UNCERTAINTY
# ------------------------
dx = np.sqrt(2/np.pi) * (sigma_x/sigma_y) * (deltaF/S0) * h
dy = np.sqrt(2/np.pi) * (sigma_y/sigma_x) * (deltaF/S0) * h


print("\nRESULTS")
print("S0 (peak)   =", S0)
print("sigma_x     =", sigma_x)
print("sigma_y     =", sigma_y)
print("deltaF      =", deltaF)
print("dx          =", dx)
print("dy          =", dy)


# ------------------------
# VISUAL CHECK
# ------------------------
cen_x = x0
cen_y = y0
dx_pix = dx / cdelt_x
dy_pix = dy / cdelt_y
plt.figure(figsize=(6,6))
plt.imshow(z, origin="lower", cmap="viridis")

plt.contour(fit_mask, levels=[0.5], colors="red", linewidths=1.5)
plt.contour(noise_mask, levels=[0.5], colors="white", linewidths=1.0, linestyles="dashed")

# centroid
plt.plot(cen_x, cen_y, marker="+", color="white", markersize=12, mew=2)

# error bars (pixel space)
plt.errorbar(
    cen_x, cen_y,
    xerr=dx_pix,
    yerr=dy_pix,
    fmt='o',
    color='white',
    ecolor='white',
    elinewidth=2,
    capsize=4
)

plt.title("Centroid + Kontar uncertainty")
plt.show()
