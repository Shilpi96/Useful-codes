### This code shows an example image with gaussian fit and error calculated from 50% of the source using Kontar error formula
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import EarthLocation,SkyCoord
import sunpy.map, pdb
from matplotlib.patches import Ellipse
from sunpy.coordinates import frames, sun

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

    # Initial guess
    A = np.nanmax(z)
    y0, x0 = np.unravel_index(np.nanargmax(z), z.shape)

    sx0 = sy0 = max(2.0, 0.1 * min(z.shape))

    p0 = [A, x0, y0, 0.0, sx0, sy0]

    bounds = (
        [0, 0, 0, -np.pi, 1, 1],
        [3*A, z.shape[1], z.shape[0], np.pi, z.shape[1], z.shape[0]]
    )

    yy, xx = np.indices(z.shape)

    # ------------------------
    # FIRST FIT (all finite pixels)
    # ------------------------
    mask = np.isfinite(z)

    popt0, pcov0 = curve_fit(
        gauss2d,
        (xx[mask], yy[mask]),
        z[mask],
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    # Build model from first fit
    model0 = gauss2d(
        (xx.ravel(), yy.ravel()),
        *popt0
    ).reshape(z.shape)

    A0 = popt0[0]

    # ------------------------
    # 50% contour mask
    # ------------------------
    fit_mask = (model0 > 0.5 * A0) & np.isfinite(z)

    # ------------------------
    # SECOND FIT (50% contour only)
    # ------------------------
    popt, pcov = curve_fit(
        gauss2d,
        (xx[fit_mask], yy[fit_mask]),
        z[fit_mask],
        p0=popt0,
        bounds=bounds,
        maxfev=20000
    )

    return popt, z, xx, yy

# ------------------------
# ΔF (background noise)
# ------------------------
def estimate_deltaF(data, model, A):

    # remove source influence (NOT 50% contour!)
    mask = model > 0.05 * A

    background = data[~mask]

    _, _, std = sigma_clipped_stats(background, sigma=3, maxiters=5)

    return std


fits_path = "/data/sbhunia/new_type_II_2024/SB370_72_3/step_iocorrect_outputs_20240713/SB370/corr_fits/SB370-t0180-image_corrWCS.fits"

with fits.open(fits_path) as hdul:
	hdr = hdul[0].header
	data = hdul[0].data
while data is not None and getattr(data, "ndim", 0) > 2:
	data = data[0]
	data = np.squeeze(data)	
freq_Hz = hdr.get("CRVAL3", None)
frequency = (freq_Hz * u.Hz) if freq_Hz is not None else (np.nan * u.Hz)

# pixel scale (deg/pix -> arcsec/pix), positive
cdelt1 = abs(hdr.get("CDELT1", np.nan)) * u.deg
cdelt2 = abs(hdr.get("CDELT2", np.nan)) * u.deg
cdelt1 = cdelt1.to(u.arcsec) if np.isfinite(cdelt1.value) else (np.nan * u.arcsec)
cdelt2 = cdelt2.to(u.arcsec) if np.isfinite(cdelt2.value) else (np.nan * u.arcsec)

# observer at site (GCRS)
obstime = Time(hdr['DATE-OBS'])
site_loc = EarthLocation(lat=47.382*u.deg, lon=2.195*u.deg)
site_gcrs = SkyCoord(site_loc.get_gcrs(obstime))

# reference sky coord from header CRVAL1/2
cunit1 = u.Unit(hdr.get("CUNIT1", "deg"))
cunit2 = u.Unit(hdr.get("CUNIT2", "deg"))
ref_gcrs = SkyCoord(hdr["CRVAL1"] * cunit1,
                hdr["CRVAL2"] * cunit2,
                frame="gcrs",
                obstime=obstime,
                obsgeoloc=site_gcrs.cartesian,
                obsgeovel=site_gcrs.velocity.to_cartesian(),
                distance=site_gcrs.hcrs.distance,)
ref_hpc = ref_gcrs.transform_to(frames.Helioprojective(observer=site_gcrs))
# rotate so solar north up
P1 = sun.P(obstime)
# reference pixel is 0-based in make_fitswcs_header
ref_pix = np.array([hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1]) * u.pixel
scale = np.array([cdelt1.value, cdelt2.value]) * (u.arcsec / u.pixel)
new_header = sunpy.map.make_fitswcs_header(
                data=data,
                coordinate=ref_hpc,
                reference_pixel=ref_pix,
                scale=scale,
                rotation_angle=-P1,
                wavelength=frequency.to(u.MHz) if np.isfinite(frequency.value) else None,
                observatory="NenuFAR (Nançay)",)

rmap = sunpy.map.Map(data, new_header)
rmap = rmap.rotate()

# -----fit the data and calculate error------
pdb.set_trace()
### fitting
popt, z, xx, yy = fit_gaussian(rmap.data) 
A, x0, y0, theta, sx, sy = popt
model = gauss2d((xx.ravel(), yy.ravel()), *popt).reshape(z.shape)  ## model
 
deltaF = estimate_deltaF(z, model, A)  ## ΔF
bmaj = abs(hdul[0].header['BMAJ']) * 3600 ## deg to arcsec
bmin = abs(hdul[0].header['BMIN']) * 3600 ## deg to arcsec
h = np.sqrt(bmaj * bmin) ## angualr resolution h
cdelt_x = abs(hdul[0].header['CDELT1']) * 3600
cdelt_y = abs(hdul[0].header['CDELT2']) * 3600 ## CDELTA value in arcsec
sigma_x = sx * cdelt_x
sigma_y = sy * cdelt_y
S0 = A
# KONTAR UNCERTAINTY
# ------------------------
dx = np.sqrt(2/np.pi) * (sigma_x/sigma_y) * (deltaF/S0) * h
dy = np.sqrt(2/np.pi) * (sigma_y/sigma_x) * (deltaF/S0) * h
print("\nRESULTS")
print("S0 (peak)   =", S0)
print("sigma_x     =", sigma_x)
print("sigma_y     =", sigma_y)
print("deltaF      =", deltaF)
print("dx_arsec          =", dx)
print("dy_arcsec          =", dy)

# ---- plot layout ----
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection=rmap)
vmin, vmax = np.nanpercentile(rmap.data, (10,99))
im = rmap.plot(axes=ax, cmap='viridis')
rmap.draw_limb(axes=ax)
rmap.draw_grid(axes=ax)
rmap.draw_contours(np.arange(50, 100, 20) * u.percent, colors="k",linewidths=1.1, alpha=0.8, axes=ax)
# plotting centroid positions
centr = rmap.pixel_to_world(x0*u.pix, y0*u.pix)  ## in arcsec
#pdb.set_trace()
ax.plot_coord(centr,'ro')
x_deg = centr.Tx.to(u.deg).value
y_deg = centr.Ty.to(u.deg).value
ax.errorbar(
    x_deg,
    y_deg,
    xerr=(dx * u.arcsec).to(u.deg).value,
    yerr=(dy * u.arcsec).to(u.deg).value,
    transform=ax.get_transform('world'),
    fmt='+',
    color='white',
    ecolor='white',
    elinewidth=2,
    capsize=4,
    markersize=10,)
# mention the position and error on image
ax.text(
    0.05, 0.95,
    f"x = {centr.Tx.value:.1f} ± {dx:.1f} arcsec\n"
    f"y = {centr.Ty.value:.1f} ± {dy:.1f} arcsec",
    transform=ax.transAxes,
    color="white",
    va="top",
    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
)
# Crop region
xlims_world = [-3000, 3000] * u.arcsec
ylims_world = [-3000, 3000] * u.arcsec
world_coords = SkyCoord(Tx=xlims_world,Ty=ylims_world,frame=rmap.coordinate_frame)
pixel_coords = rmap.world_to_pixel(world_coords)
ax.set_xlim(pixel_coords.x.value)
ax.set_ylim(pixel_coords.y.value)

# Draw beam
bmin = bmin/rmap.meta['cdelt2']  ### to convert bmin in pixel
bmaj = bmaj/rmap.meta['cdelt1'] 
bpa = hdul[0].header["BPA"]
x0 = ax.get_xlim()[0] + 0.15 * (ax.get_xlim()[1] - ax.get_xlim()[0])
y0 = ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0])
beam_center = SkyCoord(x0 * u.arcsec, y0 * u.arcsec, frame=rmap.coordinate_frame)
xpix, ypix = rmap.world_to_pixel(beam_center)
#pdb.set_trace()
xpix = float(np.atleast_1d(getattr(xpix, "value", xpix))[0])
ypix = float(np.atleast_1d(getattr(ypix, "value", ypix))[0])
e = Ellipse((x0, y0),width=bmaj,height=bmin,angle=-float(bpa),edgecolor="w",transform=ax.get_transform("pixel"),
facecolor="none",lw=1.5,)
ax.add_patch(e)
plt.show()
