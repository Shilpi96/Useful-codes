#### This script with plot td intesity profil along a slit on ratio maps
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Box2DKernel
from matplotlib.colors import Normalize

import sunpy.map,matplotlib
from sunpy.coordinates import propagate_with_solar_surface
from sunpy.coordinates.screens import SphericalScreen
from sunpy.map import pixelate_coord_path, sample_at_coords

from astropy.io import fits
from astropy.time import Time
import sunpy.visualization.colormaps as cm
sdoaia211 = matplotlib.colormaps['sdoaia211']
# ============================================================
# FUNCTIONS
# ============================================================

def exptime(fits_file):
    with fits.open(fits_file) as hdu:
        return hdu[0].header['EXPTIME']


def gettime(fits_file):
    with fits.open(fits_file) as hdu:
        return Time(hdu[0].header['DATE-OBS']).unix


def normalize(data):
    return np.array(data.data / data.exposure_time)


def smooth(data, kernel_size=5):
    kernel = Box2DKernel(kernel_size)
    return convolve(data, kernel)


def getratio(img, imgpre, vmin=0.8, vmax=1.2):
    ratio = img / imgpre

    ratio = np.nan_to_num(ratio)
    ratio[ratio > 1e3] = 1.0
    ratio[ratio < -1e3] = 1.0

    ratio = np.clip(ratio, vmin, vmax)

    ratio = (ratio - vmin) / (vmax - vmin)
    ratio = np.clip(ratio, 0, 1)

    return ratio

# ============================================================
# PATHS
# ============================================================

root = "/home/sbhunia/AIA_data/"
data_path = os.path.join(root, "211")
outdir = os.path.join(root, "td_logratio_movie")

os.makedirs(outdir, exist_ok=True)


# ============================================================
# LOAD FILES
# ============================================================

files = glob.glob(os.path.join(data_path, "*.fits"))
files = sorted(files, key=gettime)[20:]

files = [f for f in files if exptime(f) >= 1.5]

print(f"Total files: {len(files)}")


# ============================================================
# LOAD FIRST MAP
# ============================================================

sample_map = sunpy.map.Map(files[0])

# ============================================================
# SLIT DEFINITION
# ============================================================

line_coords = SkyCoord(
    [345, 537] * u.arcsec,
    [-690, -320] * u.arcsec,
    frame=sample_map.coordinate_frame
)

# Pixelize slit ONCE (important)
slit_coords = pixelate_coord_path(sample_map, line_coords)


# ============================================================
# BUILD LOG-RATIO MAPS (NO REPROJECTION)
# ============================================================

step = 4
maps = []

print("Creating log-ratio maps...")

for i in range(len(files) - step):

    m1 = sunpy.map.Map(files[i])
    m2 = sunpy.map.Map(files[i + step])

    d1 = smooth(normalize(m1), 5)
    d2 = smooth(normalize(m2), 5)

    ratio = getratio(d1, d2)

    maps.append(sunpy.map.Map(ratio, m2.meta))

print(f"Ratio maps: {len(maps)}")


# ============================================================
# SLIT SAMPLING (CO-ROTATING SLIT, NO REPROJECTION)
# ============================================================

print("Sampling slit (co-rotating)...")

intensities = []

for m in maps:

    with (
        propagate_with_solar_surface(),
        SphericalScreen(m.observer_coordinate, only_off_disk=True),
    ):
        vals = sample_at_coords(m, slit_coords)

    intensities.append(vals.value)

intensities = np.stack(intensities, axis=1)

# ============================================================
# Plotting a random ratio map to check the line
# ============================================================
print('Plotting ratio map')
i = 20
ratio_map = maps[i]

vmin, vmax = 0.2, 0.7
norm = Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection=ratio_map)

im = ratio_map.plot(axes=ax, cmap=sdoaia211, origin = 'lower', norm = norm)
ax.plot_coord(line_coords)
ax.set_title(f"Ratio Map (index {i})")
xlims_world = [100, 1225] * u.arcsec
ylims_world = [-900, 300] * u.arcsec

world_coords = SkyCoord(
Tx=xlims_world,Ty=ylims_world,frame=ratio_map.coordinate_frame)

pixel_coords = ratio_map.world_to_pixel(world_coords)

ax.set_xlim(pixel_coords.x.value)
ax.set_ylim(pixel_coords.y.value)
cbar = plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
cbar.set_label("I₂ / I₁")
#### marking the distance on the slit in AIA map
distance = slit_coords.separation(slit_coords[0]).to(u.arcsec).value

# List of (start, end) distances in arcsec
segments = [
    (221, 350),
    (9, 160),
]
for d1, d2 in segments:
    mask = (distance >= d1) & (distance <= d2)
    ax.plot_coord(slit_coords[mask], color='red', linewidth=4)

# ============================================================
# TIME–DISTANCE AXES
# ============================================================

extent = [
    maps[0].date.datetime,
    maps[-1].date.datetime,
    0,
    distance[-1],
]


# ============================================================
# FINAL TIME–DISTANCE MAP (NO MOVIE)
# ============================================================

vmin, vmax = 0.4, 1.2
norm = Normalize(vmin=vmin, vmax=vmax)

print("Building final TD map...")

# TD matrix already computed:
# intensities shape = (slit_length, num_times)

td = intensities  # no frame slicing, no accumulation

# ============================================================
# AXES
# ============================================================

time = [m.date.datetime for m in maps]
time_num = mdates.date2num(time)

distance = slit_coords.separation(slit_coords[0]).to(u.arcsec).value

extent = [
    time_num[0],
    time_num[-1],
    0,
    distance[-1],
]

# ============================================================
# PLOT FINAL TD MAP
# ============================================================

ax1 = fig.add_subplot(122)
vmin, vmax = 0.2, 0.7
norm = Normalize(vmin=vmin, vmax=vmax)
im = ax1.imshow(
    td,
    origin="lower",
    aspect="auto",
    cmap=sdoaia211,
    extent=extent,norm = norm
)

ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax1.set_xlabel("Time (UTC)")
ax1.set_ylabel("Distance (arcsec)")
ax1.set_title("Final Log-Ratio Time–Distance Map")
ax1.invert_yaxis()
cbar = plt.colorbar(im,ax=ax1,fraction=0.046,pad=0.04)
cbar.set_label("I₂ / I₁")
plt.tight_layout()
plt.show()
print("Done.")
