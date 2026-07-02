##### This script makes a movie of the temporal evolution of AIA maps and time-distance maps. Uses the example https://docs.sunpy.org/en/stable/generated/gallery/showcase/time_distance.html?__cf_chl_f_tk=PbetncirVCZtQt3B44jBNN_rzFQFA6JUkNalhHu01GQ-1782907870-1.0.1.1-23ANtCKA47wvqmuLXIUnc3VqviuCtMeMzYYNoXxQDSY
import os, glob
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm

import sunpy.map, pdb
from sunpy.coordinates import propagate_with_solar_surface
from sunpy.coordinates.screens import SphericalScreen
from sunpy.map import pixelate_coord_path, sample_at_coords


# ============================================================
# INPUT DATA
# ============================================================

root = '/home/sbhunia/AIA_data/'
files = sorted(glob.glob(root + '211/*.fits'))
#pdb.set_trace()
aia_seq = [m / m.exposure_time for m in sunpy.map.Map(files)]

print(f"Loaded {len(aia_seq)} maps")

# ============================================================
# REFERENCE SUBMAP
# ============================================================

corner = SkyCoord(
    Tx=300*u.arcsec,
    Ty=-800*u.arcsec,
    frame=aia_seq[0].coordinate_frame)

ref_sub_map = aia_seq[0].submap(
    bottom_left=corner,
    width=900*u.arcsec,
    height=1200*u.arcsec)

# ============================================================
# SLIT
# ============================================================

line_coords = SkyCoord(
    [409, 653]*u.arcsec,
    [-764, -111]*u.arcsec,
    frame=aia_seq[0].coordinate_frame)    #### (x1,x2), (y1,y2)
 
# ============================================================
# REPROJECT MAPS
# ============================================================

print("Reprojecting maps...")

reprojected_sub_maps = []

for m in aia_seq:

    with (
        propagate_with_solar_surface(),
        SphericalScreen(m.observer_coordinate, only_off_disk=True)
    ):
        reprojected_sub_maps.append(
            m.reproject_to(ref_sub_map.wcs, preserve_date_obs=True)
        )

print("Reprojection done.")

# ============================================================
# SLIT SAMPLING
# ============================================================

coords = pixelate_coord_path(reprojected_sub_maps[0], line_coords)

intensities = []

for m in reprojected_sub_maps:
    intensities.append(sample_at_coords(m, coords))

intensities = np.stack(intensities, axis=1).value

# distance axis
distance = coords.separation(coords[0]).to(u.arcsec)

# time-distance extent
extent = [
    reprojected_sub_maps[0].date.datetime,
    reprojected_sub_maps[-1].date.datetime,
    0,
    distance[-1].value
]

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

outdir = "frames_aia_time_distance_movie"
os.makedirs(root+outdir, exist_ok=True)

# fixed scaling (IMPORTANT for movies)
vmin, vmax = 50, 4000

# ============================================================
# FRAME LOOP
# ============================================================

print("Creating frames...")

for i, m in enumerate(reprojected_sub_maps):

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)

    # ---------------- LEFT PANEL (AIA IMAGE) ----------------
    ax1 = fig.add_subplot(121, projection=m)

    im = m.plot(
        axes=ax1,
        clip_interval=(1, 99.9)*u.percent
    )

    ax1.plot_coord(
        line_coords,
        color='red',
        linewidth=2
    )

    ax1.set_title(m.date.strftime("%Y-%m-%d %H:%M:%S"))

    # ---------------- RIGHT PANEL (TD MAP) ----------------
    ax2 = fig.add_subplot(122)

    td = np.full_like(intensities, np.nan)
    td[:, :i+1] = intensities[:, :i+1]

    ax2.imshow(
        td,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap=im.get_cmap(),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent
    )

    ax2.axvline(
        m.date.datetime,
        color='red',
        linewidth=2
    )

    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel("Distance (arcsec)")
    ax2.set_title("Time–Distance Map")
    plt.show()
    # ---------------- SAVE FRAME ----------------
    filename = f"{root+outdir}/frame_{i:04d}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"Saved {filename}")

print("All frames created.")

