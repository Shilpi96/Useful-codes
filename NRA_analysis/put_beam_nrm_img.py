#### Plot beam on NRH image
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import sunpy.map
from sunpy.map import Map
from sunpy.coordinates import frames, sun

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import pandas as pd

from datetime import datetime


# ============================================================
# Function: Get NRH observation times
# ============================================================

def get_time(hdul):

    times = hdul[1].data['TIME']
    date_obs_str = hdul[1].header['DATE-OBS']

    reference_time = Time(
        f"{date_obs_str}T00:00:00",
        scale='utc'
    )

    # TIME is assumed to be in milliseconds
    times_datetime = reference_time + times * u.s / 1000

    times = np.array([
        datetime.strptime(
            t,
            "%Y-%m-%dT%H:%M:%S.%f"
        )
        for t in times_datetime.isot
    ])

    return times


# ============================================================
# Function: Create SunPy map for selected NRH time
# ============================================================

def nrh_sunpy_map(nrh_file, target_time):

    with fits.open(nrh_file) as hdul:

        # Read all NRH times
        times = get_time(hdul)

        # Find image closest to requested time
        ind = np.abs(times - target_time).argmin()

        selected_time = times[ind]

        print("--------------------------------")
        print("Requested time :", target_time)
        print("Selected time  :", selected_time)
        print("NRH index      :", ind)
        print("Time difference:", abs(selected_time - target_time))
        print("--------------------------------")

        # Use the same index for the NRH image
        data = hdul[1].data['STOKESI'][ind]

        # Define solar disk center
        ref_coord = SkyCoord(
            0 * u.arcsec,
            0 * u.arcsec,
            frame=frames.Helioprojective(
                observer="earth",
                obstime=selected_time
            )
        )

        # Calculate NRH pixel scale
        solar_pix = hdul[1].header['SOLAR_R']

        solar_r_arcsec = sun.angular_radius(
            selected_time
        ).value

        cdelta = solar_r_arcsec / solar_pix

        # Create WCS header
        header = sunpy.map.make_fitswcs_header(
            data,
            ref_coord,

            reference_pixel=[
                int(hdul[1].header['CRPIX1']),
                int(hdul[1].header['CRPIX2'])
            ] * u.pixel,

            scale=[
                float(cdelta),
                float(cdelta)
            ] * u.arcsec / u.pixel,

            wavelength=(
                float(hdul[1].header['FREQ'])
                * u.MHz
            )
        )

    nrh_map = Map(data, header)

    return nrh_map, selected_time, ind

# ============================================================
# Function: Plot NRH beam
# ============================================================
def plot_nrh_beam(ax,nrh_map,major,minor,angle,x=-1300,y=50):

    """
    Plot NRH beam on NRH SunPy map.

    Parameters
    ----------
    major : float
        Beam major axis in solar radii (Rsun).

    minor : float
        Beam minor axis in solar radii (Rsun).

    angle : float
        Beam position angle in degrees.

    x, y : float
        Position of beam center in helioprojective arcsec.
    """

    # ========================================================
    # Get apparent solar radius at observation time
    # ========================================================

    solar_r_arcsec = sun.angular_radius(nrh_map.date).to_value(u.arcsec)
    # ========================================================
    # Convert beam axes:
    # solar radii -> arcsec
    # ========================================================

    major_arcsec = major * solar_r_arcsec
    minor_arcsec = minor * solar_r_arcsec

    # ========================================================
    # Define beam center in helioprojective coordinates
    # ========================================================

    beam_center = SkyCoord(Tx=x * u.arcsec,Ty=y * u.arcsec,frame=nrh_map.coordinate_frame)
    # ========================================================
    # Convert beam center to NRH image pixels
    # ========================================================
    beam_center_pix = nrh_map.world_to_pixel(beam_center)

    # ========================================================
    # Get NRH pixel scale
    # ========================================================
    scale_x = np.abs(nrh_map.scale.axis1.to_value(u.arcsec / u.pixel))
    scale_y = np.abs(nrh_map.scale.axis2.to_value(u.arcsec / u.pixel))

    # ========================================================
    # Convert beam dimensions:
    # arcsec -> pixels
    # ========================================================

    major_pix = major_arcsec / scale_x
    minor_pix = minor_arcsec / scale_y

    # ========================================================
    # Draw beam
    # ========================================================

    beam = Ellipse(xy=(beam_center_pix.x.value,beam_center_pix.y.value),width=major_pix,
        height=minor_pix,angle=angle,fill=False,
        edgecolor='red',linewidth=3,zorder=100)
    ax.add_patch(beam)

# ============================================================
# Paths
# ============================================================
root = '/home/shilpi/march_campaign/event_2025_03_28/data/'
nrh_file = (root + 'nrh2_4440_h81_20250328_151154c72_b.fts')
beam_csv = (root+ 'beam_evolution_444.csv')
# ============================================================
# Requested NRH time
# ============================================================
target_time = datetime(2025,3,28,15,14,38,720000)

# ============================================================
# Get NRH image closest to target time
#
# IMPORTANT:
# This returns "ind".
# The SAME "ind" is used below for the beam CSV.
# ============================================================

nrh_map, nrh_time, ind = nrh_sunpy_map(nrh_file,target_time)
# ============================================================
# Read beam CSV
# ============================================================
beam_df = pd.read_csv(beam_csv)

# ============================================================
# Select beam using SAME index as NRH image
# ============================================================

beam_major = beam_df["major"].iloc[ind]
beam_minor = beam_df["minor"].iloc[ind]
beam_angle = beam_df["angle"].iloc[ind]

print()
print("Same index used for image and beam:", ind)
print()
print("Beam parameters:")
print("Major =", beam_major)
print("Minor =", beam_minor)
print("Angle =", beam_angle)


# ============================================================
# Plot AIA image
# ============================================================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection=nrh_map)

# Plot NRH
nrh_map.plot(axes=ax,clip_interval=(1, 99.99) * u.percent,cmap = 'viridis')
nrh_map.draw_limb(axes=ax)
# ============================================================
# Plot NRH beam
# x and y define where the beam ellipse appears.
# ============================================================
plot_nrh_beam(ax,nrh_map,beam_major,beam_minor,beam_angle,x=-1200,y=-1000)

# ============================================================
# Zoom region
# ============================================================
xlims_world = [-1669,1096] * u.arcsec
ylims_world = [-1240,1240] * u.arcsec

# Create coordinates defining zoom region
world_coords = SkyCoord(Tx=xlims_world,Ty=ylims_world,frame=nrh_map.coordinate_frame)

# Convert world coordinates to pixels
pixel_coords = nrh_map.world_to_pixel(world_coords)

xlims_pixel = pixel_coords.x.value
ylims_pixel = pixel_coords.y.value
ax.set_xlim(xlims_pixel)
ax.set_ylim(ylims_pixel)

# ============================================================
# Title
# ============================================================
ax.set_title(f"time: {nrh_time.strftime('%H:%M:%S.%f')}")

plt.tight_layout()

plt.show()
