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
from scipy.optimize import curve_fit


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

    # TIME assumed to be in milliseconds
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
# 2D rotated Gaussian
# ============================================================

def gauss2d(coords, A, x0, y0, theta, sx, sy):

    x, y = coords

    ct = np.cos(theta)
    st = np.sin(theta)

    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)

    model = A * np.exp(
        -0.5 * (
            (xp / sx)**2 +
            (yp / sy)**2
        )
    )

    return model.ravel()


# ============================================================
# Fit Gaussian
# ============================================================

from scipy.ndimage import label

def fit_gaussian(data, threshold=0.30):

    z = np.array(data, dtype=float)
    yy, xx = np.indices(z.shape)

    # --------------------------------------------------
    # Find peak
    # --------------------------------------------------
    A = np.nanmax(z)

    y_peak, x_peak = np.unravel_index(
        np.nanargmax(z),
        z.shape
    )

    # --------------------------------------------------
    # Threshold image
    # --------------------------------------------------
    threshold_mask = (
        (z >= threshold * A)
        & np.isfinite(z)
    )

    # --------------------------------------------------
    # Find connected regions
    # --------------------------------------------------
    labels, num_features = label(threshold_mask)

    # Find label containing the peak
    source_label = labels[y_peak, x_peak]

    if source_label == 0:
        raise RuntimeError(
            "Peak is not inside threshold mask."
        )

    # Keep ONLY connected source containing peak
    fit_mask = labels == source_label

    print("Number of connected regions:", num_features)
    print("Pixels used for fit:", np.sum(fit_mask))

    # --------------------------------------------------
    # Initial Gaussian parameters
    # --------------------------------------------------
    sx0 = max(2.0, 0.1 * z.shape[1])
    sy0 = max(2.0, 0.1 * z.shape[0])

    p0 = [
        A,
        x_peak,
        y_peak,
        0.0,
        sx0,
        sy0
    ]

    bounds = (
        [
            0,
            0,
            0,
            -np.pi,
            1,
            1
        ],
        [
            3 * A,
            z.shape[1],
            z.shape[0],
            np.pi,
            z.shape[1],
            z.shape[0]
        ]
    )

    # --------------------------------------------------
    # Fit ONLY connected source
    # --------------------------------------------------
    popt, pcov = curve_fit(
        gauss2d,
        (
            xx[fit_mask],
            yy[fit_mask]
        ),
        z[fit_mask],
        p0=p0,
        bounds=bounds,
        maxfev=50000
    )

    # --------------------------------------------------
    # Full fitted Gaussian model
    # --------------------------------------------------
    model = gauss2d(
        (
            xx.ravel(),
            yy.ravel()
        ),
        *popt
    ).reshape(z.shape)

    return popt, model, xx, yy, fit_mask


# ============================================================
# Function: Create SunPy map for selected NRH time
# ============================================================

def nrh_sunpy_map(nrh_file,target_time):

    with fits.open(nrh_file) as hdul:

        times = get_time(hdul)

        # Nearest NRH frame
        ind = np.abs(times - target_time).argmin()

        selected_time = times[ind]
        # ----------------------------------------------------
        # NRH image
        # ----------------------------------------------------
        data = hdul[1].data['STOKESI'][ind]

        # ----------------------------------------------------
        # Solar reference coordinate
        # ----------------------------------------------------

        ref_coord = SkyCoord(0 * u.arcsec,0 * u.arcsec,frame=frames.Helioprojective(
                observer="earth",obstime=selected_time))

        # ----------------------------------------------------
        # NRH pixel scale
        # ----------------------------------------------------

        solar_pix = hdul[1].header[
            'SOLAR_R'
        ]

        solar_r_arcsec = sun.angular_radius(selected_time).value
        cdelta = (solar_r_arcsec/solar_pix)

        # ----------------------------------------------------
        # Create SunPy WCS header
        # ----------------------------------------------------

        header = sunpy.map.make_fitswcs_header(data,ref_coord,reference_pixel=[
                int(hdul[1].header['CRPIX1']),int(hdul[1].header['CRPIX2'])] * u.pixel,scale=[float(cdelta),float(cdelta)] * u.arcsec / u.pixel,wavelength=(
                float(hdul[1].header['FREQ'])*u.MHz))

    nrh_map = Map(data,header)
    return (nrh_map, selected_time,ind)

# ============================================================
# Function: Plot NRH beam
# ============================================================

def plot_nrh_beam(ax,nrh_map,major,minor,angle,x=-1300, y=50):

    # --------------------------------------------------------
    # Solar radius in arcsec
    # --------------------------------------------------------
    solar_r_arcsec = sun.angular_radius(nrh_map.date).to_value(u.arcsec)

    # --------------------------------------------------------
    # Beam: Rsun -> arcsec
    # --------------------------------------------------------
    major_arcsec = (major*solar_r_arcsec)
    minor_arcsec = (minor*solar_r_arcsec)

    # --------------------------------------------------------
    # Beam center
    # --------------------------------------------------------
    beam_center = SkyCoord(Tx=x * u.arcsec,Ty=y * u.arcsec,frame=nrh_map.coordinate_frame)
    beam_center_pix = nrh_map.world_to_pixel(beam_center)

    # --------------------------------------------------------
    # Pixel scale
    # --------------------------------------------------------

    scale_x = np.abs(nrh_map.scale.axis1.to_value(u.arcsec / u.pixel))
    scale_y = np.abs(nrh_map.scale.axis2.to_value(u.arcsec / u.pixel))

    # --------------------------------------------------------
    # Beam: arcsec -> pixels
    # --------------------------------------------------------

    major_pix = (major_arcsec/scale_x)
    minor_pix = (minor_arcsec/scale_y)

    # --------------------------------------------------------
    # Draw beam
    # --------------------------------------------------------

    beam = Ellipse(xy=(beam_center_pix.x.value,beam_center_pix.y.value),width=major_pix,
height=minor_pix,angle=angle,fill=False,edgecolor='red',linewidth=3,zorder=100)
    ax.add_patch(beam)

    # --------------------------------------------------------
    # Label beam
    # --------------------------------------------------------

    ax.text(beam_center_pix.x.value,beam_center_pix.y.value-minor_pix / 2-2,"Beam",color="red",fontsize=10,
ha="center",va="top",zorder=101)

    return (major_arcsec,minor_arcsec)

# ============================================================
# Paths
# ============================================================
root = ('/home/shilpi/march_campaign/event_2025_03_28/data/')
nrh_file = (root + 'nrh2_1732_h81_20250328_151154c73_b.fts')
beam_csv = (root+'beam_evolution_173.csv')
# ============================================================
# Requested time
# ============================================================
target_time = datetime(2025,3,28,15,14,38,720000)
# ============================================================
# Get NRH image
# ============================================================
nrh_map, nrh_time, ind = nrh_sunpy_map(nrh_file,target_time)
# ============================================================
# Beam parameters
# ============================================================
beam_df = pd.read_csv(beam_csv)
beam_major = beam_df["major"].iloc[ind]
beam_minor = beam_df["minor"].iloc[ind]
beam_angle = beam_df["angle"].iloc[ind]

# ============================================================
# Fit Gaussian
# ============================================================
popt, model, xx, yy, fit_mask = fit_gaussian(nrh_map.data,threshold=0.50)
A, x0, y0, theta, sx, sy = popt

# ============================================================
# Convert Gaussian sigma from pixels -> arcsec
# ============================================================
pixel_scale_x = np.abs(nrh_map.scale.axis1.to_value(u.arcsec / u.pixel))
pixel_scale_y = np.abs(nrh_map.scale.axis2.to_value(u.arcsec / u.pixel))

sigma_1_arcsec = (sx*pixel_scale_x)
sigma_2_arcsec = (sy*pixel_scale_y)

# ============================================================
# Convert sigma -> FWHM
# ============================================================
fwhm_1_arcsec = (2.35482*sigma_1_arcsec)
fwhm_2_arcsec = (2.35482*sigma_2_arcsec)

# Sort into source major/minor
source_fwhm_major = max(fwhm_1_arcsec,fwhm_2_arcsec)
source_fwhm_minor = min(fwhm_1_arcsec,fwhm_2_arcsec)

# ============================================================
# Convert beam Rsun -> arcsec
# ============================================================
solar_r_arcsec = sun.angular_radius(nrh_map.date).to_value(u.arcsec)
beam_major_arcsec = (beam_major*solar_r_arcsec)
beam_minor_arcsec = (beam_minor*solar_r_arcsec)

# ============================================================
# Source / beam ratios
# ============================================================

ratio_major = (source_fwhm_major/beam_major_arcsec)
ratio_minor = (source_fwhm_minor/beam_minor_arcsec)

# ============================================================
# Print results
# ============================================================

print()
print("================================")
print("GAUSSIAN FIT RESULTS")
print("================================")

print(f"Gaussian center x = "f"{x0:.2f} pixel")

print(f"Gaussian center y = "f"{y0:.2f} pixel")
print(f"Gaussian angle = "f"{np.degrees(theta):.2f} deg")

print()
print(f"Source FWHM major = "f"{source_fwhm_major:.2f} arcsec")
print("Source FWHM minor = "f"{source_fwhm_minor:.2f} arcsec")
print()
print(f"Beam major = "f"{beam_major_arcsec:.2f} arcsec")
print(f"Beam minor = "f"{beam_minor_arcsec:.2f} arcsec")
print()
print(f"Source / Beam major = "f"{ratio_major:.2f}")
print(f"Source / Beam minor = " f"{ratio_minor:.2f}")

print("================================")

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection=nrh_map)

# NRH image
nrh_map.plot(axes=ax,clip_interval=(1,99.99) * u.percent,cmap='viridis')
# Solar limb
nrh_map.draw_limb(axes=ax)
# Observed NRH contour
nrh_map.draw_contours(axes=ax,levels=[70] * u.percent,colors='black',linewidths=1.5)
# ============================================================
# Gaussian fit contour
#
# 70% contour of fitted Gaussian
# ============================================================
ax.contour(model,levels=[0.70*A],colors='white',linewidths=2,linestyles='--',transform=ax.get_transform('pixel'))

# ============================================================
# Gaussian centroid
# ============================================================

centroid = nrh_map.pixel_to_world(
    x0 * u.pixel,
    y0 * u.pixel
)

ax.plot_coord(
    centroid,
    'r+',
    markersize=10,
    markeredgewidth=2)


# ============================================================
# Plot beam
# ============================================================
plot_nrh_beam(ax,nrh_map,beam_major,beam_minor,beam_angle,x=-1350,y=-540)

# ============================================================
# Zoom region
# ============================================================

xlims_world = [-1669,-7] * u.arcsec
ylims_world = [-650,1175] * u.arcsec

world_coords = SkyCoord(Tx=xlims_world,Ty=ylims_world,frame=nrh_map.coordinate_frame)
pixel_coords = nrh_map.world_to_pixel(world_coords)

ax.set_xlim(pixel_coords.x.value)
ax.set_ylim(pixel_coords.y.value)

# ============================================================
# Title
# ============================================================

ax.set_title(f"time: "
    f"{nrh_time.strftime('%H:%M:%S.%f')}\n"

    f"source/beam: "
    f"{ratio_major:.2f} (major) × "
    f"{ratio_minor:.2f} (minor)")

plt.tight_layout()

plt.show()
