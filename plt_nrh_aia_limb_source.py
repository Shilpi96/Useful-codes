#### plot nrh contours of limb sources on top of AIA

from astropy.io import fits
from astropy import units as u 
from astropy.coordinates import SkyCoord
import sunpy.map
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
from scipy.io import readsav
from sunpy.coordinates import frames, screens, sun
import numpy as np
import pandas as pd
from sunpy.time import parse_time
from astropy.visualization import ImageNormalize, SqrtStretch
from matplotlib import colors
import pdb, glob
from scipy.io import readsav
from datetime import datetime
from astropy.wcs import WCS
from functools import partial
from astropy.time import Time
import multiprocessing
import sunpy.map
from sunpy.map import Map, make_fitswcs_header

def get_time(hdul):
	times = hdul[1].data['TIME']
	date_obs_str = hdul[1].header['DATE-OBS']
	reference_time = Time(f"{date_obs_str}T00:00:00", scale='utc')
	times_datetime = reference_time + times * u.s / 1000
	times = np.array([datetime.strptime(times_datetime.isot[i], "%Y-%m-%dT%H:%M:%S.%f") for i in range(times_datetime.shape[0])])
	return times
	
def nrh_sunpy_map(nrh, dtime =  datetime(2022, 11, 11, 11, 32, 1, 400000)):
    hdul = fits.open(nrh)
    times = get_time(hdul)
    ind = np.abs(times-dtime).argmin()
    data = hdul[1].data['STOKESI'][ind]
    print(times[ind])
    #pdb.set_trace()
    ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, 
                     frame=frames.Helioprojective(observer="earth", obstime=times[ind]), 
                     )
    solar_pix = hdul[1].header['SOLAR_R']
    solar_r_arcsec = sun.angular_radius(times[ind]).value
    cdelta = solar_r_arcsec/solar_pix
    
    header = sunpy.map.make_fitswcs_header(data, 
                                       ref_coord, 
                                       reference_pixel=[int(hdul[1].header['CRPIX1']), int(hdul[1].header['CRPIX2'])]*u.pixel, 
                                       scale=[float(cdelta), float(cdelta)]*u.arcsec/u.pixel, 
                                       wavelength=float(hdul[1].header['FREQ'])*u.MHz)
    #pdb.set_trace()
    nrh_map = Map(data, header)
    freq = nrh_map.meta['wavelnth']
    
    return nrh_map

def plot_contours(aia,dtime, fname):
	nrh_map = nrh_sunpy_map(fname, dtime = dtime)
	nrh_map.meta['rsun_ref'] = aia.meta['rsun_ref']
	print('start plotting the contours')
	##### if the source is at limb, then use a spherical screen
	with frames.Helioprojective.assume_spherical_screen(aia_map.observer_coordinate):
		nrh_map.draw_contours(axes = ax, levels=np.arange(90, 100, 3)*u.percent)
 

### make a list of nrh files
root1 = '/home/shilpi/march_campaign/event_2025_03_28/data/'
nrh = root1+'nrh2_4320_h80_20250328_150700c04_b.fts'
#pdb.set_trace()
#### load aia file
aia_map = sunpy.map.Map(root1+'171/aia.lev1.171A_2025_03_28T15_18_33.35Z.image_lev1.fits')
fig = plt.figure()
ax = fig.add_subplot(projection=aia_map)
cmap = plt.get_cmap(aia_map.plot_settings["cmap"])
cmap.set_bad("k")
print('plotting the aia data')
aia_map.plot(axes=ax, clip_interval=(1, 99.99)*u.percent)
collist = plt.cm.coolwarm(np.linspace(0,1, 9))

plot_contours(aia_map,datetime(2025,3,28, 15,18,34,222000),nrh)
ax.patch.set_facecolor('black')
xlims_world = [-1400, -500]*u.arcsec
ylims_world = [-100, 600]*u.arcsec
#pdb.set_trace()
world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
pixel_coords = aia_map.world_to_pixel(world_coords)

# we can then pull out the x and y values of these limits.
xlims_pixel = pixel_coords.x.value
ylims_pixel = pixel_coords.y.value

ax.set_xlim(xlims_pixel)
ax.set_ylim(ylims_pixel)

plt.show()



