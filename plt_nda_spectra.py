from astropy.io import fits
import numpy as np
from astropy.time import Time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

# =========================
# 1. READ FITS FILE
# =========================
print('Reading the file...')

fitsfile = '/Users/shilpibhunia/Documents/projects/March_2025_campaign/event_2025_03_26/orn_nda_newroutine_sun_edr_202503260756_202503261555_v1.1.fits'
hdul = fits.open(fitsfile)

# =========================
# 2. GET FREQUENCY
# =========================
freq = hdul[1].data[0][0]   # shape: (n_freq,)

# =========================
# 3. GET TIME ARRAY
# =========================
times = []
for i in range(hdul[2].data.shape[0]):
    t = Time(hdul[2].data[i][0], format='jd')
    t = datetime.strptime(t.to_value('isot'), '%Y-%m-%dT%H:%M:%S.%f')
    times.append(t)

times = np.array(times)

# =========================
# 4. SELECT TIME RANGE
# =========================
startt = datetime(2025, 3, 26, 9, 21)
endt   = datetime(2025, 3, 26, 9, 51)

startid = np.argmin(np.abs(times - startt))
endid   = np.argmin(np.abs(times - endt))

Times = times[startid:endid]
Data  = hdul[2].data[startid:endid]

print("Selected time range:", Times[0], "to", Times[-1])

# =========================
# 5. EXTRACT LL & RR (FAST)
# =========================
# Data[i][1] → (freq, polarization)
# polarization: 0 = LL, 1 = RR

ldata = np.array([row[1][:, 0] for row in Data])
rdata = np.array([row[1][:, 1] for row in Data])

# shape check
print("ldata shape:", ldata.shape)  # (time, freq)
print("rdata shape:", rdata.shape)

# =========================
# 6. STOKES I (TOTAL POWER)
# =========================
data = (ldata + rdata) / 2.0

# =========================
# 7. DATA PERCENTILE FOR PLOTTING
# =========================
# Percentile of data for colormap
vmin = np.percentile(data, 5)
vmax = np.percentile(data, 95)

# =========================
# 8. PLOT DYNAMIC SPECTRUM
# =========================
time_nums = mdates.date2num(Times)

plt.figure(figsize=(12, 6))

plt.pcolormesh(time_nums, freq, data.T,
               shading='auto',
               cmap='viridis',
               norm=LogNorm(vmin=np.percentile(data, 5),
                            vmax=np.percentile(data, 95)))

plt.colorbar(label='Intensity (log scale)')

# Format time axis
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Flip frequency axis (common for NDA)
plt.gca().invert_yaxis()

plt.xlabel('Time (UT)')
plt.ylabel('Frequency (MHz)')
plt.title('NDA Dynamic Spectrum (Stokes I)')

plt.tight_layout()
plt.show()

# =========================
# 9. SAVE ARRAYS 
# =========================
np.save('ldata.npy', ldata)
np.save('rdata.npy', rdata)
np.save('freq.npy', freq)
np.save('times.npy', Times)

print("Done!")
