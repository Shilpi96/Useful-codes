#### This code downloads RPW L3 data given date and time, and plots the spectra
from sunpy.net import Fido, attrs as a
from astropy import units as u
import matplotlib.pyplot as plt
import cdflib
from cdflib.epochs import CDFepoch
import numpy as np
from matplotlib import dates as mdates
import requests
from pathlib import Path

# ---- USER INPUT ----
year = 2025
month = 3
day = 26

# Construct filename and URL
filename = f"solo_L3_rpw-tnr-surv-flux_{year}{month:02d}{day:02d}_V01.cdf"
url = f"https://rpw-lira.obspm.fr/roc/data/pub/solo/rpw/data/L3/thr_flux/{year:04d}/{month:02d}/{filename}"

# Local path to save
local_folder = Path.home() / "Documents" / "projects" / "RPW_analysis"
local_folder.mkdir(parents=True, exist_ok=True)
local_file = local_folder / filename

# Download if not already present
if not local_file.exists():
    print(f"Downloading {filename} ...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f"Downloaded to {local_file}")
    else:
        raise ValueError(f"File not found at {url}")

else:
    print(f"Using existing file: {local_file}")

# ---- LOAD CDF FILE ----
cdf = cdflib.CDF(local_file)

epoch = cdf['Epoch'][:]
times = CDFepoch.to_datetime(epoch)
t_num = mdates.date2num(times)

freq = cdf['FREQUENCY'][:]
spec = np.squeeze(cdf['PSD_SFU'][:])

spec_to_plot = np.log10(spec)
T, F = np.meshgrid(t_num, freq, indexing='ij')

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(12, 4))
pcm = ax.pcolormesh(T, F, spec_to_plot, shading='auto', cmap="magma", vmin=2, vmax=8)
cbar = fig.colorbar(pcm, ax=ax, label='log10(PSD_SFU)')

ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [UT]')
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_yscale("log")
ax.set_ylim(4e4, 1e6)  # adjust max frequency if needed
plt.tight_layout()
plt.show()
