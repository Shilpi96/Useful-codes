import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import datetime
import pdb

filename = '/Users/shilpibhunia/Documents/ALASKA-COHOE_20240323_005959_62.fit'

hdu = pf.open(filename)
data = hdu[0].data
header = hdu[0].header
anc = hdu[1].data
freq = anc['FREQUENCY'][0]
time = anc['TIME'][0]
#pdb.set_trace()
#put in time value of header['TIME-OBS']
dt = datetime.datetime(2024, 3, 23, 00, 59, 59, 785000)

dtime = []

#some processing for nice output

#Histogram equalisation
sd = data.shape
data2 = data.copy()
for i in range(sd[0]):
    dhist, bins = np.histogram(data2[i, :], 1024, density=True)
    cdf = dhist.cumsum()
    cdf = 1023 * cdf / cdf[-1]
    data2[i, :] = np.interp(data2[i, :], bins[:-1], cdf)

data2 = (data-data.min(axis=1)[:,np.newaxis])
for t in time:
    dtime.append(dt + datetime.timedelta(seconds=t))

dtime = np.array(dtime)
fig = plt.figure()
ax = fig.add_subplot(111)
pmn = np.percentile(data2, 10)
pmx = np.percentile(data2, 90)
ax.pcolormesh(dtime, freq, data2, vmin=pmn, vmax=pmx)
#plt.yscale('log')
#ax.invert_yaxis()
plt.ylim(25,86)
plt.show()
