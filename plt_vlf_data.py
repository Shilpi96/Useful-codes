#### this script will plot Dunsink vlf data
import matplotlib.pyplot as plt 
import pandas as pd
import pdb
from matplotlib import dates
from datetime import datetime

path = '/home/shilpi/flare_ionospheric_project/'
df = pd.read_csv(path+'Dunsink_DHO38_2024-05-14.csv')

fig2 = plt.figure(figsize=(7, 7))
axs = fig2.add_subplot(1,1,1)
#df[9000:13000].plot(ax=axs)

times = [datetime.strptime(df.iloc[:, 1][9000+i], '%Y-%m-%d %H:%M:%S.%f') for i in range(5000)]
data = [df.iloc[:, 2][9000+i] for i in range(5000)]
#pdb.set_trace()
axs.plot(times,data)
axs.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.show()
