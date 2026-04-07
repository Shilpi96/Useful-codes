### plot learmonth dynamic spectra
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

print("Making dynamic spectrum\n")
# -------------------
# inputs
# -------------------
start_time = pd.to_datetime("24-10-2024 03:37:36")
end_time   = pd.to_datetime("24-10-2024 03:55:59")

pd_data = pd.read_pickle(
    "/Users/shilpibhunia/Documents/projects/MWA_event/LM241024.pd")

# -------------------
# time slicing
# -------------------
t = pd_data.columns

mask = (t >= start_time) & (t <= end_time)
sel_t = t[mask] if mask.any() else t
sel_data = pd_data[sel_t]

# -------------------
# ticks (auto 10)
# -------------------
time_idx = np.linspace(0, len(sel_t) - 1, 10, dtype=int)
freq_idx = np.linspace(0, len(pd_data.index) - 1, 10, dtype=int)

# -------------------
# plot
# -------------------
plt.figure(figsize=(10, 6))
vmin, vmax = np.nanpercentile(sel_data, [5, 98])
ax = sns.heatmap(
    sel_data,vmin = vmin, vmax = vmax,
    cmap="viridis",
    cbar_kws={"label": "Flux density (arb. unit)"},
    rasterized=True
)

ax.invert_yaxis()

plt.xticks(time_idx, pd.to_datetime(sel_t[time_idx]).time, rotation=30)
plt.yticks(freq_idx, pd_data.index[freq_idx])

ax.set_xlabel("Timestamp (UTC)")
ax.set_ylabel("Frequency (MHz)")

mid = t[len(t)//2]
ax.set_title(
    f"Learmonth dynamic spectrum {mid.day} {calendar.month_name[mid.month]} {mid.year}"
)

plt.tight_layout()
plt.show()
