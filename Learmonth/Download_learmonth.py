# Original code: Soham Dey
# Modified: Shilpi Bhunia + ChatGPT (with outdir support)
# keep srs_data.py in the same directory to convert data to pd dataframe
import pandas as pd
import sys, os, numpy as np
from scipy.signal import medfilt
from scipy import interpolate
import subprocess
import argparse


# ----------------------------
# Fill missing values in 1D array using interpolation
# ----------------------------
def fill_nan(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return arr

    try:
        inds = np.arange(arr.shape[0])
        good = np.where(np.isfinite(arr))[0]

        if len(good) == 0:
            return arr

        f = interpolate.interp1d(
            inds[good],
            arr[good],
            bounds_error=False,
            kind="linear",
            fill_value="extrapolate",
        )

        out_arr = np.where(np.isfinite(arr), arr, f(inds))

    except Exception as e:
        print(e)
        out_arr = arr

    return out_arr


# ----------------------------
# Normalize each frequency channel
# ----------------------------
def backsub(data):
    for sb in range(data.shape[0]):
        data[sb, :] = data[sb, :] / np.nanmedian(data[sb, :])
    return data


# ----------------------------
# Convert .srs → .pd
# ----------------------------
def srs_to_pd(srs_file, pd_file, bkg_sub=False, do_flag=True, flag_cal_time=True):

    print("Converting SRS file to pandas datafile...\n")

    try:
        raw_output = subprocess.check_output(
            ["python", "srs_data.py", srs_file],
            universal_newlines=True
        )
        raw_data = eval(raw_output)
    except Exception as e:
        print("Error running srs_data.py:", e)
        return None

    a_band_data = raw_data[0]
    b_band_data = raw_data[1]
    timestamps = pd.to_datetime(raw_data[2], format="%d/%m/%y, %H:%M:%S")

    freqs = list(a_band_data[0].keys()) + list(b_band_data[0].keys())
    freqs = np.round(np.array(freqs), 1)

    x = []
    for i in range(len(a_band_data)):
        row = list(a_band_data[i].values()) + list(b_band_data[i].values())
        x.append(row)

    x = np.array(x).astype("float")

    full_band_data = pd.DataFrame(x, index=timestamps, columns=freqs)
    full_band_data = full_band_data.sort_index(axis=0)
    full_band_data = full_band_data.sort_index(axis=1)

    full_band_data = full_band_data.transpose()

    freq_index = full_band_data.index
    final_data = full_band_data.to_numpy().astype("float")

    # ----------------------------
    # FLAG BAD CHANNELS
    # ----------------------------
    if do_flag:
        final_data[488:499, :] = np.nan
        final_data[524:533, :] = np.nan
        final_data[540:550, :] = np.nan
        final_data[638:642, :] = np.nan
        final_data[119:129, :] = np.nan
        final_data[108:111, :] = np.nan
        final_data[150:160, :] = np.nan
        final_data[197:199, :] = np.nan
        final_data[285:289, :] = np.nan
        final_data[621:632, :] = np.nan

        # calibration spike removal
        if flag_cal_time:
            y = np.nanmedian(final_data, axis=0)

            if not np.all(np.isnan(y)):
                kernel_size = 1001
                if len(y) < kernel_size:
                    kernel_size = len(y) if len(y) % 2 == 1 else len(y) - 1
                if kernel_size < 1:
                    kernel_size = 1

                filtered_y = medfilt(y, kernel_size)
                filtered_y[filtered_y == 0] = np.nan

                c = y / filtered_y
                c_std = np.nanstd(c)

                pos = np.where(c > 1 + (10 * c_std))
                final_data[..., pos] = np.nan

    # ----------------------------
    # Fill NaNs
    # ----------------------------
    for i in range(final_data.shape[1]):
        final_data[:, i] = fill_nan(final_data[:, i])

    if do_flag:
        final_data[780:, :] = np.nan

    if bkg_sub:
        final_data = backsub(final_data)

    full_band_data = pd.DataFrame(final_data, index=freq_index, columns=timestamps)

    # SAVE
    output_path = pd_file + ".pd"
    full_band_data.to_pickle(output_path)

    return output_path


# ----------------------------
# DOWNLOAD FUNCTION (UPDATED)
# ----------------------------
def download_learmonth(start_time="", end_time="", outdir="."):

    if not start_time:
        print("Please provide start time")
        return None

    start_time = pd.to_datetime(start_time)
    datestamp = start_time.date()

    year_stamp = str(datestamp.year)[2:]
    month_stamp = f"{datestamp.month:02d}"
    day_stamp = f"{datestamp.day:02d}"

    file_name = "LM" + year_stamp + month_stamp + day_stamp + ".srs"

    os.makedirs(outdir, exist_ok=True)
    file_path = os.path.join(outdir, file_name)

    if not os.path.exists(file_path):
        print("Downloading data...\n")
        download_link = (
            "https://downloads.sws.bom.gov.au/wdc/wdc_spec/data/learmonth/raw/"
            + year_stamp + "/" + file_name
        )
        os.system(f"wget -O {file_path} {download_link}")

    return file_path


# ----------------------------
# MAIN
# ----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--starttime", required=True)
    parser.add_argument("--endtime", default=None)

    parser.add_argument("--outdir", default=".", help="Output directory")

    parser.add_argument("--background_subtract", action="store_true", default=True)
    parser.add_argument("--no_background_subtract", action="store_false", dest="background_subtract")

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # DOWNLOAD
    srs_file = download_learmonth(
        start_time=args.starttime,
        end_time=args.endtime,
        outdir=outdir
    )

    print("Downloaded:", srs_file)

    base_name = os.path.splitext(os.path.basename(srs_file))[0]
    pd_file = os.path.join(outdir, base_name)

    output_file = pd_file + ".pd"

    # CONVERT
    if (not os.path.exists(output_file)) or args.overwrite:
        output_file = srs_to_pd(
            srs_file,
            pd_file,
            bkg_sub=args.background_subtract,
            do_flag=args.flag,
            flag_cal_time=args.flag_caltime,
        )

    print("Pandas file created:", output_file)


if __name__ == "__main__":
    main()

