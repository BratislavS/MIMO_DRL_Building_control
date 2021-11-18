"""Data preprocessing module.

Used to process data series from the NEST database.
The parameters for the functions need to be chosen heuristically.
"""
from typing import Sequence, Tuple, List, Dict

import numpy as np
import scipy.ndimage

from util.util import Num, floor_datetime_to_min


def clean_data(dat: Tuple, rem_values: Sequence = (),
               n_cons_least: int = 60,
               const_excepts: Sequence = (),
               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Removes all values with a specified value 'rem_val'
    and removes all sequences where there are at
    least 'n_cons_least' consecutive
    values having the exact same value. If the value
    occurring multiple times is in 'const_excepts' then
    it is not removed.

    Args:
        dat: The tuple with the values and the datetimes.
        rem_values: The sequence of values to remove.
        n_cons_least: The minimum number of consecutive constant values.
        const_excepts: The values that are excepted from removing.
        verbose: Whether to print info to console.

    Returns:
        Tuple with values and datetimes without removed entries.
    """
    # Check input
    assert n_cons_least > 0, f"Invalid argument n_cons_least = {n_cons_least}"

    # Extract data
    values, dates = dat
    tot_dat = values.shape[0]

    # Make copy
    new_values = np.copy(values)
    new_dates = np.copy(dates)

    # Initialize
    prev_val = np.nan
    count = 0
    num_occ = 1
    con_streak = False

    # Add cleaned values and dates
    for (v, d) in zip(values, dates):

        if v not in rem_values:

            # Monitor how many times the same value occurred
            if v == prev_val and v not in const_excepts:

                num_occ += 1
                if num_occ == n_cons_least:
                    con_streak = True
                    count -= n_cons_least - 1
            else:
                con_streak = False
                num_occ = 1

            # Add value if it has not occurred too many times
            if not con_streak:
                new_values[count] = v
                new_dates[count] = d
                count += 1
                prev_val = v

        else:
            # Reset streak
            con_streak = False
            num_occ = 1

    # Return clean data
    if verbose:
        print(f"{tot_dat - count} data points removed.")
    assert count > 0, "All data thrown away while cleaning!"
    return new_values[:count], new_dates[:count]


def remove_out_interval(dat: Tuple, interval: Tuple[Num, Num] = (0.0, 100.0)) -> None:
    """Removes values that do not lie within the interval.

    The data in `dat` will be changed, nothing will be returned.

    Args:
        dat: Raw time series tuple (values, dates).
        interval: Interval where the values have to lie within.
    """
    values, dates = dat
    values[values > interval[1]] = np.nan
    values[values < interval[0]] = np.nan


def clip_to_interval(dat: Tuple, interval: Sequence = (0.0, 100.0)) -> None:
    """Clips the values of the time series that are
    out of the interval to lie within.

    The data in `dat` will be changed, nothing will be returned.

    Args:
        dat: Raw time series tuple (values, dates).
        interval: Interval where the values will lie within.
    """
    values, dates = dat
    values[values > interval[1]] = interval[1]
    values[values < interval[0]] = interval[0]


def interpolate_time_series(dat: Tuple, dt_mins: int,
                            lin_ip: bool = False,
                            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates the given time series.

    Produces another one with equidistant timesteps
    and NaNs if values are missing.

    Args:
        dat: Raw time series tuple (values, datetimes).
        dt_mins: The number of minutes in a time interval.
        lin_ip: Whether to use linear interpolation instead of averaging.
        verbose: Verbosity

    Returns:
        The data tuple (values, datetimes) with equidistant
        datetime values.
    """

    # Unpack
    values, dates = dat

    # Datetime of first and last data point
    start_dt = floor_datetime_to_min(dates[0], dt_mins)
    end_dt = floor_datetime_to_min(dates[-1], dt_mins)
    interval = np.timedelta64(dt_mins, 'm')
    n_ts = int((end_dt - start_dt) / interval + 1)
    if verbose:
        print(f"Total: {n_ts} Timesteps")

    # Initialize
    new_values = np.empty((n_ts,), dtype=np.float32)
    new_values.fill(np.nan)
    count = 0
    last_dt = dates[0]
    last_val = values[0]
    curr_val = (last_dt - start_dt) / interval * last_val
    curr_dt = dates[0]
    v = 0.0

    # Loop over data points
    for ct, v in enumerate(values[1:]):
        curr_dt = dates[ct + 1]
        curr_upper_lim = start_dt + (count + 1) * interval
        if curr_dt >= curr_upper_lim:
            if curr_dt <= curr_upper_lim + interval:
                # Next datetime in next interval
                curr_val += (curr_upper_lim - last_dt) / interval * v
                if not lin_ip:
                    new_values[count] = curr_val
                else:
                    new_values[count] = last_val + (v - last_val) * (curr_upper_lim - last_dt) / (curr_dt - last_dt)
                count += 1
                curr_val = (curr_dt - curr_upper_lim) / interval * v
            else:
                # Data missing!
                curr_val += (curr_upper_lim - last_dt) / interval * last_val
                if not lin_ip:
                    new_values[count] = curr_val
                else:
                    new_values[count] = last_val
                count += 1
                n_data_missing = int((curr_dt - curr_upper_lim) / interval)
                if verbose:
                    print(f"Missing {n_data_missing} data points :(")
                for k in range(n_data_missing):
                    new_values[count] = np.nan
                    count += 1
                dt_start_new_iv = curr_dt - curr_upper_lim - n_data_missing * interval
                curr_val = dt_start_new_iv / interval * v

        else:
            # Next datetime still in same interval
            curr_val += (curr_dt - last_dt) / interval * v

        # Update
        last_dt = curr_dt
        last_val = v

    # Add last one
    curr_val += (end_dt + interval - curr_dt) / interval * v
    new_values[count] = curr_val

    # Return
    return new_values, start_dt


def fill_holes_linear_interpolate(time_series: np.ndarray, max_width: int = 1) -> None:
    """Fills the holes of a uniform time series
    with a width up to `max_width`
    by linearly interpolating between the previous and
    next data point.

    Mutates `time_series`, does not return anything.

    Args:
        time_series: The time series that is processed.
        max_width: Sequences of at most that many nans are removed by interpolation.
    """
    # Return if there are no NaNs
    nan_bool = np.isnan(time_series)
    if np.sum(nan_bool) == 0:
        return

    # Neglect NaNs at beginning and end
    non_nans = np.where(nan_bool == 0)[0]
    nan_bool[:non_nans[0]] = False
    nan_bool[non_nans[-1]:] = False

    # Find all indices with NaNs
    all_nans = np.argwhere(nan_bool)

    # Initialize iterators
    ind_ind = 0

    while ind_ind < all_nans.shape[0]:
        s_ind = all_nans[ind_ind][0]
        streak_len = np.where(nan_bool[s_ind:] == 0)[0][0]
        if streak_len <= max_width:

            # Interpolate values
            low_val = time_series[s_ind - 1]
            high_val = time_series[s_ind + streak_len]
            for k in range(streak_len):
                curr_val = low_val * (k + 1) + high_val * (streak_len - k)
                curr_val /= streak_len + 1
                time_series[s_ind + k] = curr_val

        ind_ind += streak_len


def remove_outliers(time_series: np.ndarray,
                    grad_clip: Num = 100.0,
                    clip_int: Sequence = None) -> None:
    """ Removes data points that lie outside
    the specified interval 'clip_int' and ones
    with a gradient larger than grad_clip.

    Mutates `time_series`, does not return anything.

    Args:
        time_series: The time series to process.
        grad_clip: The maximum gradient magnitude.
        clip_int: The interval where the data has to lie within.
    """

    # Helper functions
    def grad_fd(x1, x2):
        if x2 is None or x1 is None:
            return np.nan
        if np.isnan(x1) or np.isnan(x2):
            return np.nan
        return x2 - x1

    def is_outlier(x, x_tm1, x_tp1=None):
        g1 = grad_fd(x_tm1, x)
        g2 = grad_fd(x, x_tp1)
        if np.isnan(g1):
            return True if np.absolute(g2) > 1.5 * grad_clip else False
        if np.isnan(g2):
            return True if np.absolute(g1) > 1.5 * grad_clip else False
        rej = np.absolute(g1) > grad_clip and np.absolute(g2) > grad_clip
        rej = rej and g1 * g2 < 0
        return rej

    def reject_outliers(x, x_tm1, x_tp1=None):
        if is_outlier(x, x_tm1, x_tp1):
            return np.nan
        return x

    # First and last values
    time_series[0] = reject_outliers(time_series[0], time_series[1])
    time_series[-1] = reject_outliers(time_series[-1], time_series[-2])

    # Iterate
    for ct, el in enumerate(time_series[1:-1]):
        if el != np.nan:
            # Remove large gradient outliers
            time_series[ct + 1] = reject_outliers(el,
                                                  time_series[ct + 2],
                                                  time_series[ct])

            # Clip to interval
            if clip_int is not None:
                if el < clip_int[0] or el > clip_int[1]:
                    time_series[ct + 1] = np.nan
    return


def gaussian_filter_ignoring_nans(time_series: np.ndarray,
                                  sigma: float = 2.0) -> np.ndarray:
    """Applies 1-dimensional Gaussian Filtering ignoring occurrences of NaNs.

    From: https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Args:
        time_series: The time series to process.
        sigma: Gaussian filter standard deviation.

    Returns:
        Filtered time series.
    """

    v = time_series.copy()
    v[np.isnan(time_series)] = 0
    vv = scipy.ndimage.filters.gaussian_filter1d(v, sigma=sigma)

    w = 0 * time_series.copy() + 1
    w[np.isnan(time_series)] = 0
    ww = scipy.ndimage.filters.gaussian_filter1d(w, sigma=sigma)

    z = vv / ww
    z[np.isnan(time_series)] = np.nan
    return z


def standardize(data: np.ndarray, m: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """Removes mean and scales std to 1.0 ignoring nans.

    Stores the parameters in the meta information.

    Args:
        data: 2D Numpy array with series as columns.
        m: List of metadata dicts.

    Returns:
        Processed array and modified list of dicts.
    """
    s = data.shape
    n_feat = s[1]

    # Compute Mean and StD ignoring NaNs
    f_mean = np.nanmean(data, axis=0).reshape((1, n_feat))
    f_std = np.nanstd(data, axis=0).reshape((1, n_feat))

    # Process and store info
    proc_data = (data - f_mean) / f_std
    for k in range(n_feat):
        m[k]['mean_and_std'] = [f_mean[0, k], f_std[0, k]]

    return proc_data, m
