from abc import ABC
from functools import wraps
from typing import Tuple, Sequence, Any, List, Callable, Union

import numpy as np
import scipy.optimize

from util.util import datetime_to_np_datetime, string_to_dt, Arr, Num, tot_size, DEFAULT_TRAIN_SET


def _return_or_error(cond: bool, err_msg: str = None) -> bool:
    """Helper function for `check_shape`."""
    if not cond:
        if err_msg is not None:
            raise ValueError(err_msg)
        else:
            return False
    return True


def check_shape(arr: np.ndarray, exp_shape: Tuple[int, ...], err_msg: str = "Shape Mismatch!"):
    """Checks whether the shape of arr agrees with the expected shape `exp_shape`.

    If `err_msg` is not None, an error will be raised if the
    check fails with the given message, otherwise a bool will be returned.
    If `exp_shape` contains integers < 0, the shape can be arbitrary in that dimension.

    Args:
        arr: The array whose shape to check.
        exp_shape: The expected shape.
        err_msg: The message to show if an error should be thrown.

    Returns:
        A bool indicating whether the check failed.

    Raises:
        ValueError: If the message is not None and the check fails.
    """
    s = arr.shape
    check_passed = len(s) == len(exp_shape)
    if err_msg is not None:
        err_msg += f" Expected shape: {exp_shape}, actual shape: {s}!"
    if not _return_or_error(check_passed, err_msg):
        return False
    for n1, n2 in zip(s, exp_shape):
        if n2 >= 0 and n2 != n1:
            check_passed = False
    return _return_or_error(check_passed, err_msg)


# Error metrics
def mse(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Computes the mean square error between two arrays.

    Args:
        arr1: First array.
        arr2: Second array.

    Returns:
        MSE between `arr1` and `arr2`.
    """
    check_shape(arr1, arr2.shape)
    return np.nanmean((arr1 - arr2) ** 2).item()


def mae(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Computes the mean absolute error between two arrays.

    Args:
        arr1: First array.
        arr2: Second array.

    Returns:
        MAE between `arr1` and `arr2`.
    """
    check_shape(arr1, arr2.shape)
    return np.nanmean(np.abs(arr1 - arr2)).item()


def max_abs_err(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Computes the maximum absolute error between two arrays.

    Args:
        arr1: First array.
        arr2: Second array.

    Returns:
        Max absolute error between `arr1` and `arr2`.
    """
    check_shape(arr1, arr2.shape)
    return np.nanmax(np.abs(arr1 - arr2)).item()


Metric = Callable[[np.ndarray, np.ndarray], float]
UnitTrafo = Callable[[str], str]
ErrScaling = Callable[[Arr, float], Arr]


def lin_scaling(errs: Arr, std: float) -> Arr:
    return std * errs


def pow_scaling(power: int = 2) -> ErrScaling:

    @wraps(ErrMetric.scaling_fun)
    def scaling(errs: Arr, std: float) -> Arr:
        return (std ** power) * errs

    return scaling


def unit_trf_id(unit: str) -> str:
    return unit


def unit_trf_pow(p: int = 2) -> UnitTrafo:

    @wraps(ErrMetric.unit_trf)
    def unit_trf(unit: str) -> str:
        return f"({unit})^{p}"

    return unit_trf


class ErrMetric(ABC):
    """This is the interface for an error metric."""
    name: str  #: Short name of the metric.
    long_name: str = None  #: Long name of the metric.
    err_fun: Metric  #: Function that computes the metric.
    scaling_fun: ErrScaling = lin_scaling
    unit_trf: UnitTrafo = unit_trf_id


class MSE(ErrMetric):
    """MSE error metric."""
    name = "MSE"
    long_name = "Mean Squared Error"
    err_fun = mse
    scaling_fun = pow_scaling(2)
    unit_trf = unit_trf_pow(2)


class MAE(ErrMetric):
    """MAE error metric."""
    name = "MAE"
    long_name = "Mean Absolute Error"
    err_fun = mae


class MaxAbsEer(ErrMetric):
    """Maximum Absolute Error error metric."""
    name = "Max. abs. err."
    long_name = "Maximum Absolute Error"
    err_fun = max_abs_err


def get_metrics_eval_save_name_list(parts: List[str], dt: int,
                                    train_set: str = DEFAULT_TRAIN_SET) -> List[str]:
    """Defines the filenames for performance evaluation.

    Args:
        parts: The list with the strings specifying the parts of the dataset.
        dt: The number of minutes in a timestep.
        train_set: The part of the data the model was trained on.

    Returns:
        A list with the filenames.
    """
    train_set_ext = "" if train_set == DEFAULT_TRAIN_SET else f"_TS_{train_set}"
    ext_list = ["Inds"] + [s.capitalize() for s in parts]
    save_names = [f"Perf_{e}{train_set_ext}_dt_{dt}.txt" for e in ext_list]
    return save_names


def save_performance_extended(perf_arr: np.ndarray,
                              n_steps: Sequence[int],
                              file_names: List[str],
                              metric_names: List[str]):
    """Saves the performance of a model to a file."""
    # Check input
    n_data_parts = len(file_names) - 1
    n_metrics = len(metric_names)
    n_n_steps = len(n_steps)
    assert n_n_steps > 0 and n_metrics > 0 and n_data_parts > 0, "No data to save!"
    exp_shape = (n_data_parts, -1, n_metrics, n_n_steps)
    check_shape(perf_arr, exp_shape, err_msg="Incorrect shape!")
    n_series = perf_arr.shape[1]

    # Save n steps info
    step_data = np.array(n_steps, dtype=np.int32)
    np.savetxt(file_names[0], step_data[:], fmt="%i")

    # Save all other files
    for ct, k in enumerate(file_names[1:]):

        f_name = file_names[ct + 1]

        with open(f_name, 'w') as f:
            for m_ct, m_name in enumerate(metric_names):
                f.write(f"# Metric: {m_name}\n")

                for i in range(n_series):
                    curr_line_array = perf_arr[ct, i, m_ct, :]
                    a_str = np.array2string(curr_line_array, precision=7, separator=', ')
                    f.write(f"{a_str[1:-1]}\n")


def load_performance(path_gen_func, parts: List[str], dt: int, n_metrics: int,
                     fit_data: str = DEFAULT_TRAIN_SET) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data in the files generated by performance evaluation."""

    # Find relevant files
    n_other = len(parts)
    assert n_other > 0, "No parts specified!"
    files = get_metrics_eval_save_name_list(parts, dt, fit_data)
    f_paths = [path_gen_func(f) for f in files]

    # Load indices
    inds = np.genfromtxt(f_paths[0])

    # Load first file to get shape
    first_dat = np.genfromtxt(f_paths[1], delimiter=", ")
    s = first_dat.shape
    s0, s1 = s
    assert s0 % n_metrics == 0, "Shape mismatch!"
    assert s1 == len(inds), "Shape mismatch!"
    n_series = s0 // n_metrics

    # Initialize and fill full data
    eval_data = npf32((n_other, n_series, n_metrics, s[1]))
    for k in range(n_other):
        gft = np.genfromtxt(f_paths[k + 1], delimiter=", ")
        for i in range(n_metrics):
            eval_data[k, :, i] = gft[i * n_series: (i + 1) * n_series]

    # Return data and indices
    return eval_data, inds


def find_inds(in_inds: np.ndarray, out_inds: np.ndarray) -> np.ndarray:
    """Finds the positions of the values in `out_inds` in `in_inds`.

    Assuming no duplicate values in both arrays
    and `out_inds` must be a subset of the values in `in_inds`!

    Args:
        in_inds: Input indices (1d integer numpy array)
        out_inds: Output indices (1d integer numpy array)

    Returns:
        The positions of the values from the output indices in the input indices.
    """
    assert len(in_inds) >= len(out_inds), "Incompatible indices"
    orig_indices = in_inds.argsort()
    ndx = orig_indices[np.searchsorted(in_inds[orig_indices], out_inds)]
    return ndx


def num_nans(arr: np.ndarray) -> int:
    """Counts the number of nans in an array."""
    return np.sum(np.isnan(arr)).item()


def npf32(sh: Tuple, fill: float = None) -> np.ndarray:
    """Returns an empty numpy float32 array of specified shape.

    If `fill` is not None, the array will be filled with that
    value.

    Args:
        sh: Shape of the output array.
        fill: Value to fill the array, empty if None.

    Returns:
        The new float32 array.
    """
    empty_arr = np.empty(sh, dtype=np.float32)
    if fill is not None:
        empty_arr.fill(fill)
    return empty_arr


def has_duplicates(arr: np.ndarray) -> bool:
    """Checks if `arr` contains duplicates.

    Returns true if arr contains duplicate values else False.

    Args:
        arr: Array to check for duplicates.

    Returns:
        Whether it has duplicates.
    """
    m = np.zeros_like(arr, dtype=bool)
    m[np.unique(arr, return_index=True)[1]] = True
    return np.sum(~m) > 0


def fit_linear_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray = None):
    """Fit a linear model y = c * x + m.

    Returns coefficients m and c. If x_new
    is not None, returns the evaluated linear
    fit at x_new.

    Args:
        x_new: Where to evaluate the fitted model.
        y: Y values.
        x: X values.

    Returns:
        The parameters m and c if `x_new` is None, else
        the model evaluated at `x_new`.
    """
    n = x.shape[0]
    ls_mat = np.empty((n, 2), dtype=np.float32)
    ls_mat[:, 0] = 1
    ls_mat[:, 1] = x
    m, c = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    if x_new is None:
        return m, c
    else:
        return c * x_new + m


def fit_linear_bf_1d(x: np.ndarray, y: np.ndarray, b_fun: Callable, offset: bool = False) -> np.ndarray:
    """Fits a linear model y = alpha^T f(x).

    Args:
        x: The values on the x axis.
        y: The values to fit corresponding to x.
        b_fun: A function evaluating all basis function at the input.
        offset: Whether to add a bias term.

    Returns:
        The fitted linear parameters.
    """

    if offset:
        raise NotImplementedError("Not implemented with offset.")

    # Get shapes
    # dummy = b_fun(0.0)
    # d = dummy.shape[0]
    # n = x.shape[0]

    # Fill matrix
    # ls_mat = np.empty((n, d), dtype=np.float32)
    # for ct, x_el in enumerate(x):
    #     ls_mat[ct, :] = b_fun(x_el)
    ls_mat = b_fun(x)

    # Solve and return
    coeffs = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    return coeffs


def get_shape1(arr: np.ndarray) -> int:
    """Save version of `arr`.shape[1]

    Returns:
        The shape of the second dimension
        of an array. If it is a vector returns 1.
    """
    s = arr.shape
    if len(s) < 2:
        return 1
    else:
        return s[1]


def align_ts(ts_1: np.ndarray, ts_2: np.ndarray, t_init1: str, t_init2: str, dt: int) -> Tuple[np.ndarray, str]:
    """Aligns two time series.

    Aligns the two time series with given initial time
    and constant timestep by padding by np.nan.

    Args:
        ts_1: First time series.
        ts_2: Second time series.
        t_init1: Initial time string of series 1.
        t_init2: Initial time string of series 2.
        dt: Number of minutes in a timestep.

    Returns:
        The combined data array and the new initial time string.
    """

    # Get shapes
    n_1 = ts_1.shape[0]
    n_2 = ts_2.shape[0]
    d_1 = get_shape1(ts_1)
    d_2 = get_shape1(ts_2)

    # Compute relative offset
    interval = np.timedelta64(dt, 'm')
    ti1 = datetime_to_np_datetime(string_to_dt(t_init2))
    ti2 = datetime_to_np_datetime(string_to_dt(t_init1))

    # Bug-fix: Ugly, but working :P
    if ti1 < ti2:
        d_out, t = align_ts(ts_2, ts_1, t_init2, t_init1, dt)
        d_out_real = np.copy(d_out)
        d_out_real[:, :d_1] = d_out[:, d_2:]
        d_out_real[:, d_1:] = d_out[:, :d_2]
        return d_out_real, t

    offset = np.int(np.round((ti2 - ti1) / interval))

    # Compute length
    out_len = np.maximum(n_2 - offset, n_1)
    start_s = offset <= 0
    out_len += offset if not start_s else 0
    out = np.empty((out_len, d_1 + d_2), dtype=ts_1.dtype)
    out.fill(np.nan)

    # Copy over
    t1_res = np.reshape(ts_1, (n_1, d_1))
    t2_res = np.reshape(ts_2, (n_2, d_2))
    if not start_s:
        out[:n_2, :d_2] = t2_res
        out[offset:(offset + n_1), d_2:] = t1_res
        t_init_out = t_init2
    else:
        out[:n_1, :d_1] = t1_res
        out[-offset:(-offset + n_2), d_1:] = t2_res
        t_init_out = t_init1

    return out, t_init_out


def trf_mean_and_std(ts: Arr, mean_and_std: Sequence, remove: bool = True) -> Arr:
    """Adds or removes given  mean and std from time series.

    Args:
        ts: The time series.
        mean_and_std: Mean and standard deviance.
        remove: Whether to remove or to add.

    Returns:
        New time series with mean and std removed or added.
    """
    f = rem_mean_and_std if remove else add_mean_and_std
    return f(ts, mean_and_std)


def add_mean_and_std(ts: Arr, mean_and_std: Sequence) -> Arr:
    """Transforms the data back to having mean and std as specified.

    Args:
        ts: The series to add mean and std.
        mean_and_std: The mean and the std.

    Returns:
        New scaled series.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    return ts * mean_and_std[1] + mean_and_std[0]


def rem_mean_and_std(ts: Arr, mean_and_std: Sequence) -> Arr:
    """Whitens the data with known mean and standard deviation.

    Args:
        ts: Data to be whitened.
        mean_and_std: Container of the mean and the std.

    Returns:
        Whitened data.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    if np.any(mean_and_std[1] == 0):
        raise ZeroDivisionError("Standard deviation cannot be 0")
    return (ts - mean_and_std[0]) / mean_and_std[1]


def check_in_range(arr: np.ndarray, low: Num, high: Num) -> bool:
    """Checks if elements of an array are in specified range.

    Returns:
        True if all elements in arr are in
        range [low, high) else false.
    """
    if arr.size == 0:
        return True
    return np.max(arr) < high and np.min(arr) >= low


def split_arr(arr: np.ndarray, frac2: float) -> Tuple[Any, Any, int]:
    """Splits an array along the first axis.

    In the second part a fraction of 'frac2' elements
    is contained.

    Args:
        arr: The array to split into two parts.
        frac2: The fraction of data contained in the second part.

    Returns:
        Both parts of the array and the index where the second part
        starts relative to the whole array.
    """
    n: int = arr.shape[0]
    n_1: int = int((1.0 - frac2) * n)
    return arr[:n_1], arr[n_1:], n_1


def copy_arr_list(arr_list: Sequence[Arr]) -> Sequence[Arr]:
    """Copies a list of numpy arrays.

    Args:
        arr_list: The sequence of numpy arrays.

    Returns:
        A list with all the copied elements.
    """
    copied_arr_list = [np.copy(a) for a in arr_list]
    return copied_arr_list


def contrary_indices(inds: np.ndarray, tot_len: int = None) -> np.ndarray:
    """Returns all indices that are not in `inds`.

    If `inds` is empty, `tot_len` cannot be None.

    Args:
        inds: The original indices.
        tot_len: The highest possible index, max of `inds` if None.

    Returns:
        New indices, 1d int array.
    """
    if tot_len is None:
        assert len(inds) > 0
        tot_len = np.max(inds)
    else:
        arr_max = 0 if inds.size == 0 else np.max(inds)
        assert arr_max < tot_len

    all_inds = np.ones((tot_len, ), dtype=np.bool)
    all_inds[inds] = False

    return np.where(all_inds)[0]


def solve_ls(a_mat: np.ndarray, b: np.ndarray, offset: bool = False,
             non_neg: bool = False,
             ret_fit: bool = False):
    """Solves the least squares problem min_x ||Ax = b||.

    If offset is true, then a bias term is added.
    If non_neg is true, then the regression coefficients are
    constrained to be positive.
    If ret_fit is true, then a tuple (skl_mod, fit_values)
    is returned.

    Args:
        a_mat: The system matrix.
        b: The RHS vector.
        offset: Whether to include an offset.
        non_neg: Whether to use non-negative regression.
        ret_fit: Whether to additionally return the fitted values.

    Returns:
        The fitted parameters and optionally the fitted values.
    """

    # Choose least squares solver
    def ls_fun(a_mat_temp, b_temp):
        if non_neg:
            return scipy.optimize.nnls(a_mat_temp, b_temp)[0]
        else:
            return np.linalg.lstsq(a_mat_temp, b_temp, rcond=None)[0]

    n, m = a_mat.shape
    if offset:
        # Add a bias regression term
        a_mat_off = np.empty((n, m + 1), dtype=a_mat.dtype)
        a_mat_off[:, 0] = 1.0
        a_mat_off[:, 1:] = a_mat
        a_mat = a_mat_off

    ret_val = ls_fun(a_mat, b)
    if ret_fit:
        # Add fitted values to return value
        fit_values = np.matmul(a_mat, ret_val)
        ret_val = (ret_val, fit_values)

    return ret_val


def int_to_sin_cos(inds: Arr, tot_n_ts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the sin and cos of the time indices.

    Args:
        inds: Indices specifying timesteps.
        tot_n_ts: Total number of timesteps in a day.

    Returns:
        Periodic sin and cos.
    """
    args = 2 * np.pi * inds / tot_n_ts
    return np.sin(args), np.cos(args)


def make_periodic(arr_1d: np.ndarray, keep_start: bool = True,
                  keep_min: bool = True) -> np.ndarray:
    """Makes a data series periodic.

    By scaling it by a linearly in- / decreasing
    factor to in- / decrease the values towards the end of the series to match
    the start.

    Args:
        arr_1d: Series to make periodic.
        keep_start: Whether to keep the beginning of the series fixed.
            If False keeps the end of the series fixed.
        keep_min: Whether to keep the minimum at the same level.

    Returns:
        The periodic series.
    """
    n = len(arr_1d)
    if not keep_start:
        raise NotImplementedError("Fucking do it already!")
    if n < 2:
        raise ValueError("Too small fucking array!!")
    first = arr_1d[0]
    last = arr_1d[-1]
    d_last = last - arr_1d[-2]
    min_val = 0.0 if not keep_min else np.min(arr_1d)
    first_offs = first - min_val
    last_offs = last + d_last - min_val
    fac = first_offs / last_offs
    arr_01 = np.arange(n) / (n - 1)
    f = 1.0 * np.flip(arr_01) + fac * arr_01
    return (arr_1d - min_val) * f + min_val


def remove_nan_rows(arr_1d: np.ndarray, arr_list: Sequence[np.ndarray] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Removes nan entries from 1d array `arr_1d`.

    If additional arrays are specified, the same
    rows are removed from them.

    Args:
        arr_1d: Array with nans to be removed.
        arr_list: List with arrays with same elements to be removed.

    Returns:
        Either a single array, if `arr_list` is None, or else
        an array and a list of arrays.
    """
    check_shape(arr_1d, (-1,))
    non_nan_inds = np.where(np.logical_not(np.isnan(arr_1d)))
    arr_clean = arr_1d[non_nan_inds]
    if arr_list is None:
        return arr_clean
    arr_clean_list = [a[non_nan_inds] for a in arr_list]
    return arr_clean, arr_clean_list


def find_sequence_inds(arr: np.ndarray, include_ends: bool = True) -> np.ndarray:
    """Finds all indices where the value in the 1d array `arr` changed.

    Compared to the previous value.

    Args:
        arr: The 1D array.
        include_ends: Whether to include 0 and len(arr) in the result.

    Returns:
        Numpy index array, 1d.
    """
    check_shape(arr, (-1,))
    n = len(arr)
    inds = np.where((arr[1:] - arr[:-1]) != 0.0)[0] + 1
    if include_ends:
        inds = np.concatenate(([0], inds, [n]))
    return inds


def check_dim(a: np.ndarray, n: int) -> bool:
    """Check whether a is n-dimensional.

    Args:
        a: Numpy array.
        n: Number of dimensions.

    Returns:
        True if a is n-dim else False
    """
    return len(a.shape) == n


def nan_avg_between(t_stamps: np.ndarray, val_arr: np.ndarray, n_mins: int = 15) -> np.ndarray:
    """Computes the average of all values in `val_arr` from previous `n_mins` minutes.

    Args:
        t_stamps: The timestamps corresponding to the values.
        val_arr: The values to average.
        n_mins: The number of minutes in an interval

    Returns:
        Averaged values.
    """
    assert t_stamps.shape[0] == val_arr.shape[0], \
        f"Shape mismatch between {t_stamps} and {val_arr}"
    now = np.datetime64('now')
    prev = now - np.timedelta64(n_mins, 'm')
    read_vals = val_arr[np.logical_and(prev < t_stamps, t_stamps <= now)]
    mean_read_vals = np.nanmean(read_vals, axis=0)
    return mean_read_vals


def find_rows_with_nans(all_data: np.ndarray) -> np.ndarray:
    """Finds nans in the data.

    Returns a boolean vector indicating which
    rows of 'all_dat' contain NaNs.

    Args:
        all_data: 2d numpy array with data series as columns.

    Returns:
        1D array of bool specifying rows containing nans.

    Raises:
        ValueError: If the provided array is not 2d, or has size 0.
    """
    # Check shape of data
    sh = all_data.shape
    if not check_dim(all_data, 2) or tot_size(sh) == 0:
        raise ValueError("Invalid data")

    # Initialize
    n, m = sh
    row_has_nan = np.empty((n,), dtype=np.bool).fill(False)

    # Apply or over all columns
    for k in range(m):
        row_has_nan = np.logical_or(row_has_nan, np.isnan(all_data[:, k]))

    return row_has_nan


def extract_streak(all_data: np.ndarray, s_len: int, lag: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extracts a streak where all data is available.

    Finds the last sequence where all data is available
    for at least `s_len` + `lag` timesteps. Then splits the
    data before that last sequence and returns both parts.

    Args:
        all_data: The data.
        s_len: The sequence length.
        lag: The number of sequences the streak should contain.

    Returns:
        The data before the streak, the streak data and the index
        pointing to the start of the streak data.

    Raises:
        IndexError: If there is no streak of specified length found.
    """
    tot_s_len = s_len + lag

    # Find last sequence of length tot_s_len
    inds = find_all_streaks(find_rows_with_nans(all_data), tot_s_len)
    if len(inds) < 1:
        raise IndexError(f"No fucking streak of length {tot_s_len} found!!!")
    last_seq_start = inds[-1]

    # Extract
    first_dat = all_data[:last_seq_start, :]
    streak_dat = all_data[last_seq_start:(last_seq_start + tot_s_len), :]
    return first_dat, streak_dat, last_seq_start + lag


def nan_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Analog for np.array_equal but this time ignoring nans.

    Args:
        a: First array.
        b: Second array to compare.

    Returns:
        True if the arrays contain the exact same elements else False.
    """
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def find_all_streaks(col_has_nan: np.ndarray, s_len: int) -> np.ndarray:
    """Finds all streak of length `s_len`.

    Finds all sequences of length `s_len` where `col_has_nan`
    is never False. Then returns all indices of the start of
    these sequences in `col_has_nan`.

    Args:
        col_has_nan: Bool vector specifying where nans are.
        s_len: The length of the sequences.

    Returns:
        Index vector specifying the start of the sequences.
    """
    # Define True filter
    true_seq = np.empty((s_len,), dtype=np.int32)
    true_seq.fill(1)

    # Find sequences of length s_len
    tmp = np.convolve(np.logical_not(col_has_nan), true_seq, 'valid')
    inds = np.where(tmp == s_len)[0]
    return inds


def find_longest_streak(a1: np.ndarray,
                        last: bool = True,
                        seq_val: int = 1) -> Tuple[int, int]:
    """Finds the longest sequence of True in the bool array `a1`.

    From: https://stackoverflow.com/questions/38161606/find-the-start-position-of-the-longest-sequence-of-1s
    And: https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum/17568803 # noqa

    Args:
        a1: The 1d boolean array to analyze.
        last: Whether to return the last of the longest sequences
            when multiple occur.
        seq_val: Value of which the sequence is made off.

    Returns:
        The start index and the length of the sequence.

    Raises:
        ValueError: If there is no occurrence of `seq_val` in `a1`.
    """
    # Decide whether to use the last or the first occurrence
    max_ind = -1 if last else 0

    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False], a1 == seq_val, [False]))))[0].reshape(-1, 2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    lengths = np.diff(idx_pairs, axis=1).reshape((-1,))
    all_argmax = np.argwhere(lengths == np.amax(lengths)).reshape((-1,))
    start_seq, end_seq = idx_pairs[all_argmax][max_ind]

    return start_seq, end_seq


def cut_data(all_data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the data into sequences.

    Cuts the data contained in `all_data` into sequences of length `seq_len` where
    there are no nans in the row of `all_data`.
    Also returns the indices where to find the sequences in `all_data`.

    Args:
        all_data: The 2D numpy array containing the data series.
        seq_len: The length of the sequences to extract.

    Returns:
        3D array with sequences and 1D int array with indices where the
        sequences start with respect to `all_data`.
    """
    # Check input
    if len(all_data.shape) > 2:
        raise ValueError("Data array has too many dimensions!")

    # Find sequences
    nans = find_rows_with_nans(all_data)
    all_inds = find_all_streaks(nans, seq_len)

    # Get shapes
    n_seqs = len(all_inds)
    n_feat = get_shape1(all_data)

    # Extract sequences
    out_dat = np.empty((n_seqs, seq_len, n_feat), dtype=np.float32)
    for ct, k in enumerate(all_inds):
        out_dat[ct] = all_data[k:(k + seq_len)]
    return out_dat, all_inds


def find_disjoint_streaks(nans: np.ndarray, seq_len: int, streak_len: int,
                          n_ts_offs: int = 0) -> np.ndarray:
    """Finds streaks that are only overlapping by `seq_len` - 1 steps.

    They will be a multiple of `streak_len` from each other relative
    to the `nans` vector.

    Args:
        nans: Boolean array indicating that there is a nan if entry is true.
        seq_len: The required sequence length.
        streak_len: The length of the streak that is disjoint.
        n_ts_offs: The number of timesteps that the start is offset.

    Returns:
        Indices pointing to the start of the disjoint streaks.
    """
    n = len(nans)
    tot_len = streak_len + seq_len - 1
    start = (n_ts_offs - seq_len + 1 + streak_len) % streak_len
    n_max = (n - start) // streak_len
    inds = np.empty((n_max,), dtype=np.int32)
    ct = 0
    for k in range(n_max):
        k_start = start + k * streak_len
        curr_dat = nans[k_start: (k_start + tot_len)]
        if not np.any(curr_dat):
            inds[ct] = k_start
            ct += 1
    return inds[:ct]


def move_inds_to_back(arr: np.ndarray, inds) -> np.ndarray:
    """Moves the series specified by the `inds` to the end of the array.

    Args:
        arr: The array to transform.
        inds: The indices specifying the series to move.

    Returns:
        New array with permuted features.
    """
    n_feat = arr.shape[-1]
    mask = np.ones((n_feat,), np.bool)
    mask[inds] = False
    input_dat = np.concatenate([arr[..., mask], arr[..., inds]], axis=-1)
    return input_dat


def prepare_supervised_control(sequences: np.ndarray,
                               c_inds: np.array,
                               sequence_pred: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for supervised learning.

    Transforms a batch of sequences of constant length to
    prepare it for supervised model training. Moves control series
    to the end of the features and shifts them to the past for one
    time step.

    Args:
        sequences: Batch of sequences of same length.
        c_inds: Indices determining the features to control.
        sequence_pred: Whether to use sequence output.

    Returns:
        The prepared input and output data.
    """
    n_feat = sequences.shape[-1]

    # Get inverse mask
    mask = np.ones((n_feat,), np.bool)
    mask[c_inds] = False

    # Extract and concatenate input data
    arr_list = [sequences[:, :-1, mask],
                sequences[:, 1:, c_inds]]
    input_dat = np.concatenate(arr_list, axis=-1)

    # Extract output data
    if not sequence_pred:
        output_data = sequences[:, -1, mask]
    else:
        output_data = sequences[:, 1:, mask]

    # Return
    return input_dat, output_data
