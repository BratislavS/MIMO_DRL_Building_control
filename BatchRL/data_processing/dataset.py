"""The dataset definitions.

This module defines a class for the dataset type
that was used for storing the data and some additional information.
"""
import os
import pickle
import warnings
from typing import List, Union, Dict, Tuple

import numpy as np

from rest.client import save_dir
from util.numerics import get_shape1, check_in_range, prepare_supervised_control, align_ts, add_mean_and_std, \
    has_duplicates, trf_mean_and_std, cut_data, find_rows_with_nans, find_all_streaks, find_disjoint_streaks, \
    find_longest_streak, int_to_sin_cos, contrary_indices
from util.util import day_offset_ts, ts_per_day, datetime_to_np_datetime, string_to_dt, create_dir, Arr, \
    n_mins_to_np_dt, str_to_np_dt, np_dt_to_str, repl, yeet
from util.visualize import PLOT_DIR, plot_all

dataset_data_path = os.path.join(save_dir, "Datasets")  #: Dataset directory
no_inds = np.array([], dtype=np.int32)  #: Empty index set


class SeriesConstraint:
    """The class defining constraints for single data series.
    """

    #: The allowed constraint type
    allowed_names: List = [None, 'interval', 'exact']
    name: str = None  #: The type of the current instance
    extra_dat: np.ndarray = None  #: The interval for the `interval` type.

    def __init__(self, name: str = None, extra_dat: Union[List, np.ndarray] = None):
        """
        Initialization of the constraint.

        Args:
            name: The string specifying the type of constraint.
            extra_dat: The interval if `name` is 'interval'.
        Raises:
            ValueError: If the string is not one of the defined ones or `extra_dat` is not
                None when `name` is not 'interval'.
        """

        if name not in self.allowed_names:
            raise ValueError("Invalid name!")
        self.name = name
        if name != 'interval':
            if extra_dat is not None:
                raise ValueError(f"What the fuck are you passing?? : {extra_dat}")
        else:
            self.extra_dat = np.array(extra_dat)

    def __getitem__(self, item):
        """
        For backwards compatibility since first a namedtuple
        was used.

        Args:
            item: Either 0 or 1

        Returns:
            The name (0) or the extra data (1).

        Raises:
            IndexError: If item is not 0 or 1.
        """

        if item < 0 or item >= 2:
            raise IndexError("Index out of range!!")
        if item == 0:
            return self.name
        if item == 1:
            return self.extra_dat


# Define type
DatasetConstraints = List[SeriesConstraint]


def check_dataset_part(part_str: str):
    allowed = ["train", "val", "test", "train_val", "all"]
    if part_str not in allowed:
        yeet(f"Invalid dataset part specification: {part_str}")


def check_disjoint(set_1: str, set_2: str):
    """Checks if the two given sets are disjoint."""
    is_dis = True
    if "all" in [set_1, set_2]:
        is_dis = False
    if set_1 in set_2 or set_2 in set_1:
        is_dis = False
    if not is_dis:
        yeet(f"Sets: {set_1} and {set_2} are overlapping!")


class Dataset:
    """This class contains all infos about a given dataset and
    offers some functionality for manipulating it.
    """
    _offs: int
    _day_len: int = None

    # The split data
    split_dict: Dict[str, 'ModelDataView'] = None  #: The saved splits.
    pats_defs: List[Tuple] = None  #: List of tuples specifying the subsets of the data.

    # Basic dataset parameters
    d: int  #: The total number of series in the dataset.
    n_c: int  #: The number of control series.
    n_non_c: int  #: The number of non-control series.
    n: int  #: The number of timesteps in the dataset.

    def __init__(self, all_data: np.ndarray, dt: int, t_init, scaling: np.ndarray,
                 is_scaled: np.ndarray,
                 descs: Union[np.ndarray, List],
                 c_inds: np.ndarray = no_inds,
                 p_inds: np.ndarray = no_inds,
                 name: str = "",
                 seq_len: int = 20):
        """Base constructor."""
        assert np.array_equal(p_inds, no_inds), "Specifying `p_inds` is deprecated!"

        # Constants
        self.val_percent = 0.1
        self.seq_len = seq_len

        # Dimensions
        self.n = all_data.shape[0]
        self.d = get_shape1(all_data)
        self.n_c = c_inds.shape[0]
        self.n_non_c = self.d - self.n_c
        self.n_p = p_inds.shape[0]

        # Check that indices are in range
        if not check_in_range(c_inds, 0, self.d):
            raise ValueError("Control indices out of bound.")
        if not check_in_range(p_inds, 0, self.d):
            raise ValueError("Prediction indices out of bound.")

        # Meta data
        self.name = name
        self.dt = dt
        self.t_init = t_init
        self.is_scaled = is_scaled
        self.scaling = scaling
        self.descriptions = descs
        self.c_inds = c_inds
        self.p_inds = p_inds
        self.non_c_inds = contrary_indices(c_inds, self.d)

        # Full data
        self.data = all_data
        if self.d == 1:
            self.data = np.reshape(self.data, (-1, 1))

        # Variables for later use
        self.streak_len = None
        self.c_inds_prep = None
        self.p_inds_prep = None

    @property
    def fully_scaled(self) -> bool:
        return np.all(self.is_scaled)

    @property
    def partially_scaled(self):
        return not self.fully_scaled and np.any(self.is_scaled)

    def split_data(self) -> None:
        """Splits the data into train, validation and test set.

        Uses `pats_defs` to choose the splits.
        """
        # Get sizes
        n = self.data.shape[0]
        n_test = n_val = n - int((1.0 - self.val_percent) * n)
        n_train = n - n_test - n_val

        # Define parameters for splits
        self.pats_defs = [
            ('train', 0, n_train),
            ('val', n_train, n_val),
            ('train_val', 0, n_train + n_val),
            ('test', n_train + n_val, n_test),
            ('all', 0, n),
        ]

        # Save dict and sizes
        offs = day_offset_ts(self.t_init, self.dt)
        self._offs = offs
        self._day_len = ts_per_day(self.dt)
        self.split_dict = {p[0]: ModelDataView(self, *p) for p in self.pats_defs}

    def check_part(self, part_str: str):
        """Checks if `part_str` is a valid string for part of data specification.

        If the dataset hasn't been split it will be done here.
        Args:
            part_str: The string specifying the part.

        Raises:
            ValueError: If the string is not valid.
        """
        # Split data if it wasn't done before.
        if self.pats_defs is None:
            self.split_data()

        # Check if part exists.
        if part_str not in [s[0] for s in self.pats_defs]:
            raise ValueError(f"{part_str} is not valid for specifying a part of the data!")

    def get_streak(self, str_desc: str, n_days: int = 7,
                   other_len: int = None,
                   use_max_len: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extracts a streak from the selected part of the dataset.

        Args:
            str_desc: Part of the dataset, in ['train', 'val', 'test']
            n_days: Number of days in a streak.
            other_len: If set, returns all streaks of that length, including initial steps.
            use_max_len: Whether to use the longest available streak instead.

        Returns:
            Streak data prepared for supervised training and an offset in timesteps
            to the first element in the streak.
        """
        # Get info about data split
        self.check_part(str_desc)
        mdv = self.split_dict[str_desc]
        n_off = mdv.n
        s_len_curr = n_days * self._day_len if other_len is None else other_len
        s_len_curr += self.seq_len - 1

        # Extract, prepare and return
        sequences, streak_offs = mdv.extract_streak(s_len_curr, use_max_len=use_max_len)
        n_off += streak_offs
        input_data, output_data = prepare_supervised_control(sequences, self.c_inds, False)
        return input_data, output_data, n_off

    def get_split(self, str_desc: str, seq_out: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the required part of the data prepared for the supervised model training.

        Args:
            seq_out: Whether to return sequence output.
            str_desc: The string describing the part of the data,
                one of: ['train', 'val', 'train_val', 'test']

        Returns:
            Data prepared for training.
        """
        # Check split
        self.check_part(str_desc)

        # Get sequences and offsets
        mdv = self.split_dict.get(str_desc)
        sequences = mdv.sequences
        offs = mdv.seq_inds

        # Prepare and return
        input_data, output_data = prepare_supervised_control(sequences, self.c_inds, seq_out)
        return input_data, output_data, offs

    def get_days(self, str_desc: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Extracts all streaks of length one day that start at the beginning of the day.

        Args:
            str_desc: The string specifying what data to use.

        Returns:
            List of prepared 1-day streaks and array of indices with offsets.
        """
        # Get info about data split
        mdv = self.split_dict[str_desc]
        n_off_dis = (mdv.n + self._offs) % self._day_len

        # Extract, prepare and return
        sequences, streak_offs = mdv.extract_disjoint_streaks(self._day_len, n_off_dis)
        in_out_list = [prepare_supervised_control(s, self.c_inds, False) for s in sequences]
        return in_out_list, streak_offs + mdv.n

    @classmethod
    def fromRaw(cls, all_data: np.ndarray, m: List, name: str,
                c_inds: np.ndarray = no_inds,
                p_inds: np.ndarray = no_inds) -> 'Dataset':
        """Constructor from data and metadata dict `m`.

        Extracts the important metadata from the dict m.

        :param all_data: Numpy array with all the time series.
        :param m: List of metadata dictionaries.
        :param name: Name of the data collection.
        :param c_inds: Control indices.
        :param p_inds: Prediction indices.
        :return: Generated Dataset.
        """
        d = all_data.shape[1]

        # Extract data from m
        dt = m[0]['dt']
        t_init = m[0]['t_init']
        is_scaled = np.empty((d,), dtype=np.bool)
        is_scaled.fill(True)
        scaling = np.empty((d, 2), dtype=np.float32)
        descriptions = np.empty((d,), dtype="U100")
        for ct, el in enumerate(m):
            desc = el['description']
            descriptions[ct] = desc
            m_a_s = el.get('mean_and_std')
            if m_a_s is not None:
                scaling[ct, 0] = m_a_s[0]
                scaling[ct, 1] = m_a_s[1]
            else:
                is_scaled[ct] = False

        ret_val = cls(np.copy(all_data), dt, t_init, scaling, is_scaled, descriptions, c_inds, p_inds, name)
        return ret_val

    def __len__(self) -> int:
        """Returns the number of features per sample per timestep.

        Returns:
            Number of features.
        """
        return self.d

    def __str__(self) -> str:
        """Creates a string containing the most important information about this dataset.

        Returns:
            Dataset description string.
        """
        out_str = "Dataset(" + repr(self.data) + ", \ndt = " + repr(self.dt)
        out_str += ", t_init = " + repr(self.t_init) + ", \nis_scaled = " + repr(self.is_scaled)
        out_str += ", \ndescriptions = " + repr(self.descriptions) + ", \nc_inds = " + repr(self.c_inds)
        out_str += ", \np_inds = " + repr(self.p_inds) + ", name = " + str(self.name) + ")"
        return out_str

    @classmethod
    def copy(cls, dataset: 'Dataset') -> 'Dataset':
        """Returns a deep copy of the passed Dataset.

        Args:
            dataset: The dataset to copy.

        Returns:
            The new dataset.
        """
        return cls(np.copy(dataset.data),
                   dataset.dt,
                   dataset.t_init,
                   np.copy(dataset.scaling),
                   np.copy(dataset.is_scaled),
                   np.copy(dataset.descriptions),
                   np.copy(dataset.c_inds),
                   np.copy(dataset.p_inds),
                   dataset.name)

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """Merges dataset other into self.

        Does not commute!

        Args:
            other: The dataset to merge self with.

        Returns:
            A new dataset with the combined data.
        """
        # Check compatibility
        if self.dt != other.dt:
            raise ValueError("Datasets not compatible!")

        # Merge data
        ti1 = self.t_init
        ti2 = other.t_init
        data, t_init = align_ts(self.data, other.data, ti1, ti2, self.dt)

        # Merge metadata
        d = self.d
        scaling = np.concatenate([self.scaling, other.scaling], axis=0)
        is_scaled = np.concatenate([self.is_scaled, other.is_scaled], axis=0)
        descs = np.concatenate([self.descriptions, other.descriptions], axis=0)
        c_inds = np.concatenate([self.c_inds, other.c_inds + d], axis=0)
        p_inds = np.concatenate([self.p_inds, other.p_inds + d], axis=0)
        name = self.name + other.name

        return Dataset(data, self.dt, t_init, scaling, is_scaled, descs, c_inds, p_inds, name)

    def slice_time(self, n1: int, n2: int) -> 'Dataset':
        """Slices the dataset along the time axis.

        Args:
            n1:
            n2:

        Returns:
            New dataset.
        """
        dt_dt = n_mins_to_np_dt(self.dt)
        t_init_dt = str_to_np_dt(self.t_init)
        t_init_dt_new = t_init_dt + dt_dt * n1
        t_init_new = np_dt_to_str(t_init_dt_new)

        ds_out = Dataset.copy(self)
        ds_out.t_init = t_init_new
        ds_out.data = self.data[n1:n2]
        return ds_out

    def add_time(self, sine_cos: bool = True) -> 'Dataset':
        """Adds time to current dataset.

        Args:
            sine_cos: Whether to use sin(t) and cos(t) instead of t directly.

        Returns:
            self + the time dataset.
        """
        dt = self.dt
        t_init = datetime_to_np_datetime(string_to_dt(self.t_init))
        dt_td64 = np.timedelta64(dt, 'm')
        n_tint_per_day = ts_per_day(dt)
        floor_day = np.array([t_init], dtype='datetime64[D]')[0]
        begin_ind = int((t_init - floor_day) / dt_td64)
        dat = np.empty((self.n,), dtype=np.float32)
        for k in range(self.n):
            dat[k] = (begin_ind + k) % n_tint_per_day

        if not sine_cos:
            return self + Dataset(dat,
                                  self.dt,
                                  self.t_init,
                                  np.array([0.0, 1.0]),
                                  np.array([False]),
                                  np.array([f"Time of day [{dt} mins.]"]),
                                  no_inds,
                                  no_inds,
                                  "Time")
        else:
            all_dat = np.empty((self.n, 2), dtype=np.float32)
            all_dat[:, 0], all_dat[:, 1] = int_to_sin_cos(dat, n_tint_per_day)
            return self + Dataset(all_dat,
                                  self.dt,
                                  self.t_init,
                                  np.array([[0.0, 1.0], [0.0, 1.0]]),
                                  np.array([False, False]),
                                  np.array(["sin(Time of day)", "cos(Time of day)"]),
                                  no_inds,
                                  no_inds,
                                  "Time")
        pass

    def _get_slice(self, ind_low: int, ind_high: int) -> 'Dataset':
        """Returns a slice of the dataset.

        Helper function for `__getitem__`.
        Returns a new dataset with the columns
        'ind_low' through 'ind_high'.

        :param ind_low: Lower range index.
        :param ind_high: Upper range index.
        :return: Dataset containing series [ind_low: ind_high) of current dataset.
        """

        # warnings.warn("Prediction and control indices are lost when slicing.")
        low = ind_low

        if ind_low < 0 or ind_high > self.d or ind_low >= ind_high:
            raise ValueError("Slice indices are invalid.")
        if ind_low + 1 != ind_high:
            return Dataset(np.copy(self.data[:, ind_low: ind_high]),
                           self.dt,
                           self.t_init,
                           np.copy(self.scaling[ind_low: ind_high]),
                           np.copy(self.is_scaled[ind_low: ind_high]),
                           np.copy(self.descriptions[ind_low: ind_high]),
                           no_inds,
                           no_inds,
                           f"{self.name}[{ind_low}:{ind_high}]")
        else:
            return Dataset(np.copy(self.data[:, low:low + 1]),
                           self.dt,
                           self.t_init,
                           np.copy(self.scaling[low:low + 1]),
                           np.copy(self.is_scaled[low:low + 1]),
                           np.copy(self.descriptions[low:low + 1]),
                           no_inds,
                           no_inds,
                           f"{self.name}[{low}]")

    def __getitem__(self, key) -> 'Dataset':
        """Allows for slicing.

        Returns a copy not a view.
        Slice must be contiguous, no strides.

        Args:
            key: Specifies which series to return.

        Returns:
            New dataset containing series specified by key.
        """

        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise NotImplementedError("Only implemented for contiguous ranges!")
            return self._get_slice(key.start, key.stop)
        return self._get_slice(key, key + 1)

    def save(self) -> None:
        """Save the class object to a file.

        Uses pickle to store the instance.
        """
        create_dir(dataset_data_path)

        file_name = self.get_filename(self.name, self.dt)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def get_unscaled_data(self) -> np.ndarray:
        """Adds the mean and std back to every column and returns the data.

        Returns:
             Data array with original mean and std added back.
        """
        data_out = np.copy(self.data)
        for k in range(self.d):
            if self.is_scaled[k]:
                data_out[:, k] = add_mean_and_std(data_out[:, k], self.scaling[k, :])
        return data_out

    def scale(self, state: np.ndarray, remove_mean: bool = True,
              state_only: bool = False,
              control_only: bool = False,
              ) -> np.ndarray:
        """Scales the given data according to the own scaling.

        Dataset needs to be scaled, otherwise no scaling info is available
        and an AssertionError will be raised. If `state_only` is true,
        it is assumed that the data does not contain any control series,
        if `control_only` is true, then the data should only contain (all)
        control series.

        Args:
            state: The data to be scaled.
            remove_mean: Whether to remove the mean and the std.
            state_only: Whether only state data is given.
            control_only: Whether only control data is given.

        Returns:
            Scaled data.
        """
        # Test a few properties
        assert self.fully_scaled, f"Dataset not fully scaled!"
        assert not (state_only and control_only), "Cannot have both!"

        # Do the scaling
        sc_used = self.scaling
        inds = self.c_inds if control_only else (self.non_c_inds if state_only else None)
        if inds is not None:
            sc_used = self.scaling[inds]
        sc_pars = (sc_used[:, 0], sc_used[:, 1])
        scaled_state = trf_mean_and_std(state, sc_pars,
                                        remove=remove_mean)
        return scaled_state

    @staticmethod
    def get_filename(name: str, dt: int = 15) -> str:
        """Returns path where to store this `Dataset`."""
        dt_ext = "" if dt == 15 else f"_dt_{dt}"
        return os.path.join(dataset_data_path, name + dt_ext) + '.pkl'

    @staticmethod
    def loadDataset(name: str, dt: int = 15) -> 'Dataset':
        """Load a saved Dataset object.

        Args:
            name: Name of dataset.
            dt:

        Returns:
            Loaded dataset.
        """
        f_name = Dataset.get_filename(name, dt=dt)
        if not os.path.isfile(f_name):
            raise FileNotFoundError(f"Dataset {f_name} does not exist.")
        with open(f_name, 'rb') as f:
            ds = pickle.load(f)
            return ds

    def standardize_col(self, col_ind: int) -> None:
        """Standardizes a certain column of the data.

        Nans are ignored. If the data series is
        already scaled, nothing is done.

        Args:
            col_ind: Index of the column to be standardized.
        Raises:
            IndexError: If `col_ind` is out of range.
        """
        if col_ind >= self.d or col_ind < 0:
            raise IndexError("Column index too big!")
        if self.is_scaled[col_ind]:
            return

        # Do the actual scaling
        m = np.nanmean(self.data[:, col_ind])
        std = np.nanstd(self.data[:, col_ind])
        if std < 1e-10:
            raise ValueError(f"Std of series {col_ind} is almost zero!")
        self.data[:, col_ind] = (self.data[:, col_ind] - m) / std
        self.is_scaled[col_ind] = True
        self.scaling[col_ind] = np.array([m, std])

    def standardize(self) -> None:
        """Standardizes all columns in the data.

        Uses `standardize_col` on each column in the data.
        """
        for k in range(self.d):
            self.standardize_col(k)

    def check_inds(self, inds: np.ndarray, include_c: bool = True, unique: bool = True) -> None:
        """Checks if the in or out indices are in a valid range.

        Args:
            inds: Indices to check.
            include_c: Whether they may include control indices.
            unique: Whether to require unique elements only.

        Raises:
            ValueError: If indices are out of range.
        """
        upper_ind = self.d
        if not include_c:
            upper_ind -= self.n_c
        if not check_in_range(inds, 0, upper_ind):
            raise ValueError("Indices not in valid range!!")
        if unique and has_duplicates(inds):
            raise ValueError("Indices containing duplicates!!!")

    def to_prepared(self, inds: Arr) -> Arr:
        """Converts the indices from the original dataset
        to the indices corresponding to the prepared data.

        Since the control series are moved to the end while
        preparing the data, this is needed.

        Args:
            inds: Original indices.

        Returns:
            New indices.
        """
        new_inds = np.copy(inds)
        n_tot = self.d
        for c_ind in self.c_inds:
            new_inds[inds > c_ind] -= 1
        for ct, c_ind in enumerate(self.c_inds):
            new_inds[inds == c_ind] = n_tot - self.n_c + ct
        return new_inds

    def from_prepared(self, inds: Arr) -> Arr:
        """Converts the indices from the prepared data
        to the indices corresponding to the original dataset.

        Since the control series are moved to the end while
        preparing the data, this is needed.

        Args:
            inds: Data indices.

        Returns:
            Original indices.
        """
        new_inds = np.copy(inds)
        n_tot = self.d
        for c_ind in self.c_inds:
            new_inds[new_inds >= c_ind] += 1
        for ct, c_ind in enumerate(self.c_inds):
            new_inds[inds == n_tot - self.n_c + ct] = c_ind
        return new_inds

    def visualize_nans(self, name_ext: str = "") -> None:
        """Visualizes nans in the data.

        Visualizes where the holes are in the different
        time series (columns) of the data.

        Args:
            name_ext: Name extension.
        """
        nan_plot_dir = os.path.join(PLOT_DIR, "NanPlots")
        create_dir(nan_plot_dir)
        s_name = os.path.join(nan_plot_dir, self.name)
        not_nans = np.logical_not(np.isnan(self.data))
        scaled = not_nans * np.arange(1, 1 + self.d, 1, dtype=np.int32)
        scaled[scaled == 0] = -1
        m = [{'description': d, 'dt': self.dt} for d in self.descriptions]
        plot_all(scaled, m,
                 use_time=False,
                 show=False,
                 title_and_ylab=["Nan plot", "Series"],
                 scale_back=False,
                 save_name=s_name + name_ext)

    def transform_c_list(self, const_list: List[SeriesConstraint], remove_mean: bool = True) -> None:
        """
        Transforms the interval constraints in the sequence of constraints
        to fit the standardized / non-standardized series.

        Args:
            const_list: The list with the constraints for the series.
            remove_mean: Whether to remove or to add the given mean and std.

        Returns:
            None
        """

        if self.d != len(const_list):
            raise ValueError("Constraint List not compatible with dataset.")
        for ct, sc in enumerate(const_list):
            if sc[0] == 'interval':
                if self.is_scaled[ct]:
                    mas = self.scaling[ct]
                    iv = sc[1]
                    iv_trf = trf_mean_and_std(iv, mas, remove_mean)
                    const_list[ct] = SeriesConstraint('interval', iv_trf)

    def get_shifted_t_init(self, n: int) -> str:
        """Shifts t_init of the dataset n timesteps into the future.

        Args:
            n: The number of time steps to shift.

        Returns:
            A new t_init string with the shifted time.
        """
        dt_dt = n_mins_to_np_dt(self.dt)
        np_dt = str_to_np_dt(self.t_init)
        return np_dt_to_str(np_dt + n * dt_dt)

    def get_scaling_mul(self, ind: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Repeats scaling info of a specific series.

        Returns the scaling information of the series with index `orig_ind`
        for a possible new dataset with `n` series.

        Args:
            ind: The series to take the scaling from.
            n: Number of times to repeat scaling info.

        Returns:
            New scaling and new bool array is_scaled.
        """
        scaling = np.array(repl(self.scaling[ind], n))
        is_scd = np.copy(np.array(repl(self.is_scaled[ind], n), dtype=np.bool))
        return scaling, is_scd


class ModelDataView:
    """Container for dataset specifying parts of the original data in the dataset.

    Usable for train, val and test splits.
    """

    _d_ref: Dataset  #: Reference to dataset

    name: str  #: Name of part of data
    n: int  #: Offset in elements wrt the data in `_d_ref`
    n_len: int  #: Number of elements

    # Data for model training
    sequences: np.ndarray  #: 3D array: the relevant data cut into sequences.
    seq_inds: np.ndarray  #: 1D int array: the indices describing the offset to each sequence.

    def __init__(self, d_ref: Dataset, name: str, n_init: int, n_len: int):
        """Constructor.

        Args:
            d_ref: The underlying dataset.
            name: The name of this view, e.g. 'train'.
            n_init: The offset in timesteps.
            n_len: The number of rows.
        """

        # Store parameters
        self._d_ref = d_ref
        self.n = n_init
        self.n_len = n_len
        self.name = name
        self.s_len = d_ref.seq_len

        # Cut the relevant data
        self.sequences, self.seq_inds = self.get_sequences()

    def get_sequences(self, seq_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts sequences of length `seq_len` from the data.

        If `seq_len` is None, the sequence length of the associated
        dataset is used. Uses caching for multiple calls with
        same `seq_len`.

        Args:
            seq_len: The sequence length.

        Returns:
            The sequences and the corresponding indices.
        """
        # Use default if None
        if seq_len is None:
            seq_len = self.s_len

        ret_val = self._get_sequences(seq_len)
        return ret_val

    def _get_sequences(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extract sequences and return
        sequences, seq_inds = cut_data(self.get_rel_data(), seq_len)
        return sequences, seq_inds

    def _get_data(self, n1: int, n2: int) -> np.ndarray:
        """Returns a copy of the data[n1: n2]

        Args:
            n1: First index
            n2: Second index

        Returns:
            Numpy array of data.

        Raises:
            IndexError: If indices are out of bound.
        """
        dat_len = self._d_ref.data.shape[0]
        if n1 < 0 or n2 < 0 or n1 > dat_len or n2 > dat_len or n1 > n2:
            raise IndexError("Invalid indices!")
        return np.copy(self._d_ref.data[n1: n2])

    def get_rel_data(self) -> np.ndarray:
        """Returns the data that this class uses.

        The returned data should not be modified!

        Returns:
            Data array.
        """
        return self._get_data(self.n, self.n + self.n_len)

    def extract_streak(self, n_timesteps: int, take_last: bool = True,
                       use_max_len: bool = False) -> Tuple[np.ndarray, int]:
        """Extracts a streak of length `n_timesteps` from the associated data.

        If `take_last` is True, then the last such streak is returned,
        else the first. Uses caching for multiple calls with same signature.

        Args:
            n_timesteps: The required length of the streak.
            take_last: Whether to use the last possible streak or the first.
            use_max_len: Whether to extract the longest available sequence instead.

        Returns:
            A streak of length `n_timesteps`.
        """
        return self._extract_streak((n_timesteps, take_last, use_max_len))

    def _extract_streak(self, n: Tuple[int, bool, bool]) -> Tuple[np.ndarray, int]:
        # Extract parameters
        n_timesteps, take_last, use_max_len = n

        # Initialize i
        i = -1

        # Find nans and all streaks
        nans = find_rows_with_nans(self.get_rel_data())
        if not use_max_len:
            # Find sequence of specified size, if not found, use longest available one.
            inds = find_all_streaks(nans, n_timesteps)
            if len(inds) < 1:
                warnings.warn(f"No streak of length {n_timesteps} found, using "
                              f"longest available one instead!!")
                use_max_len = True
            i = inds[-1] if take_last else inds[0]

        if use_max_len:
            # Use the longest available sequence
            i, end_ind = find_longest_streak(nans, last=take_last, seq_val=False)
            n_timesteps = end_ind - i

        assert i != -1, "This should never ever happen, " \
                        "there should have been a ValueError before, I " \
                        "seriously fucked up here!"

        # Get the data, cut and return
        data = self.get_rel_data()[i:(i + n_timesteps)]
        ret_data, _ = cut_data(data, self.s_len)
        return ret_data, i

    def extract_disjoint_streaks(self, streak_len: int, n_offs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts disjoint streaks of length `streak_len` + seq_len - 1 and
        turns them into sequences for the use in dynamics models.
        Uses caching for same calls.

        Args:
            streak_len: The length of the disjoint streaks.
            n_offs: The offset in time steps.

        Returns:
            The sequenced streak data and the offset indices for all streaks
            relative to the associated data.
        """
        return self._extract_disjoint_streaks((streak_len, n_offs))

    def _extract_disjoint_streaks(self, n: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """This function does actually do stuff.

        Args:
            n: The arguments in a tuple.

        Returns:
            See `extract_disjoint_streaks`.
        """
        # Extract parameters
        streak_len, n_offs = n

        # Find offset
        rel_data = self.get_rel_data()
        nans = find_rows_with_nans(rel_data)
        dis_streaks = find_disjoint_streaks(nans, self.s_len, streak_len, n_offs)
        tot_len = streak_len + self.s_len - 1
        n_streaks = len(dis_streaks)
        n_feats = get_shape1(rel_data)

        # Put data together and cut it into sequences
        res_dat = np.empty((n_streaks, streak_len, self.s_len, n_feats), dtype=rel_data.dtype)
        for ct, k in enumerate(dis_streaks):
            str_dat = rel_data[k:(k + tot_len)]
            cut_dat, _ = cut_data(str_dat, self.s_len)
            res_dat[ct] = cut_dat
        return res_dat, dis_streaks
